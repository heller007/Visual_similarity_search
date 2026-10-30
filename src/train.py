"""Train the CIFAR-100 metric-learning baseline or optimizer comparison."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import get_loaders, get_splits, get_val_triplets
from src.loss import TripletLoss
from src.model import ResNet18
from src.optim import Adam, NAdam, SGD, lr_at
from src.retrieval import (
    batch_hard_metrics,
    classification_metrics,
    embed,
    retrieval_metrics,
)
from src.utils import load_checkpoint, resolve_device, seed_all


@torch.no_grad()
def fixed_triplet_accuracy(embeddings, splits, triplets):
    global_to_local = np.full(max(splits["val"]) + 1, -1, dtype=np.int64)
    global_to_local[splits["val"]] = np.arange(len(splits["val"]))
    indices = global_to_local[triplets]
    if (indices < 0).any():
        raise ValueError("Validation triplets contain an index outside the val split.")
    indices = torch.as_tensor(indices, device=embeddings.device)
    dtype = torch.float64 if embeddings.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        anchor = embeddings[indices[:, 0]].to(dtype)
        positive = embeddings[indices[:, 1]].to(dtype)
        negative = embeddings[indices[:, 2]].to(dtype)
        d_ap = (anchor - positive).square().sum(1)
        d_an = (anchor - negative).square().sum(1)
    return (d_ap < d_an).float().mean().item()


def make_optimizer(name, params, lr, weight_decay):
    if name == "adam":
        return Adam(params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=weight_decay)
    if name == "nadam":
        return NAdam(params, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=weight_decay)
    return SGD(params, lr, momentum=0.9, weight_decay=weight_decay)


def save_checkpoint(path, model, optimizer, epoch, step, config, metrics):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "config": config,
        "metrics": metrics,
    }, path)


def train_run(name, optimizer_name, lr, args, device):
    seed_all(args.seed)
    config = {
        "stage": "metric_learning",
        "seed": args.seed, "optimizer": optimizer_name, "lr": lr,
        "beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "momentum": 0.9,
        "weight_decay": args.weight_decay, "batch_size": args.P * args.K,
        "P": args.P, "K": args.K, "margin": args.margin,
        "embedding_dim": args.embedding_dim,
        "mining": "batch_all_then_batch_hard",
        "classification_loss_weight": args.classification_weight,
        "initialization_checkpoint": args.init_checkpoint,
        "batch_all_epochs": args.batch_all_epochs,
        "epochs": args.epochs, "warmup_epochs": 2, "schedule": "cosine to 1%",
        "precision": (
            "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
        ),
        "device": str(device),
    }
    run_dir = Path(args.output_root) / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=name, config=config,
            dir=str(run_dir),
        )

    download = not args.no_download
    splits = get_splits(args.seed, args.data_root, args.split_dir, download=download)
    triplets = get_val_triplets(
        args.seed, 10_000, args.data_root, args.split_dir, download=download
    )
    loaders = get_loaders(
        seed=args.seed, data_root=args.data_root, split_dir=args.split_dir,
        P=args.P, K=args.K, batch_size=args.P * args.K, num_workers=args.num_workers,
        download=download, steps_per_epoch=args.steps_per_epoch,
        pin_memory=device.type == "cuda",
    )
    model = ResNet18(args.embedding_dim).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.set_float32_matmul_precision("high")
    if args.init_checkpoint:
        initialization = load_checkpoint(args.init_checkpoint, device="cpu")
        model.load_state_dict(initialization["model"])
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr, args.weight_decay)
    criterion = TripletLoss(args.margin, "batch_all")
    warmup_steps = 2 * len(loaders["train"])
    total_steps = args.epochs * len(loaders["train"])
    step, best_map = 0, -1.0
    curve_path = run_dir / "metrics.csv"
    fields = [
        "epoch", "lr", "mining", "train_loss", "train_d_ap", "train_d_an",
        "active_fraction", "mean_pairwise_distance",
        "val_triplet_acc", "val_batch_hard_acc", "val_batch_hard_d_ap",
        "val_batch_hard_d_an", "val_r1", "val_map",
        "val_class_top1", "val_class_top5",
    ]
    previous_pairwise = None

    with curve_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(args.epochs):
            model.train()
            sampler = loaders["train"].batch_sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            criterion.mining = (
                "batch_all" if epoch < args.batch_all_epochs else "batch_hard"
            )
            mining = criterion.mining
            totals = {
                key: 0.0
                for key in ("loss", "d_ap", "d_an", "active", "pairwise")
            }
            examples, last_lr = 0, lr

            for images, labels in loaders["train"]:
                optimizer.lr = last_lr = lr_at(
                    step, base_lr=lr, warmup_steps=warmup_steps, total_steps=total_steps
                )
                images, labels = images.to(device, non_blocking=True), labels.to(device)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad()
                amp_enabled = device.type == "cuda" and torch.cuda.is_bf16_supported()
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=amp_enabled,
                ):
                    embeddings, logits = model.forward_with_logits(images)
                    metric_loss, stats = criterion(embeddings, labels)
                    classification_loss = F.cross_entropy(
                        logits, labels, label_smoothing=0.05
                    )
                    loss = metric_loss + args.classification_weight * classification_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
                count = len(labels)
                examples += count
                totals["loss"] += loss.item() * count
                totals["d_ap"] += stats["mean_d_ap"].item() * count
                totals["d_an"] += stats["mean_d_an"].item() * count
                totals["active"] += stats["active_fraction"].item() * count
                step += 1
                pairwise_distance = stats["mean_pairwise_distance"].item()
                totals["pairwise"] += pairwise_distance * count
                if step % 50 == 0:
                    print(
                        f"step {step}: mean_pairwise_distance={pairwise_distance:.3f}",
                        flush=True,
                    )
                    if wandb_run:
                        wandb_run.log(
                            {"mean_pairwise_distance": pairwise_distance}, step=step
                        )

            train_metrics = {key: value / examples for key, value in totals.items()}
            val_embeddings, val_labels = embed(model, loaders["val"], device)
            val_embeddings = val_embeddings.to(device)
            triplet_acc = fixed_triplet_accuracy(val_embeddings, splits, triplets)
            hard = batch_hard_metrics(val_embeddings, val_labels, chunk_size=64)
            retrieval = retrieval_metrics(val_embeddings, val_labels, chunk_size=64)
            val_class = classification_metrics(model, loaders["val"], device)
            metrics = {
                "epoch": epoch + 1, "lr": last_lr, "mining": mining,
                "train_loss": train_metrics["loss"], "train_d_ap": train_metrics["d_ap"],
                "train_d_an": train_metrics["d_an"], "active_fraction": train_metrics["active"],
                "mean_pairwise_distance": train_metrics["pairwise"],
                "val_triplet_acc": triplet_acc, "val_batch_hard_acc": hard["accuracy"],
                "val_batch_hard_d_ap": hard["d_ap"], "val_batch_hard_d_an": hard["d_an"],
                "val_r1": retrieval["r1"], "val_map": retrieval["map"],
                "val_class_top1": val_class["top1"],
                "val_class_top5": val_class["top5"],
            }
            writer.writerow(metrics)
            f.flush()
            save_checkpoint(run_dir / "last.pt", model, optimizer, epoch + 1, step, config, metrics)
            if retrieval["map"] > best_map:
                best_map = retrieval["map"]
                save_checkpoint(run_dir / "best.pt", model, optimizer, epoch + 1, step, config, metrics)
            print(
                f"{name} {epoch + 1:02d}/{args.epochs}: loss={metrics['train_loss']:.4f} "
                f"d_ap={metrics['train_d_ap']:.3f} d_an={metrics['train_d_an']:.3f} "
                f"pairwise={metrics['mean_pairwise_distance']:.3f} "
                f"active={metrics['active_fraction']:.3f} mining={mining} "
                f"triplet={triplet_acc:.3f} "
                f"batch-hard={hard['accuracy']:.3f} R@1={retrieval['r1']:.3f} mAP={retrieval['map']:.3f}",
                flush=True,
            )
            if wandb_run:
                wandb_run.log(metrics, step=step)
            if epoch < 5 and metrics["active_fraction"] == 0:
                if wandb_run:
                    wandb_run.finish(exit_code=1)
                raise RuntimeError("No active triplets this early; inspect margin and mining.")

            if epoch + 1 == args.batch_all_epochs:
                checks = {
                    "pairwise distance above 0.8": train_metrics["pairwise"] > 0.8,
                    "pairwise distance rising": (
                        previous_pairwise is not None
                        and train_metrics["pairwise"] > previous_pairwise
                    ),
                    "active fraction below 1": metrics["active_fraction"] < 1.0,
                    "validation triplet accuracy above 0.70": triplet_acc > 0.70,
                    "validation R@1 above 0.10": retrieval["r1"] > 0.10,
                }
                failed = [name for name, passed in checks.items() if not passed]
                if failed:
                    if wandb_run:
                        wandb_run.finish(exit_code=1)
                    raise RuntimeError(
                        "Mining warmup check failed: " + ", ".join(failed)
                        + ". Extend batch-all warmup or adjust the learning rate."
                    )

            if (
                epoch >= args.batch_all_epochs
                and previous_pairwise is not None
                and train_metrics["pairwise"] < 0.7 * previous_pairwise
            ):
                if wandb_run:
                    wandb_run.finish(exit_code=1)
                raise RuntimeError(
                    "Mean pairwise distance fell by more than 30% in one epoch. "
                    "Halve the learning rate or extend batch-all warmup."
                )
            previous_pairwise = train_metrics["pairwise"]

    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model"])
    val_metrics = best["metrics"]
    test_metrics = {}
    if not args.skip_test:
        test_embeddings, test_labels = embed(model, loaders["test"], device)
        test_metrics = retrieval_metrics(test_embeddings.to(device), test_labels, chunk_size=64)
        test_metrics.update(classification_metrics(model, loaders["test"], device))
    summary = {
        "run_name": name, "optimizer": optimizer_name, **val_metrics, "lr": lr,
        "test_r1": test_metrics.get("r1", ""), "test_map": test_metrics.get("map", ""),
        "test_class_top1": test_metrics.get("top1", ""),
        "test_class_top5": test_metrics.get("top5", ""),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if wandb_run:
        wandb_run.log(
            {"test_r1": summary["test_r1"], "test_map": summary["test_map"]},
            step=step + 1,
        )
        wandb_run.summary.update(summary)
        wandb_run.finish()
    if test_metrics:
        print(
            f"{name} final test: R@1={test_metrics['r1']:.3f} "
            f"mAP={test_metrics['map']:.3f} "
            f"class_top1={test_metrics['top1']:.3f}",
        )
    return summary


def write_comparison(root, summaries):
    columns = [
        "optimizer", "lr", "batch_size", "margin", "val_triplet_acc",
        "val_batch_hard_acc", "val_r1", "val_map", "val_class_top1",
        "test_r1", "test_map", "test_class_top1",
    ]
    rows = []
    for summary in summaries:
        config = json.loads((Path(root) / "runs" / summary["run_name"] / "config.json").read_text())
        rows.append({key: summary.get(key, config.get(key, "")) for key in columns})
    results_dir = Path(root) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "optimizer_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    table = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    table.extend("| " + " | ".join(str(row[key]) for key in columns) + " |" for row in rows)
    (results_dir / "optimizer_comparison.md").write_text("\n".join(table) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", action="store_true", help="Run Adam, NAdam, then SGD.")
    parser.add_argument("--optimizer", choices=("adam", "nadam", "sgd"), default="sgd")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--classification-weight", type=float, default=0.1)
    parser.add_argument(
        "--init-checkpoint",
        help="Optional CIFAR classifier checkpoint used to initialize the ResNet trunk.",
    )
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-all-epochs", type=int, default=3)
    parser.add_argument("--P", type=int, default=32)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--steps-per-epoch", type=int,
        help="Limit training batches per epoch for a short local smoke run.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-root", default=".")
    parser.add_argument(
        "--no-download", action="store_true",
        help="Use the local CIFAR-100 archive or extracted data; never download.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="visual-similarity-search")
    parser.add_argument("--skip-test", action="store_true", help="Skip final test evaluation for smoke runs.")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if not 0 <= args.batch_all_epochs <= args.epochs:
        parser.error("--batch-all-epochs must be between 0 and --epochs.")
    if args.P < 2 or args.K < 2:
        parser.error("--P and --K must both be at least 2.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.embedding_dim < 1:
        parser.error("--embedding-dim must be positive.")
    if args.classification_weight < 0:
        parser.error("--classification-weight cannot be negative.")
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        parser.error("--steps-per-epoch must be positive when provided.")
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    if args.compare:
        runs = [
            ("baseline", "adam", args.lr if args.lr is not None else 3e-4),
            ("nadam", "nadam", 1e-3),
            ("sgd", "sgd", 0.05),
        ]
    else:
        default_lrs = {"adam": 3e-4, "nadam": 1e-3, "sgd": 0.01}
        lr = args.lr if args.lr is not None else default_lrs[args.optimizer]
        runs = [("baseline", args.optimizer, lr)]
    summaries = [train_run(name, opt, lr, args, device) for name, opt, lr in runs]
    if args.compare:
        write_comparison(args.output_root, summaries)


if __name__ == "__main__":
    main()
