"""CIFAR-100 ResNet-18 classification pretraining for retrieval embeddings."""

import argparse
import csv
import json
import random
import time
from math import cos, pi
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import get_classification_loaders
from src.model import ResNet18
from src.utils import resolve_device, seed_all


def _scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _lr_at(step, total_steps, warmup_steps, base_lr):
    warmup_steps = min(warmup_steps, total_steps)
    if warmup_steps and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = total_steps - warmup_steps
    progress = 0.0 if decay_steps <= 1 else (step - warmup_steps) / (decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (0.01 + 0.99 * (1 + cos(pi * progress)) / 2)


def _cutmix(images, labels, alpha=0.2, probability=0.5):
    if alpha <= 0 or random.random() >= probability or len(labels) < 2:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(labels), device=images.device)
    height, width = images.shape[-2:]
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
    center_x = random.randrange(width)
    center_y = random.randrange(height)
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    images[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return images, labels, labels[permutation], lam


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct1, correct5, loss_sum = 0, 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model.classify(images)
            loss = F.cross_entropy(logits, labels)
        count = len(labels)
        total += count
        loss_sum += loss.item() * count
        correct1 += (logits.argmax(1) == labels).sum().item()
        correct5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
    return {
        "loss": loss_sum / total,
        "top1": correct1 / total,
        "top5": correct5 / total,
    }


def train(args):
    seed_all(args.seed)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda"
    loaders = get_classification_loaders(
        seed=args.seed,
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
        pin_memory=pin_memory,
        auto_augment=not args.no_auto_augment,
    )
    model = ResNet18(d=args.embedding_dim).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        torch.set_float32_matmul_precision("high")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = _scaler(device.type == "cuda")
    total_steps = args.epochs * len(loaders["train"])
    warmup_steps = args.warmup_epochs * len(loaders["train"])
    run_dir = Path(args.output_root) / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": "cifar100_classification_pretrain",
        "seed": args.seed,
        "architecture": "CIFAR ResNet-18",
        "embedding_dim": args.embedding_dim,
        "num_classes": 100,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "lr": args.lr,
        "momentum": args.momentum,
        "nesterov": True,
        "weight_decay": args.weight_decay,
        "schedule": "linear warmup then cosine to 1%",
        "warmup_epochs": args.warmup_epochs,
        "loss": "cross_entropy_with_label_smoothing",
        "label_smoothing": args.label_smoothing,
        "cutmix_alpha": args.cutmix_alpha,
        "cutmix_probability": args.cutmix_probability,
        "auto_augment": not args.no_auto_augment,
        "device": str(device),
        "official_test_set_reserved": True,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_top1", "val_top5"]
    best_top1 = -1.0
    global_step = 0
    metrics_path = run_dir / "metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(args.epochs):
            started = time.time()
            model.train()
            loss_sum, examples, last_lr = 0.0, 0, args.lr
            for images, labels in loaders["train"]:
                last_lr = _lr_at(global_step, total_steps, warmup_steps, args.lr)
                for group in optimizer.param_groups:
                    group["lr"] = last_lr
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                images, target_a, target_b, lam = _cutmix(
                    images, labels, args.cutmix_alpha, args.cutmix_probability
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    logits = model.classify(images)
                    loss = lam * F.cross_entropy(
                        logits, target_a, label_smoothing=args.label_smoothing
                    ) + (1.0 - lam) * F.cross_entropy(
                        logits, target_b, label_smoothing=args.label_smoothing
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                count = len(labels)
                loss_sum += loss.item() * count
                examples += count
                global_step += 1

            train_loss = loss_sum / examples
            val = evaluate(model, loaders["val"], device)
            row = {
                "epoch": epoch + 1,
                "lr": last_lr,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_top1": val["top1"],
                "val_top5": val["top5"],
            }
            writer.writerow(row)
            csv_file.flush()
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch + 1,
                "config": config,
                "metrics": row,
            }
            torch.save(checkpoint, run_dir / "last.pt")
            if val["top1"] > best_top1:
                best_top1 = val["top1"]
                torch.save(checkpoint, run_dir / "best.pt")
            elapsed = time.time() - started
            print(
                f"classifier {epoch + 1:03d}/{args.epochs}: "
                f"loss={train_loss:.4f} val_top1={val['top1']:.4f} "
                f"val_top5={val['top5']:.4f} lr={last_lr:.5g} "
                f"time={elapsed:.1f}s",
                flush=True,
            )

    summary = {
        "run_name": args.run_name,
        "best_val_top1": best_top1,
        "best_epoch": int(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)["epoch"]),
        "checkpoint": str(run_dir / "best.pt"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--cutmix-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-probability", type=float, default=0.5)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-root", default=".")
    parser.add_argument("--run-name", default="classifier")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-auto-augment", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if args.warmup_epochs < 0 or args.num_workers < 0:
        parser.error("--warmup-epochs and --num-workers cannot be negative.")
    if args.embedding_dim < 1:
        parser.error("--embedding-dim must be positive.")
    if args.cutmix_alpha < 0 or not 0 <= args.cutmix_probability <= 1:
        parser.error("CutMix alpha must be nonnegative and probability must be in [0, 1].")
    if not 0 <= args.label_smoothing < 1:
        parser.error("--label-smoothing must be in [0, 1).")
    try:
        train(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
