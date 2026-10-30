"""Evaluate a saved embedding model on the CIFAR-100 validation or test split."""

import argparse
import json
from pathlib import Path

from src.data import get_loaders
from src.model import ResNet18
from src.retrieval import (
    batch_hard_metrics,
    classification_metrics,
    embed,
    retrieval_metrics,
)
from src.utils import load_checkpoint, resolve_device


def evaluate_checkpoint(
    checkpoint_path,
    *,
    split="test",
    data_root="data",
    split_dir="splits",
    num_workers=0,
    device="auto",
    download=True,
):
    if split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'.")
    device = resolve_device(device)
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    config = checkpoint.get("config", {})
    seed = int(config.get("seed", 42))
    P, K = int(config.get("P", 32)), int(config.get("K", 8))
    loaders = get_loaders(
        seed=seed,
        data_root=data_root,
        split_dir=split_dir,
        P=P,
        K=K,
        num_workers=num_workers,
        download=download,
    )

    model = ResNet18(d=int(config.get("embedding_dim", 128))).to(device)
    model.load_state_dict(checkpoint["model"])
    is_classifier_checkpoint = (
        config.get("stage") == "cifar100_classification_pretrain"
    )
    embeddings, labels = embed(
        model,
        loaders[split],
        device,
        use_backbone_features=is_classifier_checkpoint,
    )
    embeddings = embeddings.to(device)
    metrics = {
        **retrieval_metrics(embeddings, labels, chunk_size=64),
        **batch_hard_metrics(embeddings, labels, chunk_size=64),
        **classification_metrics(model, loaders[split], device),
    }
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "split": split,
        "count": len(labels),
        "device": str(device),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate_checkpoint(
            args.checkpoint,
            split=args.split,
            data_root=args.data_root,
            split_dir=args.split_dir,
            num_workers=args.num_workers,
            device=args.device,
            download=not args.no_download,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
