"""Retrieval metrics, error analysis, and test-set evaluation."""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from src.retrieval import BruteForceIndex, extract_embeddings, save_test_embeddings


def average_precision(relevant):
    """Average precision for a ranked 1D or 2D boolean relevance array."""
    relevant = np.asarray(relevant, dtype=bool)
    one_row = relevant.ndim == 1
    if one_row:
        relevant = relevant[None, :]
    if relevant.ndim != 2:
        raise ValueError("relevant must be a 1D or 2D boolean array.")
    counts = relevant.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError("Every query must have at least one relevant gallery item.")
    ranks = np.arange(1, relevant.shape[1] + 1, dtype=np.float64)
    precision = np.cumsum(relevant, axis=1) / ranks
    result = (precision * relevant).sum(axis=1) / counts
    return float(result[0]) if one_row else result


def retrieval_metrics(emb, labels, metric="l2", ks=(1, 5, 10), chunk_size=128):
    """Compute leave-one-out hit-rate Recall@K, Precision@5, and mAP.

    Recall@K is the fraction of queries with at least one relevant result in
    their top K. This is query-level recall, so Precision@K <= Recall@K.
    """
    emb, labels = np.asarray(emb, dtype=np.float32), np.asarray(labels)
    if emb.ndim != 2 or len(emb) != len(labels) or len(emb) < 2:
        raise ValueError("Need at least two embeddings and one label per row.")
    if len(np.unique(labels)) < 2:
        raise ValueError("Retrieval evaluation needs at least two classes.")
    if not ks or any(int(k) != k or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers.")
    index = BruteForceIndex(emb, labels, metric=metric, chunk_size=chunk_size)
    _, class_ids, class_counts = np.unique(labels, return_inverse=True, return_counts=True)
    relevant_counts = class_counts[class_ids] - 1
    if np.any(relevant_counts <= 0):
        raise ValueError("Each query needs another same-class gallery example.")
    hits = {k: 0 for k in ks}
    precision5_sum = 0.0
    ap_sum = 0.0
    for start in range(0, len(emb), chunk_size):
        end = min(start + chunk_size, len(emb))
        rows = np.arange(start, end)
        ranked, _, _ = index.search(
            emb[start:end], len(emb) - 1, query_indices=rows
        )
        relevant = labels[ranked] == labels[start:end, None]
        for k in ks:
            hits[k] += int(relevant[:, : min(k, relevant.shape[1])].any(axis=1).sum())
        p5 = relevant[:, : min(5, relevant.shape[1])]
        precision5_sum += float(p5.mean(axis=1).sum())
        ap_sum += float(average_precision(relevant).sum())
    result = {f"recall@{k}": hits[k] / len(emb) for k in ks}
    result["precision@5"] = precision5_sum / len(emb)
    result["map"] = ap_sum / len(emb)
    # Stage 4 uses these names for best-checkpoint selection and logging.
    result["r1"] = result.get("recall@1")
    return result


def superclass_error_share(query_labels, top1_labels, fine_to_coarse):
    """Share of incorrect top-1 results that remain in the query superclass."""
    query_labels, top1_labels = np.asarray(query_labels), np.asarray(top1_labels)
    mapping = np.asarray(fine_to_coarse)
    if query_labels.shape != top1_labels.shape:
        raise ValueError("query_labels and top1_labels must have equal shapes.")
    errors = top1_labels != query_labels
    count = int(errors.sum())
    if not count:
        return {"same_superclass_error_share": None, "top1_errors": 0}
    same_superclass = mapping[query_labels] == mapping[top1_labels]
    return {
        "same_superclass_error_share": float(same_superclass[errors].mean()),
        "top1_errors": count,
    }


def load_cifar100_superclasses(data_root):
    """Read CIFAR-100's official fine-to-coarse labels and names."""
    root = Path(data_root)
    candidates = (root / "cifar-100-python", root)
    dataset_dir = next((path for path in candidates if (path / "meta").exists()), None)
    if dataset_dir is None:
        raise FileNotFoundError(f"Could not find CIFAR-100 metadata under {root}.")

    def read(name):
        with (dataset_dir / name).open("rb") as file:
            return pickle.load(file, encoding="bytes")

    meta, train = read("meta"), read("train")
    fine_to_coarse = np.full(100, -1, dtype=np.int64)
    for fine, coarse in zip(train[b"fine_labels"], train[b"coarse_labels"]):
        fine_to_coarse[fine] = coarse
    if np.any(fine_to_coarse == -1):
        raise ValueError("CIFAR metadata is missing a fine-to-coarse label.")
    decode = lambda names: [name.decode() if isinstance(name, bytes) else str(name) for name in names]
    fine_names = decode(meta[b"fine_label_names"])
    coarse_names = decode(meta[b"coarse_label_names"])
    return fine_to_coarse, fine_names, coarse_names


def faiss_topk_matches(emb, k=5):
    """Compare exact Faiss CPU and NumPy L2 neighbors, ignoring cutoff ties."""
    try:
        import faiss
    except ImportError:
        return None
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    if len(emb) <= k:
        raise ValueError("Need more gallery vectors than k for leave-one-out search.")
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    faiss_scores, faiss_indices = index.search(emb, min(k + 2, len(emb)))
    f_indices, f_scores = [], []
    for row in range(len(emb)):
        keep = faiss_indices[row] != row
        f_indices.append(faiss_indices[row, keep][: k + 1])
        f_scores.append(faiss_scores[row, keep][: k + 1])
    f_indices, f_scores = np.asarray(f_indices), np.asarray(f_scores)
    numpy_indices, numpy_scores, _ = BruteForceIndex(emb, np.arange(len(emb))).search(
        emb, min(k + 1, len(emb) - 1), query_indices=np.arange(len(emb))
    )

    # Ties at rank k may legally be returned in a different index order.
    eligible = np.ones(len(emb), dtype=bool)
    if k < len(emb) - 1:
        tied_f = f_scores[:, k - 1] == f_scores[:, k]
        tied_n = numpy_scores[:, k - 1] == numpy_scores[:, k]
        eligible &= ~(tied_f | tied_n)
    matches = np.all(
        np.sort(f_indices[:, :k], axis=1) == np.sort(numpy_indices[:, :k], axis=1),
        axis=1,
    )
    return {
        "available": True,
        "queries_checked": int(eligible.sum()),
        "topk_match": bool(matches[eligible].all()) if eligible.any() else True,
    }


def save_retrieval_grid(train_data, test_data, test_indices, embeddings, labels,
                        class_names, output_path, seed=42, count=10):
    """Save query + top-5 retrievals; green borders mark fine-label matches."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    test_indices = np.asarray(test_indices, dtype=np.int64)
    n = min(count, len(embeddings))
    query_rows = np.random.default_rng(seed).choice(len(embeddings), n, replace=False)
    neighbors, _, _ = BruteForceIndex(embeddings, labels).search(
        embeddings[query_rows], 5, query_indices=query_rows
    )
    images = np.concatenate((train_data.data, test_data.data), axis=0)
    figure, axes = plt.subplots(n, 6, figsize=(12, 2.1 * n), squeeze=False)
    for row, query_row in enumerate(query_rows):
        query_global = test_indices[query_row]
        ids = [query_global] + test_indices[neighbors[row]].tolist()
        row_labels = [int(labels[query_row])] + labels[neighbors[row]].astype(int).tolist()
        for col, (global_id, fine_label) in enumerate(zip(ids, row_labels)):
            image = images[global_id]
            axis = axes[row, col]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            color = "green" if fine_label == int(labels[query_row]) else "red"
            axis.add_patch(Rectangle((0, 0), 1, 1, transform=axis.transAxes,
                                     fill=False, edgecolor=color, linewidth=4))
            axis.set_title(("Q: " if col == 0 else "") + class_names[fine_label], fontsize=8)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return query_rows


def evaluate_checkpoint(args):
    import torch

    from src.data import get_loaders, get_splits
    from src.model import ResNet18

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                           else "cpu" if args.device == "auto" else args.device)
    splits = get_splits(args.seed, args.data_root, args.split_dir, download=not args.no_download)
    loaders = get_loaders(
        seed=args.seed, data_root=args.data_root, split_dir=args.split_dir,
        num_workers=args.num_workers, download=not args.no_download,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    model = ResNet18(config.get("embedding_dim", 128))
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    output = Path(args.output_dir)
    embeddings, labels, test_indices, embed_time = save_test_embeddings(
        model, loaders["test"], splits["test"], output, device
    )
    metrics = retrieval_metrics(embeddings, labels, metric=args.metric)
    index = BruteForceIndex(embeddings, labels, metric=args.metric)
    start = time.perf_counter()
    _, _, top_labels = index.search(
        embeddings, 5, query_indices=np.arange(len(embeddings))
    )
    search_seconds = time.perf_counter() - start
    fine_to_coarse, fine_names, coarse_names = load_cifar100_superclasses(args.data_root)
    error_analysis = superclass_error_share(labels, top_labels[:, 0], fine_to_coarse)

    random_vectors = np.random.default_rng(args.seed).normal(size=embeddings.shape).astype(np.float32)
    random_vectors /= np.maximum(np.linalg.norm(random_vectors, axis=1, keepdims=True), 1e-12)
    torch.manual_seed(args.seed)
    untrained = ResNet18(config.get("embedding_dim", 128)).to(device)
    random_emb, random_labels, _ = extract_embeddings(untrained, loaders["test"], device)
    baselines = {
        "random_gaussian": retrieval_metrics(random_vectors, labels, args.metric),
        "untrained_resnet18": retrieval_metrics(random_emb, random_labels, args.metric),
    }

    train_data, test_data = loaders["test"].dataset.pool.datasets
    grid_path = output / "retrieval_grid.png"
    save_retrieval_grid(
        train_data, test_data, test_indices, embeddings, labels, fine_names,
        grid_path, seed=args.seed,
    )
    faiss_result = faiss_topk_matches(embeddings, k=5)
    if faiss_result is None:
        faiss_result = {"available": False, "topk_match": None}
    search_ms = search_seconds * 1000 / len(embeddings)
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "metric": args.metric,
        "count": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "metrics": metrics,
        "latency_ms": {
            "embedding_per_image": embed_time["ms_per_image"],
            "search_per_query_over_test_set": search_ms,
            "embedding_plus_search_per_image": embed_time["ms_per_image"] + search_ms,
        },
        "error_analysis": error_analysis,
        "superclasses": {
            "names": coarse_names,
            "fine_to_superclass": {
                name: coarse_names[int(fine_to_coarse[i])]
                for i, name in enumerate(fine_names)
            },
        },
        "faiss_cpu": faiss_result,
        "baselines": baselines,
        "artifacts": {
            "embeddings": str(output / "test_emb.npy"),
            "labels": str(output / "test_labels.npy"),
            "indices": str(output / "test_idx.npy"),
            "qualitative_grid": str(grid_path),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--split-dir", default="splits")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metric", choices=("l2", "cosine"), default="l2")
    parser.add_argument("--no-download", action="store_true")
    evaluate_checkpoint(parser.parse_args())


if __name__ == "__main__":
    main()
