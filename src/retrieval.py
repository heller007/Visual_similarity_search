"""Exact NumPy retrieval and model embedding extraction."""

from pathlib import Path
import time

import numpy as np


class BruteForceIndex:
    """Exact chunked search. L2 scores are squared distances; cosine scores are similarities."""

    def __init__(self, emb, labels, metric="l2", chunk_size=256):
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.labels = np.asarray(labels).reshape(-1)
        if self.emb.ndim != 2 or len(self.emb) < 1 or len(self.emb) != len(self.labels):
            raise ValueError("emb must be (N, D), with one label per row.")
        if metric not in ("l2", "cosine"):
            raise ValueError("metric must be 'l2' or 'cosine'.")
        if chunk_size < 1 or not np.isfinite(self.emb).all():
            raise ValueError("chunk_size must be positive and embeddings finite.")
        self.metric, self.chunk_size = metric, chunk_size
        if metric == "cosine":
            lengths = np.linalg.norm(self.emb, axis=1, keepdims=True)
            self.emb = self.emb / np.maximum(lengths, 1e-12)
        self.norms = np.einsum("ij,ij->i", self.emb, self.emb) if metric == "l2" else None
        self.direction = 1 if metric == "l2" else -1

    def search(self, q, k, query_indices=None):
        """Return (gallery indices, scores, labels), each shaped (queries, k).

        Pass query_indices to exclude matching gallery rows, as in leave-one-out
        evaluation. Smaller L2 scores and larger cosine scores rank first.
        """
        q = np.asarray(q, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or len(q) == 0 or q.shape[1] != self.emb.shape[1]:
            raise ValueError("q must have shape (queries, embedding_dim).")
        if not isinstance(k, (int, np.integer)) or not np.isfinite(q).all() or k < 1 or k > len(self.emb):
            raise ValueError("queries must be finite and k must be between 1 and N.")
        if query_indices is not None:
            query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
            if len(query_indices) != len(q) or np.any((query_indices < 0) | (query_indices >= len(self.emb))):
                raise ValueError("query_indices must contain one valid gallery index per query.")
            if k > len(self.emb) - 1:
                raise ValueError("k must be at most N-1 when excluding self.")

        all_indices, all_scores = [], []
        for start in range(0, len(q), self.chunk_size):
            batch = q[start : start + self.chunk_size]
            if self.metric == "cosine":
                lengths = np.linalg.norm(batch, axis=1, keepdims=True)
                batch = batch / np.maximum(lengths, 1e-12)
            dots = np.einsum("ij,kj->ik", batch, self.emb, optimize=False)
            if self.metric == "l2":
                scores = np.maximum(
                    np.einsum("ij,ij->i", batch, batch)[:, None]
                    + self.norms[None, :]
                    - 2 * dots,
                    0,
                )
            else:
                scores = dots
            if query_indices is not None:
                rows = np.arange(len(batch))
                cols = query_indices[start : start + len(batch)]
                scores[rows, cols] = np.inf * self.direction
            candidates = np.argpartition(
                self.direction * scores, k - 1, axis=1
            )[:, :k]
            candidate_scores = np.take_along_axis(scores, candidates, axis=1)
            order = np.argsort(self.direction * candidate_scores, axis=1)
            indices = np.take_along_axis(candidates, order, axis=1)
            values = np.take_along_axis(candidate_scores, order, axis=1)
            all_indices.append(indices)
            all_scores.append(values)

        indices = np.concatenate(all_indices)
        return indices, np.concatenate(all_scores), self.labels[indices]


def extract_embeddings(model, loader, device=None):
    """Embed a deterministic evaluation loader and report model time per image."""
    import torch

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()
    embeddings, labels = [], []
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for images, target in loader:
            embeddings.append(model(images.to(device, non_blocking=True)).float().cpu().numpy())
            labels.append(target.cpu().numpy())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return (
        np.concatenate(embeddings).astype(np.float32, copy=False),
        np.concatenate(labels).astype(np.int64, copy=False),
        {"seconds": elapsed, "ms_per_image": elapsed * 1000 / sum(map(len, labels))},
    )


def save_test_embeddings(model, loader, test_indices, output_dir, device=None):
    """Write test_emb.npy, test_labels.npy and test_idx.npy into output_dir."""
    embeddings, labels, timing = extract_embeddings(model, loader, device)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    if len(embeddings) != len(labels) or len(labels) != len(test_indices):
        raise ValueError("Loader output and test_indices must have equal lengths.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "test_emb.npy", embeddings)
    np.save(output_dir / "test_labels.npy", labels)
    np.save(output_dir / "test_idx.npy", test_indices)
    return embeddings, labels, test_indices, timing
