"""Embedding extraction and nearest-neighbor metrics for image retrieval."""

import torch
import torch.nn.functional as F

from src.loss import pairwise_distances


@torch.no_grad()
def embed(model, loader, device, use_backbone_features=False):
    """Embed every image from a loader and return CPU embeddings and labels."""
    model.eval()
    embeddings, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        values = model.forward_features(images) if use_backbone_features else model(images)
        if use_backbone_features:
            values = F.normalize(values, dim=1)
        embeddings.append(values.cpu())
        labels.append(target.cpu())
    if not embeddings:
        raise ValueError("Cannot embed an empty data loader.")
    return torch.cat(embeddings), torch.cat(labels)


@torch.no_grad()
def classification_metrics(model, loader, device):
    """Compute top-1 and top-5 class accuracy for the auxiliary CIFAR head."""
    model.eval()
    total, correct1, correct5 = 0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model.classify(images)
        total += len(labels)
        correct1 += (logits.argmax(1) == labels).sum().item()
        correct5 += (logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
    if total == 0:
        raise ValueError("Cannot evaluate an empty data loader.")
    return {"top1": correct1 / total, "top5": correct5 / total}


def _validate_retrieval_inputs(embeddings, labels, chunk_size):
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected a 2D embedding matrix and 1D labels.")
    if len(embeddings) != len(labels) or len(labels) < 2:
        raise ValueError("Embeddings and labels must match and contain at least two items.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")


@torch.no_grad()
def retrieval_metrics(embeddings, labels, chunk_size=64):
    """Compute leave-one-out Recall@1 and mAP in query chunks."""
    _validate_retrieval_inputs(embeddings, labels, chunk_size)
    device = embeddings.device
    labels = labels.to(device=device, dtype=torch.long)
    counts = torch.bincount(labels)
    positives_per_class = counts - 1
    if (counts[counts > 0] < 2).any():
        raise ValueError("Every class in the retrieval set needs at least two images.")

    n = len(labels)
    hits, average_precision = 0, 0.0
    ranks = torch.arange(1, n + 1, device=device, dtype=embeddings.dtype)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distances = pairwise_distances(embeddings[start:end], embeddings)
        rows = torch.arange(end - start, device=device)
        own_indices = torch.arange(start, end, device=device)
        distances[rows, own_indices] = torch.inf
        order = distances.argsort(dim=1)
        relevant = (labels[order] == labels[start:end, None]) & (
            order != own_indices[:, None]
        )
        hits += relevant[:, 0].sum().item()
        precision = relevant.cumsum(1) / ranks
        ap = (precision * relevant).sum(1) / positives_per_class[labels[start:end]]
        average_precision += ap.sum().item()

    return {"r1": hits / n, "map": average_precision / n}


@torch.no_grad()
def batch_hard_metrics(embeddings, labels, chunk_size=64):
    """Compute batch-hard accuracy and average positive/negative distances."""
    _validate_retrieval_inputs(embeddings, labels, chunk_size)
    device = embeddings.device
    labels = labels.to(device=device, dtype=torch.long)
    n, correct, sum_ap, sum_an = len(labels), 0, 0.0, 0.0

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distances = pairwise_distances(embeddings[start:end], embeddings)
        rows = torch.arange(end - start, device=device)
        own_indices = torch.arange(start, end, device=device)
        same = labels[start:end, None] == labels[None, :]
        positive = same.clone()
        positive[rows, own_indices] = False
        d_ap = distances.masked_fill(~positive, -torch.inf).max(1).values
        d_an = distances.masked_fill(same, torch.inf).min(1).values
        if not torch.isfinite(d_ap).all() or not torch.isfinite(d_an).all():
            raise ValueError("Each item needs at least one positive and one negative.")
        correct += (d_ap < d_an).sum().item()
        sum_ap += d_ap.sum().item()
        sum_an += d_an.sum().item()

    return {"accuracy": correct / n, "d_ap": sum_ap / n, "d_an": sum_an / n}


@torch.no_grad()
def search_embeddings(query, gallery_embeddings, gallery_labels=None, top_k=5, exclude_index=None):
    """Return nearest gallery positions and distances for one query embedding."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if gallery_embeddings.ndim != 2 or len(gallery_embeddings) == 0:
        raise ValueError("gallery_embeddings must be a non-empty 2D tensor.")
    query = torch.as_tensor(query, device=gallery_embeddings.device)
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2 or query.shape != (1, gallery_embeddings.shape[1]):
        raise ValueError("query must contain one embedding with the gallery dimension.")
    if gallery_labels is not None and len(gallery_labels) != len(gallery_embeddings):
        raise ValueError("gallery_labels must match the gallery length.")
    if exclude_index is not None and not 0 <= exclude_index < len(gallery_embeddings):
        raise IndexError("exclude_index is outside the gallery.")

    distances = pairwise_distances(query, gallery_embeddings).squeeze(0)
    if exclude_index is not None:
        distances[exclude_index] = torch.inf
    count = min(top_k, len(distances) - (exclude_index is not None))
    if count < 1:
        raise ValueError("The gallery has no candidates after excluding the query.")
    values, indices = torch.topk(distances, count, largest=False, sorted=True)
    return indices, values
