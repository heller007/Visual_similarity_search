"""Pairwise distance, triplet mining, and loss for normalized embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distances(x, y=None):
    """Stable Euclidean distances, computed in FP32 (or FP64 for gradcheck)."""
    y = x if y is None else y
    dtype = torch.float64 if x.dtype == torch.float64 or y.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        x, y = x.to(dtype), y.to(dtype)
        differences = x[:, None, :] - y[None, :, :]
        return (differences.square().sum(dim=-1) + 1e-8).sqrt()


def distance_matrix(embeddings):
    return pairwise_distances(embeddings)


def _masks(labels):
    same = labels[:, None] == labels[None, :]
    positive = same.clone()
    positive.fill_diagonal_(False)
    negative = ~same
    if not positive.any(1).all() or not negative.any(1).all():
        raise ValueError("Each anchor needs a positive and a negative.")
    return positive, negative


def batch_hard_mining(distances, labels):
    positive, negative = _masks(labels)
    pos = distances.masked_fill(~positive, -torch.inf).argmax(1)
    neg = distances.masked_fill(~negative, torch.inf).argmin(1)
    return pos, neg


def random_mining(distances, labels, generator=None):
    positive, negative = _masks(labels)
    scores = torch.rand(distances.shape, device=distances.device, generator=generator)
    pos = scores.masked_fill(~positive, -torch.inf).argmax(1)
    neg = scores.masked_fill(~negative, -torch.inf).argmax(1)
    return pos, neg


class TripletLoss(nn.Module):
    """Triplet loss for batch-all, batch-hard, or random mining."""

    def __init__(self, margin=0.3, mining="batch_hard"):
        super().__init__()
        if mining not in ("batch_all", "batch_hard", "random"):
            raise ValueError("mining must be 'batch_all', 'batch_hard', or 'random'.")
        self.margin, self.mining = margin, mining

    def forward(self, embeddings, labels, generator=None):
        distances = distance_matrix(embeddings)

        if self.mining == "batch_all":
            positive, negative = _masks(labels)
            valid = positive[:, :, None] & negative[:, None, :]
            losses = F.relu(
                distances[:, :, None] - distances[:, None, :] + self.margin
            ).masked_fill(~valid, 0)
            active = losses > 0
            loss = losses.sum() / active.sum().clamp_min(1)
            active_fraction = active.sum() / valid.sum()
            d_ap = distances[positive].mean()
            d_an = distances[negative].mean()
        else:
            pos, neg = (
                random_mining(distances, labels, generator)
                if self.mining == "random"
                else batch_hard_mining(distances, labels)
            )
            rows = torch.arange(len(labels), device=labels.device)
            d_ap, d_an = distances[rows, pos], distances[rows, neg]
            losses = F.relu(d_ap - d_an + self.margin)
            loss = losses.mean()
            active_fraction = (losses > 0).float().mean()

        off_diagonal = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        stats = {
            "mean_d_ap": d_ap.detach().mean(),
            "mean_d_an": d_an.detach().mean(),
            "active_fraction": active_fraction.detach().float(),
            "mean_pairwise_distance": distances[off_diagonal].detach().mean(),
            "embedding_std": embeddings.detach().std(dim=0, unbiased=False).mean(),
        }
        return loss, stats
