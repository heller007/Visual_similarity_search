"""CIFAR-100 data utilities for metric-learning experiments."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler
from torchvision import datasets, transforms


SPLITS = ("train", "val", "test")


def _load_cifar100(root="data", download=True):
    train = datasets.CIFAR100(root, train=True, download=download)
    test = datasets.CIFAR100(root, train=False, download=download)
    labels = np.asarray(train.targets + test.targets, dtype=np.int64)
    return train, test, labels


def _make_split_indices(labels, seed=42):
    rng = np.random.default_rng(seed)
    parts = {name: [] for name in SPLITS}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label).copy()
        if len(indices) != 600:
            raise ValueError(f"Expected 600 examples for class {label}.")
        rng.shuffle(indices)
        parts["train"].extend(indices[:480])
        parts["val"].extend(indices[480:540])
        parts["test"].extend(indices[540:])
    return {name: np.asarray(parts[name], dtype=np.int64) for name in SPLITS}


def _index_hash(indices):
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def get_splits(seed=42, data_root="data", split_dir="splits", download=True):
    """Return global indices and save them with SHA-256 hashes to JSON."""
    train, _, labels = _load_cifar100(data_root, download)
    splits = _make_split_indices(labels, seed)
    hashes = {name: _index_hash(indices) for name, indices in splits.items()}
    payload = {
        "seed": seed,
        "dataset": "CIFAR100",
        "pool_order": "torchvision train followed by test",
        "pool_size": len(labels),
        "class_names": train.classes,
        "counts": {name: len(indices) for name, indices in splits.items()},
        "sha256": hashes,
        "indices": {name: indices.tolist() for name, indices in splits.items()},
    }
    path = Path(split_dir) / f"seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, digest in hashes.items():
        logging.info("%s split SHA-256: %s", name, digest)
    return splits


def _channel_stats(train_images, test_images, indices):
    """Compute population RGB mean/std from the selected training indices."""
    sums, squares, pixels = np.zeros(3), np.zeros(3), 0
    train_count = len(train_images)
    for start in range(0, len(indices), 1024):
        batch = np.asarray(indices[start : start + 1024])
        images = np.empty((len(batch), 32, 32, 3), dtype=np.uint8)
        is_train = batch < train_count
        images[is_train] = train_images[batch[is_train]]
        images[~is_train] = test_images[batch[~is_train] - train_count]
        values = images.astype(np.float64) / 255
        sums += values.sum((0, 1, 2))
        squares += np.square(values).sum((0, 1, 2))
        pixels += values.shape[0] * 32 * 32
    mean = sums / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 0))
    return mean.tolist(), std.tolist()


class _TransformSubset(Dataset):
    def __init__(self, pool, indices, transform):
        self.pool, self.indices, self.transform = pool, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.pool[int(self.indices[index])]
        return self.transform(image), label


class PKSampler(Sampler):
    """Yield batches with P distinct classes and K examples from each."""

    def __init__(self, labels, P=32, K=8, seed=42, num_batches=None):
        self.labels, self.P, self.K, self.seed = np.asarray(labels), P, K, seed
        if K < 2:
            raise ValueError("K must be at least 2 to provide a positive per anchor.")
        self.by_class = {
            label: np.flatnonzero(self.labels == label)
            for label in np.unique(self.labels)
        }
        if len(self.by_class) < P or any(len(ix) < K for ix in self.by_class.values()):
            raise ValueError("Not enough classes or examples for this P×K batch size.")
        self.classes = np.asarray(list(self.by_class))
        self.num_batches = num_batches or max(1, len(self.labels) // (P * K))
        self.epoch = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_batches):
            batch = []
            for label in rng.choice(self.classes, self.P, replace=False):
                batch.extend(rng.choice(self.by_class[label], self.K, replace=False))
            yield batch


def _make_val_triplets(val_indices, labels, n_triplets=10_000, seed=42):
    val_indices = np.asarray(val_indices)
    val_labels = labels[val_indices]
    classes = np.unique(val_labels)
    by_class = {c: val_indices[val_labels == c] for c in classes}
    if len(classes) < 2 or any(len(ix) < 2 for ix in by_class.values()):
        raise ValueError("Validation needs two classes and two images per class.")
    rng = np.random.default_rng(seed)
    triplets = np.empty((n_triplets, 3), dtype=np.int64)
    for row in range(n_triplets):
        anchor = rng.choice(val_indices)
        label = labels[anchor]
        positive = rng.choice(by_class[label][by_class[label] != anchor])
        negative = rng.choice(val_indices[val_labels != label])
        triplets[row] = anchor, positive, negative
    return triplets


def get_loaders(
    seed=42, data_root="data", split_dir="splits", P=32, K=8,
    batch_size=None, num_workers=0, download=True, color_jitter=False,
):
    """Return train/val/test loaders; training uses a P×K batch sampler."""
    batch_size = P * K if batch_size is None else batch_size
    if batch_size != P * K:
        raise ValueError("batch_size must equal P * K.")
    splits = get_splits(seed, data_root, split_dir, download)
    train, test, labels = _load_cifar100(data_root, download)
    pool = ConcatDataset((train, test))
    mean, std = _channel_stats(train.data, test.data, splits["train"])

    train_ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if color_jitter:
        train_ops.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.02))
    train_ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    eval_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_data = _TransformSubset(pool, splits["train"], transforms.Compose(train_ops))
    val_data = _TransformSubset(pool, splits["val"], eval_transform)
    test_data = _TransformSubset(pool, splits["test"], eval_transform)
    sampler = PKSampler(labels[splits["train"]], P, K, seed)
    return {
        "train": DataLoader(train_data, batch_sampler=sampler, num_workers=num_workers),
        "val": DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def get_val_triplets(
    seed=42, n_triplets=10_000, data_root="data", split_dir="splits", download=True
):
    """Save and return fixed-seed global-index validation triplets."""
    splits = get_splits(seed, data_root, split_dir, download)
    _, _, labels = _load_cifar100(data_root, download)
    triplets = _make_val_triplets(splits["val"], labels, n_triplets, seed)
    path = Path(split_dir) / f"val_triplets_seed{seed}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, triplets)
    return triplets
