# Visual Similarity Search

Train a CIFAR-100 ResNet-18 classifier, fine-tune its features for image retrieval, and view nearest images in a local HTML gallery.

## Set up on Windows

Use Python 3.10 or newer. From this project folder, open PowerShell and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

The `auto` device setting uses CUDA when the installed PyTorch build can access it and otherwise runs on CPU.

## Data and split

The expected archive path is `data/cifar-100-python.tar.gz`. If that local archive is present, the loader checks its checksum and extracts it offline. If it is absent, the first run downloads CIFAR-100. The split uses 45,000 official training images for training, 5,000 official training images for validation, and keeps all 10,000 official test images untouched until final evaluation.

## Train the classifier

```powershell
python -m src.train_classifier --device auto --epochs 200
```

This uses a CIFAR-specific ResNet-18 stem, SGD with momentum and Nesterov, a cosine schedule with warmup, CIFAR AutoAugment, label smoothing, and CutMix. It saves the best validation checkpoint to `runs/classifier/best.pt`.

## Fine-tune retrieval embeddings

```powershell
python -m src.train --init-checkpoint runs/classifier/best.pt --optimizer sgd --lr 0.01 --epochs 40 --batch-all-epochs 2 --P 16 --K 8 --device auto
```

This starts from the classifier’s backbone and fine-tunes unit-normalized embeddings with triplet mining plus an auxiliary class loss. It selects checkpoints by validation mAP and saves `runs/baseline/best.pt` and `runs/baseline/last.pt`.

## Evaluate and view results

Evaluate the final model on the official test split once training is complete:

```powershell
python -m src.eval --checkpoint runs/baseline/best.pt --split test --device auto
```

This prints retrieval Recall@1, mAP, batch-hard metrics, and class accuracy as JSON. To create a local nearest-neighbor gallery:

```powershell
python -m src.demo --checkpoint runs/baseline/best.pt --query-index 0 --top-k 5
```

The query index is its position in the selected split (`test` by default). The command writes `results/retrieval_demo.html`. Use `--no-download` with any command to require local data and disable network access. The installed `vss-classify`, `vss-train`, `vss-eval`, and `vss-demo` commands mirror the Python module commands.
