import math
import csv
import json

import pytest
import torch

from src.train import batch_hard_metrics, retrieval_metrics, write_comparison


def test_validation_retrieval_metrics_are_leave_one_out():
    embeddings = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    labels = torch.tensor([0, 0, 1, 1])
    retrieval = retrieval_metrics(embeddings, labels, chunk_size=2)
    hard = batch_hard_metrics(embeddings, labels, chunk_size=2)
    assert retrieval["r1"] == 0.0
    assert retrieval["map"] == pytest.approx(1 / 3)
    assert hard["accuracy"] == 0.0
    assert hard["d_ap"] == pytest.approx(2.0, abs=1e-6)
    assert hard["d_an"] == pytest.approx(math.sqrt(2), abs=1e-6)


def test_retrieval_metrics_find_nearest_same_class():
    embeddings = torch.nn.functional.normalize(torch.tensor(
        [[1., 0.], [1., 0.1], [-1., 0.], [-1., -0.1]]
    ), dim=1)
    labels = torch.tensor([0, 0, 1, 1])
    metrics = retrieval_metrics(embeddings, labels)
    assert metrics["r1"] == 1.0
    assert metrics["map"] == pytest.approx(1.0)


def test_comparison_table_records_fixed_rates_and_batch(tmp_path):
    summaries = []
    for name, optimizer, lr in (("baseline", "adam", 1e-3), ("nadam", "nadam", 1e-3), ("sgd", "sgd", 0.05)):
        run = tmp_path / "runs" / name
        run.mkdir(parents=True)
        config = {"optimizer": optimizer, "lr": lr, "batch_size": 256, "margin": 0.3}
        (run / "config.json").write_text(json.dumps(config))
        summaries.append({"run_name": name, **config, "val_r1": 0.8, "test_r1": 0.7})
    write_comparison(tmp_path, summaries)
    with (tmp_path / "results" / "optimizer_comparison.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [float(row["lr"]) for row in rows] == [1e-3, 1e-3, 0.05]
    assert all(row["batch_size"] == "256" and row["margin"] == "0.3" for row in rows)
