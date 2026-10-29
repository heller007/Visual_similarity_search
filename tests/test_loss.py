import math

import pytest
import torch
import torch.nn.functional as F

from src.loss import TripletLoss, batch_hard_mining, distance_matrix, random_mining
from src.optim import Adam


def test_batch_hard_indices_match_brute_force_for_20_batches():
    for seed in range(20):
        torch.manual_seed(seed)
        labels = torch.arange(4).repeat_interleave(3)
        embeddings = F.normalize(torch.randn(12, 7), dim=1)
        distances = distance_matrix(embeddings)
        pos, neg = batch_hard_mining(distances, labels)
        for i, label in enumerate(labels):
            positives = [j for j in range(len(labels)) if j != i and labels[j] == label]
            negatives = [j for j in range(len(labels)) if labels[j] != label]
            assert pos[i].item() == max(positives, key=lambda j: distances[i, j].item())
            assert neg[i].item() == min(negatives, key=lambda j: distances[i, j].item())


def test_hand_computed_triplet_loss_and_diagnostics():
    embeddings = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    labels = torch.tensor([0, 0, 1, 1])
    loss, diagnostics = TripletLoss(margin=0.3)(embeddings, labels)
    expected = 2 - math.sqrt(2) + 0.3
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert diagnostics["mean_d_ap"].item() == pytest.approx(2.0, abs=1e-6)
    assert diagnostics["mean_d_an"].item() == pytest.approx(math.sqrt(2), abs=1e-6)
    assert diagnostics["active_fraction"].item() == 1.0
    assert "embedding_std" in diagnostics


def test_random_mining_uses_valid_positive_and_negative():
    labels = torch.tensor([0, 0, 1, 1])
    distances = distance_matrix(F.normalize(torch.randn(4, 3), dim=1))
    pos, neg = random_mining(distances, labels, torch.Generator().manual_seed(4))
    for i in range(4):
        assert labels[pos[i]] == labels[i] and pos[i] != i
        assert labels[neg[i]] != labels[i]


def test_distance_and_triplet_paths_pass_gradcheck():
    torch.manual_seed(8)
    raw = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    assert torch.autograd.gradcheck(lambda x: distance_matrix(F.normalize(x, dim=1)), (raw,))
    loss_fn = TripletLoss(margin=0.3)
    assert torch.autograd.gradcheck(lambda x: loss_fn(F.normalize(x, dim=1), labels)[0], (raw,))


def test_identical_embeddings_have_finite_gradients():
    embeddings = torch.tensor(
        [[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]], requires_grad=True
    )
    loss, _ = TripletLoss()(embeddings, torch.tensor([0, 0, 1, 1]))
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_overfit_one_pk_batch_in_300_steps():
    torch.manual_seed(10)
    raw = torch.nn.Parameter(torch.randn(12, 8))
    labels = torch.arange(4).repeat_interleave(3)
    optimizer = Adam([raw], lr=0.03)
    loss_fn = TripletLoss(margin=0.3)

    for _ in range(300):
        optimizer.zero_grad()
        loss, _ = loss_fn(F.normalize(raw, dim=1), labels)
        loss.backward()
        optimizer.step()

    embeddings = F.normalize(raw.detach(), dim=1)
    distances = distance_matrix(embeddings)
    pos, neg = batch_hard_mining(distances, labels)
    rows = torch.arange(len(labels))
    assert loss.item() < 0.01
    assert torch.all(distances[rows, pos] < distances[rows, neg])
