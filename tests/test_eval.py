import numpy as np
import pytest

from src.eval import average_precision, retrieval_metrics, superclass_error_share
from src.retrieval import BruteForceIndex


def test_average_precision_hand_case():
    assert average_precision([1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)


def test_perfectly_separated_embeddings_have_perfect_metrics():
    embeddings = np.array([
        [1, 0], [0.99, 0.01], [-1, 0], [-0.99, 0.01],
        [0, 1], [0.01, 0.99], [0, -1], [0.01, -0.99],
    ], dtype=np.float32)
    labels = np.repeat(np.arange(4), 2)
    metrics = retrieval_metrics(embeddings, labels, chunk_size=3)
    assert metrics["recall@1"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["recall@10"] == 1.0
    assert metrics["map"] == 1.0


def test_index_self_exclusion_returns_other_images():
    embeddings = np.array([[1, 0], [1, 0], [-1, 0]], dtype=np.float32)
    index = BruteForceIndex(embeddings, [4, 4, 9], chunk_size=1)
    indices, distances, labels = index.search(
        embeddings, 1, query_indices=np.arange(3)
    )
    assert indices[0, 0] == 1 and indices[1, 0] == 0
    assert indices[2, 0] in (0, 1)
    assert all(indices[row, 0] != row for row in range(3))
    assert distances[:, 0].tolist() == pytest.approx([0, 0, 4])
    assert labels[:, 0].tolist() == [4, 4, 4]


def test_l2_search_matches_direct_distance_ranking():
    rng = np.random.default_rng(12)
    embeddings = rng.normal(size=(20, 7)).astype(np.float32)
    query_rows = np.array([2, 9, 14])
    expected = ((embeddings[query_rows, None] - embeddings[None]) ** 2).sum(axis=2)
    expected[np.arange(len(query_rows)), query_rows] = np.inf
    expected = np.argsort(expected, axis=1)[:, :5]
    actual = BruteForceIndex(embeddings, np.arange(20), chunk_size=2).search(
        embeddings[query_rows], 5, query_indices=query_rows
    )[0]
    assert np.array_equal(actual, expected)


def test_normalized_l2_and_cosine_have_same_rankings():
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(60, 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = np.arange(len(embeddings))
    l2 = BruteForceIndex(embeddings, ids, metric="l2").search(
        embeddings, 5, query_indices=ids
    )[0]
    cosine = BruteForceIndex(embeddings, ids, metric="cosine").search(
        embeddings, 5, query_indices=ids
    )[0]
    assert np.array_equal(l2, cosine)


def test_recall_precision_invariants_and_superclass_errors():
    embeddings = np.array([
        [1, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [-1, 0], [-0.9, 0.1]
    ], dtype=np.float32)
    labels = np.array([0, 0, 1, 1, 2, 2])
    result = retrieval_metrics(embeddings, labels)
    assert result["recall@1"] <= result["recall@5"] <= result["recall@10"]
    assert result["precision@5"] <= result["recall@5"]
    analysis = superclass_error_share(
        [0, 0, 2], [1, 2, 3], [0, 0, 1, 1]
    )
    assert analysis == {"same_superclass_error_share": 2 / 3, "top1_errors": 3}


def test_random_gaussian_vectors_are_near_chance():
    rng = np.random.default_rng(42)
    labels = np.repeat(np.arange(100), 60)
    embeddings = rng.normal(size=(len(labels), 64)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    result = retrieval_metrics(embeddings, labels, metric="cosine", chunk_size=128)
    expected = 59 / 5999
    assert result["map"] == pytest.approx(expected, abs=0.002)
    assert result["recall@1"] == pytest.approx(expected, abs=0.002)


def test_faiss_cpu_matches_numpy_when_available():
    pytest.importorskip("faiss")
    from src.eval import faiss_topk_matches

    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(80, 12)).astype(np.float32)
    result = faiss_topk_matches(embeddings, k=5)
    assert result["topk_match"]
