import numpy as np

from churnguard.evaluate import best_threshold, compute_metrics


def test_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["recall"] == 1.0 and m["precision"] == 1.0 and m["roc_auc"] == 1.0


def test_expensive_false_negatives_lower_the_threshold():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    cautious = best_threshold(y, proba, fn_cost=1000, fp_cost=10)
    strict = best_threshold(y, proba, fn_cost=10, fp_cost=1000)
    assert cautious < strict
