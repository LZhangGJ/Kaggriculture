from __future__ import annotations

import numpy as np

from kaggriculture_lab.expert_sampling import cluster_balanced_indices


def test_cluster_balanced_indices_equalize_imbalanced_experts() -> None:
    labels = [0, 0, 1, 1, 1, 1, 1, 1]
    order = cluster_balanced_indices(labels, np.random.default_rng(7))
    sampled = np.asarray(labels)[order]
    counts = np.bincount(sampled, minlength=2)
    assert len(order) == len(labels)
    assert abs(int(counts[0]) - int(counts[1])) <= 1
    assert set(order).issubset(set(range(len(labels))))
