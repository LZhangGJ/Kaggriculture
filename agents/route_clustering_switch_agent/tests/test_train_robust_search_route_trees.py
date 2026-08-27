from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from train_robust_search_route_trees import zero_feature_prefixes  # noqa: E402


def test_zero_feature_prefixes_preserves_schema() -> None:
    matrix = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = zero_feature_prefixes(
        matrix, ["self_money", "opponent_money", "market_price"], ("opponent_",)
    )
    assert result.tolist() == [[1.0, 0.0, 3.0]]
    assert matrix.tolist() == [[1.0, 2.0, 3.0]]
