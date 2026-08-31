import importlib.util
from pathlib import Path
import tempfile

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "export_quantized_counter_q_policy.py"
SPEC = importlib.util.spec_from_file_location("export_quantized", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_loads_compact_switch_search_states() -> None:
    feature_count = len(MODULE.route_switch_feature_names())
    states = np.arange(2 * feature_count, dtype=np.float32).reshape(
        1, 1, 1, 1, 2, feature_count
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "compact.npz"
        np.savez(path, states=states, checkpoints=np.asarray([144]))
        matrices, names = MODULE._load_feature_panel(path, [144], feature_count)
    assert names == MODULE.route_switch_feature_names()
    assert matrices[0].shape == (2, feature_count)
    assert np.array_equal(matrices[0], states.reshape(2, feature_count))


if __name__ == "__main__":
    test_loads_compact_switch_search_states()
