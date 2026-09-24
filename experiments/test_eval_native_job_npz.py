import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.eval_native_job_npz import (
    _write_exclusive,
    build_report,
)


class NativeJobNpzEvalTest(unittest.TestCase):
    def test_report_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {
                "seed": np.repeat(np.arange(10, 14), 2).astype(np.uint64),
                "seat": np.tile([0, 1], 4).astype(np.int32),
                "opponent": np.repeat([1, 2, 1, 2], 2).astype(np.int32),
                "route": np.full(8, -1, dtype=np.int32),
                "policy_seed": np.arange(100, 108, dtype=np.uint64),
            }
            parent = root / "parent.npz"
            child = root / "child.npz"
            np.savez(parent, **identity,
                     own_cash=100 + np.array([-1, -1, 3, 3, -2, -2, 4, 4]),
                     rival_cash=np.full(8, 100))
            np.savez(child, **identity,
                     own_cash=100 + np.array([2, 2, -3, -3, 5, 5, 4, 4]),
                     rival_cash=np.full(8, 100))

            report = build_report(parent, child, "test", 200, 7)
            self.assertEqual(report["overall"]["delta_wins"], 2)
            self.assertEqual(report["overall"]["rescue"], 4)
            self.assertEqual(report["overall"]["hurt"], 2)
            self.assertEqual(set(report["by_opponent"]), {"thomas", "meta"})
            output = root / "report.json"
            _write_exclusive(output, report)
            with self.assertRaises(FileExistsError):
                _write_exclusive(output, report)

            drifted = dict(identity)
            drifted["policy_seed"] = identity["policy_seed"].copy()
            drifted["policy_seed"][-1] += 1
            mismatch = root / "mismatch.npz"
            np.savez(mismatch, **drifted,
                     own_cash=np.full(8, 100), rival_cash=np.full(8, 100))
            with self.assertRaisesRegex(RuntimeError, "policy_seed"):
                build_report(parent, mismatch, "test", 10, 7)

            invalid = root / "invalid-route.npz"
            np.savez(invalid, **{**identity, "route": np.zeros(8, dtype=np.int32)},
                     own_cash=np.full(8, 100), rival_cash=np.full(8, 100))
            with self.assertRaisesRegex(RuntimeError, "route identity"):
                build_report(invalid, invalid, "test", 10, 7)


if __name__ == "__main__":
    unittest.main()
