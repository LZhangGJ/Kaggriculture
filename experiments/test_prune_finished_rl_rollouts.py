"""One safety check for completed-rollout retention."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.prune_finished_rl_rollouts import prune


def test_prune_only_completed_old_rollouts() -> None:
    with TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for round_number, status in ((211, "PASS"), (212, "FAILED"),
                                     (220, "PASS")):
            stem = f"v3-ppo-native-job-triad-v{round_number}-1536g"
            (directory / f"{stem}.rollout.npz").write_bytes(b"rollout")
            (directory / f"{stem}.pt").write_bytes(b"checkpoint")
            (directory / f"native-triad-v{round_number}-1536g.bin").write_bytes(b"weights")
            (directory / f"{stem}.metrics.json").write_text(json.dumps({
                "status": status, "games": 1536, "illegal": 0,
                "fallbacks": 0}))
        old = directory / "v3-ppo-native-job-triad-v211-1536g.rollout.npz"
        failed = directory / "v3-ppo-native-job-triad-v212-1536g.rollout.npz"
        recent = directory / "v3-ppo-native-job-triad-v220-1536g.rollout.npz"
        assert prune(directory, min_round=211, keep=8, delete=False)["files"] == 1
        assert old.exists()
        assert prune(directory, min_round=211, keep=8, delete=True)["files"] == 1
        assert not old.exists() and failed.exists() and recent.exists()


if __name__ == "__main__":
    test_prune_only_completed_old_rollouts()
