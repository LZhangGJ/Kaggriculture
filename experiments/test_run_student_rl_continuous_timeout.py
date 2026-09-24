"""Small regression check for a child that hangs after saving its rollout."""

import subprocess
import sys
import json
import tempfile
from pathlib import Path

from experiments.run_student_rl_continuous import _validated_round, _wait_child


def test_child_timeout() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    beats = []
    try:
        try:
            _wait_child(child, 0.05, lambda: beats.append(1), interval=0.02)
        except TimeoutError:
            pass
        else:
            raise AssertionError("hung child was not stopped")
        assert child.poll() is not None and beats
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_resume_metadata_gate() -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "previous.pt"
        checkpoint.touch()
        output = Path(directory) / "next.pt"
        metrics = Path(directory) / "round.json"
        metrics.write_text(json.dumps({
            "status": "PASS", "games": 1536, "illegal": 0, "fallbacks": 0,
            "checkpoint_in": str(checkpoint), "checkpoint_out": str(output),
            "environment_seed_min": 123,
        }))
        assert _validated_round(metrics, checkpoint.resolve(), output, 123)["games"] == 1536
        try:
            _validated_round(metrics, checkpoint.resolve(), output, 124)
        except RuntimeError:
            pass
        else:
            raise AssertionError("restart accepted the wrong seed block")


if __name__ == "__main__":
    test_child_timeout()
    test_resume_metadata_gate()
