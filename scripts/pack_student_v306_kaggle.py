#!/usr/bin/env python3
"""Package the sampled v306 student and replay opening as a Kaggle tarball."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from scripts.pack_kaggle_submission import BINARY, TEXT


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/student-v306/actor.pt"
METRICS = ROOT / "models/student-v306/metrics.json"
MANIFEST = ROOT / "models/student-v306/manifest.json"
EXTRA = (
    "experiments/student_action_event_agent.py",
    "experiments/student_economic_features_v1.py",
    "experiments/student_v3_runtime_model.py",
)
TOKENIZER = ROOT / "experiments/student_v306_vendor/kaggrl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "build/student-v306-kaggle.tar.gz")
    args = parser.parse_args()
    metrics = json.loads(METRICS.read_text())
    checkpoint = CHECKPOINT.read_bytes()
    if (metrics.get("status") != "PASS" or metrics.get("student_intraday") != 0 or
            hashlib.sha256(checkpoint).hexdigest() !=
            metrics.get("checkpoint_out_sha256")):
        raise ValueError("v306 checkpoint does not match the accepted RL round")
    if args.so.read_bytes()[:4] != b"\x7fELF" or b"\x3e\x00" != args.so.read_bytes()[18:20]:
        raise ValueError("expected an x86_64 ELF shared library")

    files = {name: (ROOT / name).read_bytes() for name in (*TEXT, *BINARY, *EXTRA)}
    files["policy/r1/agent.so"] = args.so.read_bytes()
    files["models/student-v306/v3-ppo-native-job-economic-v1-v306-1536g.pt"] = checkpoint
    files["models/student-v306/manifest.json"] = MANIFEST.read_bytes()
    for name in ("tokenizer.py", "constants.py", "structures.py"):
        files[f"experiments/student_v306_vendor/kaggrl/{name}"] = (
            TOKENIZER / name).read_bytes()
    files["experiments/student_v306_vendor/kaggrl/__init__.py"] = (
        TOKENIZER / "__init__.py").read_bytes()
    files["main.py"] = b"""import os
import sys
from pathlib import Path
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
root = Path(__file__).resolve().parent if '__file__' in globals() else Path(sys.path[-1])
os.environ['STUDENT_R1_BINARY'] = str(root / 'policy/r1/agent.so')
os.environ['STUDENT_CHECKPOINT'] = str(root / 'models/student-v306/v3-ppo-native-job-economic-v1-v306-1536g.pt')
os.environ['STUDENT_V3_MANIFEST'] = str(root / 'models/student-v306/manifest.json')
import torch
torch.set_num_threads(1)
from experiments.student_action_event_agent import agent
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.mode = 0o755 if name.endswith(".so") else 0o644
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    print(json.dumps({"output": str(args.output), "files": len(files),
                      "bytes": args.output.stat().st_size,
                      "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
