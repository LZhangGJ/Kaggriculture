import importlib.util
import json
from pathlib import Path
import tempfile
import zlib


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "materialize_quantized_route_q_agent.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_route_q", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_materializes_only_policy_routes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        policy = root / "policy.json"
        actions = root / "actions.zlib"
        manifest = root / "manifest.json"
        output = root / "out"
        policy.write_text(json.dumps({
            "openings": ["A"], "targets": ["A", "B"], "checkpoints": [144],
        }), encoding="utf-8")
        actions.write_bytes(zlib.compress(json.dumps({
            "A": [{"market": []}], "B": [{"market": ["BUY"]}],
            "UNUSED": [],
        }).encode()))
        manifest.write_text(json.dumps({"routes": [
            {"family": family, "base": "X", "checkpoint": 144, "donor": family}
            for family in ("A", "B", "UNUSED")
        ]}), encoding="utf-8")
        result = MODULE.materialize(policy, [(actions, manifest)], output)
        deployed = json.loads(zlib.decompress(
            (output / "route_actions.json.zlib").read_bytes()
        ))
    assert result["routes"] == ["A", "B"]
    assert list(deployed) == ["A", "B"]


if __name__ == "__main__":
    test_materializes_only_policy_routes()
