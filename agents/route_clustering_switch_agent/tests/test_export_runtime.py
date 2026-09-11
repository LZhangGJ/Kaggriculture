"""Check that the source-tree exporter preserves the selected deployment policy."""

from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_exported_runtimes_keep_the_frozen_opening_and_controller(tmp_path):
    runtime = ROOT / "runtime"
    holdout = tmp_path / "holdout.json"
    manifest = json.loads((runtime / "MANIFEST.json").read_text(encoding="utf-8"))
    holdout.write_text(json.dumps({"best": manifest["final_holdout"]}), encoding="utf-8")
    exported = tmp_path / "expanded"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/export_teammate_meta_submission.py"),
        "--base", str(runtime / "teammate_base.py"),
        "--actions", str(runtime / "route_actions.json.zlib"),
        "--metadata", str(runtime / "route_library.json"),
        "--policy", str(runtime / "route_policy.json"),
        "--nash", str(runtime / "opening_nash.json"),
        "--holdout", str(holdout), "--forced-opening", "G001",
        "--output", str(exported),
    ], check=True, capture_output=True)
    for name in ("teammate_base.py", "route_actions.json.zlib", "route_library.json"):
        assert (exported / name).read_bytes() == (runtime / name).read_bytes()
    assert json.loads((exported / "MANIFEST.json").read_text())["forced_opening"] == "G001"

    single = tmp_path / "main.py"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/export_single_file_submission.py"),
        "--source-dir", str(exported), "--forced-opening", "G001",
        "--output", str(single),
    ], check=True, capture_output=True)

    # Separate processes avoid sharing the base policy's module state between agents.
    probe = """
import json, runpy, sys
ns = runpy.run_path(sys.argv[1])
controller = ns.get('_CONTROLLER', ns.get('_S_CONTROLLER'))
expanded = ns.get('_EXPANDED', ns.get('_S_EXPANDED'))
trace = []
for step in (0, 144, 168, 216, 0):
    obs = {'step': step, 'player': 0, 'farms': [{}, {}],
           'private': {}, 'market': {}, 'shops': []}
    route, changed = controller.observe(obs, expanded.action_tapes)
    trace.append([route, changed, controller.switched])
print(json.dumps({'forced_opening': controller.forced_opening,
    'route_count': len(expanded.action_tapes),
    'checkpoints': [step for step, tree in controller.nodes['G001']],
    'trace': trace}, sort_keys=True))
"""
    snapshots = []
    for entry in (ROOT / "main.py", runtime / "main.py", exported / "main.py", single):
        output = subprocess.run(
            [sys.executable, "-I", "-c", probe, str(entry)],
            check=True, capture_output=True,
        )
        snapshots.append(json.loads(output.stdout))
    assert snapshots[0]["forced_opening"] == "G001"
    assert snapshots[0]["route_count"] == 175
    assert snapshots[0]["checkpoints"] == [144, 168, 216]
    assert all(snapshot == snapshots[0] for snapshot in snapshots[1:])
