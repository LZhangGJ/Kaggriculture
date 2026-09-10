"""Build, package, and locally validate P16 JointAFS R1 for Kaggle Public.

This script performs no network or Kaggle action.  It rebuilds the frozen AFS
source with the proven portable GCC 11 toolchain, checks behavior against the
development binary, packages the agent, and runs the extracted official
Kaggriculture 1.32.7 file entry.
"""
from __future__ import annotations

from pathlib import Path
import gzip
import hashlib
import importlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = (
    ROOT
    / "gpt_review/gpt_code/Kaggriculture_P16_JointAFS_R1_20260910"
    / "Kaggriculture_P16_JointAFS_R1_20260910"
)
OFFICIAL = ROOT / "research/official_env_update_20260815/extracted_1_32_7"
TOOLCHAIN = ROOT / "submission/pending_dp27_fixed_macro_dynamic_20260904/toolchain/root/usr"
OUT = HERE / "AFS"
REFERENCE = SRC / "policy/joint.so"
NAMESPACE = "afs_runtime"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_map(base: Path) -> dict[str, str]:
    suffixes = {".hpp", ".cpp", ".h", ".inc", ".py", ".json"}
    return {
        p.relative_to(base).as_posix(): sha(p)
        for p in sorted(base.rglob("*"))
        if p.is_file() and p.suffix in suffixes and not p.name.endswith(".build.json")
    }


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing release folder: {OUT}")
    (OUT / "source").mkdir(parents=True)
    destination = OUT / "source/policy"
    shutil.copytree(
        SRC / "policy",
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.build.json"),
    )
    frozen = json.loads((SRC / "RELEASE_BUILD.json").read_text(encoding="utf-8"))
    before = source_map(destination)
    if before != frozen["sources"]:
        raise RuntimeError("Copied AFS source does not match frozen RELEASE_BUILD.json")
    if sha(REFERENCE) != frozen["binary_sha256"]:
        raise RuntimeError("Development AFS binary does not match frozen receipt")

    compiler = TOOLCHAIN / "bin/g++-11"
    flags = [
        "-nostdinc++",
        "-isystem", str(TOOLCHAIN / "include/c++/11"),
        "-isystem", str(TOOLCHAIN / "include/x86_64-linux-gnu/c++/11"),
        "-L" + str(TOOLCHAIN / "lib/gcc/x86_64-linux-gnu/11"),
        "-B" + str(TOOLCHAIN / "lib/gcc/x86_64-linux-gnu/11") + "/",
        *frozen["flags"],
        "-static-libstdc++", "-static-libgcc",
    ]
    binary = OUT / "agent.so"
    command = [
        str(compiler), *flags,
        str(destination / "bridge.cpp"),
        str(destination / "executor/vendor/simulator.cpp"),
        "-o", str(binary),
    ]
    started = time.perf_counter()
    with (OUT / "build.log").open("w", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
    deps = subprocess.check_output(["readelf", "-d", str(binary)], text=True)
    versions = subprocess.check_output(["readelf", "--version-info", str(binary)], text=True)
    glibc = sorted({tuple(map(int, x.split("."))) for x in re.findall(r"GLIBC_([0-9.]+)", versions)})
    if "libstdc++" in deps or "libgcc_s" in deps:
        raise RuntimeError("Portable binary retained a libstdc++ or libgcc_s dependency")
    if glibc and max(glibc) > (2, 34):
        raise RuntimeError(f"Portable binary requires too-new GLIBC: {max(glibc)}")
    if source_map(destination) != before:
        raise RuntimeError("Source changed during compilation")
    save(
        OUT / "BUILD_RECEIPT.json",
        {
            "status": "PASS_PORTABLE_GCC11",
            "version": "P16 JointAFS R1",
            "command": command,
            "compiler": subprocess.check_output([str(compiler), "--version"], text=True),
            "seconds": time.perf_counter() - started,
            "source_hashes": before,
            "binary_sha256": sha(binary),
            "reference_binary_sha256": sha(REFERENCE),
            "config_sha256": sha(destination / "config.json"),
            "required_glibc_versions": glibc,
            "dynamic_dependencies": deps,
            "cpu_target": "generic x86-64; static libstdc++ and libgcc",
        },
    )


def parity() -> None:
    opponents = [x["id"] for x in json.loads((SRC / "POOL.json").read_text(encoding="utf-8"))]
    seed = 2609102991
    worker = HERE / "parity_worker.py"
    reference_file = OUT / "parity_reference.json"
    portable_file = OUT / "parity_portable.json"
    # The two shared libraries export identical C ABI names.  Loading both in
    # one process can interpose symbols despite distinct ctypes handles, so
    # each arm must run in a fresh interpreter.
    subprocess.run(
        [sys.executable, str(worker), str(REFERENCE), str(reference_file), str(seed)],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(worker), str(OUT / "agent.so"), str(portable_file), str(seed)],
        check=True,
    )
    reference_rows = json.loads(reference_file.read_text(encoding="utf-8"))
    portable_rows = json.loads(portable_file.read_text(encoding="utf-8"))
    fields = [
        "steps", "action_hash", "own_cash", "opponent_cash", "margin", "win", "tie",
        "runtime_error",
    ]
    differences = []
    for reference, portable in zip(reference_rows, portable_rows, strict=True):
        for field in fields:
            if reference.get(field) != portable.get(field):
                differences.append(
                    {
                        "opponent": reference["opponent"],
                        "seat": reference["opponent_seat"],
                        "field": field,
                        "reference": reference.get(field),
                        "portable": portable.get(field),
                    }
                )
    result = {
        "status": "PASS" if not differences else "FAIL",
        "version": "P16 JointAFS R1",
        "seed": seed,
        "opponents": opponents,
        "both_seats": True,
        "games_per_binary": len(reference_rows),
        "comparisons": len(reference_rows) * len(fields),
        "fields": fields,
        "differences": differences,
        "reference_binary_sha256": sha(REFERENCE),
        "portable_binary_sha256": sha(OUT / "agent.so"),
    }
    save(OUT / "ACTION_IDENTITY.json", result)
    if differences:
        raise RuntimeError(f"Portable build changed AFS behavior: {differences[:3]}")


def package() -> None:
    runtime = OUT / "runtime"
    package_dir = runtime / NAMESPACE
    package_dir.mkdir(parents=True)
    main = (
        '"""Portable P16 JointAFS R1 Kaggriculture entry."""\n'
        f"from {NAMESPACE}.policy import Agent as _Agent\n\n"
        "_players = {}\n\n"
        "def agent(observation, configuration=None):\n"
        "    seat = int(observation.get('player', 0))\n"
        "    if seat not in _players:\n"
        "        _players[seat] = _Agent()\n"
        "    return _players[seat](observation, configuration)\n"
    )
    (runtime / "main.py").write_text(main, encoding="utf-8")
    shutil.copy2(OUT / "source/policy/agent.py", package_dir / "policy.py")
    shutil.copy2(OUT / "source/policy/config.json", package_dir / "config.json")
    shutil.copy2(OUT / "agent.so", package_dir / "agent.so")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    names = sorted(p.relative_to(runtime).as_posix() for p in runtime.rglob("*") if p.is_file())
    expected = ["main.py"] + [f"{NAMESPACE}/{x}" for x in ("__init__.py", "agent.so", "config.json", "policy.py")]
    if names != sorted(expected):
        raise RuntimeError((names, expected))

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in names:
            data = (runtime / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    output = OUT / "submission.tar.gz"
    output.write_bytes(gzip.compress(buffer.getvalue(), mtime=0))

    check = OUT / "package_check"
    check.mkdir()
    with tarfile.open(output) as archive:
        archive.extractall(check, filter="data")
    hashes = {name: sha(runtime / name) for name in names}
    if not all(sha(check / name) == digest for name, digest in hashes.items()):
        raise RuntimeError("Archive extraction hash mismatch")
    save(
        OUT / "PACKAGE_INTEGRITY.json",
        {
            "status": "PASS",
            "version": "P16 JointAFS R1",
            "archive_sha256": sha(output),
            "bytes": output.stat().st_size,
            "files": hashes,
            "no_replays_or_credentials": True,
            "no_runtime_compilation": True,
        },
    )


def official() -> None:
    sys.path.insert(0, str(OFFICIAL))
    import kaggle_environments as ke
    from kaggle_environments.agent import get_last_callable

    entry = (OUT / "package_check/main.py").resolve()
    function = get_last_callable(entry.read_text(encoding="utf-8"), path=str(entry))
    if function.__name__ != "agent":
        raise RuntimeError("Submission entry did not resolve to agent")
    imported = importlib.import_module(NAMESPACE + ".policy")
    if Path(imported.__file__).resolve() != OUT / "package_check" / NAMESPACE / "policy.py":
        raise RuntimeError("Submission imported a non-packaged policy module")

    games = []
    for seed in (0, 2609102992):
        env = ke.make("kaggriculture", configuration={"seed": seed, "runTimeout": 1200}, debug=False)
        if env.configuration.actTimeout != 1:
            raise RuntimeError("Unexpected official action timeout")
        started = time.perf_counter()
        states = env.run([str(entry), str(entry)])
        errors = [
            {"step": step, "seat": seat, "stderr": log.get("stderr")}
            for step, logs in enumerate(env.logs)
            for seat, log in enumerate(logs)
            if log.get("stderr")
        ]
        durations = [log["duration"] for logs in env.logs for log in logs]
        row = {
            "seed": seed,
            "steps": len(states) - 1,
            "statuses": [state.status for state in states[-1]],
            "rewards": [state.reward for state in states[-1]],
            "errors": errors,
            "seconds": time.perf_counter() - started,
            "max_action_ms": max(durations) * 1000,
            "actions_over_1s": sum(value > 1 for value in durations),
        }
        row["status"] = (
            "PASS"
            if row["steps"] == 719
            and row["statuses"] == ["DONE", "DONE"]
            and not errors
            and row["actions_over_1s"] == 0
            else "FAIL"
        )
        games.append(row)
        if row["status"] != "PASS":
            raise RuntimeError(row)
    integrity = json.loads((OUT / "PACKAGE_INTEGRITY.json").read_text(encoding="utf-8"))
    save(
        OUT / "OFFICIAL_FILE_ACCEPTANCE.json",
        {
            "status": "PASS",
            "version": "P16 JointAFS R1",
            "archive_sha256": integrity["archive_sha256"],
            "games": games,
            "errors": 0,
            "max_action_ms": max(game["max_action_ms"] for game in games),
            "official_interpreter_sha256": sha(
                OFFICIAL / "kaggle_environments/envs/kaggriculture/kaggriculture.py"
            ),
            "scope": "Official 1.32.7 extracted file entry; local hardware; one-second action timeout",
        },
    )


def main() -> None:
    build()
    parity()
    package()
    official()
    print(json.dumps({"status": "READY", "folder": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
