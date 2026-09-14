"""Docker boundary. No direct host execution of submitted code."""
from __future__ import annotations

import json
import subprocess
import uuid
import threading
from pathlib import Path

from .store import file_hash, now, read, write


def preflight(cfg):
    image = cfg.get("image") or ""
    if "@sha256:" not in image and not image.startswith("sha256:"):
        raise RuntimeError("Configure an immutable Docker image digest first")
    subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True, timeout=15)


def docker_args(cfg, name, mounts, command, scratch_mb=512):
    args = ["docker", "run", "--rm", "--name", name, "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
            "--cpus", str(cfg["cpus_per_agent"]), "--memory", f"{cfg['memory_mb']}m",
            "--memory-swap", f"{cfg['memory_mb']}m", "--user", "65534:65534",
            "--tmpfs", f"/work:rw,exec,nosuid,size={scratch_mb}m,mode=1777",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m,mode=1777", "--workdir=/work",
            "--log-driver=none"]
    for source, target in mounts:
        args += ["--mount", f"type=bind,source={Path(source).resolve()},target={target},readonly"]
    return [*args, cfg["image"], *command]


def cleanup(name):
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)


def bounded_run(args, timeout, limit=1024*1024):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffers = [bytearray(), bytearray()]
    exceeded = []
    def drain(stream, buffer):
        while True:
            block = stream.read(8192)
            if not block:
                return
            if len(buffer) + len(block) > limit:
                exceeded.append(True)
                proc.kill()
                return
            buffer.extend(block)
    threads = [threading.Thread(target=drain, args=(stream, buf), daemon=True) for stream, buf in zip((proc.stdout, proc.stderr), buffers)]
    for t in threads:
        t.start()
    try:
        proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for t in threads:
            t.join(timeout=5)
        proc.stdout.close()
        proc.stderr.close()
    if exceeded:
        raise ValueError("Sandbox output limit exceeded")
    return subprocess.CompletedProcess(args, proc.returncode, bytes(buffers[0]), bytes(buffers[1]))


def validate_agent(root, aid):
    root = Path(root)
    cfg = read(root / "config.json")
    preflight(cfg)
    path = root / "agents" / f"{aid}.json"
    a = read(path)
    archive = root / "artifacts" / (a["archive"] + ".zip")
    if file_hash(archive) != a["archive"]:
        raise ValueError("Stored archive hash mismatch")
    m = a["manifest"]
    r = m.get("resources", {})
    if r.get("cpus", 1) > cfg["cpus_per_agent"] or r.get("memory_mb", 512) > cfg["memory_mb"]:
        raise ValueError("Agent exceeds configured worker limits")
    mp = root / "private" / f"manifest-{aid}.json"
    write(mp, m)
    name = "arena-check-" + uuid.uuid4().hex
    args = docker_args(cfg, name, [(archive, "/input.zip"), (mp, "/manifest.json")],
                       ["python", "/opt/arena/inside.py", "check"], r.get("scratch_mb", 512))
    try:
        result = bounded_run(args, timeout=cfg["game_timeout_seconds"])
        if result.returncode:
            raise ValueError("Sandbox build failed: " + result.stderr.decode(errors="replace")[-2000:])
        receipt = json.loads(result.stdout)
        if receipt.get("ok") is not True:
            raise ValueError("Invalid sandbox receipt")
        a.update(build_verified=True, status="pending", validated=now(), image=cfg["image"])
        write(path, a)
        return a
    finally:
        cleanup(name)
