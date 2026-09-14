"""Trusted referee with one isolated JSONL subprocess per player."""
from __future__ import annotations

import importlib.metadata
import json
import queue
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path

from .sandbox import cleanup, docker_args, preflight
from .store import file_hash, now, write


class AgentFailure(Exception):
    pass


class AgentProcess:
    def __init__(self, args, timeout):
        self.timeout = timeout
        self.q = queue.Queue(maxsize=2)
        self.diagnostics = deque(maxlen=16)
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._errors, daemon=True).start()

    def _errors(self):
        while True:
            block = self.proc.stderr.read(4096)
            if not block:
                return
            self.diagnostics.append(block.decode(errors="replace"))

    def _reader(self):
        while True:
            line = self.proc.stdout.readline(1024 * 1024 + 1)
            try:
                self.q.put(line, timeout=self.timeout)
            except queue.Full:
                self.proc.kill()
                return
            if not line or len(line) > 1024 * 1024:
                return

    def __call__(self, obs, configuration):
        try:
            data = json.dumps({"observation": obs, "configuration": configuration}, allow_nan=False).encode() + b"\n"
            # A blocked pipe must not freeze the referee indefinitely.
            failure = []
            def send():
                try:
                    self.proc.stdin.write(data)
                    self.proc.stdin.flush()
                except (OSError, ValueError) as e:
                    failure.append(e)
            thread = threading.Thread(target=send, daemon=True)
            thread.start()
            thread.join(self.timeout)
            if thread.is_alive() or failure:
                raise AgentFailure("Input timeout or broken pipe")
            line = self.q.get(timeout=self.timeout)
            if not line or len(line) > 1024 * 1024:
                raise AgentFailure("Missing/oversized JSONL action")
            result = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            if not isinstance(result, dict):
                raise AgentFailure("Action must be an object")
            return result
        except (OSError, ValueError, queue.Empty) as e:
            raise AgentFailure(str(e)) from e

    def ready(self, timeout):
        try:
            line = self.q.get(timeout=timeout)
            if json.loads(line).get("arena_ready") is not True:
                raise RuntimeError("Sandbox startup receipt missing")
        except (queue.Empty, ValueError, AttributeError) as e:
            raise RuntimeError("Sandbox build/startup failed") from e

    def close(self):
        self.proc.kill()
        self.proc.wait(timeout=10)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            stream.close()


def official_game(agents, game, contract):
    if importlib.metadata.version("kaggle-environments") != contract["version"]:
        raise RuntimeError("Referee version mismatch")
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"seed": game["seed"], "episodeSteps": contract["episode_steps"]}, debug=False)
    failures = {}
    wrappers = []
    start = time.monotonic()
    for i, agent in enumerate(agents):
        def call(obs, config, index=i, policy=agent):
            if time.monotonic() - start > contract["game_timeout_seconds"]:
                raise RuntimeError("Game deadline exceeded")
            try:
                return policy(obs, config)
            except AgentFailure as e:
                failures[index] = str(e)[:300]
                raise
        wrappers.append(call)
    env.run(wrappers)
    state = env.state
    status = [s.status for s in state]
    invalid = [i for i, s in enumerate(status) if s in ("ERROR", "INVALID", "TIMEOUT")]
    invalid = sorted(set(invalid) | set(failures))
    if len(invalid) == 2:
        # Two broken agents must never collect a draw as valid evidence.
        return {"resolved": False, "reason": "both_agents_failed", "statuses": status}
    if invalid:
        return {"resolved": True, "outcome": "win1" if invalid[0] == 0 else "win0",
                "cash": None, "reason": "agent_failure", "statuses": status, "terminal": False}
    if not all(s == "DONE" for s in status):
        raise RuntimeError("Referee did not reach terminal state")
    rewards = [float(s.reward) for s in state]
    return {"resolved": True, "outcome": "draw" if rewards[0] == rewards[1] else ("win0" if rewards[0] > rewards[1] else "win1"),
            "cash": rewards, "reason": "terminal", "terminal": True, "statuses": status, "steps": len(env.steps)}


def play(root, manifest, game, cfg):
    root = Path(root)
    preflight(cfg)
    processes, names = [], []
    try:
        for seat, aid in enumerate(game["agents"]):
            a = manifest["agents"][aid]
            archive = root / "artifacts" / (a["archive"] + ".zip")
            if file_hash(archive) != a["archive"] or a["image"] != cfg["image"]:
                raise RuntimeError("Artifact/image changed since validation")
            mpath = root / "private" / ("manifest-" + aid + ".json")
            write(mpath, a["manifest"])
            name = "arena-" + game["id"][:24] + "-" + str(seat)
            cleanup(name)  # Recover a container left by an interrupted attempt.
            names.append(name)
            args = docker_args(cfg, name, [(archive, "/input.zip"), (mpath, "/manifest.json")],
                               ["python", "/opt/arena/inside.py", "run", str(game["agent_seed"])],
                               a["manifest"].get("resources", {}).get("scratch_mb", 512))
            # Docker stdin must remain open for the JSONL stream.
            args.insert(2, "-i")
            processes.append(AgentProcess(args, cfg["turn_timeout_seconds"]))
            processes[-1].ready(cfg["game_timeout_seconds"])
        result = official_game(processes, game, manifest["contract"])
        result["private_diagnostics"] = ["".join(p.diagnostics)[-16000:] for p in processes]
    finally:
        for p in processes:
            p.close()
        for name in names:
            cleanup(name)
    return {**result, "game": game["id"], "agents": game["agents"], "finished": now()}


if __name__ == "__main__":
    import sys
    from .store import read
    root, run_id, game_id, destination = sys.argv[1:]
    manifest = read(Path(root) / "runs" / run_id / "manifest.json")
    game = next(g for g in manifest["games"] if g["id"] == game_id)
    cfg = read(Path(root) / "config.json")
    if cfg["image"] != manifest["contract"]["image"]:
        raise RuntimeError("Image changed during frozen run")
    cfg.update({k: manifest["contract"][k] for k in ("turn_timeout_seconds", "game_timeout_seconds", "cpus_per_agent", "memory_mb")})
    write(destination, play(root, manifest, game, cfg))
