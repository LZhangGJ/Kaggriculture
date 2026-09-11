from __future__ import annotations

import copy
import json
import sys
import time
import traceback
from pathlib import Path


def load_package(package: Path):
    sys.path.insert(0, str(package))
    import arena
    return arena


def game(job: tuple) -> dict:
    package, name, seed, seat, binary = job
    arena = load_package(Path(package))
    started = time.monotonic()
    public = agent = None
    row = {
        "opponent": name,
        "seed": seed,
        "opponent_seat": seat,
        "binary": binary,
        "runtime_error": None,
    }
    try:
        public = arena.FreshPublic(name)
        config = json.loads((Path(package) / "policy/config.json").read_text())
        agent = arena.codec.Agent(config=config, binary_path=binary)
        env = arena.LocalGame(seed, arena.load_engine())
        actions = []
        latency_max = 0.0
        while not env.done and env.t < 719:
            observations = [env.observation(i) for i in (0, 1)]
            own = observations[1 - seat]
            tick = time.monotonic()
            own_action = agent(own)
            latency_max = max(latency_max, time.monotonic() - tick)
            rival_action = public.call(observations[seat], copy.deepcopy(env.configuration))
            issued = [None, None]
            issued[1 - seat] = own_action
            issued[seat] = rival_action
            actions.append(copy.deepcopy(issued))
            env.advance(issued)
        observations = [env.observation(i) for i in (0, 1)]
        cash = [farm["money"] for farm in observations[0]["farms"]]
        margin = cash[1 - seat] - cash[seat]
        debug = agent.debug()
        row.update(
            steps=env.t,
            own_cash=cash[1 - seat],
            opponent_cash=cash[seat],
            margin=margin,
            win=margin > 0,
            tie=margin == 0,
            action_hash=arena.digest(actions),
            latency_max=latency_max,
            workflow=debug.get("workflow", {}),
            workflow_diagnostic=debug.get("workflow_diagnostic", {}),
        )
    except Exception:
        row["runtime_error"] = traceback.format_exc()
    finally:
        if agent:
            agent.close()
        if public:
            public.close()
    row["seconds"] = time.monotonic() - started
    return row
