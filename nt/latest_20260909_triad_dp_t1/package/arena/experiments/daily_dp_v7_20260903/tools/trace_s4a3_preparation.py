"""Trace only causal preparation-pipeline actions; diagnostic, never selection."""
from pathlib import Path
import argparse, json, sys

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "native/build"))
import _dp7_native as native


def pass_action(obs, seat):
    return {"farmer": ["PASS"], "hands": [["PASS"] for _ in obs["farms"][seat]["hands"]], "market": []}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seed", type=int, default=20262701)
    p.add_argument("--seat", type=int, default=0)
    p.add_argument("--through-day", type=int, default=6)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(a.configs).read_text())[a.label]
    env = native.Env(a.seed)
    ctl = native.Controller(config)
    rows = []
    while not env.done:
        observations = [env.observation(0), env.observation(1)]
        obs = observations[a.seat]
        if obs["day"] > a.through_day:
            break
        before = ctl.debug()
        own = obs["farms"][a.seat]
        positions = [list(own["farmer"])] + [list(h) for h in own["hands"]]
        actions = [pass_action(observations[0], 0), pass_action(observations[1], 1)]
        actions[a.seat] = ctl.act(env, a.seat)
        after = ctl.debug()
        unit_actions = [actions[a.seat]["farmer"]] + actions[a.seat]["hands"]
        if after["preparation_pipeline_checks"] > before["preparation_pipeline_checks"]:
            rows.append({
                "step": obs["step"], "day": obs["day"], "hour": obs["hour"],
                "positions": positions, "unit_actions": unit_actions,
                "market": actions[a.seat]["market"], "debug": after,
            })
        env.step(actions)
    payload = {"status": "DIAGNOSTIC_ONLY", "seed": a.seed, "seat": a.seat, "label": a.label, "rows": rows}
    (out / "trace.json").write_text(json.dumps(payload, indent=2), encoding="utf8")
    for row in rows:
        print(json.dumps({k: row[k] for k in ("step", "day", "hour", "positions", "unit_actions", "market")}))


if __name__ == "__main__":
    main()
