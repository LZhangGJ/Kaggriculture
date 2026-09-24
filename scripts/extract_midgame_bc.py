#!/usr/bin/env python3
"""Stream kaggriculture-bc-v1 traces into post-handoff, day-level BC records."""

import argparse
import gzip
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path


FORMAT = "kaggriculture-midgame-bc-v1"


def open_text(path, mode="rt"):
    return gzip.open(path, mode, encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def observation(frame, seat):
    public = dict(frame["public"])
    if "private" in public or "player" in public:
        raise ValueError("public state contains a private/player field")
    public["private"] = frame["private"][seat]
    public["player"] = seat
    return public


def day_samples(path, policy_source, diagnostic, start_step=288,
                include_opponent_private_target=False):
    if start_step < 288 or start_step % 24:
        raise ValueError("start_step must be a day boundary at step 288 or later")
    path = Path(path)
    with open_text(path) as stream:
        try:
            meta = json.loads(next(stream))
        except StopIteration as exc:
            raise ValueError(f"empty trajectory: {path}") from exc
        if meta.get("type") != "meta" or meta.get("format") != "kaggriculture-bc-v1":
            raise ValueError(f"unsupported trajectory header: {path}")
        seat = meta.get("seat")
        if seat not in (0, 1):
            raise ValueError(f"invalid seat in {path}: {seat!r}")

        provenance = {
            "policy_source": policy_source,
            "policy_label": meta.get("label"),
            "diagnostic": diagnostic,
            "engine": meta.get("engine"),
            "config_override": meta.get("config_override"),
            "handoff_selector": meta.get("handoff_selector"),
        }
        # Split-only metadata is deliberately outside state: opponent identity is
        # needed to prevent leakage across train/validation, never as a feature.
        opponent_group = meta.get("source_family") or meta.get("bot")
        split_group = {"group_id": f"{meta.get('seed')}:{opponent_group}",
                       "seed": meta.get("seed"), "opponent_group": opponent_group}
        replicate = {"seat": seat, "future_seed": meta.get("future_seed")}
        current_day = None
        decisions = []
        day_state = None
        last_step = None
        terminal_seen = False

        def emit():
            if not decisions:
                return None
            return {
                "type": "day",
                "format": FORMAT,
                "step": decisions[0]["step"],
                "day": current_day,
                "state": day_state,
                "behavior_target": {"hourly_actions": [row["action"] for row in decisions]},
                "decisions": decisions,
                "provenance": provenance,
                "split_only": split_group,
                "replicate": replicate,
                "complete_day": len(decisions) == 24,
            }

        for line_number, line in enumerate(stream, 2):
            row = json.loads(line)
            kind = row.get("type")
            if terminal_seen:
                raise ValueError(f"record after terminal at {path}:{line_number}")
            if kind == "terminal":
                terminal_seen = True
                continue
            if kind == "error":
                raise ValueError(f"failed trajectory {path}: {row.get('error')}")
            if kind != "step":
                raise ValueError(f"unexpected record at {path}:{line_number}: {kind!r}")
            step = row.get("step")
            if last_step is not None and step != last_step + 1:
                raise ValueError(f"non-consecutive steps in {path}: {last_step} -> {step}")
            last_step = step
            if step < start_step:
                continue
            if len(row.get("private", ())) != 2 or len(row.get("actions", ())) != 2:
                raise ValueError(f"invalid player arrays at {path}:{line_number}")
            day = row["public"].get("day")
            if current_day is not None and day != current_day:
                sample = emit()
                if sample:
                    yield sample
                decisions, day_state = [], None
            current_day = day
            own_observation = observation(row, seat)
            if day_state is None:
                day_state = own_observation
            decision = {"step": step, "observation": own_observation, "action": row["actions"][seat]}
            if include_opponent_private_target:
                decision["auxiliary_target"] = {"opponent_private": row["private"][1 - seat]}
            decisions.append(decision)

        sample = emit()
        if sample:
            yield sample
        if not terminal_seen:
            raise ValueError(f"trajectory has no terminal record: {path}")


@contextmanager
def output_stream(path):
    if path is None:
        yield None
        return
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    stream = open_text(temporary, "wt")
    try:
        yield stream
        stream.close()
        os.replace(temporary, path)
    except BaseException:
        stream.close()
        temporary.unlink(missing_ok=True)
        raise


def extract(paths, policy_source, diagnostic, start_step=288,
            include_opponent_private_target=False, output=None):
    summary = {"format": FORMAT, "trajectories": 0, "days": 0, "decisions": 0,
               "complete_days": 0, "seats": [], "policy_source": policy_source,
               "diagnostic": diagnostic}
    seats = set()
    with output_stream(output) as sink:
        if sink:
            sink.write(json.dumps({"type": "meta", "format": FORMAT,
                                   "start_step": start_step,
                                   "policy_source": policy_source,
                                   "diagnostic": diagnostic}, separators=(",", ":")) + "\n")
        for path in paths:
            for sample in day_samples(path, policy_source, diagnostic, start_step,
                                      include_opponent_private_target):
                summary["days"] += 1
                summary["decisions"] += len(sample["decisions"])
                summary["complete_days"] += sample["complete_day"]
                seats.add(sample["replicate"]["seat"])
                if sink:
                    sink.write(json.dumps(sample, separators=(",", ":")) + "\n")
            summary["trajectories"] += 1
    summary["seats"] = sorted(seats)
    return summary


def smoke_paths():
    root = Path(__file__).resolve().parents[1]
    return sorted(root.glob("work/bc-recorder-smoke-thomas-seed200-trajectories/candidate/thomas_2945/*.jsonl.gz"))


def self_check():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl.gz"
        rows = [
            {"type": "meta", "format": "kaggriculture-bc-v1", "label": "candidate",
             "bot": "split-only-opponent", "seed": 7, "seat": 1, "engine": "fast",
             "future_seed": None, "config_override": None, "handoff_selector": "0"},
            {"type": "step", "step": 288,
             "public": {"step": 288, "day": 12, "hour": 0, "farms": [], "market": {}, "town": {}},
             "private": [{"shed": {"WHEAT": 1}}, {"shed": {"WHEAT": 2}}],
             "actions": [{"farmer": ["PASS"]}, {"farmer": ["NORTH"]}]},
            {"type": "terminal", "rewards": [1, 2], "final": {}},
        ]
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.writelines(json.dumps(row) + "\n" for row in rows)
        sample, = day_samples(path, "synthetic-policy", False)
        assert sample["state"]["private"]["shed"]["WHEAT"] == 2
        assert sample["behavior_target"]["hourly_actions"] == [{"farmer": ["NORTH"]}]
        assert sample["split_only"]["group_id"] == "7:split-only-opponent"
        assert sample["replicate"]["seat"] == 1 and "auxiliary_target" not in sample["decisions"][0]

    paths = smoke_paths()
    smoke = {"trajectories": 0, "days": 0, "decisions": 0}
    if paths:
        samples = [sample for path in paths for sample in day_samples(path, "deployed_dynamic", False)]
        assert len(paths) == 2 and len(samples) == 36
        assert sum(len(x["decisions"]) for x in samples) == 862
        assert {x["replicate"]["seat"] for x in samples} == {0, 1}
        assert len({x["split_only"]["group_id"] for x in samples}) == 1
        smoke = {"trajectories": 2, "days": 36, "decisions": 862}
    print(json.dumps({"status": "PASS", "synthetic": True, "smoke": smoke}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--policy-source", help="stable teacher/policy identifier")
    parser.add_argument("--data-kind", choices=("behavior", "diagnostic"))
    parser.add_argument("--start-step", type=int, default=288)
    parser.add_argument("--include-opponent-private-target", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.paths or not args.policy_source or not args.data_kind:
        parser.error("paths, --policy-source and --data-kind are required")
    if args.start_step < 288 or args.start_step % 24:
        parser.error("learned midgame data must start on a day boundary at step 288 or later")
    summary = extract(args.paths, args.policy_source, args.data_kind == "diagnostic",
                      args.start_step, args.include_opponent_private_target, args.output)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
