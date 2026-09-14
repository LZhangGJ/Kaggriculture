"""CLI. Invoke from repository root with python -m tools.arena.manage."""
import argparse
import json
from pathlib import Path

from . import controller, gates, intake, reporting, sandbox, schedule, store


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(".arena"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    s = sub.add_parser("submit")
    s.add_argument("--archive", required=True)
    s.add_argument("--manifest", required=True)
    s = sub.add_parser("import-issue")
    s.add_argument("--event", required=True)
    s.add_argument("--archive", required=True)
    s = sub.add_parser("validate")
    s.add_argument("agent")
    s = sub.add_parser("roster")
    s.add_argument("file")
    s = sub.add_parser("plan")
    s.add_argument("run")
    s.add_argument("--kind", choices=["daily", "placement", "confirmation", "benchmark", "topup"], default="daily")
    s.add_argument("--seeds", type=int)
    s.add_argument("--candidate")
    s.add_argument("--references", nargs="+")
    sub.add_parser("tick")
    sub.add_parser("check")
    sub.add_parser("events")
    s = sub.add_parser("ack")
    s.add_argument("event_id")
    s = sub.add_parser("report")
    s.add_argument("--bootstrap", type=int, default=500)
    s = sub.add_parser("experiment")
    s.add_argument("id")
    s.add_argument("--brief", required=True, help="JSON with hypothesis, owner, parent and scope")
    s = sub.add_parser("search")
    s.add_argument("query")
    s = sub.add_parser("finish")
    s.add_argument("id")
    s.add_argument("--result", required=True)
    s = sub.add_parser("compare-plan")
    s.add_argument("run")
    s.add_argument("--candidate", required=True)
    s.add_argument("--incumbent", required=True)
    s.add_argument("--panel", required=True)
    s.add_argument("--seeds", type=int, default=256)
    for name in ("gate", "promote", "certify"):
        s = sub.add_parser(name)
        s.add_argument("run")
    s = sub.add_parser("topup")
    s.add_argument("run")
    s.add_argument("--source", required=True)
    s.add_argument("--candidate", required=True)
    s.add_argument("--opponent", required=True)
    s.add_argument("--target", type=int, required=True)
    args = p.parse_args(argv)
    with store.locked(args.root):
        cmd = args.command
        if cmd == "init":
            store.init(args.root)
            result = {"initialized": str(args.root), "execution_enabled": False}
        elif cmd == "submit":
            result = intake.submit(args.root, args.archive, store.read(args.manifest))
        elif cmd == "import-issue":
            result = intake.import_issue(args.root, args.event, args.archive)
        elif cmd == "validate":
            result = sandbox.validate_agent(args.root, store.ident(args.agent))
        elif cmd == "roster":
            intake.set_roster(args.root, store.read(args.file))
            result = {"roster_saved": True}
        elif cmd == "plan":
            m = schedule.plan(args.root, args.run, args.kind, args.seeds, args.candidate, args.references)
            result = {"run": m["id"], "games": len(m["games"])}
        elif cmd == "tick":
            result = controller.tick(args.root)
        elif cmd == "compare-plan":
            result = gates.plan_comparison(args.root, args.run, args.candidate, args.incumbent, store.read(args.panel), args.seeds)
            result = {"run": result["id"], "games": len(result["games"])}
        elif cmd in ("gate", "promote", "certify"):
            result = {"gate": gates.compare, "promote": gates.promote, "certify": gates.certify}[cmd](args.root, args.run)
        elif cmd == "topup":
            m = schedule.topup(args.root, args.run, args.source, args.candidate, args.opponent, args.target)
            result = {"run":m["id"], "new_games":len(m["games"])}
        elif cmd == "check":
            import importlib.metadata
            cfg = store.read(args.root / "config.json")
            result = {"referee_installed": importlib.metadata.version("kaggle-environments"), "referee_required": cfg["contract"]["version"], "execution_enabled": cfg["execution_enabled"]}
            try:
                sandbox.preflight(cfg)
                result["sandbox"] = "ready"
            except Exception as e:
                result["sandbox"] = str(e)
        elif cmd == "report":
            data = reporting.build(args.root, args.bootstrap)
            result = {"page": str(args.root / "site/index.html"), "runs": len(data["runs"])}
        elif cmd == "events":
            result = [e for e in store.records(args.root, "events") if not e["acknowledged"]]
        elif cmd == "ack":
            path = args.root / "events" / (store.ident(args.event_id) + ".json")
            record = store.read(path)
            record["acknowledged"] = True
            store.write(path, record)
            result = {"acknowledged": args.event_id}
        elif cmd == "experiment":
            brief = store.read(args.brief)
            if not all(brief.get(k) for k in ("hypothesis", "owner", "parent", "scope")):
                raise ValueError("hypothesis, owner, parent, scope required")
            path = args.root / "experiments" / store.ident(args.id) / "brief.json"
            if path.exists():
                raise ValueError("Experiment ID already exists")
            for previous in (args.root / "experiments").glob("*/brief.json"):
                b = store.read(previous)
                if b["scope"] == brief["scope"] and not (previous.parent / "result.json").exists():
                    raise ValueError("Scope already assigned")
            store.write(path, {**brief, "created": store.now()})
            result = {"experiment": args.id}
        elif cmd == "search":
            result = [{"path": str(path.relative_to(args.root)), "record": store.read(path)}
                      for path in (args.root / "experiments").glob("*/*.json") if args.query.casefold() in path.read_text(encoding="utf-8").casefold()]
        elif cmd == "finish":
            path = args.root / "experiments" / store.ident(args.id)
            if not (path / "brief.json").exists() or (path / "result.json").exists():
                raise ValueError("Missing or finished experiment; use a new attempt ID")
            record = store.read(args.result)
            if not all(k in record for k in ("variants", "evidence", "conclusion", "limitations")):
                raise ValueError("variants, evidence, conclusion, limitations required")
            store.write(path / "result.json", record)
            store.event(args.root, "research_complete", args.id, {"experiment": args.id})
            result = {"finished": args.id}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
