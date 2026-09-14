"""Private static reports. Explicit export allowlist excludes seeds/source/paths."""
from collections import defaultdict
from html import escape
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .ratings import summary
from .schedule import validate_result
from .store import digest, now, read, records, write


def run_report(root, m, bootstrap):
    root = Path(root)
    rows = []
    fingerprint = []
    for g in m["games"]:
        r = read(root / "runs" / m["id"] / "games" / (g["id"] + ".json"))
        if r:
            validate_result(g, r)
            fingerprint.append(r)
            if r["resolved"]:
                rows.append({**r, "seed": g["seed"]})
    cache = root / "runs" / m["id"] / "summary.json"
    effective_bootstrap = bootstrap if len(rows) == len(m['games']) else 0
    key = digest([fingerprint, effective_bootstrap, 1])
    old = read(cache)
    if old and old.get("fingerprint") == key:
        if "updated" not in old:
            old["updated"] = datetime.fromtimestamp(cache.stat().st_mtime, timezone.utc).isoformat()
            write(cache, old)
        return old
    result = {"run": m["id"], "kind": m["kind"], "created": m["created"], "fingerprint": key, "updated": now(),
              "planned": len(m["games"]), "completed": len(rows), "complete": len(rows) == len(m["games"]),
              "contract": m["contract_hash"], **summary(rows, sorted(m["agents"]), effective_bootstrap)}
    # No seed values or local paths go into summary JSON.
    write(cache, result)
    return result


def pct(value):
    return "—" if value is None else f"{value*100:.1f}%"


def build(root, bootstrap=500):
    root = Path(root)
    agents = {a["id"]: a for a in records(root, "agents")}
    runs = [run_report(root, m, bootstrap) for p in sorted((root / "runs").glob("*/manifest.json"))
            if (m:=read(p))['kind']!='continuous']
    public_ids = {e["agent"] for e in read(root / "roster.json", []) if e["category"] == "public"}
    refreshes = [read(p) for p in sorted((root / "private/public-refresh").glob("*.json"))]
    today = datetime.now(ZoneInfo(read(root / "config.json")["timezone"])).date().isoformat()
    refresh_by_notebook = {r["notebook"]:r for r in refreshes}
    # Explicit public-safe data schema; no recursive dump of private objects.
    public_agents = [{"id": aid, "name": a["manifest"]["name"], "author": a["manifest"]["author"],
                      "version": a["manifest"]["version"], "status": a["status"],
                      "validation_failed": bool(a.get("validation_failure"))} for aid, a in agents.items()]
    for row in public_agents:
        a = agents[row["id"]]
        origin = a["manifest"].get("origin", {})
        is_public = origin.get("kind") == "public" or a.get("public_notebook", False) or row["id"] in public_ids
        ref = refresh_by_notebook.get(origin.get("notebook"), {})
        row.update(agent_type="public" if is_public else "team", notebook=origin.get("notebook"),
                   notebook_version=origin.get("version"), last_checked=ref.get("attempted"),
                   refresh_status=("stale" if ref and ref.get("date") != today else ref.get("status", "not configured")))
    metadata = {a["id"]: a for a in public_agents}
    champion = read(root / "champion.json", {})
    discovery = read(root/'private/discovery.json',{})
    public_decisions = [read(p) for p in sorted((root/'runs').glob('public-*/public-decision.json'))]
    data = {"schema": 1, "updated": now(), "agents": public_agents, "runs": runs,
            "continuous_elo":read(root/'continuous-elo.json',{'ratings':[]}),
            "continuous_status":read(root/'continuous-status.json',{'status':'not started'}),
            "champion": champion.get("agent"), "champion_evidence": champion.get("evidence"),
            "roster": [{"agent": r["agent"], "category": r["category"]} for r in read(root / "roster.json", [])],
            "public_refresh": [{k:r.get(k) for k in ("id","notebook","date","status","version","attempted","successful")} for r in refreshes],
            "public_discovery": {"checked":discovery.get('checked'),"found":discovery.get('found',0),
                "truncated":discovery.get('truncated',[]),"decisions":public_decisions}}
    write(root / "site/data.json", data)
    def name(aid):
        label = escape(agents.get(aid, {}).get("manifest", {}).get("name", aid[:12]))
        meta = metadata.get(aid, {})
        if meta.get("agent_type") == "public":
            label += ' <strong class="public-badge">Public notebook</strong>'
            if meta.get("notebook"):
                notebook = escape(meta["notebook"], quote=True)
                label += f' <small><a href="https://www.kaggle.com/code/{notebook}">{notebook}</a> · v{escape(meta["notebook_version"])}</small>'
        return label
    sections = ["<h1>Kaggriculture arena</h1>", "<p>Private preview · Updated " + escape(data["updated"]) + "</p>",
                "<h2>Verified champion</h2><p>" + (name(data["champion"]) if data["champion"] else "Not nominated") + "</p>",
                '<p>Daily, placement and fixed-benchmark results are separate. Ratings from different rosters are not directly comparable.</p>']
    active={e['agent'] for e in data['roster']}
    elo_rows=[r for r in data['continuous_elo']['ratings'] if r['agent'] in active]
    sections += ['<h2>Continuous Elo</h2><p>Mini PC round robin. Daily tournament results do not enter Elo. New versions start at 1500. K=32 per completed seat-swapped pair; draws count half.</p>',
                 '<p>Last updated: '+escape(data['continuous_elo'].get('updated', 'Not yet'))+' (UTC)</p>',
                 '<p>Status: '+escape(data['continuous_status']['status'])+'</p>',
                 '<table><tr><th>Agent</th><th>Elo</th><th>Games</th><th>Score</th><th>Contract</th></tr>']
    for r in elo_rows:
        sections.append(f"<tr><td>{name(r['agent'])}</td><td>{r['elo']:.1f}</td><td>{r['games']}</td><td>{pct(r['score'])}</td><td>{r['contract'][:10]}</td></tr>")
    sections.append('</table>')
    sections += ["<h2>Public notebook refresh</h2><p>Checked daily before the next roster freeze. Failed refreshes retain the last validated version.</p>"]
    sections.append(f"<p>Discovery last checked: {escape(str(discovery.get('checked','not yet')))}. Found {discovery.get('found',0)} notebooks. Newcomers need a completed paired comparison before replacing a public agent.</p>")
    if not refreshes:
        sections.append("<p>No public notebook refresh sources configured.</p>")
    for r in refreshes:
        status = "stale (" + r["status"] + ")" if r.get("date") != today else r["status"]
        sections.append(f"<p>{escape(r['notebook'])}: {escape(status)} · checked {escape(r['attempted'])} · last success {escape(r.get('successful', 'never'))}</p>")
    sections += ["<h2>Submissions</h2><table><tr><th>Agent / source</th><th>Author</th><th>Status</th></tr>"]
    for a in public_agents:
        sections.append(f"<tr><td>{name(a['id'])} ({escape(a['version'])})</td><td>{escape(a['author'])}</td><td>{'Validation failed' if a['validation_failed'] else escape(a['status'])}</td></tr>")
    sections.append("</table>")
    daily = [r for r in runs if r["kind"] == "daily" and r["complete"]]
    if len(daily) >= 2:
        previous, latest = daily[-2:]
        shared = set(previous["matrix"]) & set(latest["matrix"])
        changes = [(latest["matrix"][k]["win_rate"] - previous["matrix"][k]["win_rate"], k)
                   for k in shared if latest["matrix"][k]["games"] and previous["matrix"][k]["games"]]
        sections.append("<h2>Largest daily matchup changes</h2><p>Same versions only. Descriptive changes, not significance tests.</p><ul>")
        for change, key in sorted(changes, key=lambda v: -abs(v[0]))[:6]:
            a, b = key.split(":")
            sections.append(f"<li>{name(a)} vs {name(b)}: {change*100:+.1f} points ({previous['matrix'][key]['games']} → {latest['matrix'][key]['games']} games)</li>")
        sections.append("</ul>")
    for aid in agents:
        contracts = {r["contract"] for r in runs}
        for kind in ("daily", "benchmark"):
            for contract in contracts:
                series = [(r["run"], r["stats"][aid]["win_rate"]) for r in runs if r["complete"] and r["kind"] == kind and r["contract"] == contract and aid in r["stats"]]
                if len(series) >= 2:
                    sections.append("<h3>" + name(aid) + " — " + kind + " strict win rate</h3><p>Changes in opponents can affect daily results.</p>" + sparkline(series))
    for run in reversed(runs):
        connected = len(run["components"]) == 1
        order = sorted(run["ratings"], key=lambda a: -(run["ratings"][a] or 0))
        title = "Completed" if run["complete"] else "Provisional"
        body = [f"<h2>{escape(run['run'])} — {title}</h2><p>{run['completed']:,}/{run['planned']:,} games · {escape(run['kind'])}</p>"]
        body.append('<p>Bradley–Terry leaderboard · Last updated: '+escape(run['updated'])+' (UTC)</p>')
        if not connected:
            body.append("<p>Disconnected comparisons: ratings are only comparable within each component.</p>")
        if connected and run["complete"] and order:
            body.append("<p>Highest fitted rating: " + name(order[0]) + ". Check uncertainty before claiming superiority.</p>")
        body.append("<table><tr><th>Agent</th><th>Rating [95% interval]</th><th>W/L/D</th><th>Strict wins</th><th>Score</th><th>Cash margin</th></tr>")
        for aid in order:
            st = run["stats"][aid]
            val, ci = run["ratings"][aid], run["intervals"][aid]
            rating = "—" if val is None else f"{val:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else " [insufficient data]")
            margin = "—" if st["cash_margin"] is None else f"{st['cash_margin']:,.0f}"
            body.append(f"<tr><td>{name(aid)}</td><td>{rating}</td><td>{st['wins']}/{st['losses']}/{st['draws']}</td><td>{pct(st['win_rate'])}</td><td>{pct(st['score'])}</td><td>{margin}</td></tr>")
        body.append("</table><details><summary>Matchups and seats</summary><table><tr><th>Agent</th><th>Opponent</th><th>Win rate (games)</th><th>Seat 0 W/G</th><th>Seat 1 W/G</th></tr>")
        for key, st in run["matrix"].items():
            a, b = key.split(":")
            seats = st["seats"]
            body.append(f"<tr><td>{name(a)}</td><td>{name(b)}</td><td>{pct(st['win_rate'])} ({st['games']})</td><td>{seats['0']['wins']}/{seats['0']['games']}</td><td>{seats['1']['wins']}/{seats['1']['games']}</td></tr>")
        body.append("</table></details>")
        sections.extend(body)
        (root / "site" / (run["run"] + ".html")).write_text(document("".join(body)), encoding="utf-8")
    events = [e for e in records(root, "events") if not e["acknowledged"]]
    sections.append(f"<h2>Controller attention</h2><p>{len(events)} pending events. Private details remain in the local event queue.</p>")
    (root / "site/index.html").write_text(document("".join(sections)), encoding="utf-8")
    def label(aid):
        return agents[aid]['manifest']['name'].replace('|', '/').replace('\n',' ')
    md = ["# Kaggriculture arena", "", "Updated: " + data["updated"], "",
          "CPU evaluation on WRX90 and the mini PC. Results below are local; they are not Kaggle leaderboard scores.", "",
          "[Submit an agent or join the workflow](../workflows/pro8_arena/START.md)", "",
          "Arena-certified champion: " + (label(data['champion']) if data['champion'] else "None yet. AFS R2 remains a historical reference, not a new certification."), "",
          "Public notebooks refresh daily. Exact versions keep separate results. Incomplete tournaments are provisional; small launch checks do not establish strength.", "",
          "## Roster", "", "| Agent | Type | Version | Status |", "|---|---|---|---|"]
    for a in public_agents:
        md.append(f"| {label(a['id'])} | {'**PUBLIC**' if a['agent_type']=='public' else 'Team'} | {a['version']} | {a['status']} |")
    md += ['', '## Continuous Elo', '',
           'Mini PC round robin. Daily tournaments stay separate. New versions start at 1500; K=32 per completed seat-swapped pair. Draws count half. Compare ratings only within the same contract.', '',
           'Last updated: '+data['continuous_elo'].get('updated', 'Not yet')+' (UTC)', '',
           'Status: '+data['continuous_status']['status']+' · Last sync: '+data['continuous_status'].get('at','Never'), '',
           '| Agent | Elo | Games | Score | Contract |','|---|---:|---:|---:|---|']
    for r in elo_rows:
        badge=' **PUBLIC**' if metadata[r['agent']]['agent_type']=='public' else ''
        md.append(f"| {label(r['agent'])}{badge} | {r['elo']:.1f} | {r['games']} | {pct(r['score'])} | {r['contract'][:10]} |")
    md += ["", "## Public refresh", "", "| Notebook | Status | Last successful check |", "|---|---|---|"]
    for r in refreshes:
        md.append(f"| {r['notebook']} | {r['status']} | {r.get('successful','Never')} |")
    md += ['', '## Public discovery and replacements', '',
           f"Last scan: {discovery.get('checked','Not yet')}. Found {discovery.get('found',0)} notebooks; {discovery.get('eligible',0)} updated in the last 24 hours. Discovery uses Kaggle public-score order, highest first. Up to four new outputs enter evaluation per day. Update time uses Kaggle's lastRunTime. A completed daily panel is required to select the weakest public agent.", '',
           'Each challenger and the proposed replacement face the same other agents on 128 fresh seeds in both seats. Replacement requires at least a two-point win-rate gain and a positive approximate 95% lower bound. This is a roster decision, not a 90% strength certificate.', '']
    if discovery.get('truncated'):md.append('Discovery reached its page limit; the scan was not exhaustive.')
    for d in public_decisions:
        md.append(f"- {label(d['candidate'])} vs {label(d['incumbent'])}: {'replaced' if d['replaced'] else 'retained incumbent'}; gain {d['gain']*100:+.1f} points, lower bound {d['lower']*100:+.1f} points.")
    md += ["", "## Run coverage", "", "| Run | Status | Games |", "|---|---|---|"]
    md += [f"| {r['run']} | {'Complete' if r['complete'] else 'Provisional'} | {r['completed']}/{r['planned']} |" for r in runs]
    for run in reversed([r for r in runs if r['kind']=='daily'][-2:]):
        md += ["", '## '+run['run']+(' — complete' if run['complete'] else ' — provisional'), "",
               "Bradley–Terry leaderboard · Last updated: "+run["updated"]+" (UTC)", "",
               "| Agent | BT rating | W / L / D | Strict win rate | Cash margin |", "|---|---:|---:|---:|---:|"]
        for aid in sorted(run['ratings'],key=lambda a:-(run['ratings'][a] or 0)):
            s=run['stats'][aid];v=run['ratings'][aid];ci=run['intervals'][aid]
            rating='—' if v is None else f'{v:.2f}'+(f' [{ci[0]:.2f}, {ci[1]:.2f}]' if ci else '')
            margin='—' if s['cash_margin'] is None else f"{s['cash_margin']:,.0f}"
            md.append(f"| {label(aid)} | {rating} | {s['wins']} / {s['losses']} / {s['draws']} | {pct(s['win_rate'])} | {margin} |")
        md += ["", "<details><summary>Win rate by opponent and seat</summary>", "",
               "| Agent | Opponent | Wins / games | Win rate | Seat 0 W/G | Seat 1 W/G |", "|---|---|---:|---:|---:|---:|"]
        for key,s in run['matrix'].items():
            a,b=key.split(':');seats=s['seats']
            md.append(f"| {label(a)} | {label(b)} | {s['wins']}/{s['games']} | {pct(s['win_rate'])} | {seats['0']['wins']}/{seats['0']['games']} | {seats['1']['wins']}/{seats['1']['games']} |")
        md += ["", "</details>", ""]
    if len(daily)>=2:
        md += ["", "## Largest daily matchup changes", "", "Same versions only; descriptive changes, not significance tests.", ""]
        for change,key in sorted(changes,key=lambda v:-abs(v[0]))[:6]:
            a,b=key.split(':');md.append(f"- {label(a)} vs {label(b)}: {change*100:+.1f} percentage points.")
    (root / "site/README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return data


def sparkline(series):
    coords = [(20+i*560/max(1,len(series)-1), 130-100*v) for i, (_, v) in enumerate(series)]
    points = " ".join(f"{x:.1f},{y:.1f}" for x,y in coords)
    labels = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3"><title>{escape(label)}: {pct(value)}</title></circle>' for (x,y),(label,value) in zip(coords,series))
    return f'<svg viewBox="0 0 600 160" role="img" aria-label="Win rate history" style="max-width:700px"><path d="M20 20V130H580" fill="none" stroke="#abb8ca"/><polyline points="{points}" fill="none" stroke="#1763b5" stroke-width="2"/>{labels}<text x="20" y="155">{escape(series[0][0])}</text><text x="410" y="155">{escape(series[-1][0])}</text></svg>'


def document(body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Kaggriculture arena</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#182334;background:#f7f9fc}table{border-collapse:collapse;width:100%;background:white;margin:1rem 0}th,td{text-align:left;border-bottom:1px solid #dce3ed;padding:.55rem}details{margin:1rem 0}h2{margin-top:2rem}</style><main>' + body + '</main></html>'
