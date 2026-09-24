"""Generate an English teammate report from the same verified acceptance JSON."""
from pathlib import Path
import html
import json

HERE=Path(__file__).resolve().parent


def main():
    r=json.loads((HERE/'ACCEPTANCE.json').read_text())
    s=json.loads((HERE/'SERIAL_VERIFICATION.json').read_text())
    audit=json.loads((HERE/'SOURCE_AUDIT.json').read_text())
    cfg=r['configuration'];opening=r['opening_overlay']['opening_liquidity']
    lines=['# A06 dual-panel fusion: independent acceptance','',
           f"Candidate: `{r['candidate']}`. Status: **{r['status']}**.",'',
           '50 new shared seeds, both seats, 100 games per opponent. Internal: 13 original variants / 1,300 games. External: 10 frozen public entries / 1,000 games. Official 1.32.7 Python rules, live policies, 719 state transitions per completed game, zero execution errors. Draws do not count as wins.','',
           '| Panel | Wins / games | Strict win rate | 95% seed-cluster bootstrap interval | Mean cash margin |',
           '|---|---:|---:|---:|---:|']
    for name,g in r['groups'].items():
        lo,hi=g['seed_bootstrap_95'];lines.append(f"| {name} | {g['wins']}/{g['games']} | {g['strict_win_rate']:.2%} | {lo:.2%}–{hi:.2%} | {g['mean_margin']:+,.1f} |")
    unique=r['external_unique_source_metrics']
    lines+=['',f"External result after deduplicating identical main.py files ({r['external_unique_source_count']} source entries): **{unique['strict_win_rate']:.2%}**, {unique['wins']}/{unique['games']}.",'',
            'The target is a 90% average strict win rate on each frozen panel. It is not a 90% guarantee against every opponent or a 90% confidence lower bound. The public pool is a September 23–24, 2026 snapshot; this is not a current online leaderboard or Kaggle sandbox certification.','',
            '## What changed','',
            'The native Cashflow C++ planner, executor, binary and public-supply sale wrapper are unchanged. Cashflow retains its heuristic that shifts half of positive field output value to the following day; this is a forecast, not future information.',
            f'An opening wheat buy/sell intent ({opening} units) uses cash, order-count and warehouse guards. Intraday admission/procurement of new projects is disabled. The hired-worker cap is {cfg["max_hands"]}. New-animal candidate ranking is multiplied by {cfg["animal_bias"]}; this is a preference adjustment, not a fixed animal count or a fixed crop route.',
            'No ML/RL, trained weights, opponent identity, opponent source code, hidden rival inventory or test seed is used for decision making. The learned_value header is a disabled stub. The agent still plans from the observed state.','',
            '## Remaining weaknesses','',
            '| Opponent | Wins-draws-losses / 100 | Strict win rate | Mean cash margin |',
            '|---|---:|---:|---:|']
    for name,g in r['per_opponent'].items():
        lines.append(f"| {r['display_names'].get(name,name)} | {g['wins']}-{g['ties']}-{g['losses']} | {g['strict_win_rate']:.1%} | {g['mean_margin']:+,.1f} |")
    paired=r.get('paired_external_comparison')
    if paired:
        lo,hi=paired['seed_bootstrap_delta_95']
        lines+=['','## Matched-seed external comparison','',
                f"On the same 50 seeds and both seats: frozen V1 {paired['v1_wins']}/1000; frozen V2 {paired['v2_wins']}/1000. Paired win-rate difference: {paired['v2_minus_v1_win_rate']*100:+.2f} percentage points; 95% seed-cluster interval {lo*100:+.2f} to {hi*100:+.2f} points.",
                'This is a post-acceptance diagnosis without retuning. Do not attribute the raw difference between the original V1 and V2 reports to the parameter change: those original reports used different seed panels. Both runtimes are included in the release.']
    lines+=['','## Reproduction and timing','',
            f"All {s['games']} serial reruns reproduced both players' terminal cash and results exactly. Maximum local serial agent call in those checks: {s['max_serial_action_s']:.3f} seconds. This is a local timing observation, not platform certification.",
            'The release includes the frozen runtime/source, SHA-256 manifest, official local referee, compressed per-game receipts and reproduce_one.py. The opponents come from the previously shared repository branch research/a06-dynamic-variants-public10-20260924, commit c00bcb4c5ed0cb43d750ad5a684e9f1ae318f80a.',
            'Run under compatible Linux x86-64 / WSL. The inherited binary requires compatible GLIBC_2.32 and GLIBCXX_3.4.31 libraries. Rebuild from C++20 source if necessary, then recheck exact results.','',
            '```text',
            'python reproduce_one.py --pool /path/to/research/a06_dynamic_variants_public10_20260924 --opponent internal/r14_cashflow --seat 0',
            '```','',
            'The default seed is the first final holdout seed. The script verifies the runtime/referee and opponent hashes, then checks the exact recorded terminal cash. The historical README and reports inside agent/ belong to the parent package; use the root release README and evidence/ACCEPTANCE.json for this fusion.','']
    (HERE/'ACCEPTANCE_EN.md').write_text('\n'.join(lines),encoding='utf-8')
    cards=''.join(f'<article><small>{name.title()}</small><strong>{g["strict_win_rate"]:.2%}</strong><span>{g["wins"]} wins / {g["games"]} games</span><p>95% seed interval {g["seed_bootstrap_95"][0]:.2%}–{g["seed_bootstrap_95"][1]:.2%}</p></article>' for name,g in r['groups'].items())
    rows=''.join(f'<tr><td>{html.escape(r["display_names"].get(name,name))}</td><td>{g["wins"]}-{g["ties"]}-{g["losses"]}</td><td>{g["strict_win_rate"]:.1%}</td><td>{g["mean_margin"]:+,.0f}</td></tr>' for name,g in r['per_opponent'].items())
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A06 fusion acceptance</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#172033;background:#f8fafc}.cards{display:flex;gap:20px;margin:25px 0}article{flex:1;background:white;padding:24px;border:1px solid #dae3ee;border-radius:12px}strong{display:block;font-size:38px;color:#15605d}article p{font-size:14px;color:#536277}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{text-align:left;padding:11px;border-bottom:1px solid #e2e8f0}a{color:#17578c}.note{background:#edf3fa;padding:18px;border-radius:10px}@media(max-width:650px){.cards{display:block}article{margin-bottom:12px}body{padding:0 10px}td{padding:6px;word-break:break-word}}</style>'''
    page+=f'<small>Frozen September 23–24, 2026 pool · Official 1.32.7 local rules</small><h1>A06 dual-panel fusion acceptance</h1><p><code>{html.escape(r["candidate"])}</code> · {html.escape(r["status"])}</p><div class="cards">{cards}</div>'
    page+=f'<div class="note">50 new seeds, both seats, 2,300 completed games, zero execution errors. Draws are not wins. External source-deduplicated win rate: <b>{unique["strict_win_rate"]:.2%}</b> ({r["external_unique_source_count"]} entries). The target is each panel’s average; neither every opponent nor a confidence lower bound is required to reach 90%.</div>'
    page+=f'<h2>Implementation</h2><p>Unchanged Cashflow native planner/executor and sale wrapper, plus an opening {opening}-unit wheat buy/sell intent, intraday new-project procurement disabled, {cfg["max_hands"]}-worker cap, and animal candidate ranking multiplier {cfg["animal_bias"]}. No ML/RL or opponent identity branches.</p>'
    page+='<h2>Per-opponent results</h2><table><thead><tr><th>Opponent</th><th>Wins-draws-losses</th><th>Win rate</th><th>Mean cash margin</th></tr></thead><tbody>'+rows+'</tbody></table>'
    if paired:
        lo,hi=paired['seed_bootstrap_delta_95']
        page+=f'<h2>Matched-seed comparison</h2><p>Same 50 seeds and both seats: V1 {paired["v1_wins"]}/1000; V2 {paired["v2_wins"]}/1000. Paired difference {paired["v2_minus_v1_win_rate"]*100:+.2f} percentage points (95% seed interval {lo*100:+.2f} to {hi*100:+.2f}). Both versions remained frozen; this is diagnostic, not a new tuning round.</p>'
    page+=f'<h2>Reproduction</h2><p>{s["games"]} serial reruns matched both terminal cash values and results exactly. Maximum tested serial call: {s["max_serial_action_s"]:.3f}s. This is not Kaggle sandbox or live leaderboard certification.</p><p><a href="ACCEPTANCE_EN.md">Full English notes and reproduction command</a> · <a href="ACCEPTANCE.json">Verified JSON</a> · <a href="ACCEPTANCE_ZH.html">中文报告</a></p></html>'
    (HERE/'ACCEPTANCE_EN.html').write_text(page,encoding='utf-8')
    print('English report generated from verified acceptance and serial checks')


if __name__=='__main__':main()
