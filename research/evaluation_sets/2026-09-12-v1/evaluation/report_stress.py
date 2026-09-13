"""Export the completed stress panel from the unchanged frozen analysis."""
import gzip
import hashlib
import json
from pathlib import Path
import time

from analyze_panels import LABELS, ORDER

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'outputs/stress_results'
OUT.mkdir(exist_ok=True)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()

def dump(path, data):
    path.write_bytes((json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())

analysis_path = HERE / 'development_results/RESULTS.json'
analysis = json.loads(analysis_path.read_bytes())
release = json.loads((HERE / 'HOLDOUT_RELEASE.json').read_bytes())
roster = json.loads((HERE / 'roster.json').read_bytes())
assert analysis['status'] == 'complete_development_holdout_pending'
assert analysis['analysis_sha256'] == sha(HERE / 'analyze_panels.py') == release['analysis_code_sha256']
assert analysis['roster_sha256'] == sha(HERE / 'roster.json') == release['roster_sha256']
assert sha(HERE / 'panel_runner.py') == release['runner_sha256']
for name, digest in roster['runtime_files'].items():
    assert sha(HERE / 'runtime' / name) == digest, name
receipt = analysis['run_receipts']['development-v1']
assert receipt['status']['complete'] and receipt['status']['completed'] == 73728
assert receipt['status']['invalid'] == receipt['status']['missing'] == 0
assert receipt['rows_sha256'] == sha(HERE / 'runs/development-v1/rows.jsonl')

# Reconcile every already-published representative cell summary and interval.
old = json.loads((HERE.parent / 'outputs/representative_results/RESULTS.json').read_bytes())
rep = analysis['panels']['representative']
def compare_group(current, prior):
    assert set(current) == set(prior) == set(ORDER)
    for cid in ORDER:
        left, right = current[cid], prior[cid]
        for field in ('games', 'wins', 'draws', 'losses', 'win_rate', 'mean_cash', 'mean_margin'):
            assert left[field] == right[field], (cid, field)
        assert left['win_rate_ci95'] == right['ci95'], cid
        for baseline in ORDER[:2]:
            a, b = left['paired_differences'][baseline], right['paired_differences'][baseline]
            assert a['win_rate_difference'] == b['delta'] and a['ci95'] == b['ci95']

compare_group(rep['overall']['candidates'], old['overall'])
for group in ('by_opponent', 'by_seat'):
    assert set(rep[group]) == set(old[group])
    for name in rep[group]:
        compare_group(rep[group][name]['candidates'], old[group][name])
for opponent in rep['by_opponent_and_seat']:
    for seat in ('0', '1'):
        compare_group(rep['by_opponent_and_seat'][opponent][seat]['candidates'], old['by_opponent_and_seat'][opponent][seat])
assert rep['bootstrap']['index_matrix_sha256'] == old['bootstrap']['index_matrix_sha256']

stress = analysis['panels']['stress']
assert stress['overall']['distinct_seeds'] == 128
assert len(stress['by_opponent']) == 16
overall = stress['overall']['candidates']
for cid, row in overall.items():
    assert row['games'] == row['wins'] + row['draws'] + row['losses'] == 4096
    for grouping in ('by_opponent', 'by_seat'):
        for field in ('games', 'wins', 'draws', 'losses'):
            assert sum(g['candidates'][cid][field] for g in stress[grouping].values()) == row[field]

raw_hash = hashlib.sha256()
raw_count = 0
with (HERE / 'runs/development-v1/rows.jsonl').open('rb') as source, (OUT / 'rows.jsonl.gz').open('wb') as target:
    with gzip.GzipFile(filename='', mode='wb', fileobj=target, mtime=0) as compressed:
        for line in source:
            row = json.loads(line)
            if row['panel'] == 'stress':
                compressed.write(line)
                raw_hash.update(line)
                raw_count += 1
assert raw_count == 24576

qualification = ('Stress covers 128 selected extremes and contrasting conditions from a 4,096-seed pool. '
                 'Its win rate does not estimate the default seed distribution, and selected extremes need not be harder for every policy. '
                 'Reference shop labels use PASS/PASS; actual shop and price paths depend on both policies. '
                 'Sixteen opponent names include shared strategy families. No holdout outcome enters this report; all six policies remain frozen.')
report = dict(status='stress_complete_holdout_pending', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              scheduled=24576, completed=24576, invalid=0, missing=0, duplicate_cells=0,
              candidate_labels=LABELS, panel='stress', **stress, qualification=qualification,
              roster_sha256=analysis['roster_sha256'], frozen_analysis_sha256=analysis['analysis_sha256'],
              source_analysis_sha256=sha(analysis_path), source_run_receipt=receipt,
              stress_manifest_sha256=sha(HERE.parent / 'outputs/seed_sets_v1/manifests/stress.json'),
              stress_rows_sha256=raw_hash.hexdigest(), raw_gzip_sha256=sha(OUT / 'rows.jsonl.gz'),
              representative_snapshot_reconciliation='PASS: every count, rate, interval and paired difference',
              report_script_sha256=sha(Path(__file__)))
dump(OUT / 'RESULTS.json', report)
for name in ('REFERENCE_CONDITIONS.json', 'REALIZED_MARKETS.json'):
    data = json.loads((HERE / 'development_results' / name).read_bytes())
    dump(OUT / name, {'stress': data['stress']})

ranking = sorted(ORDER, key=lambda cid: overall[cid]['wins'], reverse=True)
text = '# Stress results\n\nAll 24,576 stress games are complete: 4,096 per candidate, using 128 seeds, 16 opponents and both seats. Zero invalid, missing or duplicate cells. Holdout results are pending.\n\n'
text += '| Candidate | Wins | Draws | Losses | Strict win rate | 95% seed-cluster interval |\n|---|---:|---:|---:|---:|---:|\n'
for cid in ranking:
    row = overall[cid]
    lo, hi = row['win_rate_ci95']
    text += f"| {LABELS[cid]} | {row['wins']:,} | {row['draws']:,} | {row['losses']:,} | {row['win_rate']:.2%} | {lo:.2%}–{hi:.2%} |\n"
text += '\nDraws count as zero wins. The unchanged analysis uses 4,000 paired bootstrap draws, resampling whole seeds. These are local CPU results under the official 1.32.7 rules.\n\n'
text += 'v37 leads the aggregate, but its paired lead over Day9 is 3.32 percentage points with a 95% interval of −0.20 to +7.15 points. Against native AFS R2, v37 wins 58/256 games (22.66%); Day9 wins 243/256 (94.92%). Day9 and Day3 v1 each win 3,561 games, while Day3 v2 wins four fewer. The paired comparisons do not establish a gain over Day9 for either Day3 variant.\n\n'
text += '[Win rates by opponent](BY_OPPONENT.md) · [Opponent-by-seat tables](BY_OPPONENT_AND_SEAT.md) · [Exact counts and paired differences](RESULTS.json).\n\n'
text += qualification + '\n\n'
text += '[Reference-condition results](REFERENCE_CONDITIONS.json) and [realized market summaries](REALIZED_MARKETS.json) retain the frozen analysis definitions. Realized market groups are descriptive because policy actions affect them.\n\n'
text += 'The v37 policy uses the public route-based parent and does not establish the team’s Three-Layer architecture goal. Optional terminal-suffix debug JSON errors are recorded separately; action and terminal checks passed. No candidate is promoted by this report.\n'
(OUT / 'RESULTS.md').write_bytes(text.encode())

def opponent_table(by_seat=False):
    title = '# Stress win rates by opponent' + (' and candidate seat' if by_seat else '')
    txt = title + '\n\nCells show strict win rate and wins/games. Draws count as zero wins. All 128 stress seeds are complete; exact draws, losses and intervals appear in RESULTS.json.\n'
    for seat in ('0', '1') if by_seat else (None,):
        txt += '\n' + ('## Candidate seat ' + seat + '\n\n' if by_seat else '')
        txt += '| Opponent | ' + ' | '.join(LABELS[cid] for cid in ORDER) + ' |\n|---|' + '---:|' * 6 + '\n'
        for opponent in stress['by_opponent']:
            group = stress['by_opponent_and_seat'][opponent][seat] if by_seat else stress['by_opponent'][opponent]
            rows = group['candidates']
            txt += '| ' + opponent + ' | ' + ' | '.join(f"{rows[cid]['win_rate']:.2%} ({rows[cid]['wins']}/{rows[cid]['games']})" for cid in ORDER) + ' |\n'
    return txt

(OUT / 'BY_OPPONENT.md').write_bytes(opponent_table().encode())
(OUT / 'BY_OPPONENT_AND_SEAT.md').write_bytes(opponent_table(True).encode())
print(json.dumps({'completed': raw_count, 'representative_reconciliation': 'PASS', 'stress_win_rates': {LABELS[cid]: overall[cid]['win_rate'] for cid in ranking}}))
