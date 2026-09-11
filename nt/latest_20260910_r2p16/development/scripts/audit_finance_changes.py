"""Account for adverse full-game effects, not just a successful local HIRE."""
from pathlib import Path
from collections import defaultdict
import gzip
import json
import audit_trace
import run_panel as panel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read_audit(row,variant):
    key=f"{row['opponent']}_{row['seed']}_{row['opponent_seat']}"
    folder=HERE/'baseline_audit'/key if variant=='old' else HERE/'finance_screen16'/'audit'/key
    if not (folder/'AUDIT.json.gz').exists():
        audit_trace.run_case(row,folder)
    return json.loads(gzip.decompress((folder/'AUDIT.json.gz').read_bytes()))


def main():
    baseline={(r['opponent'],r['seed'],r['opponent_seat']):r for r in json.loads((HERE/'baseline_rows_all11.json').read_text())}
    matched=json.loads((HERE/'finance_screen16/PAIRED.json').read_text())['rows']
    changed={(r['opponent'],r['seed'],r['opponent_seat']) for r in matched if not r['same_actions']}
    rows=json.loads((HERE/'finance_screen16/rows.json').read_text())
    results=[]
    for row in rows:
        key=(row['opponent'],row['seed'],row['opponent_seat'])
        if key not in changed:continue
        row=dict(row,trace=str((HERE/'finance_screen16'/row['trace']).relative_to(ROOT)))
        before=read_audit(baseline[key],'old');after=read_audit(row,'new')
        # Persist the exact ledger deltas for subsequent review; schema is
        # deliberately read from the audit, never reconstructed from requests.
        results.append(dict(key=key,old_summary=before['summary'],new_summary=after['summary'],
                            old_transactions=before['transactions'],new_transactions=after['transactions']))
    panel.save(HERE/'finance_screen16/FINANCE_LEDGER.json',results)
    print(json.dumps(dict(completed=len(results))),flush=True)


if __name__=='__main__':main()
