"""Verify the selected rules agent differs from Cashflow only where declared."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    freeze = json.loads((HERE / 'FINAL_FREEZE.json').read_text())
    current = HERE / 'candidates' / freeze['candidate']
    parent = ROOT / 'experiments/a06_r12_gptpro_round_robin_20260924/agents/r14_cashflow'
    matched = []
    for name, digest in freeze['files'].items():
        assert sha(current / name) == digest, name
        if name in ('main.py', 'overlay.json', 'policy/config.json'):
            continue
        original_name = 'main.py' if name == 'base_entry.py' else name
        assert sha(parent / original_name) == digest, name
        matched.append(name)
    before = json.loads((parent / 'policy/config.json').read_text())
    after = json.loads((current / 'policy/config.json').read_text())
    delta = {k: [before.get(k), after.get(k)] for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    definitions={r['id']:r for r in json.loads((HERE/'CANDIDATES.json').read_text())}
    lineage=[];ident=freeze['candidate']
    while ident in definitions:
        row=definitions[ident];lineage.append(row);ident=row['parent']
    assert ident=='r14_cashflow'
    expected=before.copy();quantity=0
    for row in reversed(lineage):
        expected.update(row.get('config_changes',{}))
        if row.get('opening_overlay'):quantity=10
        quantity=row.get('opening_quantity',quantity)
    assert after==expected,delta
    assert json.loads((current / 'overlay.json').read_text()) == {'opening_liquidity': quantity}
    result = dict(candidate=freeze['candidate'], frozen_files=len(freeze['files']),
                  identical_parent_files=len(matched), changed_config=delta,opening_quantity=quantity,
                  new_files=['main.py', 'overlay.json'],
                  renamed_unchanged={'main.py': 'base_entry.py'},
                  native_binary_unchanged=True, native_sha256=sha(current / 'policy/a06.so'),
                  note='The Python opening wrapper is new. All native planning/execution code, the native binary and the Cashflow sale wrapper are byte-identical to the frozen parent. No retraining or learned weights were introduced.')
    (HERE / 'SOURCE_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
