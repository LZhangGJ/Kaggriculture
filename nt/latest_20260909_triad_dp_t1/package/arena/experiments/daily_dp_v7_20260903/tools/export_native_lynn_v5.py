"""Static export of the frozen full Lynn V5 package; no downloaded code executed."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import zlib
from export_native_boatlee_v29 import put

EXP = Path(__file__).resolve().parents[1]
PACKAGE = EXP / 'opponents/lynn_v5/output/generated_submission'
ENTRY_SHA = 'e8498c67914ecc607ae69fde25a728361eb5acea94c00ecc85deffbdafe50413'


def literal(path, name):
    tree = ast.parse(path.read_text())
    nodes = [n.value for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
             and ((isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
                  or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name))]
    assert len(nodes) == 1, (path, name)
    return ast.literal_eval(nodes[0])


def main():
    assert hashlib.sha256((PACKAGE / 'main.py').read_bytes()).hexdigest() == ENTRY_SHA
    hashes = {p.relative_to(PACKAGE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.suffix in ('.py', '.json')}
    tapes = []
    for path in sorted(PACKAGE.glob('artifacts/*/*.py')):
        actions = json.loads(zlib.decompress(base64.b85decode(literal(path, '_PAYLOAD'))))
        assert len(actions) in (719, 720)
        # Require the current importer to represent every original quantity.
        for action in actions:
            assert isinstance(action, dict)
            for cmd in [action.get('farmer', ['PASS']), *action.get('hands', []), *action.get('market', [])]:
                assert isinstance(cmd, list) and cmd
                if len(cmd) > 2: assert isinstance(cmd[2], int), cmd
        tapes.append(actions)
    assert len(tapes) == 2
    old, active = tapes
    physical = lambda a: (a.get('farmer'), a.get('hands', []))
    non_sell = lambda a: [x for x in a.get('market', []) if x[0] != 'SELL']
    boundaries = [next(i for i in range(min(len(old), len(active))) if f(old[i]) != f(active[i]))
                  for f in (lambda x: x, non_sell, physical)]
    assert boundaries == [72, 72, 86], boundaries
    payload = dict(source_sha256=ENTRY_SHA, source_hashes=hashes, actions=active,
                   bundles=literal(PACKAGE / 'agents/e773a_demand_aligned_pasture_network.py', 'BUNDLES'),
                   price_params=literal(PACKAGE / 'agents/e749a_niklita_consensus_network.py', 'PARAMS'))
    blob = zlib.compress(json.dumps(payload, separators=(',', ':')).encode(), 9)
    put(EXP / 'native/lynn_v5_frozen.json.zlib', blob)
    receipt = dict(status='STATIC_EXPORT_ONLY', source_executed=False,
                   source_sha256=ENTRY_SHA, source_hashes=hashes,
                   asset_sha256=hashlib.sha256(blob).hexdigest(), bytes=len(blob),
                   action_count=len(active), source_boundary=boundaries, native_parity='PENDING')
    put(EXP / 'receipts/native_lynn_v5_static_export.json', json.dumps(receipt, indent=2).encode())
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__': main()
