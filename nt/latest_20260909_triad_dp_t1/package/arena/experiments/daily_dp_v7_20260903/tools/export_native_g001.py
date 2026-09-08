"""Unpack the complete frozen G001 opponent, not a selected route substitute."""
from pathlib import Path
import hashlib, importlib.util, json, sys, zlib

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/main.py'
EXPECTED = '9b4fdf7a7c92d3eefc75c693c4949825567588478bef0e53529bdec82944dda7'

def main():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED, digest
    spec = importlib.util.spec_from_file_location('frozen_native_opponent_export', SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    controller, expanded = module._S_CONTROLLER, module._S_EXPANDED
    families = list(controller.route_by_family)
    assert controller.forced_opening == 'G001', controller.forced_opening
    k320 = expanded.namespace['_BASE'].__dict__['_K320'].__dict__
    moon = expanded.namespace['_MOON'].__dict__
    labels = ('10C4S_3Q', '8C6S_3Q', '6C8S_3Q', '6C12S_4Q_FIRST_YARN', '6C12S_4Q_SECOND_YARN')
    def market_tape(values):
        return [{'farmer':['PASS'], 'hands':[], 'market':list(m or [])} for m in values]
    nodes = []
    for checkpoint, tree in controller.nodes[controller.forced_opening]:
        targets = []
        for cls in tree.classes:
            family = cls if cls in controller.route_by_family else controller.targets[int(cls)]
            targets.append(families.index(family))
        nodes.append(dict(checkpoint=checkpoint, targets=targets, left=tree.left.tolist(), right=tree.right.tolist(),
                          feature=tree.feature.tolist(), threshold=tree.threshold.tolist(), value=tree.value.tolist()))
    payload = dict(source_sha256=digest, families=families, opening=families.index('G001'), nodes=nodes,
        routes=[expanded.action_tapes[controller.route_by_family[f]] for f in families],
        r5=market_tape(k320['_V17_R5_MARKETS']), md=market_tape(k320['_V17_MD_MARKETS']),
        moon=[moon['_ACTIONS_'+l] for l in labels], moon_legacy=[moon['_LEGACY_ACTIONS_'+l] for l in labels])
    path=EXP/'native/g001_frozen.json.zlib'
    path.write_bytes(zlib.compress(json.dumps(payload,separators=(',',':')).encode(),9))
    receipt=dict(source_sha256=digest, asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 families=len(families), opening='G001', checkpoints=[n['checkpoint'] for n in nodes], bytes=path.stat().st_size)
    (EXP/'receipts/native_g001_export.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(receipt))

if __name__=='__main__': main()
