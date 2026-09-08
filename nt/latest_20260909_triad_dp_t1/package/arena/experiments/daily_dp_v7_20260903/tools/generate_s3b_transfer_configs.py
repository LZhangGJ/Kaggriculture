"""Predefined external-idea ablations, not an exact port of GPT's v7."""
from pathlib import Path
import argparse, copy, hashlib, json

EXP = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    dest = Path(args.out)
    assert not dest.exists(), dest
    bases = {'S1': {}, 'S3C03': json.loads((EXP/'profiles/S3C03.json').read_text())}
    configs = {}
    for label, base in bases.items():
        for opening in ('control', 'portfolio', 'portfolio_layout'):
            for weight in (0., .25, .5, .75, 1.):
                p = copy.deepcopy(base)
                if opening != 'control':
                    p.update(opening_animals=['SHEEP', 'SHEEP', 'SHEEP', 'COW'],
                             opening_crops=[5, 0, 0, 3, 7])
                if opening == 'portfolio_layout':
                    # Crop enum: WHEAT, CARROT, TOMATO, STRAWBERRY, MELON.
                    p['opening_crop_order'] = [4, 0, 3, 2, 1]
                p['hold_opponent_supply_weight'] = weight
                configs[f'{label}_{opening}_hold{weight:g}'] = p
        # Separate the investment forecast from inventory-holding valuation.
        if label == 'S1':
            for opening in ('control', 'portfolio_layout'):
                for hold in (0., .75):
                    p = copy.deepcopy(configs[f'{label}_{opening}_hold{hold:g}'])
                    p['opponent_supply_weight'] = .75
                    configs[f'{label}_{opening}_investment075_hold{hold:g}'] = p
    dest.write_text(json.dumps(configs, indent=2), encoding='utf-8')
    print(json.dumps({'configs': len(configs), 'path': str(dest),
                      'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
