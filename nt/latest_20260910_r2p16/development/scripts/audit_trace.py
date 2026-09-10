"""Read-only policy audit: exact recorded actions through official 1.32.7.

Instrumentation calls original transition functions unchanged and checks all saved
daily states and final cash. No action edits, policy tuning or counterfactuals.
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import copy
import gzip
import hashlib
import json
import statistics
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = ROOT/'experiments/ahmed_v27_vs_r2_200_20260909'
REF = ROOT/'gpt_review/gpt_code/Kaggriculture_T2_RecordedCandidate_Recovered_20260909/Kaggriculture_T2_RecordedCandidate_Recovered_20260909/rebuild_checks/referee'
sys.path.insert(0, str(REF))
from cpu_runtime import LocalGame, load_engine


def save(name, obj):
    (HERE/(name+'.gz')).write_bytes(gzip.compress(json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode(),compresslevel=1))


def stock(private):
    total = Counter(private['shed'])
    for inv in private['inventories']:
        total.update(inv)
    return total


def counts(farm):
    out = Counter()
    for line in farm['tiles']:
        for t in line:
            if isinstance(t, dict):
                out[t.get('crop', t.get('animal', t['kind']))] += 1
    return dict(out)


def run_case(selected, output):
    global HERE
    HERE = Path(output)
    HERE.mkdir(parents=True, exist_ok=True)
    assert not (HERE/'AUDIT.json.gz').exists(), 'Completed audit exists; do not overwrite'
    trace = json.loads(gzip.decompress((ROOT/selected['trace']).read_bytes()))
    engine = load_engine()
    g = LocalGame(selected['seed'], engine)
    original = {n: getattr(engine, n) for n in ['_commit_unit', '_do_hire', '_do_buy_land',
        '_apply_unit_action', '_drop_inventories_to_shed', '_daily_refresh_plants', '_daily_refresh_animals']}
    transactions = defaultdict(lambda: {'quantity': 0, 'cash_delta': 0, 'prices': []})
    rejects, unit_events, overflow, expiration = [], [], [], []
    day_end = []

    def seat_of(obj, private=False):
        objs = [s.observation.private for s in g.state] if private else g.state[0].observation.farms
        return next(i for i, other in enumerate(objs) if obj is other)

    def commit(op, item, price, farm, private, market, shed_capacity=100):
        p = seat_of(farm)
        before = farm['money']
        ok = original['_commit_unit'](op, item, price, farm, private, market, shed_capacity)
        if ok:
            t = transactions[(g.t, p, op, item)]
            t['quantity'] += 1
            t['cash_delta'] += farm['money']-before
            t['prices'].append(price)
        else:
            reason = ('no_shed_stock' if op == 'SELL' and private['shed'].get(item, 0) <= 0
                      else 'no_cash' if op != 'SELL' and before < price
                      else 'shed_full' if op in ('BUY_PRODUCT','BUY_ANIMAL') and sum(private['shed'].values()) >= shed_capacity
                      else 'other')
            rejects.append({'step': g.t, 'seat': p, 'op': op, 'item': item, 'cash': before, 'reason': reason})
        return ok

    def atomic(name, label, farm, *args):
        p, before = seat_of(farm), farm['money']
        result = original[name](farm, *args)
        delta = farm['money']-before
        if delta:
            t = transactions[(g.t, p, label, label)]
            t['quantity'] += 1
            t['cash_delta'] += delta
            t['prices'].append(-delta)
        else:
            rejects.append({'step': g.t, 'seat': p, 'op': label, 'cash': before, 'reason': 'no_effect'})
        return result

    def apply_unit(farm, private, idx, action, *args):
        p = seat_of(farm)
        pos = engine._farmer_position(farm, idx)
        before_private = copy.deepcopy(private)
        before_farm = copy.deepcopy(farm)
        result = original['_apply_unit_action'](farm, private, idx, action, *args)
        op = action[0] if action else 'EMPTY'
        effect = farm != before_farm or private != before_private
        if op not in engine.FARMER_MOVES and op != 'PASS':
            oldtile = before_farm['tiles'][pos[1]][pos[0]] if pos else None
            newtile = farm['tiles'][pos[1]][pos[0]] if pos else None
            item = (oldtile.get('crop', oldtile.get('animal', oldtile.get('kind'))) if isinstance(oldtile, dict) else None)
            before_stock, after_stock = stock(before_private), stock(private)
            delta = {k: after_stock[k]-before_stock[k] for k in before_stock.keys() | after_stock.keys()
                     if after_stock[k] != before_stock[k]}
            unit_events.append({'step': g.t, 'seat': p, 'unit': idx, 'action': action, 'pos': pos,
                'effect': effect, 'target': item, 'stock_delta': delta,
                'new_crop': newtile.get('crop') if isinstance(newtile, dict) else None})
        return result

    def drop(private, capacity):
        p, before = seat_of(private, True), stock(private)
        result = original['_drop_inventories_to_shed'](private, capacity)
        after = stock(private)
        loss = {k: before[k]-after[k] for k in before if before[k] > after[k]}
        if loss:
            overflow.append({'step': g.t, 'seat': p, 'items': loss})
        return result

    def refresh(name, farm, *args):
        p = seat_of(farm)
        before = copy.deepcopy(farm['tiles'])
        result = original[name](farm, *args)
        for y, line in enumerate(before):
            for x, tile in enumerate(line):
                if not isinstance(tile, dict):
                    continue
                new = farm['tiles'][y][x]
                if ('animal' in tile and (not isinstance(new,dict) or 'animal' not in new)) or ('crop' in tile and (not isinstance(new,dict) or 'crop' not in new)):
                    expiration.append({'step': g.t, 'seat': p, 'source': name, 'pos': [x,y], 'before': tile, 'after': new})
        return result

    engine._commit_unit = commit
    engine._do_hire = lambda farm,*a: atomic('_do_hire','HIRE',farm,*a)
    engine._do_buy_land = lambda farm,*a: atomic('_do_buy_land','BUY_LAND',farm,*a)
    engine._apply_unit_action = apply_unit
    engine._drop_inventories_to_shed = drop
    engine._daily_refresh_plants = lambda farm,*a: refresh('_daily_refresh_plants',farm,*a)
    engine._daily_refresh_animals = lambda farm,*a: refresh('_daily_refresh_animals',farm,*a)
    expected = {d['step']: d['observations'] for d in trace['days']}
    checked, steps = 0, []
    for t, actions in enumerate(trace['actions']):
        assert g.t == t
        if t in expected:
            for p in (0,1):
                assert g.observation(p) == expected[t][p], (t,p)
            checked += 1
        # Record before last action of each day for actual active workers; next-day
        # snapshots otherwise reset all hands and obscure daily employment.
        if (t+1)%24 == 0 or t == 718:
            day_end.append({'day': t//24+1, 'step_before_last_action': t,
                'farms': [{'counts': counts(f), 'cash': f['money'], 'hands': len(f['hands']),
                           'land': len(f['unlocked_quadrants'])} for f in g.state[0].observation.farms]})
        g.advance(actions)
        steps.append({'step': t, 'cash': [f['money'] for f in g.state[0].observation.farms],
            'counts': [counts(f) for f in g.state[0].observation.farms],
            'land': [len(f['unlocked_quadrants']) for f in g.state[0].observation.farms],
            'hands': [len(f['hands']) for f in g.state[0].observation.farms],
            'prices': dict(g.state[0].observation.market['prices'])})
    assert g.done and g.t == 719
    for p in (0,1):
        assert g.observation(p) == expected[719][p]
    expected_cash = [None,None]
    expected_cash[selected['opponent_seat']] = selected['opponent_cash']
    expected_cash[1-selected['opponent_seat']] = selected['r2_cash']
    assert [g.state[0].observation.farms[p]['money'] for p in (0,1)] == expected_cash
    tx = [{'step': k[0], 'seat': k[1], 'op': k[2], 'item': k[3], 'quantity': v['quantity'],
           'cash_delta': v['cash_delta'], 'min_price': min(v['prices']), 'max_price': max(v['prices'])}
          for k,v in sorted(transactions.items())]
    summary = []
    for p in (0,1):
        net = defaultdict(float)
        for z in tx:
            if z['seat'] == p: net[z['op']+':'+z['item']] += z['cash_delta']
        assert 3000+sum(net.values()) == g.state[0].observation.farms[p]['money']
        products = {}
        for item in engine.PRODUCTS:
            ss = [z for z in tx if z['seat'] == p and z['op']=='SELL' and z['item']==item]
            q, cash = sum(z['quantity'] for z in ss), sum(z['cash_delta'] for z in ss)
            products[item] = {'quantity': q, 'revenue': cash, 'mean_price': cash/q if q else None,
                'first_sale_step': min((z['step'] for z in ss), default=None), 'last_sale_step': max((z['step'] for z in ss), default=None)}
        summary.append({'seat': p, 'cash_ledger': dict(net), 'products': products,
            'no_effect_units': dict(Counter(e['action'][0] for e in unit_events if e['seat']==p and not e['effect'])),
            'final_private': g.observation(p)['private'], 'final_counts': counts(g.state[0].observation.farms[p]),
            'rejects': [z for z in rejects if z['seat']==p], 'overflow': [z for z in overflow if z['seat']==p]})
    save('AUDIT.json', {'status':'PASS','seed':selected['seed'],'source_row':selected,
        'saved_frames_identical': checked+1,'steps':719,'cash_checks':'PASS',
        'official_sha256':hashlib.sha256((REF/'official/kaggriculture.py').read_bytes()).hexdigest(),
        'summary':summary, 'transactions':tx, 'unit_events':unit_events,
        'expiration':expiration,'daily_before_last_action':day_end,'post_steps':steps})
    return {'status':'PASS','opponent':selected['opponent'],'seed':selected['seed'],
        'opponent_seat':selected['opponent_seat'],'r2_win':selected['r2_win'],
        'r2_margin':selected['r2_margin'],'saved_frames_identical':checked+1,
        'summary':summary,'output':str(HERE/'AUDIT.json.gz')}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--row-file',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run_case(json.loads(args.row_file.read_text()),args.output),ensure_ascii=False))
