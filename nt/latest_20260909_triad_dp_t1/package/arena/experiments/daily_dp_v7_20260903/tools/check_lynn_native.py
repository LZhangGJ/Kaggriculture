"""Six-layer original Lynn V5 differential check with frozen official referee."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import sys
import time
import uuid
import zlib

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PACKAGE = EXP / 'opponents/lynn_v5/output/generated_submission'
sys.path.insert(0, str(EXP / 'native/build'))
sys.path.insert(0, str(ROOT / 'gpt_review/codex/G001_CPU_FOR_GPT_20260903'))
import _dp7_native as native
from cpu_runtime import LocalGame
from check_native import canon, normalized


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


class Reference:
    def __init__(self):
        # The original uses module-level monkeypatch contexts. Reload every
        # package-owned module between games; never share it between threads.
        paths = {p.resolve() for p in PACKAGE.rglob('*.py')}
        for name, mod in list(sys.modules.items()):
            file = getattr(mod, '__file__', None)
            if file and Path(file).resolve() in paths: del sys.modules[name]
        for p in (str(PACKAGE), str(PACKAGE / 'agents')):
            if p not in sys.path: sys.path.insert(0, p)
        sys.dont_write_bytecode = True
        name = 'lynn_reference_' + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, PACKAGE / 'main.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        self.agent = module.kaggriculture_agent
        self.top = module._policy.__globals__['_MODULE']
        self.late = self.top.PARENT
        self.fixed = self.late.PARENT
        self.latent = self.fixed.PARENT
        self.terminal = self.latent.PARENT
        self.network = self.terminal.PARENT
        self.engine = self.network.ENGINE
        self.stages = [None] * 6
        for index, layer in enumerate((self.engine, self.terminal, self.latent, self.fixed, self.late, self.top)):
            original = layer.agent
            def capture(obs, configuration=None, fn=original, i=index):
                action = fn(obs, configuration)
                self.stages[i] = copy.deepcopy(action)
                return action
            layer.agent = capture

    def state(self, seat):
        n = self.network._STATE[seat]
        b = self.engine._STATE[seat]
        result = {k: copy.deepcopy(n[k]) for k in (
            'last_step', 'assignments', 'cow_to_sheep', 'sheep_to_cow',
            'cow_window_target', 'cow_window_shops', 'deferred_cow_signal', 'decision_rows')}
        result.update(base_last=b['last_step'], weed_transactions=copy.deepcopy(b['weed_transactions']),
                      delivery=[copy.deepcopy(layer._STATE[seat]) for layer in (self.latent, self.fixed, self.late)],
                      stages=[normalized(x) for x in self.stages])
        for r in result['weed_transactions'].values():
            r['intended'] = normalized(dict(farmer=r['intended']))['farmer']
        return canon(result)


def native_state(st):
    result = canon(st.debug())
    result['stages'] = [normalized(x) for x in result['stages']]
    for r in result['weed_transactions'].values():
        r['intended'] = normalized(dict(farmer=r['intended']))['farmer']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', default=str(EXP / 'profiles/candidates.json'))
    p.add_argument('--label', default='S3C03')
    p.add_argument('--seeds', default='20261401,20261404')
    p.add_argument('--seats', default='0,1')
    p.add_argument('--mirror', action='store_true')
    p.add_argument('--out', required=True)
    p.add_argument('--save-replays', action='store_true')
    args = p.parse_args()
    out = Path(args.out);out.mkdir(parents=True, exist_ok=False)
    params = json.loads(Path(args.configs).read_text())[args.label]
    asset = EXP / 'native/lynn_v5_frozen.json.zlib'
    data = json.loads(zlib.decompress(asset.read_bytes()))
    for rel, h in data['source_hashes'].items(): assert digest(PACKAGE / rel) == h, rel
    opponent = native.LynnV5(data)
    build = json.loads((EXP / 'native/build/build_receipt.json').read_text())
    assert digest(Path(native.__file__)) == build['binary_sha256']
    for rel, h in build['source_hashes'].items(): assert digest(EXP / rel) == h, rel
    receipt = dict(status='RUNNING', args=vars(args), params=params, build=build,
                   source_hashes=data['source_hashes'], asset_sha256=digest(asset), rows=[])
    (out / 'plan.json').write_text(json.dumps(receipt, indent=2))
    def fail(kind, **details):
        (out / 'failure.json').write_text(json.dumps(dict(kind=kind, **details), indent=2))
        raise AssertionError((kind, details.get('seed'), details.get('seat'), details.get('step')))
    started = time.perf_counter()
    for seed in map(int, args.seeds.split(',')):
        for seat in map(int, args.seats.split(',')):
            env, official = native.Env(seed), LocalGame(seed)
            own, st, mirror_st = native.Controller(params), native.LynnState(), native.LynnState()
            ref = Reference();trace=[];weed_frames=0;max_action=0
            for step in range(719):
                ours = opponent.act(env, seat, mirror_st) if args.mirror else own.act(env, seat)
                tic = time.perf_counter();actual = opponent.act(env, 1-seat, st)
                max_action = max(max_action, time.perf_counter()-tic)
                expected = ref.agent(official.observation(1-seat))
                cs, ps = native_state(st), ref.state(1-seat)
                if normalized(actual) != normalized(expected):
                    fail('opponent_action', seed=seed, seat=seat, step=step, actual=actual, expected=expected,
                         native_state=cs, reference_state=ps, observation=canon(official.observation(1-seat)))
                if cs != ps:
                    fail('opponent_state_or_layer_action', seed=seed, seat=seat, step=step, actual=cs, expected=ps,
                         differing_fields=[k for k in cs if cs[k] != ps[k]], observation=canon(official.observation(1-seat)))
                weed_frames += bool(cs['weed_transactions'])
                actions = [None, None];actions[seat], actions[1-seat] = ours, actual
                expected_actions = copy.deepcopy(actions);expected_actions[1-seat] = expected
                if args.save_replays: trace.append(dict(step=step, observation=canon(env.observation(seat)), actions=actions, opponent_state=cs))
                env.step(actions);official.advance(expected_actions)
                for side in (0, 1):
                    ao, eo = canon(env.observation(side)), canon(official.observation(side))
                    for field in ('farms', 'private', 'market', 'town', 'day', 'hour', 'step'):
                        if ao[field] != eo[field]: fail('official_state', seed=seed, seat=seat, step=step+1, field=field, actual=ao[field], expected=eo[field])
            assert env.done and official.done and env.step_count == 719
            farms = env.observation(seat)['farms']
            row = dict(seed=seed, seat=seat, steps=719, cash=farms[seat]['money'], opponent_cash=farms[1-seat]['money'],
                       opponent_state=native_state(st), weed_repair_frames=weed_frames,
                       action_mismatches=0, layer_action_mismatches=0, memory_mismatches=0, official_state_mismatches=0,
                       max_native_opponent_action_seconds=max_action)
            receipt['rows'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'opponent_state'}), flush=True)
            if args.save_replays: (out / f'{seed}_seat{seat}.json.gz').write_bytes(gzip.compress(json.dumps(dict(trace=trace, final=canon(env.observation(seat)))).encode()))
            (out / 'progress.json').write_text(json.dumps(receipt, indent=2))
    receipt.update(status='PASS', seconds=time.perf_counter()-started,
                   check='original_full_package_six_layer_actions_all_behavior_memory_and_official_719_step_state',
                   caveat='Finite differential tests, not proof for every legal observation; local referee is not full Kaggle sandbox.')
    (out / 'acceptance.json').write_text(json.dumps(receipt, indent=2))


if __name__ == '__main__': main()
