"""Official-engine default parity against a locally rebuilt pre-change binary.

Run with --baseline-binary work/agent-before-marginal.so. Both seats are checked
on complete trajectories; this is compatibility testing, not win-rate evidence.
"""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
from kaggle_environments import make

ROOT = Path(__file__).resolve().parents[1]

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-binary', type=Path, required=True)
    parser.add_argument('--candidate-binary', type=Path, default=ROOT / 'policy/r1/agent.so')
    parser.add_argument('--seed', type=int, default=2610500000)
    args = parser.parse_args()
    os.environ.pop('R1_CONFIG_OVERRIDES', None)
    module = load(ROOT / 'policy/r1/agent.py', 'marginal_agent')
    frames = 0
    for seat in (0, 1):
        before = module.Agent(binary_path=args.baseline_binary)
        after = module.Agent(binary_path=args.candidate_binary)
        rival = load(ROOT / 'opponents/thomas_2945/main.py', f'parity_rival_{seat}').agent
        env = make('kaggriculture', configuration={'seed': args.seed}, debug=True)
        state = env.reset()
        try:
            while not env.done:
                observations = []
                for player in (0, 1):
                    obs = json.loads(json.dumps(state[player].observation))
                    obs['player'] = player
                    obs['step'] = obs['day'] * 24 + obs['hour']
                    observations.append(obs)
                old_action = before(observations[seat])
                new_action = after(observations[seat])
                if old_action != new_action:
                    raise AssertionError((seat, observations[seat]['step'], old_action, new_action))
                actions = [None, None]
                actions[seat] = old_action
                actions[1-seat] = rival(observations[1-seat])
                state = env.step(actions)
                frames += 1
            assert after.debug()['marginal_value_updates'] == 0
        finally:
            before.close(); after.close()
    # A C caller built against the previous settings prefix must still get
    # the new field's default, without reading beyond its supplied array.
    prefix = module._ORDER[:-1]
    params = (ctypes.c_double * len(prefix))(*(float(after.config[k]) for k in prefix))
    handle = after.lib.td_new(params, len(params))
    if not handle:
        raise AssertionError('new binary rejected the previous settings prefix')
    after.lib.td_delete(handle)
    for value in (-1, .5, 2):
        try:
            invalid = module.Agent(config={'marginal_value': value}, binary_path=args.candidate_binary)
        except ValueError:
            pass
        else:
            invalid.close()
            raise AssertionError(f'invalid switch accepted: {value}')
    try:
        invalid = module.Agent(config={'marginal_value': 1}, binary_path=args.baseline_binary)
    except RuntimeError as exc:
        assert 'rebuild' in str(exc)
    else:
        invalid.close()
        raise AssertionError('old binary silently accepted the new switch')
    print(json.dumps({'status': 'PASS', 'equal_frames': frames, 'seats': 2,
                      'seed': args.seed, 'switch': 'marginal_value'}))

if __name__ == '__main__':
    main()
