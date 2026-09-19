"""Shared learned policy and data codec for joint training and actual matches."""
from pathlib import Path
import json
import os
import sys
ROOT = Path(__file__).resolve().parent
SOURCE = Path('F:/Kaggriculture/experiments/local_teacher_bc_20260917')
sys.path.insert(0, str(ROOT / 'frozen_code'))
import numpy as np
import torch
from model_joint_numeric import load_checkpoint, loss_and_counts
from student import observation_batch, UNIT_TYPES
from numeric_features import exact_numbers, GLOBAL_NAMES
from features import encode_action, CROPS


def save(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    os.replace(temp, path)


def arrays(split):
    keys = ('board', 'global_features', 'units', 'unit_count', 'step', 'unit_y', 'market_y', 'previous_u', 'previous_m')
    result = {k: np.load(SOURCE / f'data/packed/{split}/{k}.npy', mmap_mode='r') for k in keys}
    result.update({k: np.load(SOURCE / f'data/numeric/{split}/{k}.npy', mmap_mode='r') for k in ('global_exact', 'unit_exact')})
    assert len({len(a) for a in result.values()}) == 1
    return result


def batch(data, indices, device):
    return {k: torch.from_numpy(np.array(a[indices], copy=True)).to(device) for k, a in data.items()}


def canonical_unit_history(u, b):
    """History codec only: mirror encode_action's atomic seed-demand cancellation."""
    result = u.clone()
    for crop in CROPS:
        selected = u[:, :, 0] == UNIT_TYPES.index(('PLANT', crop))
        blocked = selected.sum(1) > b['global_exact'][:, GLOBAL_NAMES.index('seeds_'+crop)]
        mask = selected & blocked[:, None]
        result[:, :, 0].masked_fill_(mask, UNIT_TYPES.index(('PASS', None)))
        result[:, :, 1].masked_fill_(mask, 0)
    return result


class Agent:
    def __init__(self, model, codec, device='cpu'):
        self.model, self.codec, self.device = model, codec, device
        self.previous_u = np.zeros((32, 2), np.int16)
        self.previous_m = np.zeros((11, 2), np.int16)
        self.next_step = 0
        self.cancellations = 0

    @torch.inference_mode()
    def __call__(self, obs):
        assert int(obs['step']) == self.next_step and 'seed' not in obs
        b = observation_batch(obs, self.previous_u, self.previous_m)
        b['global_exact'], b['unit_exact'] = exact_numbers(obs)
        n = int(b['unit_count'])
        b = {k: torch.as_tensor(a)[None].to(self.device) for k, a in b.items()}
        u, m = self.model.predict(b)
        action = self.codec.decode(u[0].cpu().numpy(), m[0].cpu().numpy(), n)
        # Store actual submitted actions with the same canonical rule used by the dataset.
        tokens, _, stats = encode_action(action, n, obs['private']['seeds'])
        self.previous_u, self.previous_m = self.codec.encode(tokens, n)
        self.cancellations += stats['atomic_plant_cancellations']
        self.next_step += 1
        return action
