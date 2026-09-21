"""shaped-reward-v1: scalar baseline head and exact-state migration.

exact_model.py is hashed into the feature-cache identity, so the head is
attached from here at model construction time instead of being declared in the
architecture module. Registration order puts it last in state_dict and in
named_parameters, which the optimizer grouping and migration rely on.
"""
import torch
from torch import nn
from exact_model import SCHEMA

SHAPED_HEAD_KEYS = ('value_shaped.weight', 'value_shaped.bias')


def attach_shaped_head(model):
    """Zero-initialised Linear(256,1) on the critic state; V_shaped == 0 until trained."""
    if hasattr(model, 'value_shaped'):
        raise ValueError('Model already carries the shaped value head')
    head = nn.Linear(256, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    model.value_shaped = head
    return model


def with_shaped_head(weights, model):
    """Add zero scalar-head tensors when a state dict predates shaped-reward-v1."""
    weights = dict(weights)
    own = model.state_dict()
    for key in SHAPED_HEAD_KEYS:
        if key not in weights:
            weights[key] = torch.zeros_like(own[key], device='cpu')
    return weights


def without_shaped_head(weights):
    """Export format for consumers pinned to the pre-shaped-reward model contract."""
    return {k: v for k, v in weights.items() if k not in SHAPED_HEAD_KEYS}


def load_exact_with_head(model, checkpoint):
    """Same architecture/schema checks as ExactWorkerMarketPolicyV1.load_exact, tolerant of the missing head."""
    if checkpoint.get('architecture') != model.architecture or checkpoint.get('schema') != SCHEMA:
        raise ValueError('An exact worker/market checkpoint is required')
    model.load_state_dict(with_shaped_head(checkpoint['model'], model), strict=True)


def freeze_actor_gradients(module, optimizer):
    """Warm-up: drop every gradient outside the critic AdamW group so only value heads step."""
    critic = {id(p) for g in optimizer.param_groups[1:] for p in g['params']}
    dropped = 0
    for p in module.parameters():
        if id(p) not in critic and p.grad is not None:
            p.grad = None
            dropped += 1
    return dropped


def migrate_state(saved, model):
    """Pad a pre-shaped-reward FULL checkpoint in place: zero head tensors and two
    fresh critic-group AdamW entries. Every existing tensor, Adam moment, step
    counter, RNG blob, league/reference state and matchup record is untouched."""
    own = model.state_dict()
    if any(k in saved['model'] for k in SHAPED_HEAD_KEYS):
        raise ValueError('Checkpoint already carries the shaped value head')
    for key in SHAPED_HEAD_KEYS:
        saved['model'][key] = torch.zeros_like(own[key], device='cpu')
    opt = saved['optimizer']
    groups = opt['param_groups']
    if len(groups) != 2:
        raise ValueError('Expected actor/critic parameter groups')
    total = sum(len(g['params']) for g in groups)
    names = [n for n, _ in model.named_parameters()]
    critic = [n for n in names if n.startswith(('critic_memory.', 'value.'))]
    if names[-2:] != list(SHAPED_HEAD_KEYS):
        raise ValueError('Scalar head must be the last registered parameters')
    if len(groups[1]['params']) != len(critic) or total != len(names) - 2:
        raise ValueError('Optimizer parameter count does not match the pre-migration model')
    new_ids = [total, total + 1]
    if any(i in opt['state'] for i in new_ids):
        raise ValueError('Optimizer state already has entries for the new parameters')
    # shaped-reward-v4: third group for the residual head at 100x the critic LR; hyperparameters copied from the critic group.
    head_group = {k: v for k, v in groups[1].items() if k != 'params'}
    head_group['lr'] = groups[1]['lr'] * 100.
    if 'initial_lr' in head_group:
        head_group['initial_lr'] = head_group['lr']
    head_group['params'] = new_ids
    groups.append(head_group)
    return dict(kind='shaped-reward-v4', added_model_keys=list(SHAPED_HEAD_KEYS), added_optimizer_param_ids=new_ids,
                head_group_lr=head_group['lr'], critic_group_size=len(groups[1]['params']), optimizer_state_entries=len(opt['state']),
                preserved=['all existing model tensors', 'all Adam moments and step counters', 'CPU/CUDA RNG', 'league state', 'reference state', 'matchups'])
