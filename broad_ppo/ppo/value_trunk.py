"""critic-fix-v1 (source-v45): the scalar value_shaped loss may train the critic trunk; MSE option for that loss.

Defaults reproduce source-v43 exactly (value_trunk_grad 0, value_loss huber): replay.py and train.py only call into
this module when a flag is on, and the contract drops both keys while they hold their defaults.

--value-trunk-grad 1
    source-v43 fits V_shaped = utility(value(c)).detach() + value_shaped(c.detach()): the Huber/MSE gradient reaches only
    the Linear(256,1) head, and critic_memory is shaped only by the win/draw/loss cross-entropy. With the flag on, the
    head reads a second run of critic_memory over the DETACHED encoder token, from the same initial state and with the
    same burn-in detach. Its forward values equal the first run (same weights, same inputs), so V_shaped, GAE and the
    cross-entropy are unchanged at the moment the flag turns on; the shaped-return gradient now reaches
        critic_memory.{weight_ih, weight_hh, bias_ih, bias_hh} and value_shaped.{weight, bias}
    and nothing else: the token is detached (no encoder / actor trunk / feature-ext gradient), utility() stays detached
    (no gradient into the value.* classifier), and actor_memory / heads are not on this path.
--value-loss mse
    0.5*(V - G)^2 per turn. Equal to Huber(delta=1) whenever |V - G| <= 1, so --vf-shaped keeps its meaning; beyond that
    it keeps the quadratic gradient, removing the median-seeking bias of Huber on bimodal (win/loss) returns.
"""
import torch

KEYS = ('value_trunk_grad', 'value_loss')
DEFAULTS = dict(value_trunk_grad=0, value_loss='huber')
MIGRATION_KIND = 'critic-fix-v1'
# Config keys a critic-fix-v1 migration may change (old and new values are recorded in the receipt).
MIGRATION_KEYS = KEYS + ('policy_warmup_until', 'critic_lr')


def add_arguments(p):
    p.add_argument('--value-trunk-grad', type=int, default=0, choices=(0, 1),
                   help='critic-fix-v1: 1 = the value_shaped return loss also trains critic_memory (token detached: no encoder/actor gradient)')
    p.add_argument('--value-loss', default='huber', choices=('huber', 'mse'),
                   help='critic-fix-v1: per-turn loss of the scalar shaped value (mse = 0.5*err^2, equal to Huber(1) inside |err|<=1)')


def configure(loss_module, args):
    """Called once on the LossModule before any DDP wrap. With defaults nothing is set, so v43 attribute lookups fall through."""
    if int(getattr(args, 'value_trunk_grad', 0)): loss_module.scorer.value_trunk_grad = True
    if getattr(args, 'value_loss', 'huber') != 'huber': loss_module.value_loss = args.value_loss


def contract_config(contract, args):
    """Defaults leave the run contract exactly as source-v43."""
    for k in KEYS:
        if getattr(args, k, DEFAULTS[k]) == DEFAULTS[k]: contract['config'].pop(k, None)


def critic_for_value(chunk, token, initial_critic, burn):
    """Critic states for the value_shaped head with a gradient path into critic_memory only (see module doc)."""
    token = token.detach()
    if chunk.sequence_memory:
        from ppo.sequence_memory import run_sequence
        states, _ = run_sequence(chunk.critic_sequence, token, initial_critic, burn)
        return states.flatten(0, 1)
    ch = initial_critic; states = []
    for t in range(token.shape[0]):
        if t == burn: ch = ch.detach()
        ch = chunk.model.critic_memory(token[t], ch); states.append(ch)
    return torch.stack(states).flatten(0, 1)


def shaped_value_loss(prediction, target, kind):
    if kind == 'mse': return .5 * (prediction - target).square()
    raise ValueError('Unknown value loss: ' + str(kind))


def migration_allows(old_config, new_config, migration):
    """critic-fix-v1: only MIGRATION_KEYS may differ, and exactly as the receipt records (absent key = v43 default)."""
    if migration.get('kind') != MIGRATION_KIND: return False
    prior = dict(old_config); current = dict(new_config)
    old = {k: prior.pop(k, DEFAULTS.get(k)) for k in MIGRATION_KEYS}
    new = {k: current.pop(k, DEFAULTS.get(k)) for k in MIGRATION_KEYS}
    return prior == current and old == migration.get('old_values') and new == migration.get('new_values')


def apply_resume_learning_rate(optimizer, args, migration):
    """A critic-fix-v1 migration that changes critic_lr applies it to the critic group (the saved Adam groups keep the old LR otherwise)."""
    if not migration or migration.get('kind') != MIGRATION_KIND: return None
    if migration['old_values'].get('critic_lr') == migration['new_values'].get('critic_lr'): return None
    optimizer.param_groups[1]['lr'] = args.critic_lr
    return args.critic_lr
