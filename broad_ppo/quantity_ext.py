"""quantity-ext-v1 (audit): sharper quantity scoring, attached like shaped_reward.attach_shaped_head.

exact_model.py stays byte-identical. Each ActionHead gets
  q_extra: MLP(EXTRA -> 64 -> 64), final layer zero-initialised, fed with residual features appended by QuantityFeatures
  q_index: Parameter [vocab, 64], zero-initialised (a learned per-value embedding)
and its `quantities` method is replaced by one that adds both to the unchanged quantity_embedding(features[..., :12]).
With zero init the log-probabilities equal the original head exactly, so existing checkpoints load and behave identically.
"""
import types
import torch
from torch import nn
from torch.nn import functional as F

EXTRA = 9
KEYS = ('q_extra.0.weight', 'q_extra.0.bias', 'q_extra.2.weight', 'q_extra.2.bias', 'q_index')


def _quantities(self, actor, prefix, chosen, ledger, features, legal):
    if legal.device.type == 'cpu' and not legal.any(-1).all():
        raise ValueError('Each quantity row needs at least one legal choice')
    q = self.quantity_context(torch.cat((actor, prefix, chosen, self.ledger(ledger)), -1))
    base, extra = features[..., :12], features[..., 12:]
    emb = self.quantity_embedding(base) + self.q_extra(extra) + self.q_index[None]
    logits = (emb * q[:, None]).sum(-1) / 8
    return F.log_softmax(logits.float().masked_fill(~legal, -torch.inf), -1)


def attach_quantity_extension(model, worker_vocab, market_vocab):
    for head, v in ((model.worker_head, worker_vocab), (model.market_head, market_vocab)):
        if hasattr(head, 'q_extra'):
            raise ValueError('quantity extension already attached')
        head.q_extra = nn.Sequential(nn.Linear(EXTRA, 64), nn.SiLU(), nn.Linear(64, 64))
        nn.init.zeros_(head.q_extra[2].weight); nn.init.zeros_(head.q_extra[2].bias)
        head.q_index = nn.Parameter(torch.zeros(len(v), 64))
        head.quantities = types.MethodType(_quantities, head)
    return model


def fill_missing(weights, model):
    """State dicts from before the extension get the zero-init tensors (behaviour unchanged)."""
    weights = dict(weights)
    own = model.state_dict()
    for k, v in own.items():
        if k not in weights and any(k.endswith(s) for s in KEYS):
            weights[k] = v.detach().cpu().clone()
    return weights
