"""Shared spatial/entity encoder with separate worker and market decoders.

The two action prefixes reset each turn. Only actor/critic state spans turns.
Loss weighting is external; these methods return unweighted policy densities.
"""
import torch
import hashlib
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from model_v2 import MacroPolicyV2, gather, mlp

SCHEMA = 'exact-worker-market-bc-v1'
ENCODER_NAMES = ('board', 'pool', 'global_encoder', 'farm_encoder',
    'worker_encoder', 'market_encoder', 'program_encoder', 'history_encoder',
    'types', 'farm_role', 'layers')
ENCODER_PARAMETERS = ('position', 'spatial_queries')


def factor_log_probs(logits, gate, legal):
    """Index zero is PASS/END; all other entries share the ACT gate."""
    if legal.device.type == 'cpu' and not legal.any(-1).all():
        raise ValueError('Each action row needs at least one legal choice')
    any_action = legal[:, 1:].any(-1)
    gate = gate.masked_fill(torch.stack((~legal[:, 0], ~any_action), -1), -torch.inf)
    gate_logp = F.log_softmax(gate.float(), -1)
    conditional = logits[:, 1:].float().masked_fill(~legal[:, 1:], -torch.inf)
    conditional = conditional.masked_fill(~any_action[:, None], 0)
    command_logp = F.log_softmax(conditional, -1)
    joint = torch.cat((gate_logp[:, :1], gate_logp[:, 1:] + command_logp), -1)
    return joint.masked_fill(~legal, -torch.inf), gate_logp, command_logp


class ActionHead(nn.Module):
    def __init__(self, raw_width, ledger_width, extra_width):
        super().__init__()
        self.raw = mlp(raw_width, 192, 256)
        self.source_cell = nn.Linear(192, 256, bias=False)
        self.target_cell = nn.Linear(192, 256, bias=False)
        self.item = nn.Linear(192, 256, bias=False)
        self.worker = nn.Linear(192, 256, bias=False)
        self.norm = nn.LayerNorm(256)
        self.ledger = mlp(ledger_width, 128, 64)
        self.query = mlp(512 + 64 + extra_width, 256, 256)
        self.gate = mlp(512 + 64 + extra_width, 128, 2)
        self.bias = nn.Linear(raw_width, 1)
        self.quantity_context = mlp(832, 128, 64)
        self.quantity_embedding = mlp(12, 64, 64)
        self.feedback = nn.GRUCell(256, 256)
        self.amount = nn.Linear(1, 256, bias=False)
        self.delta = nn.Linear(ledger_width, 256, bias=False)

    def score(self, actor, prefix, context, raw, source_cells, target_cells, items, workers,
              ledger, extra, legal):
        candidates = self.norm(self.raw(raw)
            + self.source_cell(gather(context['cells'], source_cells))
            + self.target_cell(gather(context['cells'], target_cells))
            + self.item(gather(context['market'], items))
            + self.worker(gather(context['workers'], workers)))
        query_input = torch.cat((actor, prefix, self.ledger(ledger), extra), -1)
        logits = (candidates * self.query(query_input)[:, None]).sum(-1) / 16
        logits = logits + self.bias(raw).squeeze(-1)
        joint, gates, commands = factor_log_probs(logits, self.gate(query_input), legal)
        return joint, gates, commands, candidates

    def quantities(self, actor, prefix, chosen, ledger, features, legal):
        if legal.device.type == 'cpu' and not legal.any(-1).all():
            raise ValueError('Each quantity row needs at least one legal choice')
        q = self.quantity_context(torch.cat((actor, prefix, chosen, self.ledger(ledger)), -1))
        logits = (self.quantity_embedding(features) * q[:, None]).sum(-1) / 8
        return F.log_softmax(logits.float().masked_fill(~legal, -torch.inf), -1)

    def advance(self, prefix, chosen, quantity, delta):
        # Compute before narrowing to BF16/FP16, preserving sign for no-effect
        # requests whose parsed quantities are zero or negative.
        q = quantity.float()
        amount = (torch.sign(q) * torch.log1p(q.abs()) / 12).to(chosen.dtype).reshape(-1, 1)
        return self.feedback(chosen + self.amount(amount) + self.delta(delta), prefix)


class ExactWorkerMarketPolicyV1(nn.Module):
    architecture = 'ExactWorkerMarketPolicyV1'
    schema = SCHEMA

    def __init__(self, worker_ledger_width=128):
        super().__init__()
        old = MacroPolicyV2()
        for name in ENCODER_NAMES:
            self.add_module(name, getattr(old, name))
        for name in ENCODER_PARAMETERS:
            self.register_parameter(name, getattr(old, name))
        self.actor_memory = nn.GRUCell(192, 256)
        self.critic_memory = nn.GRUCell(192, 256)
        self.value = mlp(256, 128, 3)
        self.worker_head = ActionHead(128, worker_ledger_width, 192)
        self.market_head = ActionHead(128, 112, 256)
        # Market receives resolved state, never the worker request prefix.
        self.post_worker_context = mlp(112 + 96, 256, 256)
        assert sum(p.numel() for p in self.parameters()) < 10_000_000

    encode_features = MacroPolicyV2.encode_features
    encode = MacroPolicyV2.encode

    def summarize_post_worker(self, ledger, own_farm):
        return self.post_worker_context(torch.cat((ledger, own_farm), -1))

    def warm_start_encoder(self, checkpoint):
        """Initialization only. Never loads recurrence, action heads, or optimizer."""
        if checkpoint.get('architecture') != 'MacroPolicyV2' or checkpoint.get('schema') != 'macro-bc-v2':
            raise ValueError('Encoder source must be MacroPolicyV2 / macro-bc-v2')
        source = checkpoint['model']
        own = self.state_dict()
        names = [name for name in own if name.split('.')[0]
                 in ENCODER_NAMES + ENCODER_PARAMETERS]
        for name in names:
            if name not in source or source[name].shape != own[name].shape:
                raise ValueError('Incompatible encoder tensor: ' + name)
        # Validate the entire whitelist before changing any tensor.
        with torch.no_grad():
            for name in names:
                own[name].copy_(source[name])
            own['worker_encoder.0.weight'][:, 34:64].zero_()
            own['farm_encoder.weight'][:, 94].zero_()
        return dict(copied_tensors=names, worker_input_copied_columns=[0, 34],
                    worker_input_zeroed_columns=[34, 64], farm_input_zeroed_indices=[94],
                    optimizer_loaded=False)

    def warm_start_encoder_file(self, path):
        path = Path(path)
        # Hash and load the same open file. This records initialization evidence,
        # not an exact training-resume contract.
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
            source.seek(0)
            checkpoint = torch.load(source, map_location='cpu', weights_only=True)
        return dict(self.warm_start_encoder(checkpoint), source_checkpoint_sha=digest)

    def load_exact(self, checkpoint):
        if checkpoint.get('architecture') != self.architecture or checkpoint.get('schema') != SCHEMA:
            raise ValueError('An exact worker/market checkpoint is required')
        self.load_state_dict(checkpoint['model'], strict=True)
