"""Full-resolution, entity-grounded recurrent macro policy (PyTorch 2.11).

The state encoder runs once per observation; candidate/quantity scoring uses the
current request ledger after every selection. No attention over job candidates.
"""
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'macro-bc-v2'


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


def gather(tokens, indices):
    """-1 is a missing reference, never an alias for the last entity."""
    selected = tokens.gather(1, indices.clamp_min(0)[..., None].expand(-1, -1, tokens.shape[-1]))
    return selected * (indices >= 0)[..., None]


class MacroPolicyV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.board = nn.Sequential(nn.Conv2d(36, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.SiLU(), nn.Conv2d(128, 192, 3, padding=1))
        self.position = nn.Parameter(torch.randn(1, 100, 192) * .01)
        self.spatial_queries = nn.Parameter(torch.randn(1, 4, 192) * .01)
        self.pool = nn.MultiheadAttention(192, 4, batch_first=True)
        self.global_encoder = nn.Linear(64, 192)
        self.farm_encoder = nn.Linear(96, 192)
        self.worker_encoder = mlp(64, 192, 192)
        self.market_encoder = mlp(64, 192, 192)
        self.program_encoder = mlp(96, 192, 192)
        self.history_encoder = mlp(48, 128, 192)
        self.types = nn.Embedding(7, 192)
        self.farm_role = nn.Embedding(2, 192)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(192, 4, 768,
            dropout=0., activation='gelu', batch_first=True, norm_first=True) for _ in range(3)])
        self.actor_memory = nn.GRUCell(192, 256)
        self.critic_memory = nn.GRUCell(192, 256)
        self.candidate = mlp(128, 192, 256)
        self.cell_ref = nn.Linear(192, 256, bias=False)
        self.item_ref = nn.Linear(192, 256, bias=False)
        self.worker_ref = mlp(392, 192, 256)
        self.plan_ref = nn.Linear(192, 256, bias=False)
        self.candidate_norm = nn.LayerNorm(256)
        self.query = nn.Sequential(nn.Linear(512, 256), nn.SiLU())
        self.ledger_encoder = mlp(112, 128, 64)
        self.ledger_query = nn.Linear(64, 256, bias=False)
        self.phase = nn.Embedding(2, 64)
        self.gate = mlp(640, 128, 2)
        self.candidate_bias = nn.Linear(128, 1)
        self.feedback = nn.GRUCell(256, 256)
        self.amount_feedback = nn.Linear(1, 256, bias=False)
        self.ledger_feedback = nn.Linear(112, 256, bias=False)
        self.phase_feedback = nn.Linear(64, 256, bias=False)
        self.quantity_context = mlp(832, 128, 64)
        self.quantity_embedding = mlp(12, 64, 64)
        self.value = mlp(256, 128, 3)
        nn.init.zeros_(self.ledger_query.weight)
        nn.init.zeros_(self.ledger_feedback.weight)
        nn.init.zeros_(self.phase_feedback.weight)

    def encode_features(self, x):
        b = x['boards'].shape[0]
        cells = self.board(x['boards'].reshape(b * 2, 36, 10, 10)).flatten(2).transpose(1, 2)
        cells = cells + self.position
        summaries = self.pool(self.spatial_queries.expand(b * 2, -1, -1), cells, cells,
                              need_weights=False)[0].reshape(b, 2, 4, 192)
        summaries = (summaries + self.farm_role.weight[None, :, None]).reshape(b, 8, 192)
        groups = [self.global_encoder(x['global_features'])[:, None], self.farm_encoder(x['farms']),
                  summaries, self.worker_encoder(x['workers']), self.market_encoder(x['market']),
                  self.program_encoder(x['programs']), self.history_encoder(x['history'])]
        groups = [v + self.types.weight[i] for i, v in enumerate(groups)]
        groups[1] = groups[1] + self.farm_role.weight[None]
        tokens = torch.cat(groups, 1)
        padding = torch.cat([torch.zeros(b, 11, device=tokens.device, dtype=torch.bool),
            ~x['worker_valid'], torch.zeros(b, 12, device=tokens.device, dtype=torch.bool),
            ~x['program_valid'], ~x['history_valid']], 1)
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=padding)
        w = x['workers'].shape[1]
        context = dict(cells=cells.reshape(b, 200, 192), workers=tokens[:, 11:11+w],
                       market=tokens[:, 11+w:23+w], programs=tokens[:, 23+w:39+w])
        return tokens[:, 0], context

    def encode(self, x, actor_h, critic_h):
        token, context = self.encode_features(x)
        actor_h = self.actor_memory(token, actor_h)
        critic_h = self.critic_memory(token, critic_h)
        return actor_h, critic_h, self.value(critic_h), context

    def score(self, actor_h, prefix, context, candidates, ledger, phase):
        raw, refs = candidates['raw'], candidates['refs']
        worker = torch.cat([gather(context['workers'], refs[..., 2]),
                            gather(context['workers'], refs[..., 3]), candidates['relations']], -1)
        encoded = self.candidate_norm(self.candidate(raw)
            + self.cell_ref(gather(context['cells'], refs[..., 0]))
            + self.item_ref(gather(context['market'], refs[..., 1]))
            + self.worker_ref(worker) + self.plan_ref(gather(context['programs'], refs[..., 4])))
        led = self.ledger_encoder(ledger)
        ph = self.phase.weight[phase].expand(len(actor_h), -1)
        query = self.query(torch.cat([actor_h, prefix], -1)) + self.ledger_query(led)
        logits = (encoded * query[:, None]).sum(-1) / 16 + self.candidate_bias(raw).squeeze(-1)
        legal = candidates['admissible']
        logits = logits.masked_fill(~legal, -torch.inf)
        gates = self.gate(torch.cat([actor_h, prefix, led, ph], -1))
        gates = gates.masked_fill(torch.stack([torch.zeros_like(legal[:, 0]), ~legal[:, 1:].any(-1)], -1), -torch.inf)
        # END is index 0. Its probability is independent of candidate count.
        conditional = logits[:, 1:]
        safe = conditional.masked_fill(~legal[:, 1:].any(-1)[:, None], 0)
        joint = torch.cat([F.log_softmax(gates, -1)[:, :1],
            F.log_softmax(gates, -1)[:, 1:] + F.log_softmax(safe, -1)], -1)
        joint = joint.masked_fill(~legal, -torch.inf)
        return joint, encoded

    def quantities(self, actor_h, prefix, chosen, ledger, features, legal):
        ctx = self.quantity_context(torch.cat([actor_h, prefix, chosen, self.ledger_encoder(ledger)], -1))
        scores = (self.quantity_embedding(features) * ctx[:, None]).sum(-1) / 8
        return scores.masked_fill(~legal, -torch.inf)

    def advance(self, prefix, chosen, quantity, delta, phase):
        q = torch.log1p(quantity.to(chosen.dtype)).reshape(-1, 1) / 12
        ph = self.phase.weight[phase].expand(len(chosen), -1)
        return self.feedback(chosen + self.amount_feedback(q) + self.ledger_feedback(delta)
                             + self.phase_feedback(ph), prefix)

    def warm_start(self, old):
        """Copy only shape-compatible weights, plus expanded input columns.

        This is initialization, never optimizer/checkpoint resume.
        """
        own = self.state_dict(); copied = []
        for k, v in old.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v); copied.append(k)
        for k in ('board.0.weight', 'candidate.0.weight'):
            if k in old:
                columns=57 if k=='candidate.0.weight' else 24
                own[k].zero_(); own[k][:, :columns].copy_(old[k][:, :columns]); copied.append(k)
        self.load_state_dict(own)
        return copied
