"""Compile action feedback using the documented PyTorch GRUCell equations."""
import torch
from torch.nn.functional import linear


def feedback(head, prefix, chosen, quantity, delta):
    q = quantity.float()
    amount = (torch.sign(q) * torch.log1p(q.abs()) / 12).to(chosen.dtype).reshape(-1, 1)
    x = chosen + head.amount(amount) + head.delta(delta)
    cell = head.feedback
    ir, iz, inn = linear(x, cell.weight_ih, cell.bias_ih).chunk(3, -1)
    hr, hz, hn = linear(prefix, cell.weight_hh, cell.bias_hh).chunk(3, -1)
    reset = torch.sigmoid(ir + hr)
    update = torch.sigmoid(iz + hz)
    candidate = torch.tanh(inn + reset * hn)
    return candidate + update * (prefix - candidate)


def compiled_feedback(head):
    def advance(prefix, chosen, quantity, delta):
        return feedback(head, prefix, chosen, quantity, delta)
    return torch.compile(advance, fullgraph=True, dynamic=False,
                         options={'triton.cudagraphs': False})
