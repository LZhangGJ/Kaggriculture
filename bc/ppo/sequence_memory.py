"""Run unchanged GRUCell weights through PyTorch's sequence GRU backend."""
import torch
from torch import nn


def sequence_view(cell):
    # Meta construction avoids allocations and changes to the sampling RNG.
    sequence=nn.GRU(cell.input_size,cell.hidden_size,bias=cell.bias,device='meta')
    for name in ('weight_ih','weight_hh','bias_ih','bias_hh'):
        if getattr(cell,name,None) is not None:
            setattr(sequence,name+'_l0',getattr(cell,name))
    # The first public forward refreshes GRU's cached parameter references and
    # may repack their storage. Do it before collection captures CUDA pointers.
    with torch.no_grad():
        sequence(cell.weight_ih.new_zeros(1,1,cell.input_size),
                 cell.weight_ih.new_zeros(1,1,cell.hidden_size))
    return sequence


def run_sequence(sequence,token,state,burn):
    sequence.train(torch.is_grad_enabled())
    parts=[];hidden=state.unsqueeze(0).contiguous()
    if burn:
        prefix,hidden=sequence(token[:burn].contiguous(),hidden)
        parts.append(prefix)
    hidden=hidden.detach()
    if burn<len(token):
        suffix,hidden=sequence(token[burn:].contiguous(),hidden)
        parts.append(suffix)
    return torch.cat(parts,0),hidden[0]
