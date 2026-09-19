"""Teacher-free shared-encoder joint student entry."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import torch
from common import Agent, load_checkpoint
torch.set_num_threads(1)
torch.set_float32_matmul_precision('highest')
model, codec, _ = load_checkpoint(Path(__file__).resolve().parent / 'selected.pt', 'cuda')
agents = {}


def agent(observation, configuration=None):
    seat = int(observation['player'])
    if int(observation['step']) == 0:
        agents[seat] = Agent(model, codec, 'cuda')
    return agents[seat](observation)
