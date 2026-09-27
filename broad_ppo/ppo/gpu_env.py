"""GPU simulator bridge for full-season PPO collection.

JAX owns immutable simulator state. Torch views must not be modified in place.
Only initialization and explicit diagnostics may read data on the host.
"""
import os
import jax
if os.environ.get('PPO_JAX_ONE_DEVICE') == '1' and os.environ.get('JAX_CUDA_VISIBLE_DEVICES'):  # scale-out: jax 0.11 ignores the
    jax.config.update('jax_cuda_visible_devices', os.environ['JAX_CUDA_VISIBLE_DEVICES'])  # env var; one CUDA context per process
import jax.numpy as jnp
import torch
import hashlib
from pathlib import Path
from kaggriculture_jax.state import reset, load_tables, load_event_bank, events_for_seed
from kaggriculture_jax.simulator import batched_step_sync
from kaggriculture_jax.types import Action, Events


def runtime_identity():
    """Pin simulator rules and tables in PPO checkpoints, not just adapters."""
    import jaxlib
    from kaggriculture_jax import state
    root=Path(state.__file__).parent
    h=hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        h.update(path.name.encode());h.update(path.read_bytes())
    with state.DEFAULT_TABLES_PATH.open('rb') as f:
        tables=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(source_sha256=h.hexdigest(),tables_sha256=tables,
                jax=jax.__version__,jaxlib=jaxlib.__version__,torch=str(torch.__version__),
                cuda=torch.version.cuda)


class GpuGameBatch:
    def __init__(self, seeds, device='cuda:0', events=None):
        self.device = torch.device(device)
        if self.device.type != 'cuda' or not seeds:
            raise ValueError('A nonempty seed list and CUDA device are required')
        _gpus = jax.devices('gpu')  # scale-out: with PPO_JAX_ONE_DEVICE only this rank's GPU is visible to JAX
        self.jax_device = _gpus[self.device.index or 0] if len(_gpus) > (self.device.index or 0) else _gpus[0]
        self.reset_state = jax.jit(jax.vmap(reset))
        self.advance = jax.jit(batched_step_sync,donate_argnums=(0,) if os.environ.get('PPO_DONATE_SIM_STATE')=='1' else ())
        with jax.default_device(self.jax_device):
            self.tables = load_tables()
        self.reset(seeds,events)

    def reset(self,seeds,events=None):
        with jax.default_device(self.jax_device):
            if events is None:
                from ppo.gpu_events import prepare_events
                events = prepare_events(seeds)
            weed,shops=events
            if weed.shape != (len(seeds),30,200) or shops.shape != (len(seeds),30,201):
                raise ValueError('Event shape does not match the full seed batch')
            self.events = Events(jnp.asarray(weed),jnp.asarray(shops))
            self.state = self.reset_state(jnp.asarray(seeds, dtype=jnp.int32))

    def tensors(self):
        return {name: torch.utils.dlpack.from_dlpack(value)
                for name, value in zip(self.state._fields, self.state)}

    def step(self, action, return_state=True):
        if set(action) != set(Action._fields):
            raise ValueError('Expected the complete numeric simulator action schema')
        if any(value.device != self.device for value in action.values()):
            raise ValueError('Actions must already reside on the simulator GPU')
        # DLPack handles stream synchronization. Clone on GPU because contiguous
        # time slices can have offsets that violate XLA buffer alignment.
        values = Action(*(jax.dlpack.from_dlpack(action[name].contiguous().clone(), copy=False)
                          for name in Action._fields))
        self.state = self.advance(self.state, values, self.events, self.tables)
        return self.tensors() if return_state else None
