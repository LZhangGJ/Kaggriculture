"""Fast local simulation helpers for Kaggriculture."""

from .fast_env import (
    ENGINE_VERSION,
    EpisodeResult,
    FastKaggricultureEnv,
    StepResult,
    VectorFastEnv,
    run_duel,
    run_fast_episode,
)

try:
    from .gpu_engine import CudaKaggricultureEnv, GpuEngineConfig, TensorActions, TensorState, TensorStep
except ImportError:  # PyTorch is an optional dependency.
    CudaKaggricultureEnv = None
    GpuEngineConfig = None
    TensorActions = None
    TensorState = None
    TensorStep = None

__all__ = [
    "ENGINE_VERSION",
    "EpisodeResult",
    "FastKaggricultureEnv",
    "StepResult",
    "VectorFastEnv",
    "run_duel",
    "run_fast_episode",
    "CudaKaggricultureEnv",
    "GpuEngineConfig",
    "TensorActions",
    "TensorState",
    "TensorStep",
]
