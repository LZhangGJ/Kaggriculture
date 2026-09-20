# Licensed under the Apache License, Version 2.0.
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import sysconfig

ROOT = Path(__file__).resolve().parent
TORCH_INCLUDE = Path(sysconfig.get_paths()["purelib"]) / "torch" / "include"

setup(
    name="fast-kaggriculture",
    version="0.1.0",
    packages=["fast_kaggriculture"],
    package_dir={"fast_kaggriculture": "python/fast_kaggriculture"},
    ext_modules=[Extension(
        "fast_kaggriculture._fast_kaggriculture",
        [
            str(ROOT / "src/bindings.cpp"),
            str(ROOT / "src/simulator.cpp"),
            str(ROOT / "src/native_teammate.cpp"),
            str(ROOT / "src/native_general_market.cpp"),
            str(ROOT / "src/native_phased_market.cpp"),
            str(ROOT / "../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/productionObligationCpp/src/production_obligation.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/productionForecastCpp/src/production_forecast.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/legacyBaselineForecastCpp/src/legacy_baseline_forecast.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/plannerInputCpp/src/planner_input.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/economicIntentBridgeCpp/src/economic_intent_bridge.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/protectedQueueCpp/src/protected_queue.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/queueInvariantProofCpp/src/queue_invariant_proof.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/robustCertificateCpp/src/robust_certificate.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/phaseInputCpp/src/phase_input.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/phasedTakeoverCpp/src/phased_takeover.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/publicBeliefRuntimeCpp/src/public_belief_runtime.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/nativeObservationAdapterCpp/src/native_observation_adapter.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/nativeSelectiveInputCpp/src/native_selective_input.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/selectiveRuntimeCpp/src/selective_runtime.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/general_econ_audit/src/adaptive_execution.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/src/general_planner.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/src/rolling_optimizer.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/src/dump_scenarios.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/src/persistent_options.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/src/belief.cpp"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/src/market.cpp"),
        ],
        include_dirs=[
            str(ROOT / "src"),
            str(ROOT / "../native_deps/findNewRoad/repairMechanismCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/productionObligationCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/productionForecastCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/legacyBaselineForecastCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/plannerInputCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/economicIntentBridgeCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/protectedQueueCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/queueInvariantProofCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/robustCertificateCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/phaseInputCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/phasedTakeoverCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/publicBeliefRuntimeCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/nativeObservationAdapterCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/nativeSelectiveInputCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/selectiveRuntimeCpp/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/general_econ_audit/include"),
            str(ROOT / "../native_deps/findNewRoad/marketMechanismCpp/include"),
            str(TORCH_INCLUDE),
        ],
        language="c++",
        extra_compile_args=["-O3", "-DNDEBUG", "-std=c++20", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )],
)
