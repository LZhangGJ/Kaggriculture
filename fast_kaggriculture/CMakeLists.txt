cmake_minimum_required(VERSION 3.18)
project(fast_kaggriculture LANGUAGES CXX)

option(ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(ENABLE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'include')"
  OUTPUT_VARIABLE TORCH_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)

set(MARKET_ROOT ../native_deps/findNewRoad/marketMechanismCpp)
set(NATIVE_SELECTIVE_SOURCES
  ${MARKET_ROOT}/nativeObservationAdapterCpp/src/native_observation_adapter.cpp
  ${MARKET_ROOT}/nativeSelectiveInputCpp/src/native_selective_input.cpp
  ${MARKET_ROOT}/selectiveRuntimeCpp/src/selective_runtime.cpp
  ${MARKET_ROOT}/publicBeliefRuntimeCpp/src/public_belief_runtime.cpp
  ${MARKET_ROOT}/productionObligationCpp/src/production_obligation.cpp
  ${MARKET_ROOT}/productionForecastCpp/src/production_forecast.cpp
  ${MARKET_ROOT}/legacyBaselineForecastCpp/src/legacy_baseline_forecast.cpp
  ${MARKET_ROOT}/plannerInputCpp/src/planner_input.cpp
  ${MARKET_ROOT}/economicIntentBridgeCpp/src/economic_intent_bridge.cpp
  ${MARKET_ROOT}/protectedQueueCpp/src/protected_queue.cpp
  ${MARKET_ROOT}/queueInvariantProofCpp/src/queue_invariant_proof.cpp
  ${MARKET_ROOT}/robustCertificateCpp/src/robust_certificate.cpp
  ${MARKET_ROOT}/phaseInputCpp/src/phase_input.cpp
  ${MARKET_ROOT}/phasedTakeoverCpp/src/phased_takeover.cpp
  ${MARKET_ROOT}/general_econ_audit/src/adaptive_execution.cpp
  ${MARKET_ROOT}/src/general_planner.cpp
  ${MARKET_ROOT}/src/rolling_optimizer.cpp
  ${MARKET_ROOT}/src/dump_scenarios.cpp
  ${MARKET_ROOT}/src/persistent_options.cpp
  ${MARKET_ROOT}/src/belief.cpp
  ${MARKET_ROOT}/src/market.cpp)
set(NATIVE_SELECTIVE_INCLUDES
  ${MARKET_ROOT}/nativeObservationAdapterCpp/include
  ${MARKET_ROOT}/nativeSelectiveInputCpp/include
  ${MARKET_ROOT}/selectiveRuntimeCpp/include
  ${MARKET_ROOT}/publicBeliefRuntimeCpp/include
  ${MARKET_ROOT}/productionObligationCpp/include
  ${MARKET_ROOT}/productionForecastCpp/include
  ${MARKET_ROOT}/legacyBaselineForecastCpp/include
  ${MARKET_ROOT}/plannerInputCpp/include
  ${MARKET_ROOT}/economicIntentBridgeCpp/include
  ${MARKET_ROOT}/protectedQueueCpp/include
  ${MARKET_ROOT}/queueInvariantProofCpp/include
  ${MARKET_ROOT}/robustCertificateCpp/include
  ${MARKET_ROOT}/phaseInputCpp/include
  ${MARKET_ROOT}/phasedTakeoverCpp/include
  ${MARKET_ROOT}/general_econ_audit/include
  ${MARKET_ROOT}/include)

add_library(_fast_kaggriculture MODULE
  src/bindings.cpp
  src/simulator.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(_fast_kaggriculture PRIVATE cxx_std_20)
target_compile_options(_fast_kaggriculture PRIVATE -O3 -DNDEBUG -march=native)
target_include_directories(_fast_kaggriculture PRIVATE
  ${TORCH_INCLUDE}
  src
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
target_link_libraries(_fast_kaggriculture PRIVATE Python3::Module OpenMP::OpenMP_CXX)
set_target_properties(_fast_kaggriculture PROPERTIES PREFIX "" OUTPUT_NAME "_fast_kaggriculture")
if(DEFINED Python3_EXTENSION_SUFFIX)
  set_target_properties(_fast_kaggriculture PROPERTIES SUFFIX "${Python3_EXTENSION_SUFFIX}")
endif()

add_executable(native_general_market_tests
  src/native_general_market_tests.cpp
  src/native_general_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/productionObligationCpp/src/production_obligation.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/src/market.cpp)
target_compile_features(native_general_market_tests PRIVATE cxx_std_20)
target_compile_options(native_general_market_tests PRIVATE -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic)
target_include_directories(native_general_market_tests PRIVATE
  src
  ../native_deps/findNewRoad/marketMechanismCpp/productionObligationCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/include)

enable_testing()
add_test(NAME native_general_market_tests COMMAND native_general_market_tests)

add_executable(native_phased_market_tests
  src/native_phased_market_tests.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(native_phased_market_tests PRIVATE cxx_std_20)
target_compile_options(native_phased_market_tests PRIVATE
  -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic)
target_include_directories(native_phased_market_tests PRIVATE
  src
  ${NATIVE_SELECTIVE_INCLUDES})
add_test(NAME native_phased_market_tests COMMAND native_phased_market_tests)

add_executable(current_unit_soft_sanitation_tests
  src/current_unit_soft_sanitation_tests.cpp
  src/current_unit_soft_sanitation.cpp
  src/simulator.cpp)
target_compile_features(current_unit_soft_sanitation_tests PRIVATE cxx_std_20)
target_compile_options(current_unit_soft_sanitation_tests PRIVATE
  -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic)
target_include_directories(current_unit_soft_sanitation_tests PRIVATE src)
add_test(NAME current_unit_soft_sanitation_tests
  COMMAND current_unit_soft_sanitation_tests)

add_executable(native_route_cursor_eval
  src/native_route_cursor_eval.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/route_loader.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(native_route_cursor_eval PRIVATE cxx_std_20)
target_compile_options(native_route_cursor_eval PRIVATE
  -O3 -DNDEBUG -march=native -Wall -Wextra -Wpedantic)
target_include_directories(native_route_cursor_eval PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
target_compile_definitions(native_route_cursor_eval PRIVATE
  NATIVE_RC_TAPES="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_actions.json.zlib"
  NATIVE_RC_LIBRARY="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_library.json")
target_link_libraries(native_route_cursor_eval PRIVATE OpenMP::OpenMP_CXX ZLIB::ZLIB)
add_test(NAME native_route_cursor_default_off_parity
  COMMAND native_route_cursor_eval
    --seed-begin 993001 --seeds 1 --threads 1
    --output ${CMAKE_CURRENT_BINARY_DIR}/route-cursor-smoke/report.json)

add_executable(native_event_local_repair_tests
  src/native_event_local_repair_tests.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(native_event_local_repair_tests PRIVATE cxx_std_20)
target_compile_options(native_event_local_repair_tests PRIVATE
  -O2 -Wall -Wextra -Wpedantic)
target_include_directories(native_event_local_repair_tests PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
add_test(NAME native_event_local_repair_tests
  COMMAND native_event_local_repair_tests)

add_executable(component_shadow_mapper_tests
  src/component_shadow_mapper_tests.cpp
  src/component_shadow_mapper.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/effectCertificateSemanticsCpp/src/effect_certificate_semantics.cpp
  src/simulator.cpp)
target_compile_features(component_shadow_mapper_tests PRIVATE cxx_std_20)
target_compile_options(component_shadow_mapper_tests PRIVATE
  -O2 -Wall -Wextra -Wpedantic)
target_include_directories(component_shadow_mapper_tests PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/effectCertificateSemanticsCpp/include
  ../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/include)
add_test(NAME component_shadow_mapper_tests COMMAND component_shadow_mapper_tests)

add_executable(component_shadow_eval
  src/component_shadow_eval.cpp
  src/component_shadow_mapper.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/effectCertificateSemanticsCpp/src/effect_certificate_semantics.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/src/component_scoped_route_planner.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/src/exact_move_slot_planner.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/src/event_local_elastic_day.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/route_loader.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(component_shadow_eval PRIVATE cxx_std_20)
target_compile_options(component_shadow_eval PRIVATE
  -O2 -Wall -Wextra -Wpedantic)
target_include_directories(component_shadow_eval PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/effectCertificateSemanticsCpp/include
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
target_compile_definitions(component_shadow_eval PRIVATE
  NATIVE_G001_TAPES="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_actions.json.zlib"
  NATIVE_G001_LIBRARY="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_library.json")
target_link_libraries(component_shadow_eval PRIVATE OpenMP::OpenMP_CXX ZLIB::ZLIB)
add_test(NAME component_shadow_eval_smoke
  COMMAND component_shadow_eval --normal-seeds 1
    --output ${CMAKE_CURRENT_BINARY_DIR}/component-shadow-smoke.json)

# Independent read-only audit for the bit-64 rolling MOVE skeleton. It shares
# the production implementation but never changes the native seam.
add_executable(native_route_skeleton_v3_adversarial_audit
  src/native_route_skeleton_v3_adversarial_audit.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(native_route_skeleton_v3_adversarial_audit PRIVATE
  cxx_std_20)
target_compile_options(native_route_skeleton_v3_adversarial_audit PRIVATE
  -O2 -Wall -Wextra -Wpedantic)
target_include_directories(native_route_skeleton_v3_adversarial_audit PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
add_test(NAME native_route_skeleton_v3_adversarial_audit
  COMMAND native_route_skeleton_v3_adversarial_audit)

add_executable(native_route_skeleton_v3_forced_audit
  src/native_route_skeleton_v3_forced_audit.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/route_loader.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(native_route_skeleton_v3_forced_audit PRIVATE cxx_std_20)
target_compile_options(native_route_skeleton_v3_forced_audit PRIVATE
  -O2 -Wall -Wextra -Wpedantic)
target_include_directories(native_route_skeleton_v3_forced_audit PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
target_compile_definitions(native_route_skeleton_v3_forced_audit PRIVATE
  NATIVE_G001_TAPES="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_actions.json.zlib"
  NATIVE_G001_LIBRARY="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_library.json")
target_link_libraries(native_route_skeleton_v3_forced_audit PRIVATE
  OpenMP::OpenMP_CXX ZLIB::ZLIB)

# Read-only full-episode regression over the actual G001 tape. This target
# deliberately stays separate from native_teammate.cpp/bindings.cpp seams.
add_executable(native_g001_forced_regression
  src/native_g001_forced_regression.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/route_loader.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(native_g001_forced_regression PRIVATE cxx_std_20)
target_compile_options(native_g001_forced_regression PRIVATE
  -O2 -Wall -Wextra -Wpedantic)
target_include_directories(native_g001_forced_regression PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
target_compile_definitions(native_g001_forced_regression PRIVATE
  NATIVE_G001_TAPES="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_actions.json.zlib"
  NATIVE_G001_LIBRARY="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_library.json")
target_link_libraries(native_g001_forced_regression PRIVATE
  OpenMP::OpenMP_CXX ZLIB::ZLIB)

# Paired, dual-seat bit-8 treatment evaluation at the official weed rate.
# The forced-trigger executable above remains an invariant stress only.
add_executable(native_g001_bit8_normal_eval
  src/native_g001_bit8_normal_eval.cpp
  src/native_teammate.cpp
  src/native_general_market.cpp
  src/native_phased_market.cpp
  src/simulator.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/repair.cpp
  ../native_deps/findNewRoad/repairMechanismCpp/src/route_loader.cpp
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/src/unified_market_allocator.cpp
  ${NATIVE_SELECTIVE_SOURCES})
target_compile_features(native_g001_bit8_normal_eval PRIVATE cxx_std_20)
target_compile_options(native_g001_bit8_normal_eval PRIVATE
  -O2 -Wall -Wextra -Wpedantic)
target_include_directories(native_g001_bit8_normal_eval PRIVATE
  src
  ../native_deps/findNewRoad/repairMechanismCpp/include
  ../native_deps/findNewRoad/repairMechanismCpp/eventTriggeredLocalRepairCpp/include
  ../native_deps/findNewRoad/jointFixedMoveOracleCpp/include
  ../native_deps/findNewRoad/marketMechanismCpp/unifiedMarketAllocatorCpp/include
  ${NATIVE_SELECTIVE_INCLUDES})
target_compile_definitions(native_g001_bit8_normal_eval PRIVATE
  NATIVE_G001_TAPES="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_actions.json.zlib"
  NATIVE_G001_LIBRARY="${CMAKE_CURRENT_LIST_DIR}/../teammate_meta_route_submission_v1/route_library.json")
target_link_libraries(native_g001_bit8_normal_eval PRIVATE
  OpenMP::OpenMP_CXX ZLIB::ZLIB)
