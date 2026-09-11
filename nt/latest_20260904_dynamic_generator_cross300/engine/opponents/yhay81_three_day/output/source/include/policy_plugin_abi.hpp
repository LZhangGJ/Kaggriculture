// Stable C ABI between the native tournament and separately compiled policies.
#pragma once

#include "runtime_types.hpp"

#include <cstdint>

namespace kag::native {

inline constexpr uint32_t POLICY_PLUGIN_ABI_VERSION = 1;

using PluginAbiVersion = uint32_t (*)();
using PluginCreate = void* (*)();
using PluginDestroy = void (*)(void* context);
using PluginAct = int (*)(
    void* context,
    const State* state,
    const Config* config,
    int seat,
    Action* output);

}  // namespace kag::native
