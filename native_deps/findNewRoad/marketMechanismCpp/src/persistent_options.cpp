#include "persistent_options.hpp"

#include <algorithm>
#include <stdexcept>

namespace g001::option {

const char* name(Kind kind) {
    switch (kind) {
        case Kind::Baseline: return "baseline";
        case Kind::Hold: return "hold";
        case Kind::Drip: return "drip";
        case Kind::PriceTarget: return "price-target";
        case Kind::InventoryTarget: return "inventory-target";
        case Kind::PreDump: return "pre-dump";
        case Kind::Clear: return "clear";
    }
    return "unknown";
}

DripTransition settle_drip(DripState state, int available, int filled) {
    state.remaining_quota = std::max(0, state.remaining_quota);
    state.remaining_windows = std::max(0, state.remaining_windows);
    state.debt = std::max(0, state.debt);
    available = std::max(0, available);
    const auto base = state.remaining_windows == 0 ? 0 :
        (state.remaining_quota + state.remaining_windows - 1) / state.remaining_windows;
    const auto requested = std::min(available, base + state.debt);
    if (filled < 0 || filled > requested) {
        throw std::invalid_argument("DRIP fill outside requested range");
    }
    DripTransition result;
    result.base_due = base;
    result.requested = requested;
    result.next.remaining_quota = std::max(0, state.remaining_quota - base);
    result.next.remaining_windows = std::max(0, state.remaining_windows - 1);
    result.next.debt = std::max(0, state.debt + base - filled);
    return result;
}

}  // namespace g001::option
