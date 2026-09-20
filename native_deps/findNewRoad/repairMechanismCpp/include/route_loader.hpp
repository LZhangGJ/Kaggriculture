#pragma once

#include "simulator.hpp"

#include <string>
#include <vector>

namespace g001::repair {

[[nodiscard]] std::vector<fastkag::PlayerAction> load_route(
    const std::string& compressed_tapes_path,
    const std::string& route_library_path,
    const std::string& family_or_route_id
);

}  // namespace g001::repair
