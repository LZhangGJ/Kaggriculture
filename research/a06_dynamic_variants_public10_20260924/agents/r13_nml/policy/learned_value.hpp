#pragma once
// A06R13 is strictly rule-based. No fitted model, weights, training, or inference.
#include <vector>
#include <stdexcept>
namespace triad::learned {
inline constexpr bool available=false;
inline double score(const std::vector<double>&,bool) {
 throw std::logic_error("ML/RL is disabled in A06R13");
}
}
