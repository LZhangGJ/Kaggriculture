#pragma once
// No trained model or learned parameters are shipped in this agent.
#include <vector>
namespace triad::learned {
inline constexpr bool available=false;
inline double score(const std::vector<double>&,bool){return 0.;}
}
