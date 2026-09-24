#pragma once
// R13: trained ranker removed. This build cannot execute ML/RL inference.
#include <vector>
namespace triad::learned {
inline constexpr bool available=false;
inline double score(const std::vector<double>&,bool){return 0.;}
}
