#pragma once
// R13 pure-rule build. No learned weights, training data or ranker.
#include <vector>
namespace triad::learned {
inline constexpr bool available=false;
inline double predict(const std::vector<double>&){return 0.;}
inline double score(const std::vector<double>&,bool){return 0.;}
}
