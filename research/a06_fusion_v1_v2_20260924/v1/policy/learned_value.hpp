#pragma once
#include <vector>
#include <stdexcept>
namespace triad::learned {
inline constexpr bool available=false;
inline double score(const std::vector<double>&,bool){throw std::logic_error("ML/RL disabled in R13");}
}
