#pragma once
// No learned parameters, training data, ML or RL inference are shipped in R13.
// The legacy negative-scenario interface deliberately fails rather than falling back.
#include <vector>
#include <stdexcept>
namespace triad::learned {
inline constexpr bool available=false;
inline double score(const std::vector<double>&,bool){
 throw std::logic_error("A06 R13: learned inference is disabled; scenario must be nonnegative");
}
}
