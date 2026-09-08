#include "../policy/learned_value.hpp"
extern "C" double predict(const double*x,int n){return triad::learned::predict(std::vector<double>(x,x+n));}
