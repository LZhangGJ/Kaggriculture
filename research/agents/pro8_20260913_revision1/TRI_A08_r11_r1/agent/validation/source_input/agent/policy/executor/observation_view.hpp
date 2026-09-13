#pragma once
#include "vendor/simulator.hpp"
namespace dp7 {
// The sole information boundary supplied to the autonomous policy.
struct View {
 int step,day,hour;
 const fastkag::Farm& own;
 const fastkag::Farm& opponent;
 const fastkag::PrivateState& priv;
 const fastkag::Market& market;
 const std::vector<int8_t>& shops;
};
}
