#pragma once
// Isolated frozen opponent. These source-specific dates are NOT our planner.
#include "policy.hpp"
#include <map>
#include <string>

namespace lynn5 {
struct Data {std::vector<fastkag::PlayerAction> actions;};
struct Recovery {int start=0;fastkag::Action intended;};
struct Delivery {
  int last=-1,animal=-1;
  bool active=false,cancelled=false;
  std::string reason;
  std::vector<int> interventions;
};
struct Decision {
  int bundle=0,target=0,window=-1;
  double wool=0,dairy=0,milk_price=0,wool_price=0,balance=0,signal=0;
  bool mature=false,deferred=false,reused=false;
  std::vector<int8_t> shops;
};
struct State {
  int seat=-1,base_last=-1,last=-1,cow_to_sheep=0,sheep_to_cow=0,window=-1;
  std::array<int,5> assignments{-1,-1,-1,-1,-1};
  std::map<int,Recovery> weed;
  std::vector<int8_t> window_shops,deferred_shops;
  int deferred_bundle=-1;double deferred_signal=0;
  std::vector<Decision> decisions;
  std::array<Delivery,3> delivery;
  // Last actions of six semantic layers; checked against all Python wrappers.
  std::array<fastkag::PlayerAction,6> stages;
};
class Agent {
 public:
  explicit Agent(Data d):data(std::move(d)){}
  const Data data;
  fastkag::PlayerAction act(const dp7::View&,int seat,State&) const;
 private:
  fastkag::PlayerAction source(const dp7::View&,int step,const State&) const;
};
}
