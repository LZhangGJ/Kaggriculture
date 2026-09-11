#pragma once
// Frozen opponent implementation only. Never a branch in dp7::Controller.
#include "policy.hpp"
#include <map>
#include <string>
#include <stdexcept>

namespace boatlee29 {
struct Data {
  std::vector<fastkag::PlayerAction> actions;
  std::vector<int> items;
  std::array<double,12> base_price{};
  std::array<std::array<int,12>,8> shop_demand{};
  std::map<std::string,double> config;
  int weed_replay_steps=8;
  double c(const char* key) const {return config.at(key);}
};
struct Recovery {int start=0;fastkag::Action intended;};
struct State {
  int seat=-1,weed_last=-1,am_last=-1,near_mirror=-1;
  std::map<int,Recovery> active;
  std::array<int,12> last_inventory{},last_sold{},added{};
  std::array<double,12> pressure{};
  std::vector<int8_t> last_shops;
};
class Agent {
 public:
  explicit Agent(Data d):data(std::move(d)){}
  const Data data;
  fastkag::PlayerAction act(const dp7::View&,int seat,State&) const;
 private:
  void weed(const dp7::View&,int step,fastkag::PlayerAction&,State&) const;
  void observe(const dp7::View&,int step,State&) const;
  void market(const dp7::View&,int step,fastkag::PlayerAction&,State&) const;
  int demand(const std::vector<int8_t>&,int item) const;
};
}
