#pragma once
// Frozen public OPPONENT, not a strategy option in our dp7::Controller.
#include "policy.hpp"
#include <map>
#include <string>

namespace kaito58 {
using Products = std::array<int,9>;
struct Data {
  std::array<std::vector<fastkag::PlayerAction>,10> routes;
  std::array<std::string,10> names;
  std::map<std::string,double> config;
  double c(const char* key) const { return config.at(key); }
};
struct Recovery { int start; fastkag::Action intended; };
struct PolicyState {
  int last=-1, near_streak=0, evidence=0;
  bool near=false;
  double confidence=0;
  Products inventory{},net{};
  std::array<double,9> supply{};
  std::vector<int8_t> shops;
  std::map<int,Products> due;
  std::map<int,Recovery> weed;
  fastkag::PlayerAction emitted;
  int collisions=0,preempt_turns=0,preempt_units=0,repaid=0;
};
struct State {
  int seat=-1,last=-1,mirror_streak=0,selected=0;
  std::string mode;
  bool clone=false,known_yarn=false;
  std::array<PolicyState,10> policies;
};
class Agent {
 public:
  explicit Agent(Data d):data(std::move(d)){}
  const Data data;
  fastkag::PlayerAction act(const dp7::View&,int seat,State&) const;
  // Exposed for differential unit tests, receives only legal public/own state.
  int select(const dp7::View&,State&) const;
  fastkag::PlayerAction policy(const dp7::View&,int route,PolicyState&) const;
};
}
