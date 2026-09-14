#pragma once
// Bounded public-information scenario, NOT a clone of a real game's future.
#include "observation_view.hpp"
#include <cmath>
#include <stdexcept>

namespace fastkag {
class ObservedDayScenario {
 public:
  explicit ObservedDayScenario(const dp7::View& v):env_(Config{},0),start_day_(v.day) {
    if(v.step<0||v.step>=719||v.day!=v.step/24||v.hour!=v.step%24)
      throw std::invalid_argument("day scenario clock");
    validate_farm(v.own);validate_farm(v.opponent);
    const auto n=v.own.hands.size()+1;
    if(v.priv.inventories.size()!=n||v.priv.inventory_order.size()!=n)
      throw std::invalid_argument("day scenario own inventory shape");
    for(int q:v.priv.shed)if(q<0)throw std::invalid_argument("negative shed");
    for(int q:v.priv.seeds)if(q<0)throw std::invalid_argument("negative seed");
    for(size_t u=0;u<n;u++){
      std::array<bool,12>seen{};
      for(int i:v.priv.inventory_order[u]){
        if(i<0||i>=12||seen[i]||v.priv.inventories[u][i]<=0)
          throw std::invalid_argument("day scenario inventory order");
        seen[i]=true;
      }
      for(int i=0;i<12;i++)if(v.priv.inventories[u][i]<0||
          (v.priv.inventories[u][i]>0)!=seen[i])
        throw std::invalid_argument("day scenario incomplete inventory order");
    }
    for(int shop:v.shops)if(shop<0||shop>=8)throw std::invalid_argument("day scenario shop");
    if(v.shops.size()>8)throw std::invalid_argument("day scenario shop count");
    env_.step_=v.step;
    env_.farms_[0]=v.own;env_.farms_[1]=v.opponent;
    env_.privates_[0]=v.priv;
    env_.privates_[1]=PrivateState{};
    env_.privates_[1].inventories.resize(v.opponent.hands.size()+1);
    env_.privates_[1].inventory_order.resize(v.opponent.hands.size()+1);
    env_.market_=v.market;env_.shops_=v.shops;
    known_shops_=v.shops;
    // Only unknown random metadata is suppressed. Existing weeds, crop decay,
    // deterministic maintenance and capacity/order semantics stay unchanged.
    env_.cfg_.weed_spawn_chance=0.;
  }
  bool finished()const {return env_.done()||env_.day()!=start_day_;}
  dp7::View view()const {
    if(finished())throw std::logic_error("no next-day planning in day scenario");
    return {env_.step_count(),env_.day(),env_.hour(),env_.farms()[0],env_.farms()[1],
      env_.privates()[0],env_.market(),env_.shops()};
  }
  void advance(const PlayerAction& own) {
    if(finished())throw std::logic_error("day scenario horizon exceeded");
    env_.step({own,PlayerAction{}});
    if(env_.day()!=start_day_)env_.shops_=known_shops_;
    ++ticks_;
  }
  const Farm& own()const{return env_.farms()[0];}
  const PrivateState& inventory()const{return env_.privates()[0];}
  const Market& market()const{return env_.market();}
  const std::vector<int8_t>& shops()const{return env_.shops();}
  const std::vector<int32_t>& fills()const{return env_.last_market_fills()[0];}
  // Exact current-turn own unit prefix only. Opponent private state remains
  // empty, and no market, clock, RNG or next-day transition is performed.
  Simulator project_units(const std::vector<Action>& actions,int count)const{
    return env_.project_unit_phase(0,actions,count);
  }
  // Diagnostic only: sell all CURRENT warehouse products at this scenario's
  // endpoint, including feed. Never adds phantom future harvest or bank cash.
  // At terminal the engine forbids further orders, so its quote is zero.
  double liquidation_quote()const{
    std::vector<Action>orders;
    for(int i=0;i<9;i++)if(inventory().shed[i]>0)
      orders.push_back({Op::SELL,Item(i),inventory().shed[i]});
    return env_.project_own_market(0,orders).farms()[0].money-own().money;
  }
  int overflow()const{return env_.last_end_of_day_overflow()[0];}
  int ticks()const{return ticks_;}
  int step_count()const{return env_.step_count();}
 private:
  Simulator env_;
  int start_day_,ticks_=0;
  std::vector<int8_t>known_shops_;
  static void validate_farm(const Farm& f){
    if(f.tiles.size()!=100||!std::isfinite(f.money)||f.money<0)
      throw std::invalid_argument("day scenario farm");
    auto position=[](Position p){if(p.x<0||p.x>=10||p.y<0||p.y>=10)
      throw std::invalid_argument("day scenario position");};
    position(f.farmer);for(auto p:f.hands)position(p);
  }
};
} // namespace fastkag
