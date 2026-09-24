#pragma once
#include "planner.hpp"
#include "sale_clock.hpp"
#ifndef R2_SHOP_BRANCH_AUDIT
#define R2_SHOP_BRANCH_AUDIT 0
#endif
// Conditional forecast, NOT a clone of the live environment. No live Simulator,
// hidden rival private state, actual random seed, or future events are accepted.
namespace fastkag {
class PublicFlowScenario {
 Simulator env_;std::vector<int8_t>known_;competitive::Flow flows_{};
 std::array<double,9>fraction_{};int start_day_;double scale_;
 int shop_branch_=-1;bool branch_applied_=false;
 const triad::SaleClock*clock_=nullptr;
 public:
 struct FlowAudit {
  std::array<double,9>requested{};
  std::array<int,9>rounded{},filled{},market_delta{};
  std::array<double,9>cash_delta{};
 };
 private:
 FlowAudit audit_{};
 public:
 PublicFlowScenario(const dp7::View&v,const competitive::Flow&flows,double scale,const triad::SaleClock*clock=nullptr,int shop_branch=-1):env_(Config{},0),known_(v.shops),flows_(flows),start_day_(v.day),scale_(scale),clock_(clock){
#if R2_SHOP_BRANCH_AUDIT
  if(shop_branch<-1||shop_branch>7)throw std::invalid_argument("shop branch");
  shop_branch_=shop_branch;
#else
  if(shop_branch!=-1)throw std::invalid_argument("shop branch diagnostic disabled");
#endif
  if(v.day!=v.step/24||v.hour!=v.step%24)throw std::invalid_argument("public scenario clock");
  env_.step_=v.step;env_.farms_[0]=v.own;env_.farms_[1]=v.opponent;env_.privates_[0]=v.priv;
  env_.privates_[1]=PrivateState{};env_.privates_[1].inventories.resize(v.opponent.hands.size()+1);env_.privates_[1].inventory_order.resize(v.opponent.hands.size()+1);
  env_.market_=v.market;env_.shops_=v.shops;env_.cfg_.weed_spawn_chance=0.;
 }
 dp7::View view()const{return{env_.step_,env_.day(),env_.hour(),env_.farms_[0],env_.farms_[1],env_.privates_[0],env_.market_,env_.shops_};}
 bool done()const{return env_.done_;}
#if R2_SHOP_BRANCH_AUDIT
 const std::vector<int32_t>& own_market_fills()const{return env_.last_market_fills()[0];}
#endif
 void advance(const PlayerAction&own){
  int s=env_.step_,d=env_.day(),h=env_.hour();PlayerAction rival;
#if R2_FLOW_AUDIT
  std::array<int,9>order,sign{};order.fill(-1);
#endif
  // Two deterministic arrival windows: a scenario assumption, not observed orders.
  if(clock_||h==1||h==17){for(int i=0;i<9;i++){
   double x=flows_[d][i]*scale_;int q=clock_?clock_->quantity(i,x,h):h==1?int(std::floor(x*.5)):int(std::round(x))-int(std::floor(x*.5));
#if R2_FLOW_AUDIT
   audit_.requested[i]+=clock_?q:x*.5;audit_.rounded[i]+=q;
#endif
   if(q>0){env_.privates_[1].shed[i]+=q;
#if R2_FLOW_AUDIT
    order[i]=rival.market.size();sign[i]=1;
#endif
    rival.market.push_back({Op::SELL,Item(i),q});}
   else if(q<0&&(i==0||i==8)){
#if R2_FLOW_AUDIT
    order[i]=rival.market.size();sign[i]=-1;
#endif
    rival.market.push_back({Op::BUY_PRODUCT,Item(i),-q});}
  }}
  // Public visible service is projected, never recovered from rival inventory.
  for(auto&t:env_.farms_[1].tiles){if(t.kind==TileKind::ANIMAL){t.fed_today=true;t.cared_today=true;t.yield_units=0;t.fertilizer_available=false;}
   if(t.kind==TileKind::PLANT){t.watered_today=true;t.yield_units=0;t.max_lifespan_step=-1;}}
  env_.step({own,rival});
#if R2_FLOW_AUDIT
  for(int i=0;i<9;i++)if(order[i]>=0){int k=order[i];
   audit_.filled[i]+=sign[i]*env_.last_market_fills()[1][k];
   audit_.cash_delta[i]+=env_.last_market_cash_deltas()[1][k];
   audit_.market_delta[i]+=env_.last_market_inventory_deltas()[1][k];
  }
#endif
  if(s%4==0){
   // Unknown shops are IID with replacement. Add their EXPECTED demand only.
   static const double average[9]={5./8,3./8,2./8,4./8,0,2./8,3./8,2./8,0};
   int unknown=std::max(0,std::min(8,d/3)-int(known_.size()));
   for(int i=0;i<9;i++){fraction_[i]+=unknown*average[i];int n=int(std::floor(fraction_[i]+1e-9));env_.market_.inventory[i]-=n;fraction_[i]-=n;}
   env_.refresh_prices();
  }
#if R2_SHOP_BRANCH_AUDIT
  // Reveal exactly the next unlock; all later unopened shops retain the old expected-demand model.
  if(shop_branch_>=0&&!branch_applied_&&h==23&&(d+1)%3==0&&known_.size()<8){known_.push_back(int8_t(shop_branch_));branch_applied_=true;}
#endif
  env_.shops_=known_; // Discard synthetic RNG draws rather than exposing them.
 }
 double own_cash()const{return env_.farms_[0].money;}
 double rival_cash()const{return env_.farms_[1].money;}
 const FlowAudit&audit()const{return audit_;}
 std::array<int,9>rival_shed()const{std::array<int,9>x{};for(int i=0;i<9;i++)x[i]=env_.privates_[1].shed[i];return x;}
};
}
