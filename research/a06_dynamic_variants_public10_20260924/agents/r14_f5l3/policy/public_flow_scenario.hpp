#pragma once
#include "planner.hpp"
#include "sale_clock.hpp"
// Conditional forecast, NOT a clone of the live environment. No live Simulator,
// hidden rival private state, actual random seed, or future events are accepted.
namespace fastkag {
class PublicFlowScenario {
 Simulator env_;std::vector<int8_t>known_;competitive::Flow flows_{};
 std::array<double,9>fraction_{};int start_day_;double scale_;
 const triad::SaleClock*clock_=nullptr;
 std::array<int,100> finish_{}; int flags_=0;
 public:
 PublicFlowScenario(const dp7::View&v,const competitive::Flow&flows,double scale,const triad::SaleClock*clock=nullptr,const competitive::Planner*planner=nullptr,int flags=0):env_(Config{},0),known_(v.shops),flows_(flows),start_day_(v.day),scale_(scale),clock_(clock){
  if(v.day!=v.step/24||v.hour!=v.step%24)throw std::invalid_argument("public scenario clock");
  env_.step_=v.step;env_.farms_[0]=v.own;env_.farms_[1]=v.opponent;env_.privates_[0]=v.priv;
  env_.privates_[1]=PrivateState{};env_.privates_[1].inventories.resize(v.opponent.hands.size()+1);env_.privates_[1].inventory_order.resize(v.opponent.hands.size()+1);
  env_.market_=v.market;env_.shops_=v.shops;env_.cfg_.weed_spawn_chance=0.;
  flags_=flags;finish_.fill(30);
  if((flags_&4)&&planner)for(int p=0;p<100;p++){
   const auto&t=v.opponent.tiles[p];int k=int(t.crop);
   if(t.kind!=TileKind::PLANT)continue;
   if(k==dp7::W||k==dp7::C||k==dp7::M){
    int age=v.day-t.planted_day,h=planner->core.h_age(k);double total=0;
    for(int a=std::max(0,age);a<30;a++)total+=planner->rival_harvest_weights[k][a];
    if(total>0){double cumulative=0;for(int a=std::max(0,age);a<30;a++){cumulative+=planner->rival_harvest_weights[k][a];if(cumulative>=total*.5){h=a;break;}}}
    finish_[p]=std::clamp(int(t.planted_day)+h,v.day,29);
   }else if(k==dp7::T||k==dp7::S){finish_[p]=std::clamp(int(t.planted_day)+dp7::first[k]+3*dp7::interval[k],v.day,29);}
  }

 }
 dp7::View view()const{return{env_.step_,env_.day(),env_.hour(),env_.farms_[0],env_.farms_[1],env_.privates_[0],env_.market_,env_.shops_};}
 bool done()const{return env_.done_;}
 void advance(const PlayerAction&own){
  int s=env_.step_,d=env_.day(),h=env_.hour();PlayerAction rival;
  // Two deterministic arrival windows: a scenario assumption, not observed orders.
  if(clock_||h==1||h==17){for(int i=0;i<9;i++){
   double x=flows_[d][i]*scale_;int q=clock_?clock_->quantity(i,x,h):h==1?int(std::floor(x*.5)):int(std::round(x))-int(std::floor(x*.5));
   if(q>0){env_.privates_[1].shed[i]+=q;rival.market.push_back({Op::SELL,Item(i),q});}
   else if(q<0&&(i==0||i==8))rival.market.push_back({Op::BUY_PRODUCT,Item(i),-q});
  }}
  // Public visible service is projected, never recovered from rival inventory.
  for(auto&t:env_.farms_[1].tiles){if(t.kind==TileKind::ANIMAL){t.fed_today=true;t.cared_today=true;t.yield_units=0;t.fertilizer_available=false;}
   if(t.kind==TileKind::PLANT){t.watered_today=true;t.yield_units=0;t.max_lifespan_step=-1;}}
  // When this conditional forecast has delivered the final crop lot, remove
  // that synthetic incumbent before a tail planner can value it again.
  // This is an explicit forecast assumption, never a mutation of the referee.
  if((flags_&4)&&h==17)for(int p=0;p<100;p++)if(d>=finish_[p])env_.farms_[1].tiles[p]=Tile{};
  env_.step({own,rival});
  if(s%4==0){
   // Unknown shops are IID with replacement. Add their EXPECTED demand only.
   static const double average[9]={5./8,3./8,2./8,4./8,0,2./8,3./8,2./8,0};
   int unknown=std::max(0,std::min(8,d/3)-int(known_.size()));
   for(int i=0;i<9;i++){fraction_[i]+=unknown*average[i];int n=int(std::floor(fraction_[i]+1e-9));env_.market_.inventory[i]-=n;fraction_[i]-=n;}
   env_.refresh_prices();
  }
  env_.shops_=known_; // Discard synthetic RNG draws rather than exposing them.
 }
 double own_cash()const{return env_.farms_[0].money;}
 double rival_cash()const{return env_.farms_[1].money;}
};
}
