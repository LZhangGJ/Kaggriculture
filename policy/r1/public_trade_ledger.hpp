#pragma once
#include "executor/observed_day_scenario.hpp"
#ifndef R2_OBSERVE_PUBLIC_TRADES
#define R2_OBSERVE_PUBLIC_TRADES 0
#endif
#ifndef R2_PENDING_STOCK_WEIGHT
#define R2_PENDING_STOCK_WEIGHT 0.0
#endif
namespace triad {
// An observable inventory UPPER BOUND, not access to the opponent's warehouse.
// Only non-buyable products are tracked. For these, within-step market supply
// is monotone until known town consumption, allowing certified sale inference
// when no $1 sale can have been hidden from the market inventory counter.
struct PublicTradeLedger {
 static constexpr bool enabled=R2_OBSERVE_PUBLIC_TRADES || R2_PENDING_STOCK_WEIGHT>0;
 static constexpr double weight=R2_PENDING_STOCK_WEIGHT;
 static_assert(weight>=0&&weight<=1);
 struct Snapshot {
  int step=-1;dp7::Farm rival{};fastkag::Market market{};
  std::array<int,9> own_after_units{};std::vector<int8_t> shops;
 } previous;
 std::array<double,9> lower{},upper{},added{},certain_added{};
 std::array<int,9> rival_net{},valid{};
 std::array<int,9> own_sales{},own_valid{};
 int observation_step=-1,source_step=-1,accepted=0,skipped_floor=0,skipped_boundary=0,invalid_market=0;
 static bool tracked(int i){return i>=1&&i<=7;}
 static std::array<int,9> total(const fastkag::PrivateState& p){
  std::array<int,9> x{};for(int i=0;i<9;++i){x[i]=p.shed[i];for(auto&v:p.inventories)x[i]+=v[i];}return x;
 }
 static std::array<int,9> consumption(int step,const std::vector<int8_t>&shops){
  std::array<int,9> x{};
  static const std::array<std::vector<int>,8> ps{{{dp7::E,dp7::W},{dp7::E,dp7::W,dp7::S},
    {dp7::W,dp7::C,dp7::T,dp7::S},{dp7::S,dp7::MI,dp7::W},{dp7::C,dp7::C},
    {dp7::MI,dp7::T,dp7::W},{dp7::S,dp7::MI},{dp7::WO,dp7::WO}}};
  if(step%4==0)for(int s:shops)for(int i:ps[s])++x[i];
  if(step%24==0)for(int i=0;i<8;++i)++x[i];return x;
 }
 static int available_upper(const dp7::Tile&t,int day){
  if(dp7::animal(t))return t.yield_units;
  if(!dp7::plant(t))return 0;
  int k=int(t.crop),q=t.yield_units;
  if(dp7::ongoing(k))return q;
  int age=day-t.planted_day,last=k==dp7::W?4:k==dp7::C?3:12;
  // A different worker can water immediately before the harvest. This bound
  // includes that legal gain; we do not mistake current yield for a maximum.
  // Fertilize -> water -> harvest can all occur in the same unit phase using
  // different workers; after removal we cannot observe the intermediate tile.
  if(!t.watered_today&&age>=(last+1)/2&&age<=last)q=std::min(k==dp7::C?4:6,q+2);
  return q;
 }
 void observe(const dp7::View&o){
  if constexpr(!enabled)return;
  observation_step=o.step;source_step=previous.step;added={};certain_added={};rival_net={};valid={};own_sales={};own_valid={};
  if(previous.step<0)return;
  if(o.step!=previous.step+1)throw std::runtime_error("nonconsecutive public trade ledger");
  const bool boundary=previous.step%24==23;
  for(int pos=0;pos<100;++pos){
   const auto&a=previous.rival.tiles[pos];const auto&b=o.opponent.tiles[pos];
   int item=dp7::animal(a)?dp7::product[int(a.animal)-9]:dp7::plant(a)?int(a.crop):-1;
   if(!tracked(item))continue;
   int gain=0,certain=0;
   if(boundary){
    // Refresh can hide a harvest; any previously available output might have
    // been taken. Never claim this is an observed successful harvest.
    gain=available_upper(a,previous.step/24);
   }else if(dp7::animal(a)){
    bool same=dp7::animal(b)&&a.animal==b.animal&&a.placed_day==b.placed_day;
    gain=same?std::max(0,int(a.yield_units)-int(b.yield_units)):int(a.yield_units);
    certain=same?gain:0;
   }else{
    bool same=dp7::plant(b)&&a.crop==b.crop&&a.planted_day==b.planted_day;
    gain=same?std::max(0,int(a.yield_units)-int(b.yield_units)):available_upper(a,previous.step/24);
    certain=same?gain:0;
    // DIG and decay are indistinguishable from some disappearance signals;
    // adding them is safe only as an upper bound, not as certain stock.
   }
   added[item]+=gain;certain_added[item]+=certain;lower[item]+=certain;upper[item]+=gain;
  }
  auto now=total(o.priv),consume=consumption(previous.step,previous.shops);
  for(int i=1;i<=7;++i){
   if(boundary){++skipped_boundary;lower[i]=0;upper[i]=std::min(upper[i],100.0);continue;}
   own_sales[i]=previous.own_after_units[i]-now[i];own_valid[i]=own_sales[i]>=0;
   // Before consumption this is the largest inventory in a sell-only market.
   // Equality at $1 is rejected conservatively, including a possible last
   // quote of $2 that reaches the saturation boundary after that sale.
   const int maximum_inventory=o.market.inventory[i]+consume[i];
   if(dp7::price(i,previous.market.inventory[i])<=1||dp7::price(i,maximum_inventory)<=1){++skipped_floor;lower[i]=0;continue;}
   int own_net=previous.own_after_units[i]-now[i];
   int all_net=o.market.inventory[i]-previous.market.inventory[i]+consume[i];
   int rival=all_net-own_net;
   // A diagnostic failure must not stop a live farm. Audits require this
   // counter to remain zero; production falls back to an uncertain bound.
   if(rival<0){++invalid_market;lower[i]=0;continue;}
   rival_net[i]=rival;valid[i]=1;++accepted;lower[i]=std::max(0.0,lower[i]-rival);upper[i]=std::max(0.0,upper[i]-rival);
  }
 }
 void record(const dp7::View&o,const fastkag::PlayerAction&action){
  if constexpr(!enabled)return;
  fastkag::ObservedDayScenario world(o);auto projected=world.project_units(action.units,-1);
  previous.step=o.step;previous.rival=o.opponent;previous.market=o.market;previous.shops=o.shops;
  previous.own_after_units=total(projected.privates()[0]);
 }
 void apply(competitive::Planner&model,int day,double supply)const{
  if constexpr(weight>0){
   model.extra_rival={};if(supply<=0)return;
   for(int i=1;i<=7;++i)model.extra_rival[day][i]=weight*upper[i]/supply;
  }
 }
};
}
