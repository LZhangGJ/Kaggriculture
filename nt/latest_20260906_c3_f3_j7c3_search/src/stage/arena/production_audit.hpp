#pragma once
// Offline read-only projections. These counters never enter the policy View.
#include "policy.hpp"
#include "investment_audit.hpp"
#include <stdexcept>
namespace dp7audit {
using namespace fastkag;
using Stock=std::array<int,12>;
struct ProductionDay {
 Stock generated{},acquired{},used{},unit_discard{},environment_loss{},drop_loss{},eod_loss{},bought{},sold{},end_private{},end_field{};
 std::array<int,5>planted{},fertilized{},watered{},seed_bought{},end_seeds{};
 std::array<int,3>placed{},fed{},cared{},escaped{};
 std::array<int,24>attempts{},no_effect{};
 int checks=0;
};
struct NoEffect {int step,player,unit,op,position,item;};
inline Stock private_stock(const Simulator&s,int p){Stock q=s.privates()[p].shed;for(auto&inv:s.privates()[p].inventories)dp7::add(q,inv);return q;}
inline Stock tile_stock(const Tile&t){Stock q{};if(dp7::plant(t))q[int(t.crop)]+=t.yield_units;else if(dp7::animal(t)){q[dp7::product[int(t.animal)-9]]+=t.yield_units;q[8]+=t.fertilizer_available;}return q;}
inline Stock field_stock(const Simulator&s,int p){Stock q{};for(auto&t:s.farms()[p].tiles)dp7::add(q,tile_stock(t));return q;}
inline int position(const Simulator&s,int p,int u){return dp7::cell(u?s.farms()[p].hands.at(u-1):s.farms()[p].farmer);}
inline Simulator units(const Simulator&env,int p,const PlayerAction&a,ProductionDay&d,std::vector<NoEffect>&examples,InvestmentAudit*investments=nullptr){
 Simulator previous=env;size_t n=std::min(a.units.size(),env.farms()[p].hands.size()+1);
 for(size_t u=0;u<n;u++){
  auto next=env.project_unit_phase(p,a.units,int(u+1));auto action=a.units[u];int op=int(action.op),item=int(action.item),pos=position(previous,p,u);
  auto before=private_stock(previous,p),after=private_stock(next,p),field_before=field_stock(previous,p),field_after=field_stock(next,p);
  auto&t=previous.farms()[p].tiles[pos];auto&nt=next.farms()[p].tiles[pos];Stock harvest{};bool effect=false;
  d.attempts[op]++;
  switch(action.op){
   case Op::HARVEST:case Op::COLLECT_FERTILIZER:
    for(int i=0;i<9;i++){harvest[i]=std::max(0,after[i]-before[i]);d.acquired[i]+=harvest[i];effect|=harvest[i]>0;}break;
   case Op::FEED:case Op::FERTILIZE:{int i=action.op==Op::FEED?0:8;int consumed=before[i]-after[i];d.used[i]+=consumed;effect=consumed>0;if(effect){if(i==0)d.fed[int(t.animal)-9]++;else d.fertilized[int(t.crop)]++;}break;}
   case Op::PLANT:
    if(item>=0&&item<5){int count=previous.privates()[p].seeds[item]-next.privates()[p].seeds[item];d.planted[item]+=count;effect=count>0;}break;
   case Op::WATER:effect=dp7::plant(t)&&!t.watered_today&&nt.watered_today;if(effect)d.watered[int(t.crop)]++;break;
   case Op::CARE:effect=dp7::animal(t)&&!t.cared_today&&nt.cared_today;if(effect)d.cared[int(t.animal)-9]++;break;
   case Op::PLACE:
    if(item>=9&&item<12&&before[item]>after[item]){d.used[item]+=before[item]-after[item];d.placed[item-9]++;effect=true;}
    else if(item>=0&&item<12)effect=previous.privates()[p].inventories[u][item]>next.privates()[p].inventories[u][item];break;
   case Op::DROP:
    for(int i=0;i<12;i++)d.drop_loss[i]+=before[i]-after[i];effect=dp7::sum(previous.privates()[p].inventories[u])>dp7::sum(next.privates()[p].inventories[u]);break;
   case Op::PICKUP:if(item>=0&&item<12)effect=previous.privates()[p].inventories[u][item]<next.privates()[p].inventories[u][item];break;
   case Op::DIG:effect=t.kind!=nt.kind;break;
   case Op::BUILD_COOP:case Op::BUILD_PASTURE:effect=t.kind!=nt.kind;break;
   case Op::NORTH:case Op::SOUTH:case Op::EAST:case Op::WEST:effect=pos!=position(next,p,u);break;
   case Op::PASS:effect=true;break;
   default:break;
  }
  if(!effect){d.no_effect[op]++;if(examples.size()<32)examples.push_back({env.step_count(),p,int(u),op,pos,item});}
  for(int i=0;i<9;i++){
   int delta=field_after[i]-field_before[i]+harvest[i];
   if(delta>=0)d.generated[i]+=delta;else d.unit_discard[i]-=delta;
  }
  if(investments)investments->unit(previous,next,p,int(u),action);
  previous=std::move(next);
 }
 return previous;
}
inline void finish(const Simulator&before,const std::array<Simulator,2>&projected,const Simulator&after,
                   const std::array<PlayerAction,2>&actions,std::array<ProductionDay,2>&d){
 for(int p=0;p<2;p++){
  auto available=private_stock(projected[p],p);auto&fills=after.last_market_fills()[p];
  for(size_t j=0;j<fills.size();j++){
   auto a=actions[p].market.at(j);int i=int(a.item),n=fills[j];if(!n)continue;
   if(a.op==Op::BUY_PRODUCT||a.op==Op::BUY_ANIMAL){available[i]+=n;d[p].bought[i]+=n;}
   if(a.op==Op::SELL){available[i]-=n;d[p].sold[i]+=n;}
   if(a.op==Op::BUY_SEED)d[p].seed_bought[i]+=n;
  }
  auto actual=private_stock(after,p),sf=field_stock(projected[p],p),ef=field_stock(after,p);
  for(int i=0;i<12;i++){
   int lost=available[i]-actual[i];if(lost<0)throw std::runtime_error("unaccounted inventory gain");
   if(lost>0&&before.hour()!=23)throw std::runtime_error("inventory loss outside day end");d[p].eod_loss[i]+=lost;
  }
  // Per-tile deltas: production on another tile must not hide this tile's decay.
  for(size_t k=0;k<projected[p].farms()[p].tiles.size();k++){
   auto a=tile_stock(projected[p].farms()[p].tiles[k]),b=tile_stock(after.farms()[p].tiles[k]);
   for(int i=0;i<9;i++){int delta=b[i]-a[i];if(delta>=0)d[p].generated[i]+=delta;else d[p].environment_loss[i]-=delta;}
  }
  for(size_t i=0;i<projected[p].farms()[p].tiles.size();i++){auto&t=projected[p].farms()[p].tiles[i];if(dp7::animal(t)&&!dp7::animal(after.farms()[p].tiles[i]))d[p].escaped[int(t.animal)-9]++;}
  d[p].end_private=actual;d[p].end_field=ef;d[p].end_seeds=after.privates()[p].seeds;d[p].checks++;
 }
}
}
