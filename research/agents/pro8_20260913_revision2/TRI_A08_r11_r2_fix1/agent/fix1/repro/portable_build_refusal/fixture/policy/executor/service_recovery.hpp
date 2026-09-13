#pragma once
// Re-enter currently intended maintenance after actual resource availability.
// No simulator, future events, rival identity, or full-game suffix evaluation.
#include "policy.hpp"
namespace dp7::service {
struct Need {int pos,input;Op op;};
inline std::vector<Need> missing(const Controller&c,const View&o,const PlayerAction*issued=nullptr){
 std::set<std::pair<int,Op>>covered;std::set<int>changed;
 auto take=[&](Action a,int pos){if(pos<0)return;if(a.op==Op::DIG||a.op==Op::PLANT||(a.op==Op::PLACE&&int(a.item)>=9))changed.insert(pos);covered.emplace(pos,a.op);};
 for(const auto&pl:c.plans)for(size_t k=pl.index;k<pl.a.size();k++)take(pl.a[k],pl.target[k]);
 if(issued)for(size_t u=0;u<issued->units.size();u++)take(issued->units[u],cell(u?o.own.hands[u-1]:o.own.farmer));
 std::vector<Need>result;
 for(const auto&j:c.jobs(o)){
  if(changed.count(j.pos))continue;
  if(std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::DIG||a.op==Op::PLANT||a.op==Op::PLACE;}))continue;
  auto&t=o.own.tiles[j.pos];
  for(auto a:j.actions){
   if(covered.count({j.pos,a.op}))continue;
   bool feed=a.op==Op::FEED&&animal(t)&&!t.fed_today;
   bool fert=a.op==Op::FERTILIZE&&plant(t)&&ongoing(int(t.crop))&&c.fertilize_due(t);
   if(feed||fert){result.push_back({j.pos,feed?W:F,a.op});covered.emplace(j.pos,a.op);}
  }
 }
 // Existing maintenance semantics, not a new macro allocation preference.
 std::stable_sort(result.begin(),result.end(),[](auto a,auto b){return a.input<b.input;});return result;
}
inline Counts free_cargo(const Controller&c,const View&o,int u){
 Counts q=o.priv.inventories.at(u);
 const auto&pl=c.plans.at(u);
 for(size_t k=pl.index;k<pl.a.size();k++){
  auto a=pl.a[k];int i=int(a.item);
  if(a.op==Op::DROP)q={};
  if(a.op==Op::FEED)q[W]=std::max(0,q[W]-1);
  if(a.op==Op::FERTILIZE)q[F]=std::max(0,q[F]-1);
  if(a.op==Op::PLACE&&i>=0&&i<12)q[i]=std::max(0,q[i]-a.quantity);
  // Neither future pickup nor future harvest is credited as already owned.
 }return q;
}
inline Plan remaining(const Controller&c,int u){const auto&old=c.plans.at(u);return {Acts(old.a.begin()+old.index,old.a.end()),std::vector<int>(old.target.begin()+old.index,old.target.end()),0};}
inline Plan append(const Controller&c,const View&o,int u,Need n,bool pickup){
 Plan pl=remaining(c,u);int pos=Controller::plan_end(o,u,pl);
 if(pickup){int dep=depot[0];for(int d:depot)if(dist(pos,d)+dist(d,n.pos)<dist(pos,dep)+dist(dep,n.pos))dep=d;
  Controller::walk(pl,pos,dep);pl.a.push_back(action(Op::PICKUP,n.input));pl.target.push_back(dep);}
 Controller::walk(pl,pos,n.pos);pl.a.push_back(action(n.op));pl.target.push_back(n.pos);return pl;
}
inline void recover(Controller&c,const View&o){
 liverepair::feed(c,o);
 if(!c.p.recover_service_inputs||c.phase!=3||o.day>=29||c.plans.size()!=o.priv.inventories.size())return;
 c.service_checks++;
 for(auto n:missing(c,o)){
  auto r=intraday::reserved(c);int best=-1,steps=1000;Plan chosen;
  for(int u=0;u<int(c.plans.size());u++){
   bool pickup=free_cargo(c,o,u)[n.input]<1;
   if(pickup&&o.priv.shed[n.input]<=r.shed[n.input])continue;
   auto pl=append(c,o,u,n,pickup);int size=int(pl.a.size());
   if(o.hour+size<=24&&size<steps){best=u;steps=size;chosen=std::move(pl);}
  }
  if(best>=0){c.plans[best]=std::move(chosen);c.intraday_units.insert(best);
   if(n.input==W)c.service_recovered_feed++;else c.service_recovered_fertilize++;}
 }
}
inline void protect(const Controller&c,const View&o,PlayerAction&a){
 if(!c.p.recover_service_inputs)return;
 // Protect all still-reserved pickups, including an action issued this step.
 auto r=intraday::reserved(c);for(auto x:a.units)if(x.op==Op::PICKUP&&int(x.item)>=0&&int(x.item)<12)r.shed[int(x.item)]+=x.quantity;
 Counts allowed{};for(int i=0;i<12;i++)allowed[i]=std::max(0,o.priv.shed[i]-r.shed[i]);
 // Include only explicit quantitative deposits; DROP has order/capacity loss.
 int room=std::max(0,100-sum(o.priv.shed));
 for(size_t u=0;u<a.units.size();u++){auto x=a.units[u];int i=int(x.item),pos=cell(u?o.own.hands[u-1]:o.own.farmer);
  if(at_depot(pos)&&x.op==Op::PLACE&&i>=0&&i<9){int q=std::min({x.quantity,o.priv.inventories[u][i],room});allowed[i]+=q;room-=q;}}
 // This is a reservation guard, not a replacement for the market compiler.
 // Unreserved goods may arrive via a current DROP before their SELL executes.
 // Capping those orders at pre-unit inventory destroys terminal liquidation.
 for(auto&x:a.market)if(x.op==Op::SELL&&int(x.item)>=0&&int(x.item)<9&&r.shed[int(x.item)]>0){int i=int(x.item);x.quantity=std::min(x.quantity,allowed[i]);allowed[i]-=x.quantity;}
 a.market.erase(std::remove_if(a.market.begin(),a.market.end(),[](auto x){return x.quantity<=0;}),a.market.end());
}
// A bounded projection of current own logistics followed by already issued
// orders. It deliberately excludes harvests/collection and opponent orders.
inline std::optional<Controller::PreparedState> prefix(const Controller&c,const View&o,const PlayerAction&a){
 auto pr=o.priv;
 for(size_t u=0;u<a.units.size();u++){
  auto x=a.units[u];int i=int(x.item),pos=cell(u?o.own.hands[u-1]:o.own.farmer);
  if(x.op==Op::DROP)return {}; // Cannot assume which overflowed goods survive.
  if(at_depot(pos)&&x.op==Op::PICKUP&&i>=0&&i<12){int q=std::min(x.quantity,pr.shed[i]);pr.shed[i]-=q;pr.inventories[u][i]+=q;}
  if(at_depot(pos)&&x.op==Op::PLACE&&i>=0&&i<9){int q=std::min({x.quantity,pr.inventories[u][i],std::max(0,100-sum(pr.shed))});pr.shed[i]+=q;pr.inventories[u][i]-=q;}
 }
 Controller projection(c.p);projection.phase=1;projection.queue=a.market;
 return projection.project_preparation(View{o.step,o.day,o.hour,o.own,o.opponent,pr,o.market,o.shops});
}
inline void procure(Controller&c,const View&o,PlayerAction&a){
 if(!c.p.recover_service_inputs||!c.p.procure_service_inputs||c.phase!=3||o.day>=29||a.market.size()>=10)return;
 auto r=intraday::reserved(c);auto future=prefix(c,o,a);if(!future)return;
 for(auto n:missing(c,o,&a)){
  bool useful=false;
  for(int u=0;u<int(c.plans.size());u++){
   // A current moving/working unit is not assumed to have already moved.
   // One extra turn is reserved before assigning any newly filled resource.
   if(a.units[u].op==Op::PASS&&o.hour+1+int(append(c,o,u,n,true).a.size())<=24)useful=true;
  }
  if(!useful)continue;
  if(future->priv.shed[n.input]>r.shed[n.input])continue;
  bool already_owned=false;for(int u=0;u<int(c.plans.size());u++)if(free_cargo(c,o,u)[n.input]>0)already_owned=true;
  if(already_owned)continue;
  int cost=price(n.input,future->market.inventory[n.input]-1);
  if(sum(future->priv.shed)<100&&future->farm.money>=cost){a.market.push_back(action(Op::BUY_PRODUCT,n.input));c.service_buys++;return;}
  if(!c.p.finance_service_inputs||a.market.size()>8)continue;
  // Only already owned unreserved goods; never collateralise future output.
  PlayerAction funded=a;bool financed=false,deposited=false;
  for(int i=0;i<9&&!financed;i++)if(i!=n.input){
   int free=std::max(0,future->priv.shed[i]-r.shed[i]);
   int q=0;double revenue_sum=0;
   while(q<free&&future->farm.money+revenue_sum<cost)revenue_sum+=price(i,future->market.inventory[i]+q++);
   if(q>0&&future->farm.money+revenue_sum>=cost){funded.market.push_back(action(Op::SELL,i,q));financed=true;}
  }
  if(!financed)for(int u=0;u<int(c.plans.size())&&!financed;u++){
   int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
   if(!at_depot(pos)||a.units[u].op!=Op::PASS||c.plans[u].index<c.plans[u].a.size())continue;
   // Existing quantitative deposits must not compete for the same room.
   bool other_deposit=std::any_of(a.units.begin(),a.units.end(),[](auto x){return x.op==Op::DROP||x.op==Op::PLACE;});if(other_deposit)continue;
   auto cargo=free_cargo(c,o,u);int room=std::max(0,100-sum(o.priv.shed));
   for(int i=0;i<9&&!financed;i++)if(i!=n.input){
    int q=0;double rev=0;while(q<std::min(room,cargo[i])&&future->farm.money+rev<cost)rev+=price(i,future->market.inventory[i]+q++);
    if(q>0&&future->farm.money+rev>=cost){funded.units[u]=action(Op::PLACE,i,q);funded.market.push_back(action(Op::SELL,i,q));financed=true;deposited=true;}
   }
  }
  if(!financed)continue;
  auto sold=prefix(c,o,funded);if(!sold||sum(sold->priv.shed)>=100||sold->farm.money<price(n.input,sold->market.inventory[n.input]-1))continue;
  funded.market.push_back(action(Op::BUY_PRODUCT,n.input));a=std::move(funded);c.service_buys++;c.service_financed++;c.service_deposits+=deposited;return;
 }
}
}
