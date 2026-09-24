#pragma once
// Join existing economic intent to live execution. No seed, identity or future.
#include "policy.hpp"
namespace dp7::liverepair {
inline void market(Controller&c,const View&o,PlayerAction&a){
 if(!c.p.continuous_market_execution||o.day>=29)return;
 c.live_market_checks++;auto desired=c.competition_sales(o);Counts issued{},queued{},pickup{};
 for(auto x:a.market)if(x.op==Op::SELL&&int(x.item)>=0&&int(x.item)<9)issued[int(x.item)]+=x.quantity;
 for(auto x:c.queue)if(x.op==Op::SELL&&int(x.item)>=0&&int(x.item)<9)queued[int(x.item)]+=x.quantity;
 for(auto x:a.units)if(x.op==Op::PICKUP&&int(x.item)>=0&&int(x.item)<9)pickup[int(x.item)]+=x.quantity;
 auto reserved=intraday::reserved(c);
 auto order=c.sales_sorted(o,desired);
 for(auto wish:order){int i=int(wish.item);if(i<=W||i>=F||queued[i]>0)continue;
  int available=std::max(0,o.priv.shed[i]-reserved.shed[i]-pickup[i]);
  int extra=std::max(0,std::min(desired[i],available)-issued[i]);if(!extra)continue;
  auto found=std::find_if(a.market.begin(),a.market.end(),[&](auto x){return x.op==Op::SELL&&int(x.item)==i;});
  if(found!=a.market.end())found->quantity+=extra;
  else if(a.market.size()<10)a.market.push_back(action(Op::SELL,i,extra));
  else {c.live_market_slots_blocked++;continue;}
  issued[i]+=extra;c.live_market_added+=extra;
 }
}
inline bool wheat_feasible(const View&o,int u,const Plan&p){
 int held=o.priv.inventories[u][W];
 for(auto a:p.a){
  if(a.op==Op::PICKUP&&int(a.item)==W)held+=a.quantity;
  if(a.op==Op::DROP)held=0;
  if(a.op==Op::FEED)held--;
  if(a.op==Op::PLACE&&int(a.item)==W)held-=a.quantity;
  if(held<0)return false;
 }return true; // Never credit future harvests or another worker's cargo.
}
inline Plan inserted(const View&o,int u,const Plan&base,int at,int target,bool pickup){
 auto sem=Controller::plan_semantics(base);Plan out;int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
 for(int k=0;k<=int(sem.size());k++){
  if(k==at){
   if(pickup){int dep=depot[0];for(int d:depot)if(dist(pos,d)+dist(d,target)<dist(pos,dep)+dist(dep,target))dep=d;
    Controller::walk(out,pos,dep);out.a.push_back(action(Op::PICKUP,W));out.target.push_back(dep);}
   Controller::walk(out,pos,target);out.a.push_back(action(Op::FEED));out.target.push_back(target);
  }
  if(k<int(sem.size())){Controller::walk(out,pos,sem[k].second);out.a.push_back(sem[k].first);out.target.push_back(sem[k].second);}
 }return out;
}
inline void feed(Controller&c,const View&o){
 if(!c.p.insert_missing_feed||c.phase!=3||o.day>=29||c.plans.size()!=o.priv.inventories.size())return;
 auto needs=service::missing(c,o);if(needs.empty())return;c.feed_insert_checks++;
 for(auto n:needs){if(n.op!=Op::FEED)continue;
  std::vector<Plan>base;for(int u=0;u<int(c.plans.size());u++)base.push_back(service::remaining(c,u));
  auto original_order=exchange::order(base);auto reserved=intraday::reserved(c);
  Plan best;int owner=-1;std::tuple<int,int,int,int>rank{100000,100000,100000,100000};
  for(int u=0;u<int(base.size());u++)for(int at=0;at<=int(Controller::plan_semantics(base[u]).size());at++)for(bool pickup:{false,true}){
   if(pickup&&o.priv.shed[W]<=reserved.shed[W])continue;
   auto trial=inserted(o,u,base[u],at,n.pos,pickup);int length=int(trial.a.size());
   if(o.hour+length>24||!wheat_feasible(o,u,trial))continue;
   auto all=base;all[u]=trial;if(!exchange::deadlines(o,all))continue;
   // Inserting a feed may delay another worker's same-plot crop operation.
   // Compare the old cross-worker ordering after removing only this new task.
   auto old_only=all;
   for(size_t k=0;k<old_only[u].a.size();k++)if(old_only[u].target[k]==n.pos&&old_only[u].a[k].op==Op::FEED)old_only[u].target[k]=-1;
   if(exchange::order(old_only)!=original_order)continue;
   auto score=std::tuple(length-int(base[u].a.size()),length,u,at);
   if(score<rank){rank=score;owner=u;best=std::move(trial);}
  }
  if(owner<0){c.feed_insert_unresolved++;continue;}
  c.plans[owner]=std::move(best);c.feed_insert_applied++;c.intraday_units.insert(owner);
 }
}
}
