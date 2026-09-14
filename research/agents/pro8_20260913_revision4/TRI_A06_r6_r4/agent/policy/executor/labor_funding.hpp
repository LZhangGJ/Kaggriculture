#pragma once
// TRI_A06_r6_r3: funding and observation-confirmed recovery of the selected crew.
// Current own/public state only. No rival orders, future tape or borrowed inputs.
#include "observed_day_scenario.hpp"
namespace dp7::labor {
inline bool service_op(Op op){return op==Op::WATER||op==Op::FEED||op==Op::CARE||op==Op::FERTILIZE;}
inline std::vector<Job> missing(const Controller&c,const View&o){
 std::set<Controller::TaskKey>covered;std::set<int>changing;
 for(const auto&pl:c.plans)for(size_t k=pl.index;k<pl.a.size();k++){
  auto a=pl.a[k];int pos=pl.target[k];if(pos<0)continue;
  covered.insert(Controller::task_key(pos,a));
  if(a.op==Op::DIG||a.op==Op::PLANT||(a.op==Op::PLACE&&int(a.item)>=9))changing.insert(pos);
 }
 if(c.pending_admission.active)changing.insert(c.pending_admission.pos);
 std::vector<Job>out;
 for(auto j:c.jobs(o)){
  if(changing.count(j.pos)||j.actions.empty()||(!plant(o.own.tiles[j.pos])&&!animal(o.own.tiles[j.pos])))continue;
  // Never split a partially assigned input/lifecycle chain or steal its owner.
  bool safe=true;
  for(auto a:j.actions)if(!service_op(a.op)||covered.count(Controller::task_key(j.pos,a)))safe=false;
  if(safe)out.push_back(std::move(j));
 }
 return out;
}
inline PrivateState unreserved(const Controller&c,const View&o){
 auto pr=o.priv;auto r=intraday::reserved(c);
 if(c.pending_admission.active){const auto&q=c.pending_admission;
  if(q.day==o.day&&q.unit>=0&&q.unit<int(pr.inventories.size())){
   auto j=intraday::project_job(c,o,q.pos,q.kind);
   for(int i=0;i<12;i++)r.shed[i]+=std::max(0,j.needs[i]-pr.inventories[q.unit][i]);
   for(int i=0;i<5;i++)r.seeds[i]+=j.seeds[i];
  }
 }
 for(int i=0;i<12;i++)pr.shed[i]=std::max(0,pr.shed[i]-r.shed[i]);
 for(int i=0;i<5;i++)pr.seeds[i]=std::max(0,pr.seeds[i]-r.seeds[i]);
 return pr;
}
// A conservative service heuristic, NOT recovered cash or a terminal guarantee.
// No scheduling priority bonus is treated as money. Existing survival receives
// replacement-cost credit only while the incumbent service policy still opts in.
inline double value(const Controller&c,const View&o,const Job&j){
 const auto&t=o.own.tiles[j.pos];double v=0;
 for(auto a:j.actions){
  if(a.op==Op::WATER&&plant(t)){
   int k=int(t.crop);v+=o.market.prices[k]*(t.fertilized_until_day>=o.day?2:1);
   if(t.consecutive_unwatered>=1)v+=seed_price[k];
  }else if(a.op==Op::FEED&&animal(t)){
   int k=int(t.animal)-9;v+=o.market.prices[product[k]];
   if(t.consecutive_unfed>=1)v+=animal_price[k];
  }else if(a.op==Op::CARE&&animal(t))v+=o.market.prices[product[int(t.animal)-9]];
  else if(a.op==Op::FERTILIZE&&plant(t))v+=2*o.market.prices[int(t.crop)];
 }
 for(int i=0;i<9;i++)v-=j.needs[i]*o.market.prices[i];
 return std::max(0.,v);
}
struct Allocation{std::vector<Route>routes;double utility=0;int jobs=0;};
inline Allocation allocate(const Controller&c,const View&o,int first,int count){
 Allocation r;if(count<=0||o.hour>=24)return r;
 auto priv=unreserved(c,o);View v{o.step,o.day,o.hour,o.own,o.opponent,priv,o.market,o.shops};
 auto probe=c;auto js=probe.reserve(v,missing(c,o));
 js.erase(std::remove_if(js.begin(),js.end(),[&](const auto&j){return value(c,o,j)<=0;}),js.end());
 if(js.empty())return r;
 std::vector<int>starts;for(int u=first;u<first+count;u++)starts.push_back(cell(u?o.own.hands.at(u-1):o.own.farmer));
 auto packed=c.pack(js,starts,24-o.hour,false,true,true);
 r.routes=std::move(packed.first);
 for(auto&route:r.routes){
  double benefit=0;int work=0;for(const auto&j:route.jobs){benefit+=value(c,o,j);work+=j.actions.size();}
  double net=benefit-c.p.action_shadow*std::max(0,route.total(false)-work);
  if(net<=0){route=Route(route.unit,route.start);continue;}
  r.utility+=net;r.jobs+=route.jobs.size();
 }
 return r;
}
inline Plan materialize(const Route&r){
 Plan p;int pos=r.start;std::vector<int>items;
 for(int i=0;i<12;i++)if(r.needs[i]>0)items.push_back(i);
 std::sort(items.begin(),items.end(),[](int a,int b){return name(a)<name(b);});
 for(int i:items){p.a.push_back(action(Op::PICKUP,i,r.needs[i]));p.target.push_back(pos);}
 for(const auto&j:r.jobs){Controller::walk(p,pos,j.pos);for(auto a:j.actions){p.a.push_back(a);p.target.push_back(j.pos);}}
 return p;
}
inline void observe(Controller&c,const View&o){
 if(!c.p.funded_labor||c.phase!=3)return;
 int old=c.plans.size(),now=1+o.own.hands.size();
 if(now<=old)return;
 // Never install hypothetical workers/routes. The real roster is authoritative.
 c.plans.resize(now);c.labor_arrivals+=now-old;
 auto chosen=allocate(c,o,old,now-old);
 for(const auto&r:chosen.routes){int u=old+r.unit;c.plans[u]=materialize(r);c.intraday_units.insert(u);}
 c.labor_jobs+=chosen.jobs;
}
inline void procure(Controller&c,const View&o,PlayerAction&out){
 if(!c.p.funded_labor||c.phase!=3||o.day>=29||o.hour>=23||
    int(o.own.hands.size())>=std::min(c.labor_target,c.p.max_hands)||out.market.size()>=10||
    c.plans.size()!=o.priv.inventories.size())return;
 // One outstanding hiring decision per page; observe any existing HIRE first.
 if(std::any_of(out.market.begin(),out.market.end(),[](auto a){return a.op==Op::HIRE;}))return;
 c.labor_checks++;
 ObservedDayScenario known(o);auto post=known.project_units(out.units,-1);
 // Split only the current ten-order page. Keep every existing order, with
// current sales and material maintenance inputs ahead of wages and investment.
 Acts support,optional;
 for(auto a:out.market){
  if(a.op==Op::SELL||a.op==Op::BUY_PRODUCT||a.op==Op::HIRE)support.push_back(a);
  else optional.push_back(a);
 }
 auto funded=post.project_own_market(0,support);
 int existing=funded.farms()[0].hands.size();
 int count=std::min({c.labor_target-existing,c.p.max_hands-existing,10-int(out.market.size())});
 if(count<=0)return;
 Acts best;double best_net=0;int best_count=0;
 for(int n=1;n<=count;n++){
  Acts prefix=support;for(int i=0;i<n;i++)prefix.push_back(action(Op::HIRE));
  auto paid=post.project_own_market(0,prefix);
  if(int(paid.farms()[0].hands.size())!=existing+n)break;
  // Hiring consumes this market phase: first service work is NEXT observation.
  View next{o.step+1,o.day,o.hour+1,paid.farms()[0],o.opponent,paid.privates()[0],paid.market(),o.shops};
  auto allocation=allocate(c,next,1+existing,n);
  double wage=funded.farms()[0].money-paid.farms()[0].money;
  double net=allocation.utility-wage;
  if(allocation.jobs>0&&net>best_net+1e-6){best_net=net;best_count=n;best=std::move(prefix);}
 }
 if(best_count){
  best.insert(best.end(),optional.begin(),optional.end());out.market=std::move(best);
  c.labor_offers+=best_count;c.labor_last_utility=best_net;
 }
}
} // namespace dp7::labor
