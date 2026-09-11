#pragma once
// Execution-only repair. Observed own state + public board; no live environment,
// replay, seed, opponent identity or opponent private inventory is accepted.
// The conditional day model disables unknown randomness and is NOT a promise
// about future opposing trades. All actual effects are read again next tick.
#include "policy.hpp"
#include "observed_day_scenario.hpp"
namespace dp7::t3repair {
inline int position(const View&o,int u){return cell(u?o.own.hands.at(u-1):o.own.farmer);}
inline std::vector<Plan> remaining(const Controller&c,const View&o){
 std::vector<Plan> out;
 for(int u=0;u<int(c.plans.size());u++){
  auto&x=c.plans[u];size_t k=std::min(x.index,x.a.size());
  out.push_back({Acts(x.a.begin()+k,x.a.end()),std::vector<int>(x.target.begin()+k,x.target.end()),0});
 }
 return out;
}
inline bool covered_water(const std::vector<Plan>&plans,int p,int hour){
 for(const auto&pl:plans)for(size_t k=pl.index;k<pl.a.size();k++)
  if(hour+int(k-pl.index)<24&&pl.target[k]==p&&pl.a[k].op==Op::WATER)return true;
 return false;
}
inline bool safe_boundary(const std::vector<std::pair<Action,int>>&s,int at){
 // Never insert into an establishment/material chain on the same plot.
 return at==0||at==int(s.size())||s[at-1].second!=s[at].second;
}
inline bool water_order(const View&o,const std::vector<Plan>&all,int target,int owner,int at){
 auto when=std::pair(at,owner);const auto&t=o.own.tiles[target];
 for(int u=0;u<int(all.size());u++)for(int k=0;k<int(all[u].a.size());k++){
  if(all[u].target[k]!=target)continue;auto a=all[u].a[k];auto time=std::pair(k,u);
  if(a.op==Op::FERTILIZE&&time>=when)return false;
  if(a.op==Op::PLANT||a.op==Op::DIG||(a.op==Op::PLACE&&int(a.item)>=9))return false;
  if(a.op==Op::HARVEST&&plant(t)&&!ongoing(int(t.crop))&&time<when)return false;
 }
 return true;
}
inline void obligations(Controller&c,const View&o){
 if constexpr(T3_OBLIGATION_REPAIR==0)return;
 if(c.t3.suspended||c.phase!=3||o.day>=29||c.plans.size()!=o.priv.inventories.size()||c.t3.obligation_last_step==o.step)return;
 c.t3.obligation_last_step=o.step;c.t3.obligation_checks++;c.t3.missing_water.fill(0);
 auto base=remaining(c,o);auto tasks=c.jobs(o);std::vector<int> missing;
 for(const auto&j:tasks){
  const auto&t=o.own.tiles[j.pos];if(!plant(t)||t.watered_today||covered_water(base,j.pos,o.hour))continue;
  bool water=false,establishment=false;
  for(auto a:j.actions){water|=a.op==Op::WATER;establishment|=a.op==Op::PLANT||a.op==Op::DIG||a.op==Op::PLACE;}
  if(water&&!establishment&&std::find(missing.begin(),missing.end(),j.pos)==missing.end())missing.push_back(j.pos);
 }
 std::stable_sort(missing.begin(),missing.end(),[&](int a,int b){
  const auto&ta=o.own.tiles[a];const auto&tb=o.own.tiles[b];
  return std::tuple(-int(ta.consecutive_unwatered),-o.market.prices[int(ta.crop)]*(1+ta.yield_units),a)
       < std::tuple(-int(tb.consecutive_unwatered),-o.market.prices[int(tb.crop)]*(1+tb.yield_units),b);
 });
 for(int p:missing){
  base=remaining(c,o);Job j;j.pos=p;j.actions={action(Op::WATER)};
  auto order=exchange::order(base);int owner=-1;Plan best;std::tuple<int,int,int,int> rank{100000,100000,100000,100000};
  for(int u=0;u<int(base.size());u++){
   auto sem=Controller::plan_semantics(base[u]);
   for(int at=0;at<=int(sem.size());at++){
    if(!safe_boundary(sem,at))continue;
    auto trial=Controller::insert_service(o,u,base[u],j,at);
    if(o.hour+int(trial.a.size())>24){continue;}
    int wi=-1;for(int k=0;k<int(trial.a.size());k++)if(trial.target[k]==p&&trial.a[k].op==Op::WATER){wi=k;break;}
    auto all=base;all[u]=trial;
    if(wi<0||!water_order(o,all,p,u,wi)){continue;}
    if(!exchange::deadlines(o,all)){continue;}
    // Keep every old cross-worker causal order; only the new water is removed
    // from the comparison. Water insertion does not allocate any material.
    all[u].target[wi]=-1;if(exchange::order(all)!=order){continue;}
    auto score=std::tuple(int(trial.a.size())-int(base[u].a.size()),wi,int(trial.a.size()),u);
    if(score<rank){rank=score;owner=u;best=std::move(trial);}
   }
  }
  if(owner>=0){c.plans[owner]=std::move(best);c.t3.water_insertions++;}
  else {c.t3.water_unresolved++;c.t3.missing_water[p]=1;
  }
 }
}

struct Projection {int overflow=0;double loss_quote=0,cash=0,bankable=0;int ticks=0;};
inline Projection project(const Controller&c,const View&o,const std::vector<Plan>&plans){
 ObservedDayScenario world(o);auto ctl=c;ctl.plans=plans;ctl.workflow_runtime=false;Projection answer;
 ctl.t3.suspended=true;ctl.last_step=o.step-1;ctl.resume_compiled_tick=true;
 // The same native worker loop, including handoff and recoordination. Static
 // tape-only forecasts miss the work these state-feedback passes add later.
 // No NEW capital projects inside this logistics comparison.
 ctl.p.intraday_admission=false;ctl.admission_inspection=nullptr;
 while(!world.finished()){
  auto v=world.view();auto a=ctl.act(v);
  auto post=world.project_units(a.units,-1);const auto&pr=post.privates()[0];
  auto reserved=intraday::reserved(ctl).shed;
  if(v.hour<23){int animals=0,heldfeed=0;for(auto&t:post.farms()[0].tiles)animals+=animal(t);
   for(auto&iv:pr.inventories)heldfeed+=iv[W];reserved[W]=std::max(reserved[W],std::max(0,animals-heldfeed));}
  Counts sell{};for(int i=0;i<9;i++)sell[i]=std::max(0,pr.shed[i]-reserved[i]);
  Acts buys;for(auto x:a.market)if(x.op!=Op::SELL)buys.push_back(x);
  if(ctl.t3.project_sales)ctl.t3.project_sales(v,pr,buys,sell);
  a.market=ctl.sales_sorted(v,sell);if(a.market.size()+buys.size()>10)a.market.resize(10-buys.size());a.market.insert(a.market.end(),buys.begin(),buys.end());
  world.advance(a);answer.overflow+=world.overflow();answer.ticks++;
 }
 answer.cash=world.own().money;answer.bankable=answer.cash+world.liquidation_quote();
 return answer;
}
inline Plan insert_drop(const View&o,int u,const Plan&base,int at,int depot_pos){
 auto sem=Controller::plan_semantics(base);Plan out;int pos=position(o,u);
 for(int k=0;k<=int(sem.size());k++){
  if(k==at){Controller::walk(out,pos,depot_pos);out.a.push_back(action(Op::DROP));out.target.push_back(depot_pos);}
  if(k<int(sem.size())){Controller::walk(out,pos,sem[k].second);out.a.push_back(sem[k].first);out.target.push_back(sem[k].second);}
 }
 return out;
}
inline void capacity(Controller&c,const View&o){
 if constexpr(T3_CAPACITY_REPAIR==0)return;
 if(c.t3.suspended||c.phase!=3||o.day>=29||o.hour>21||c.plans.size()!=o.priv.inventories.size()||c.t3.capacity_last_step==o.step)return;
 c.t3.capacity_last_step=o.step;
 // Cheap conservative screen. Include actual shed, every worker, and yields
 // obtainable by today's scheduled finite-crop watering (old model omitted it).
 int upper=sum(o.priv.shed);for(auto&iv:o.priv.inventories)upper+=sum(iv);
 std::set<int>harvested,collected;
 for(auto&pl:c.plans)for(size_t k=pl.index;k<pl.a.size();k++)if(pl.target[k]>=0){
  int p=pl.target[k];auto&t=o.own.tiles[p];
  if(pl.a[k].op==Op::HARVEST&&harvested.insert(p).second){upper+=t.yield_units;
   if(plant(t)&&!ongoing(int(t.crop))&&!t.watered_today)upper+=2;}
  if(pl.a[k].op==Op::COLLECT_FERTILIZER&&collected.insert(p).second)upper+=t.fertilizer_available;
 }

 if(upper<=100)return;
 // Recheck every four ticks and during the last feasible transport window.
 if(o.hour<16&&o.hour%4!=0)return;
 c.t3.capacity_checks++;auto base=remaining(c,o);auto original=project(c,o,base);c.t3.capacity_rollouts++;

 if(original.overflow==0)return;
 struct Candidate {int u,at,dep,extra,q;double score;};std::vector<Candidate> choices;
 auto causal=exchange::order(base);
 for(int u=0;u<int(base.size());u++){
  auto sem=Controller::plan_semantics(base[u]);Counts bag=o.priv.inventories[u];
  int prev=position(o,u),time=0;std::set<int>seen_h,seen_f;
  for(int at=0;at<=int(sem.size());at++){
   int q=sum(bag);bool inputs=false;for(int i=9;i<12;i++)inputs|=bag[i]>0;
   for(int k=at;k<int(sem.size());k++){
    inputs|=sem[k].first.op==Op::FEED&&bag[W]>0;
    inputs|=sem[k].first.op==Op::FERTILIZE&&bag[F]>0;
   }
   if(q>0&&q<=100&&!inputs&&safe_boundary(sem,at))for(int d:depot){
    int extra=dist(prev,d)+1+(at<int(sem.size())?dist(d,sem[at].second)-dist(prev,sem[at].second):0);
    if(o.hour+time+dist(prev,d)>21||o.hour+int(base[u].a.size())+extra>24)continue;
    double quote=0;for(int i=0;i<9;i++)quote+=bag[i]*o.market.prices[i];
    double score=quote*std::min(q,original.overflow)/q/std::max(1,extra);
    choices.push_back({u,at,d,extra,q,score});
   }
   if(at==int(sem.size()))break;
   auto [a,p]=sem[at];time+=dist(prev,p)+1;prev=p;int i=int(a.item);const auto&t=o.own.tiles[p];
   if(a.op==Op::PICKUP&&i>=0&&i<12)bag[i]+=a.quantity;
   if(a.op==Op::DROP)bag={};
   if(a.op==Op::FEED)bag[W]=std::max(0,bag[W]-1);
   if(a.op==Op::FERTILIZE)bag[F]=std::max(0,bag[F]-1);
   if(a.op==Op::PLACE&&i>=0&&i<12)bag[i]=std::max(0,bag[i]-a.quantity);
   if(a.op==Op::HARVEST&&seen_h.insert(p).second){if(plant(t))bag[int(t.crop)]+=t.yield_units;else if(animal(t))bag[product[int(t.animal)-9]]+=t.yield_units;}
   if(a.op==Op::COLLECT_FERTILIZER&&seen_f.insert(p).second)bag[F]+=t.fertilizer_available;
  }
 }
 std::stable_sort(choices.begin(),choices.end(),[](auto&a,auto&b){return a.score>b.score;});
 int selected=-1;Plan best;Projection result=original;
 std::set<std::tuple<int,int,int>>tested;
 for(auto x:choices){
  if(tested.size()>=6)break;
  // Closely adjacent center tiles with the same route length are duplicates.
  if(!tested.emplace(x.u,x.at,x.extra).second)continue;
  auto trial=base;trial[x.u]=insert_drop(o,x.u,base[x.u],x.at,x.dep);
  if(!exchange::deadlines(o,trial)||exchange::order(trial)!=causal)continue;
  auto next=project(c,o,trial);c.t3.capacity_rollouts++;
  // This only orders equal-semantic logistics. Do not drop work or acquire
  // new production to game the conditional forecast. Require both cash and
  // overflow to improve/non-regress in this deliberately bounded model.
  if(next.overflow<result.overflow&&next.cash>=original.cash-1e-6){selected=x.u;best=std::move(trial[x.u]);result=next;}
 }
 if(selected>=0){c.plans[selected]=std::move(best);c.t3.delivery_insertions++;c.t3.projected_overflow_saved+=original.overflow-result.overflow;}
}

inline std::multiset<Controller::TaskKey> field_keys(const Controller&c){
 std::multiset<Controller::TaskKey>keys;
 for(auto&pl:c.plans)for(size_t k=pl.index;k<pl.a.size();k++){
  auto a=pl.a[k];if(pl.target[k]>=0&&!Controller::movement(a.op)&&a.op!=Op::PICKUP&&a.op!=Op::DROP)keys.insert(Controller::task_key(pl.target[k],a));
 }return keys;
}
inline std::optional<PlayerAction> receipt(Controller&c,const View&o){
 if constexpr(T3_RECEIPT_REPAIR==0)return {};
 if(c.t3.suspended||c.phase!=2||o.day>=29||o.hour>5||c.t3.receipt_attempt_day==o.day)return {};
 c.t3.receipt_checks++;
 // After actual procurement: native staffing may have been based on a
 // different seed/worker state. Admit ONE additional worker only if the
 // conditional schedule retains ALL old field actions and rescues a dying
 // crop. A whole extra preparation tick and actual escalating wage are paid.
 auto oldplan=c;oldplan.compile_base(o);auto old=field_keys(oldplan);
 // Test complete-day hauling admission jointly with actual staffing. A
 // transport route must fit with the SAME retained field actions, after
 // paying one real purchase tick and the marginal Fibonacci wages.
 int gross=sum(c.forecast(c.jobs(o)))+sum(o.priv.shed);for(auto&iv:o.priv.inventories)gross+=sum(iv);
 if(T3_CAPACITY_REPAIR&&gross>100){
  auto baseline=project(c,o,oldplan.plans);c.t3.capacity_rollouts++;
  if(baseline.overflow>0){
   int best_h=-1;double best_value=baseline.bankable;
   for(int extra=0;extra<=2;extra++){
    if(int(o.own.hands.size())+extra>c.p.max_hands||int(o.own.hires_today)+extra>int(fib.size()))continue;
    double cost=0;for(int i=0;i<extra;i++)cost+=fib[o.own.hires_today+i];if(cost>o.own.money)continue;
    ObservedDayScenario prep(o);PlayerAction buy;buy.units.resize(1+o.own.hands.size());for(int i=0;i<extra;i++)buy.market.push_back(action(Op::HIRE));
    if(extra)prep.advance(buy);if(prep.finished())continue;
    auto v=prep.view();auto trial=c;trial.compile_base(v);auto keys=field_keys(trial);
    if(!std::includes(keys.begin(),keys.end(),old.begin(),old.end()))continue;
    auto score=project(trial,v,trial.plans);c.t3.capacity_rollouts++;
    if(score.overflow<baseline.overflow&&score.bankable>best_value+1e-6){best_value=score.bankable;best_h=extra;}
   }
   if(best_h>=0){
    c.t3.transport_admissions++;c.t3.receipt_attempt_day=o.day;
    if(best_h){PlayerAction out;out.units.resize(1+o.own.hands.size());for(int i=0;i<best_h;i++)out.market.push_back(action(Op::HIRE));c.t3.receipt_hires+=best_h;return out;}
    return {};
   }
  }
 }
 std::vector<int> thirsty;
 for(auto&j:c.jobs(o))if(plant(o.own.tiles[j.pos])&&!o.own.tiles[j.pos].watered_today&&o.own.tiles[j.pos].consecutive_unwatered>=1){
  bool want=false;for(auto a:j.actions)want|=a.op==Op::WATER;
  if(want&&!old.count(Controller::task_key(j.pos,action(Op::WATER))))thirsty.push_back(j.pos);
 }
 int hi=o.own.hires_today;
 if(T3_OBLIGATION_REPAIR&&!thirsty.empty()&&hi<int(fib.size())&&int(o.own.hands.size())<c.p.max_hands&&fib[hi]<=o.own.money){
  ObservedDayScenario scenario(o);PlayerAction hire;hire.units.resize(1+o.own.hands.size());hire.market={action(Op::HIRE)};scenario.advance(hire);
  if(!scenario.finished()&&scenario.own().hands.size()>o.own.hands.size()){
   auto v=scenario.view();auto replacement=c;replacement.compile_base(v);auto next=field_keys(replacement);
   bool keeps=std::includes(next.begin(),next.end(),old.begin(),old.end());double recovered=0;int saved=0;
   for(int pos:thirsty)if(next.count(Controller::task_key(pos,action(Op::WATER)))){
    auto&t=o.own.tiles[pos];recovered+=std::max(1,int(t.yield_units))*o.market.prices[int(t.crop)];saved++;
   }
   if(keeps&&saved&&recovered>fib[hi]){c.t3.receipt_attempt_day=o.day;c.t3.receipt_hires++;return hire;}
  }
 }
 c.t3.receipt_attempt_day=o.day;
 Counts need{},repair{};auto jobs=c.jobs(o);for(auto&j:jobs)add(need,j.seeds);
 int money=0;for(int k=0;k<5;k++){repair[k]=std::max(0,need[k]-o.priv.seeds[k]);money+=repair[k]*seed_price[k];}
 int wanted=c.t3.receipt_day==o.day?std::max(0,c.t3.receipt_expected_hands-int(o.own.hands.size())):0;
 int hires=0;while(hires<wanted&&int(o.own.hires_today)+hires<int(fib.size())&&int(o.own.hands.size())+hires<int(c.p.max_hands)){
  int cost=fib[int(o.own.hires_today)+hires];if(money+cost>o.own.money)break;money+=cost;hires++;
 }
 if(money<=0||money>o.own.money)return {};
 Acts orders;for(int k=0;k<5;k++)if(repair[k]>0)orders.push_back(action(Op::BUY_SEED,k,repair[k]));for(int i=0;i<hires;i++)orders.push_back(action(Op::HIRE));
 if(orders.empty()||orders.size()>10)return {};
 auto baseline=c;baseline.compile_base(o);auto oldkeys=field_keys(baseline);
 // Compare AFTER a real shopping turn using fixed-price resources only.
 // No speculative sale revenue/extra buffer and no unit work in the same turn.
 auto helper=c;helper.queue=orders;helper.phase=1;auto state=helper.project_preparation(o);
 View next{o.step+1,o.day,o.hour+1,state.farm,o.opponent,state.priv,state.market,o.shops};
 auto stocked=c;stocked.compile_base(next);auto newkeys=field_keys(stocked);
 if(!std::includes(newkeys.begin(),newkeys.end(),oldkeys.begin(),oldkeys.end()))return {};
 if(newkeys.size()<=oldkeys.size())return {};
 PlayerAction out;out.units.resize(1+o.own.hands.size());out.market=orders;
 c.seed_reconcile_checked=true;c.t3.receipt_seeds+=sum(repair);c.t3.receipt_hires+=hires;
 return out;
}
}
