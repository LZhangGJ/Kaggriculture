#pragma once
// r4: shared rolling admission for terminal service-to-sale chains.
// Only View (current public facts and own private resources) enters this code.
// The two quote scenarios are explicitly conditional, not live future states.
#include "observed_day_scenario.hpp"
#include <map>
namespace dp7::terminalroll {
inline bool produces(Op op){return op==Op::HARVEST||op==Op::COLLECT_FERTILIZER;}
inline int terminal_ticks(const View&o){return std::max(0,719-o.step);}

// Keep all original semantic atoms in their original order. Inserting before
// an existing DROP reuses it; appending after one creates a second deposit.
// Current cargo is also a delivery obligation, including unsold input surplus.
inline bool close_delivery(const View&o,int u,Plan&pl,int limit){
 bool cargo=sum(o.priv.inventories.at(u))>0;
 for(const auto a:pl.a){
  if(produces(a.op)||a.op==Op::PICKUP)cargo=true;
  else if(a.op==Op::DROP)cargo=false;
 }
 if(cargo){int pos=Controller::plan_end(o,u,pl),dest=depot[0];
  for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;
  Controller::walk(pl,pos,dest);pl.a.push_back(action(Op::DROP));pl.target.push_back(dest);
 }
 return pl.a.size()==pl.target.size()&&int(pl.a.size())<=limit;
}
// DROP names a shared warehouse, not a private fixed waypoint. Rebase it
// after insertion exactly as repair_plan() does on the next real observation.
// Material pickups and all semantic order remain unchanged.
inline Plan insert_chain(const View&o,int u,const Plan&base,const Job&job,int at){
 auto sem=Controller::plan_semantics(base);Plan out;
 int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
 for(int k=0;k<=int(sem.size());k++){
  if(k==at){Controller::walk(out,pos,job.pos);for(auto x:job.actions){out.a.push_back(x);out.target.push_back(job.pos);}}
  if(k<int(sem.size())){int dest=sem[k].second;
   if(sem[k].first.op==Op::DROP){dest=depot[0];for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;}
   Controller::walk(out,pos,dest);out.a.push_back(sem[k].first);out.target.push_back(dest);
  }
 }
 return out;
}
inline bool chain_order(const View&o,const std::vector<Plan>&plans){
 constexpr int absent=1000000;std::array<int,100>fert,water,harvest,feed,care;
 for(auto*a:{&fert,&water,&harvest,&feed,&care})a->fill(absent);
 const int n=int(plans.size());
 for(int u=0;u<n;u++)for(size_t k=0;k<plans[u].a.size();k++){
  int pos=plans[u].target[k];if(pos<0)continue;int when=int(k)*n+u;
  auto set=[&](auto&a){a[pos]=std::min(a[pos],when);};
  switch(plans[u].a[k].op){case Op::FERTILIZE:set(fert);break;case Op::WATER:set(water);break;
   case Op::HARVEST:set(harvest);break;case Op::FEED:set(feed);break;case Op::CARE:set(care);break;default:break;}
 }
 for(int pos=0;pos<100;pos++){
  const auto&t=o.own.tiles[pos];
  if(plant(t)&&!t.watered_today){
   if(fert[pos]<absent&&water[pos]<absent&&fert[pos]>=water[pos])return false;
   if(!ongoing(int(t.crop))&&water[pos]<absent&&harvest[pos]<absent&&water[pos]>=harvest[pos])return false;
  }
  if(animal(t)&&!t.fed_today&&feed[pos]<absent&&care[pos]<absent&&feed[pos]>=care[pos])return false;
 }
 return true;
}
struct Quote {
 double cash=0.;bool known=false;Counts terminal_stock{};
 // Successful effects, not action attempts. Preserve all previously executable
 // crop/animal work, including harvest quantity after intervening decay.
 std::array<int,100>harvest{},collect{},water{},fert{},feed{},care{};
};
inline Quote quote(const Controller&source,const View&o,const std::vector<Plan>&plans){
 Quote q;ObservedDayScenario scenario(o);Controller c=source;
 c.plans=plans;for(auto&pl:c.plans)pl.index=0;
 c.phase=3;c.day=o.day;c.last_step=o.step-1;c.queue.clear();
 c.admission_inspection=nullptr;c.admission_values={};c.pending_admission={};
 // Evaluate the proposed fixed commitments, not a recursive planner. The
 // ordinary valid(), aggregate resource guards, deposits and terminal market
 // executor remain in use. No new investments/workers can be introduced.
 c.p.idle_task_handoff=false;c.p.intraday_admission=false;
 while(!scenario.finished()){
  auto v=scenario.view();c.resume_compiled_tick=true;auto a=c.act(v);
  // Use ordered unit-prefix effects: WATER and HARVEST can occur on the
  // same tile in the same tick, so end-of-tick tile subtraction is insufficient.
  auto before=scenario.project_units(a.units,0);
  for(int u=0;u<int(a.units.size());u++){
   auto after=scenario.project_units(a.units,u+1);
   int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer);
   const auto&t=before.farms()[0].tiles[pos];const auto&nt=after.farms()[0].tiles[pos];
   auto op=a.units[u].op;
   if(op==Op::HARVEST){int i=animal(t)?product[int(t.animal)-9]:plant(t)?int(t.crop):-1;
    if(i>=0)q.harvest[pos]+=std::max(0,int(after.privates()[0].inventories[u][i]-before.privates()[0].inventories[u][i]));}
   if(op==Op::COLLECT_FERTILIZER)q.collect[pos]+=std::max(0,int(after.privates()[0].inventories[u][F]-before.privates()[0].inventories[u][F]));
   if(op==Op::WATER)q.water[pos]+=int(!t.watered_today&&nt.watered_today);
   if(op==Op::FERTILIZE)q.fert[pos]+=int(nt.fertilized_until_day>t.fertilized_until_day);
   if(op==Op::FEED)q.feed[pos]+=int(!t.fed_today&&nt.fed_today);
   if(op==Op::CARE)q.care[pos]+=int(!t.cared_today&&nt.cared_today);
   before=std::move(after);
  }
  scenario.advance(a);
 }
 if(scenario.step_count()!=719)return q;
 q.cash=scenario.own().money;add(q.terminal_stock,scenario.inventory().shed);
 for(const auto&bag:scenario.inventory().inventories)add(q.terminal_stock,bag);
 q.known=std::isfinite(q.cash);return q;
}
inline bool preserves(const Quote&base,const Quote&trial){
 if(!base.known||!trial.known)return false;
 for(int i=0;i<12;i++)if(trial.terminal_stock[i]>base.terminal_stock[i])return false;
 for(int p=0;p<100;p++)if(trial.harvest[p]<base.harvest[p]||trial.collect[p]<base.collect[p]||
    trial.water[p]<base.water[p]||trial.fert[p]<base.fert[p]||trial.feed[p]<base.feed[p]||trial.care[p]<base.care[p])return false;
 return true;
}
// A deliberately adverse early-supply quote: currently visible, mature rival
// output is assumed already sold at the start. No hidden stock, future yields,
// identities, random seeds or actual rival actions are accepted. This is only
// price sensitivity; its synthetic cash is never written to the live agent.
inline Market pressure_market(const View&o){
 Market m=o.market;
 for(const auto&t:o.opponent.tiles){int item=-1;
  if(animal(t))item=product[int(t.animal)-9];
  else if(plant(t)&&o.day-t.planted_day>=first[int(t.crop)])item=int(t.crop);
  if(item>=0)m.inventory[item]+=std::max(0,int(t.yield_units));
 }
 for(int i=0;i<9;i++)m.prices[i]=price(i,m.inventory[i]);return m;
}
inline std::vector<Job>missing_jobs(const std::vector<Plan>&plans,const std::vector<Job>&fresh){
 std::multiset<Controller::TaskKey>covered;
 for(const auto&pl:plans)for(size_t k=0;k<pl.a.size();k++)if(pl.target[k]>=0&&!Controller::movement(pl.a[k].op))covered.insert(Controller::task_key(pl.target[k],pl.a[k]));
 std::vector<Job>out;
 for(const auto&j:fresh){Job part=j;part.actions.clear();part.needs={};part.seeds={};part.out={};auto available=covered;
  for(auto a:j.actions){auto key=Controller::task_key(j.pos,a);auto it=available.find(key);
   if(it!=available.end())available.erase(it);else part.actions.push_back(a);}
  if(part.actions.empty()||!Controller::resource_free_service(part))continue;
  for(auto a:part.actions)if(produces(a.op)){part.out=j.out;break;}
  out.push_back(std::move(part));
 }
 return out;
}
struct Proposal {int unit=-1,slot=0,job=-1,delta=0;double hint=0.;Plan plan;};
inline int augment(Controller&c,const View&o,std::vector<Plan>&plans,const std::vector<Job>&fresh,int limit){
 if(o.day!=29||limit<=0||limit!=terminal_ticks(o)||plans.size()!=o.priv.inventories.size())return 0;
 c.r4_route_checks++;const auto initial=plans;int applied=0,previews=0;double accepted_gain=0.;
 try{
  auto stress=pressure_market(o);View sv{o.step,o.day,o.hour,o.own,o.opponent,o.priv,stress,o.shops};
  bool different=stress.inventory!=o.market.inventory;
  Quote base=quote(c,o,plans),base_stress=different?quote(c,sv,plans):base;
  c.r4_route_previews+=1+different;
  // Deterministic computational bounds, not seed-dependent policy switches.
  // Unconsidered work stays available at the next genuine observation.
  constexpr int max_previews=96,max_additions=4,per_job_options=4;
  for(int pass=0;pass<max_additions&&previews<max_previews;pass++){
   auto jobs=missing_jobs(plans,fresh);std::vector<Proposal>proposals;
   for(int j=-1;j<int(jobs.size());j++){
    std::vector<Proposal>local;
    for(int u=0;u<int(plans.size());u++){
     int slots=j<0?1:int(Controller::plan_semantics(plans[u]).size())+1;
     for(int at=0;at<slots;at++){
      Plan trial=j<0?plans[u]:insert_chain(o,u,plans[u],jobs[j],at);
      if(!close_delivery(o,u,trial,limit)||Controller::same_remaining(trial,plans[u]))continue;
      auto all=plans;all[u]=trial;if(!chain_order(o,all))continue;
      int delta=int(trial.a.size())-int(plans[u].a.size());double value=0.;
      if(j>=0){for(int i=0;i<9;i++)if(jobs[j].out[i]>0)value+=revenue(i,o.market.inventory[i],jobs[j].out[i]);}
      else{for(int i=0;i<9;i++)value+=revenue(i,o.market.inventory[i],o.priv.inventories[u][i]);}
      // Zero-output WATER can raise a harvest owned by another unit. Keep it
      // eligible; the exact endpoint quote, not this search ordering, decides.
      bool duplicate=false;for(const auto&old:local)if(old.unit==u&&Controller::same_remaining(trial,old.plan)){duplicate=true;break;}
      if(!duplicate)local.push_back({u,at,j,delta,value/std::max(1,delta),std::move(trial)});
     }
    }
    std::stable_sort(local.begin(),local.end(),[](const auto&a,const auto&b){return std::tuple(a.delta,a.plan.a.size(),a.unit,a.slot)<std::tuple(b.delta,b.plan.a.size(),b.unit,b.slot);});
    if(local.size()>per_job_options)local.resize(per_job_options);
    for(auto&x:local)proposals.push_back(std::move(x));
   }
   std::stable_sort(proposals.begin(),proposals.end(),[](const auto&a,const auto&b){return std::tuple(-a.hint,a.delta,a.unit,a.slot,a.job)<std::tuple(-b.hint,b.delta,b.unit,b.slot,b.job);});
   int best=-1;double best_gain=1e-6;Quote win,win_stress;
   for(int k=0;k<int(proposals.size())&&previews<max_previews;k++){
    auto all=plans;all[proposals[k].unit]=proposals[k].plan;
    auto v=quote(c,o,all);previews++;c.r4_route_previews++;
    if(!preserves(base,v)||v.cash<=base.cash+1e-6){c.r4_route_rejected++;continue;}
    auto vs=different?quote(c,sv,all):v;c.r4_route_previews+=different;
    if(!preserves(base_stress,vs)||vs.cash<=base_stress.cash+1e-6){c.r4_route_rejected++;continue;}
    double gain=std::min(v.cash-base.cash,vs.cash-base_stress.cash);
    if(gain>best_gain+1e-6){best=k;best_gain=gain;win=std::move(v);win_stress=std::move(vs);}
   }
   if(best<0)break;
   plans[proposals[best].unit]=std::move(proposals[best].plan);base=std::move(win);base_stress=std::move(win_stress);
   applied++;accepted_gain+=best_gain;
  }
 }catch(const std::exception&){plans=initial;c.r4_route_errors++;return 0;}
 c.r4_route_insertions+=applied;c.r4_route_conditional_gain+=accepted_gain;return applied;
}
} // namespace dp7::terminalroll
