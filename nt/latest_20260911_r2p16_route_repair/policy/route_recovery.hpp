#pragma once
// Event-driven recovery of today's intent from authoritative inventory/cash.
// This does not rerun the dawn planner or assume today's workers start at home.
namespace triad::recovery_detail {
using namespace route_detail;
inline std::vector<Job> jobs(const triad::Controller&c,const View&o){
 auto raw=c.core;raw.p.shared_task_atoms_v2=raw.p.split_service_jobs=false;
 return raw.jobs(o);
}
inline Work work(const std::vector<Job>&js){
 Work r;for(const auto&j:js)for(auto a:j.actions)if(field(a))r[dp7::Controller::task_key(j.pos,a)]++;return r;
}
inline Acts deficits(const View&o,const std::vector<Job>&js){
 Counts need{},seed{};for(const auto&j:js){add(need,j.needs);add(seed,j.seeds);}
 for(const auto&bag:o.priv.inventories)for(int i=0;i<12;i++)need[i]=std::max(0,need[i]-bag[i]);
 Acts a;
 for(int i:{W,F,G,CO,SH})if(need[i]>o.priv.shed[i])a.push_back(action(i>=9?Op::BUY_ANIMAL:Op::BUY_PRODUCT,i,need[i]-o.priv.shed[i]));
 for(int i=0;i<5;i++)if(seed[i]>o.priv.seeds[i])a.push_back(action(Op::BUY_SEED,i,seed[i]-o.priv.seeds[i]));
 return a;
}
inline bool filled(const Acts&a,const std::vector<int32_t>&q){
 if(a.size()!=q.size())return false;
 for(size_t i=0;i<a.size();i++)if(procurement(a[i])&&wanted(a[i])>q[i])return false;return true;
}
// Reuse the packer and complete-chain repair, then prove the resulting routes
// from actual positions. Backpacks are credited only to their physical owner.
inline std::optional<Plans> schedule(const triad::Controller&source,const View&o){
 budget::check();
 auto js=jobs(source,o);auto due=work(js);auto c=source;
 c.core.route_inspection=nullptr;c.core.admission_inspection=nullptr;
 std::vector<int>starts{cell(o.own.farmer)};for(auto p:o.own.hands)starts.push_back(cell(p));
 int budget=std::min(24-o.hour,719-o.step);
 int atoms=0;for(const auto&j:js)atoms+=j.actions.size();
 if(atoms>budget*int(starts.size()))return std::nullopt;
 // Incumbent harvest/maintenance may share workers as in the released R2.
 // Only new investment chains require a single worker's ordered sequence.
 auto packed=c.core.pack(c.core.jobs(o),starts,budget,false,true);
 Plans ps(starts.size());
 for(const auto&r:packed.first){
  int pos=starts[r.unit];Plan p;
  for(const auto&j:r.jobs){dp7::Controller::walk(p,pos,j.pos);for(auto a:j.actions){p.a.push_back(a);p.target.push_back(pos);}}
  ps[r.unit]=supplied(o,r.unit,p);
 }
 auto proof=verify(o,ps);
 if(!proof.valid||!covers(proof.completed,due)){
  if(!proof.valid)ps.assign(starts.size(),{});
  c.core.plans=ps;c.core.phase=3;c.logistics.required=due;c.logistics.completed.clear();
  c.logistics.witness.clear();c.logistics.witness_step=-1;
  c.route_restore(o);ps=c.core.plans;proof=verify(o,ps);
 }
 if(!proof.valid||!covers(proof.completed,due))return std::nullopt;
 return ps;
}
struct Choice {Acts orders;Plans plans;bool valid=false;PlayerAction first;bool advanced=false;};
inline Choice choose(const triad::Controller&c,const View&o){
 budget::check();
 // Preview only the unit phase. Existing workers can continue while market
 // sales fund inputs and hires later in this same step.
 auto preview=c;preview.logistics.probe=true;preview.logistics.recovery_allowed=false;
 preview.previous_step=preview.core.last_step=o.step-1;
 preview.core.phase=3;preview.core.queue.clear();preview.core.resume_compiled_tick=true;
 if(preview.core.plans.empty())preview.core.plans.resize(o.priv.inventories.size());
 auto units=preview.act(o).units;
 auto allowed=work(jobs(c,o));
 for(int u=0;u<int(units.size());u++)if(field(units[u])){
  int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
  if(!allowed.contains(dp7::Controller::task_key(pos,units[u])))units[u]=action(Op::PASS);
 }
 fastkag::ObservedDayScenario initial(o);auto post=initial.project_units(units,-1);
 View after{o.step,o.day,o.hour,post.farms()[0],post.farms()[1],post.privates()[0],post.market(),o.shops};
 auto buy=deficits(after,jobs(c,after));
 if(buy.size()>10)return {};
 // Already paid workers are sunk cost. Add only the minimum further workers
 // for which a complete remaining-day route has actually been constructed.
 int limit=std::min({c.core.p.max_hands-int(o.own.hands.size()),10-int(buy.size()),int(fib.size())-o.own.hires_today});
 for(int extra=0;extra<=std::max(0,limit);extra++){
  budget::check();
  Acts orders=buy;for(int i=0;i<extra;i++)orders.push_back(action(Op::HIRE));
  if(orders.empty()){if(auto plans=schedule(c,o))return {orders,*plans,true};continue;}
  // An additional order consumes a real step, even if its worker is cheap.
  if(o.hour>=23||o.step>=718)break;
  PlayerAction a;a.units=units;a.market=orders;
  auto funded=c;funded.core.phase=3;funded.route_finance(o,a,true);
  // A funding sale must not silently displace a required purchase at the
  // ten-order limit. All original orders must still occur and fill.
  auto remaining=a.market;bool retained=true;
  for(auto order:orders){auto it=std::find_if(remaining.begin(),remaining.end(),[&](auto x){return dp7::Controller::same_action(x,order);});
   if(it==remaining.end()){retained=false;break;}remaining.erase(it);}
  if(!retained)break;
  fastkag::ObservedDayScenario world(o);world.advance(a);
  if(!filled(a.market,world.fills())||world.finished())break;
  if(auto plans=schedule(c,world.view()))return {a.market,*plans,true,a,true};
 }
 return {};
}
inline void defer(triad::Controller&c,const View&o,int pos){
 c.core.plant_not_before[pos]=o.day+1;
 c.core.recovery_not_before[pos]=o.day+1;
 // Removing an unplaced animal investment cannot remove a living animal.
 c.core.target.erase(std::remove_if(c.core.target.begin(),c.core.target.end(),[&](auto x){return x.first==pos&&!plant(o.own.tiles[pos])&&!animal(o.own.tiles[pos]);}),c.core.target.end());
}
inline void install(triad::Controller&c,const triad::Controller&chosen,const View&o,const Choice&v,bool complete=true){
 auto&l=c.logistics;
 auto before=work(jobs(c,o)),after=work(jobs(chosen,o));
 for(auto[key,n]:before)if(n>after[key]){l.deferred_work[key]+=n-after[key];l.recovery_deferred+=n-after[key];}
 c.core.target=chosen.core.target;c.core.plant_not_before=chosen.core.plant_not_before;
 c.core.recovery_not_before=chosen.core.recovery_not_before;
 c.core.pending_admission.active=false;c.core.intraday_units.clear();c.core.queue.clear();
 c.core.planned_land=std::popcount(unsigned(o.own.unlocked_mask));
 {
  c.core.plans=v.plans;c.core.phase=3;c.core.resume_compiled_tick=true;
  c.core.seed_reconcile_checked=true;
  l.recovery_pending=!complete;l.recovery_waiting=!complete;l.recovery_installs++;
  if(v.advanced){l.recovery_orders++;l.recovery_joint++;}
  else{triad::Controller::RouteValue promised;promised.completed=verify(o,v.plans).completed;promise(c,promised);}
  l.witness=v.plans;l.witness_step=o.step+int(v.advanced);l.changes++;
  if(!complete){l.recovery_partial++;l.retry_cash=o.own.money;l.retry_hands=o.own.hands.size();
   l.retry_resources=o.priv.shed;for(const auto&b:o.priv.inventories)add(l.retry_resources,b);l.retry_seeds=o.priv.seeds;}
 }
}
} // namespace triad::recovery_detail
namespace triad {
inline std::optional<PlayerAction> Controller::route_recover(const View&o){
 using namespace recovery_detail;
 if(!logistics.recovery_allowed||!logistics.recovery_pending)return std::nullopt;
 budget::check();
 if(logistics.recovery_waiting){
  Counts resources=o.priv.shed;for(const auto&b:o.priv.inventories)add(resources,b);
  bool changed=o.own.money>logistics.retry_cash+1e-6||int(o.own.hands.size())>logistics.retry_hands;
  for(int i=0;i<12;i++)changed|=resources[i]>logistics.retry_resources[i];
  for(int i=0;i<5;i++)changed|=o.priv.seeds[i]>logistics.retry_seeds[i];
  if(!changed)return std::nullopt;
  logistics.recovery_retries++;
 }
 logistics.recovery_checks++;
 // Register today's original intent without rewriting previous completions.
 // Deferred tasks remain in required, so the native audit still reports them.
 auto original=work(jobs(*this,o));
 for(auto[key,n]:original)logistics.required[key]=std::max(logistics.required[key],logistics.completed[key]+n);
 logistics.changes++;
 // Secure incumbent maintenance before spending the search budget on new
 // projects. An interrupted expansion search can then install useful work.
 std::vector<std::pair<double,int>>optional;
 for(const auto&j:jobs(*this,o))if(std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::PLANT||(a.op==Op::PLACE&&int(a.item)>=9);}))
  optional.emplace_back(core.job_consequence(o,j),j.pos);
 std::stable_sort(optional.rbegin(),optional.rend());
 auto maintenance=*this;for(auto[value,pos]:optional)defer(maintenance,o,pos);
 auto chosen=maintenance;Choice best=choose(chosen,o);
 auto score=[&](const Controller&policy,const Choice&choice){
  auto trial=policy;install(trial,policy,o,choice);
  auto value=trial.route_evaluate(o,true,choice.advanced?&choice.first:nullptr);
  logistics.evaluations++;return value.score;
 };
 if(best.valid)try{
  double bestscore=score(chosen,best);
  // Bound admission work: test the full portfolio, then bisect the ranked
  // prefix. Feasibility is proved for each accepted candidate, not assumed
  // monotone; no claim of a globally optimal portfolio is made.
  int low=0,high=optional.size();
  for(int trial_index=0;trial_index<4&&high>low;trial_index++){
   budget::check();int count=trial_index==0?high:(low+high+1)/2;
   auto trial=*this;for(int i=count;i<int(optional.size());i++)defer(trial,o,optional[i].second);
   auto candidate=choose(trial,o);
   if(candidate.valid){low=count;double value=score(trial,candidate);
    if(value>bestscore+1e-6){chosen=std::move(trial);best=std::move(candidate);bestscore=value;}}
   else high=count-1;
  }
  auto keep=route_evaluate(o,true);logistics.evaluations++;
  if(covers(keep.completed,original)&&keep.unfilled==0&&keep.score>=bestscore-1e-6){
   logistics.recovery_pending=logistics.recovery_waiting=false;logistics.recovery_kept++;return std::nullopt;
  }
 }catch(const budget::Exhausted&){logistics.budget_stops++;}
 if(best.valid){
  install(*this,chosen,o,best);
  if(!best.advanced)return std::nullopt;
  return best.first;
 }
 // Incumbent work itself may be impossible (no feed/cash/time). Stop retrying
 // the unfunded investments and let existing service recovery do useful work.
 // Do not issue speculative hires, or label the unmet original debt complete.
 best.orders.clear();best.plans.assign(o.priv.inventories.size(),{});
 auto due=work(jobs(chosen,o));auto probe=chosen;probe.core.plans=best.plans;
 probe.logistics.required=due;probe.logistics.completed.clear();probe.route_restore(o);
 auto proof=verify(o,probe.core.plans);
 if(proof.valid)best.plans=probe.core.plans;
 install(*this,chosen,o,best,proof.valid&&covers(proof.completed,due));
 return std::nullopt;
}
} // namespace triad
