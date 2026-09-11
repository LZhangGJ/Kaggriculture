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
 auto js=jobs(source,o);auto due=work(js);auto c=source;
 c.core.route_inspection=nullptr;c.core.admission_inspection=nullptr;
 std::vector<int>starts{cell(o.own.farmer)};for(auto p:o.own.hands)starts.push_back(cell(p));
 int budget=std::min(24-o.hour,719-o.step);
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
struct Choice {Acts orders;Plans plans;bool valid=false;};
inline Choice choose(const triad::Controller&c,const View&o){
 auto buy=deficits(o,jobs(c,o));
 if(buy.size()>10)return {};
 // Already paid workers are sunk cost. Add only the minimum further workers
 // for which a complete remaining-day route has actually been constructed.
 int limit=std::min({c.core.p.max_hands-int(o.own.hands.size()),10-int(buy.size()),int(fib.size())-o.own.hires_today});
 for(int extra=0;extra<=std::max(0,limit);extra++){
  Acts orders=buy;for(int i=0;i<extra;i++)orders.push_back(action(Op::HIRE));
  if(orders.empty()){if(auto plans=schedule(c,o))return {orders,*plans,true};continue;}
  // An additional order consumes a real step, even if its worker is cheap.
  if(o.hour>=23||o.step>=718)break;
  PlayerAction a;a.units.resize(o.priv.inventories.size());a.market=orders;
  // No sale proceeds or future harvest appear in this cash test.
  if(intraday::committed_market_cost(c.core,o,a)>o.own.money+1e-6)break;
  fastkag::ObservedDayScenario world(o);world.advance(a);
  if(!filled(orders,world.fills())||world.finished())break;
  if(auto plans=schedule(c,world.view()))return {orders,*plans,true};
 }
 return {};
}
inline void defer(triad::Controller&c,const View&o,int pos){
 c.core.plant_not_before[pos]=o.day+1;
 c.core.recovery_not_before[pos]=o.day+1;
 // Removing an unplaced animal investment cannot remove a living animal.
 c.core.target.erase(std::remove_if(c.core.target.begin(),c.core.target.end(),[&](auto x){return x.first==pos&&!plant(o.own.tiles[pos])&&!animal(o.own.tiles[pos]);}),c.core.target.end());
}
inline Acts sales(const triad::Controller&c,const View&o){
 auto reserve=reserves(c,o);Counts need{};
 for(const auto&j:jobs(c,o))add(need,j.needs);
 for(const auto&bag:o.priv.inventories)for(int i=0;i<12;i++)need[i]=std::max(0,need[i]-bag[i]);
 Counts available{};for(int i=0;i<9;i++)available[i]=std::max(0,o.priv.shed[i]-std::max(reserve[i],need[i]));
 return c.core.sales_sorted(o,available);
}
inline void install(triad::Controller&c,const triad::Controller&chosen,const View&o,const Choice&v){
 auto&l=c.logistics;
 auto before=work(jobs(c,o)),after=work(jobs(chosen,o));
 for(auto[key,n]:before)if(n>after[key]){l.deferred_work[key]+=n-after[key];l.recovery_deferred+=n-after[key];}
 c.core.target=chosen.core.target;c.core.plant_not_before=chosen.core.plant_not_before;
 c.core.recovery_not_before=chosen.core.recovery_not_before;
 c.core.pending_admission.active=false;c.core.intraday_units.clear();c.core.queue.clear();
 c.core.planned_land=std::popcount(unsigned(o.own.unlocked_mask));
 if(v.orders.empty()){
  c.core.plans=v.plans;c.core.phase=3;c.core.resume_compiled_tick=true;
  c.core.seed_reconcile_checked=true;
  l.recovery_pending=false;l.recovery_installs++;
  triad::Controller::RouteValue promised;promised.completed=verify(o,v.plans).completed;promise(c,promised);
  l.witness=v.plans;l.witness_step=o.step;l.changes++;
 }else{c.core.phase=2;l.recovery_orders++;}
}
} // namespace triad::recovery_detail
namespace triad {
inline std::optional<PlayerAction> Controller::route_recover(const View&o){
 using namespace recovery_detail;
 if(!logistics.recovery_allowed||!logistics.recovery_pending)return std::nullopt;
 logistics.recovery_checks++;
 // Register today's original intent without rewriting previous completions.
 // Deferred tasks remain in required, so the native audit still reports them.
 auto original=work(jobs(*this,o));
 for(auto[key,n]:original)logistics.required[key]=std::max(logistics.required[key],logistics.completed[key]+n);
 logistics.changes++;logistics.witness.clear();logistics.witness_step=-1;core.queue.clear();
 auto chosen=*this;Choice best=choose(chosen,o);
 if(!best.valid&&o.hour<22&&o.step<717){
  auto offers=sales(*this,o);
  if(!offers.empty()){
   PlayerAction sale;sale.units.resize(o.priv.inventories.size());sale.market=offers;
   fastkag::ObservedDayScenario funded(o);funded.advance(sale);
   // Selling is useful only if it leads to a complete feasible plan. The
   // predicted proceeds merely select a sale; purchases wait for real cash.
   if(choose(*this,funded.view()).valid){logistics.recovery_sales++;return sale;}
  }
 }
 if(!best.valid){
  std::vector<std::pair<double,int>>optional;
  for(const auto&j:jobs(*this,o))if(std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::PLANT||(a.op==Op::PLACE&&int(a.item)>=9);}))
   optional.emplace_back(core.job_consequence(o,j),j.pos);
  std::stable_sort(optional.begin(),optional.end());
  for(auto[value,pos]:optional){defer(chosen,o,pos);best=choose(chosen,o);if(best.valid)break;}
 }
 if(best.valid){
  install(*this,chosen,o,best);
  if(best.orders.empty())return std::nullopt;
  PlayerAction out;out.units.resize(o.priv.inventories.size());out.market=best.orders;return out;
 }
 // Incumbent work itself may be impossible (no feed/cash/time). Stop retrying
 // the unfunded investments and let existing service recovery do useful work.
 // Do not issue speculative hires, or label the unmet original debt complete.
 best.orders.clear();best.plans.assign(o.priv.inventories.size(),{});
 auto due=work(jobs(chosen,o));auto probe=chosen;probe.core.plans=best.plans;
 probe.logistics.required=due;probe.logistics.completed.clear();probe.route_restore(o);
 auto proof=verify(o,probe.core.plans);
 if(proof.valid)best.plans=probe.core.plans;
 install(*this,chosen,o,best);
 return std::nullopt;
}
} // namespace triad
