#pragma once
// Execution-only workflow integrity layer adapted from teammate commit db109ce.
// It never changes targets, purchases, hires, sales, or project values.  It
// protects only field work that this AFS instance already proved schedulable at
// the beginning of the work phase.  Unknown RNG, replay ids, opponent identity,
// and opponent-private state are not inputs.
#include "observed_day_scenario.hpp"

namespace dp7::workflow {
using Plans=std::vector<Plan>;
using Work=WorkflowRepairState::Work;
using Key=WorkflowRepairState::TaskKey;

inline constexpr bool enabled=P16_AFS_WORKFLOW_REPAIR||P16_AFS_WORKFLOW_AUDIT;

inline bool field(const Action&a){
 return !Controller::movement(a.op)&&a.op!=Op::PASS&&a.op!=Op::PICKUP&&a.op!=Op::DROP&&
        !(a.op==Op::PLACE&&int(a.item)<9);
}
inline Plans remaining(const Controller&c){
 Plans out;out.reserve(c.plans.size());
 for(const auto&old:c.plans){size_t k=std::min(old.index,old.a.size());
  out.push_back({Acts(old.a.begin()+k,old.a.end()),
    std::vector<int>(old.target.begin()+k,old.target.end()),0});
 }
 return out;
}
inline Work work(const Plans&ps){
 Work out;for(const auto&p:ps)for(size_t k=p.index;k<p.a.size();k++)
  if(p.target[k]>=0&&field(p.a[k]))out[Controller::task_key(p.target[k],p.a[k])]++;
 return out;
}
inline bool same_plans(const Plans&a,const Plans&b){
 if(a.size()!=b.size())return false;
 for(size_t u=0;u<a.size();u++){
  size_t ai=std::min(a[u].index,a[u].a.size()),bi=std::min(b[u].index,b[u].a.size());
  if(a[u].a.size()-ai!=b[u].a.size()-bi)return false;
  for(size_t k=0;k<a[u].a.size()-ai;k++)if(!Controller::same_action(a[u].a[ai+k],b[u].a[bi+k])||a[u].target[ai+k]!=b[u].target[bi+k])return false;
 }
 return true;
}
inline bool covers(const Work&have,const Work&need){
 for(auto[key,n]:need){auto it=have.find(key);if(it==have.end()||it->second<n)return false;}return true;
}
inline Work debt(const WorkflowRepairState&s){
 Work out;for(auto[key,n]:s.required){auto it=s.completed.find(key);
  int done=it==s.completed.end()?0:it->second;if(n>done)out[key]=n-done;}return out;
}
inline int missing_count(const Work&have,const Work&need){
 int n=0;for(auto[key,want]:need){auto it=have.find(key);n+=std::max(0,want-(it==have.end()?0:it->second));}return n;
}
inline bool fits(const View&o,const Plans&ps){
 int limit=std::min(24-o.hour,719-o.step);
 for(const auto&p:ps)if(int(p.a.size()-std::min(p.index,p.a.size()))>limit)return false;return true;
}

inline int effect(const fastkag::Simulator&before,const fastkag::Simulator&after,int u,const Action&a){
 int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer),i=int(a.item);
 const auto&t=before.farms()[0].tiles[pos];const auto&z=after.farms()[0].tiles[pos];
 const auto&b=before.privates()[0].inventories[u];const auto&e=after.privates()[0].inventories[u];
 switch(a.op){
  case Op::HARVEST:{int n=0;for(int k=0;k<8;k++)n+=std::max(0,e[k]-b[k]);return n;}
  case Op::COLLECT_FERTILIZER:return std::max(0,e[F]-b[F]);
  case Op::WATER:return plant(t)&&!t.watered_today&&z.watered_today;
  case Op::CARE:return animal(t)&&!t.cared_today&&z.cared_today;
  case Op::FEED:return animal(t)&&!t.fed_today&&z.fed_today&&b[W]>e[W];
  case Op::FERTILIZE:return plant(t)&&b[F]>e[F]&&z.fertilized_until_day>=t.fertilized_until_day;
  case Op::PLANT:return !plant(t)&&plant(z)&&int(z.crop)==i;
  case Op::DIG:return t.kind!=TileKind::EMPTY&&z.kind==TileKind::EMPTY;
  case Op::BUILD_COOP:return t.kind==TileKind::EMPTY&&z.kind==TileKind::COOP;
  case Op::BUILD_PASTURE:return t.kind==TileKind::EMPTY&&z.kind==TileKind::PASTURE;
  case Op::PLACE:return i>=9&&!animal(t)&&animal(z)&&int(z.animal)==i;
  default:return 0;
 }
}
struct Proof {Work completed,harvested;bool valid=true;int fail_tick=-1,fail_unit=-1,fail_pos=-1,fail_op=-1;};
inline Proof verify(const View&o,const Plans&ps){
 Proof result;if(ps.size()!=o.priv.inventories.size()){result.valid=false;result.fail_tick=-2;result.fail_unit=int(ps.size());result.fail_pos=int(o.priv.inventories.size());return result;}
 if(!fits(o,ps)){result.valid=false;result.fail_tick=-3;return result;}
 fastkag::ObservedDayScenario world(o);
 for(size_t tick=0;!world.finished();tick++){
  PlayerAction out;out.units.resize(ps.size());bool any=false;
  for(size_t u=0;u<ps.size();u++)if(tick<ps[u].a.size()){out.units[u]=ps[u].a[tick];any=true;}
  if(!any)break;
  auto before=world.project_units(out.units,0);
  for(size_t u=0;u<ps.size();u++){
   auto a=out.units[u];auto after=world.project_units(out.units,int(u)+1);
   if(field(a)){
    int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer);
    int n=effect(before,after,int(u),a);
    if(n<=0||tick>=ps[u].target.size()||pos!=ps[u].target[tick]){
     result.valid=false;if(result.fail_tick<0){result.fail_tick=int(tick);result.fail_unit=int(u);result.fail_pos=pos;result.fail_op=int(a.op);}
    }
    if(n>0){auto key=Controller::task_key(pos,a);result.completed[key]++;
     if(a.op==Op::HARVEST)result.harvested[key]+=n;}
   }
   before=std::move(after);
  }
  world.advance(out);
 }
 return result;
}

inline bool inputs(const View&o,const Plans&ps){
 Counts pickups{},seeds{};std::set<int>harvested,collected;
 for(size_t u=0;u<ps.size();u++){
  auto held=o.priv.inventories[u];
  for(auto[a,pos]:Controller::plan_semantics(ps[u])){
   int i=int(a.item);
   if(a.op==Op::PICKUP){if(i<0||i>=12)return false;pickups[i]+=a.quantity;held[i]+=a.quantity;}
   if(a.op==Op::DROP)held={};
   if(a.op==Op::FEED&&--held[W]<0)return false;
   if(a.op==Op::FERTILIZE&&--held[F]<0)return false;
   if(a.op==Op::PLACE&&(i<0||i>=12||(held[i]-=a.quantity)<0))return false;
   if(a.op==Op::PLANT){if(i<0||i>=5)return false;seeds[i]++;}
   if(pos>=0&&a.op==Op::COLLECT_FERTILIZER&&collected.insert(pos).second&&o.own.tiles[pos].fertilizer_available)held[F]++;
   if(pos>=0&&a.op==Op::HARVEST&&harvested.insert(pos).second){const auto&t=o.own.tiles[pos];
    int item=animal(t)?product[int(t.animal)-9]:plant(t)?int(t.crop):-1;if(item>=0)held[item]+=t.yield_units;}
  }
 }
 for(int i=0;i<12;i++)if(pickups[i]>o.priv.shed[i])return false;
 for(int i=0;i<5;i++)if(seeds[i]>o.priv.seeds[i])return false;
 return true;
}

inline Plan supplied(const View&o,int u,const Plan&base){
 auto held=o.priv.inventories[u];Counts missing{};std::set<int>harvested,collected;
 for(auto[a,pos]:Controller::plan_semantics(base)){
  int i=int(a.item);if(a.op==Op::PICKUP)held[i]+=a.quantity;if(a.op==Op::DROP)held={};
  int resource=a.op==Op::FEED?W:a.op==Op::FERTILIZE?F:a.op==Op::PLACE?i:-1;
  if(resource>=0){int q=a.op==Op::PLACE?a.quantity:1;int deficit=std::max(0,q-held[resource]);missing[resource]+=deficit;held[resource]+=deficit-q;}
  const auto&t=o.own.tiles[pos];
  if(a.op==Op::COLLECT_FERTILIZER&&collected.insert(pos).second&&t.fertilizer_available)held[F]++;
  if(a.op==Op::HARVEST&&harvested.insert(pos).second){int item=animal(t)?product[int(t.animal)-9]:plant(t)?int(t.crop):-1;if(item>=0)held[item]+=t.yield_units;}
 }
 if(!sum(missing))return base;
 auto sem=Controller::plan_semantics(base);Plan out;int pos=cell(u?o.own.hands[u-1]:o.own.farmer),dest=depot[0];
 int next=sem.empty()?pos:sem.front().second;
 for(int d:depot)if(dist(pos,d)+dist(d,next)<dist(pos,dest)+dist(dest,next))dest=d;
 Controller::walk(out,pos,dest);
 for(int i=0;i<12;i++)if(missing[i]){out.a.push_back(action(Op::PICKUP,i,missing[i]));out.target.push_back(dest);}
 for(auto[a,target]:sem){Controller::walk(out,pos,target);out.a.push_back(a);out.target.push_back(target);}
 return out;
}

inline void finalize(WorkflowRepairState&s){
 if(s.day<0||s.finalized_day==s.day||!s.captured)return;
 auto left=debt(s);int n=0;for(auto[key,q]:left)n+=q;
 if(n){s.deadline_misses+=n;s.deadline_days++;auto[key,q]=*left.begin();s.final_miss_pos=std::get<0>(key);s.final_miss_op=std::get<1>(key);s.final_miss_item=std::get<2>(key);}
 s.finalized_day=s.day;
}
inline void reset_day(WorkflowRepairState&s,int day){
 if(s.day==day)return;finalize(s);s.day=day;s.captured_step=-1;s.captured=false;
 s.required.clear();s.completed.clear();s.witness.clear();s.latest_cover.clear();s.latest_cover_step=-1;
}
inline void capture(Controller&c,const View&o,const Plans&base){
 auto&s=c.workflow;s.verify_calls++;auto proof=verify(o,base);
 if(!proof.valid){s.certificate_failures++;return;}
 s.required=proof.completed;s.completed.clear();s.witness=base;s.latest_cover=base;s.latest_cover_step=o.step;s.captured=true;s.captured_step=o.step;s.captures++;
}

inline Job chain_for(const Controller&c,const View&o,int pos,const Work&wanted,const Plans&base){
 Job chain;chain.pos=pos;Work left=wanted;
 auto take=[&](const Action&a,int p){if(p!=pos||!field(a))return;auto key=Controller::task_key(pos,a);auto it=left.find(key);if(it!=left.end()&&it->second>0){chain.actions.push_back(a);it->second--;}};
 auto raw=c;raw.p.shared_task_atoms_v2=false;raw.p.split_service_jobs=false;
 for(const auto&j:raw.jobs(o))if(j.pos==pos)for(auto a:j.actions)take(a,pos);
 for(const auto&p:c.workflow.witness)for(size_t k=p.index;k<p.a.size();k++)take(p.a[k],p.target[k]);
 for(const auto&p:base)for(size_t k=p.index;k<p.a.size();k++)take(p.a[k],p.target[k]);
 bool complete=true;for(auto[key,n]:left)complete&=n==0;
 if(!complete)chain.actions.clear();return chain;
}

// Workflow-only route rebuild.  The legacy repair_plan groups consecutive
// semantic atoms by their original target, then retargets the whole group to a
// nearby depot when its first atom is PICKUP/DROP.  That is unsafe for a causal
// group such as PICKUP(W,2)@depot -> FEED@animal: the material atom may move to
// another depot, but the field atom must stay on the animal tile.  Keep the
// public controller untouched (and therefore keep the switch-off baseline
// byte-for-byte compatible); use this split-destination rebuild only while
// restoring a previously certified workflow.
inline Plan repair_plan_precise(const Controller&c,const View&o,int unit,const Plan&old,int&stale){
 struct Group{int pos=-1;Acts actions;};std::vector<Group>groups;
 for(size_t k=old.index;k<old.a.size();k++){
  if(Controller::movement(old.a[k].op)||old.target[k]<0)continue;
  if(groups.empty()||groups.back().pos!=old.target[k])groups.push_back({old.target[k],{}});
  groups.back().actions.push_back(old.a[k]);
 }
 Plan out;int pos=cell(unit?o.own.hands.at(unit-1):o.own.farmer);bool projected_cargo=sum(o.priv.inventories.at(unit))>0;
 for(auto&g:groups){size_t first_needed=0;while(first_needed<g.actions.size()){
   const auto&a=g.actions[first_needed];bool needed=a.op==Op::DROP?projected_cargo:c.semantic_needed(o,unit,a,g.pos);
   if(needed)break;first_needed++;stale++;
  }
  if(first_needed==g.actions.size())continue;
  for(size_t k=first_needed;k<g.actions.size();k++){
   const auto&a=g.actions[k];int dest=g.pos;
   if(a.op==Op::PICKUP||a.op==Op::DROP){dest=depot[0];for(int d:depot)
     if(dist(pos,d)+dist(d,g.pos)<dist(pos,dest)+dist(dest,g.pos))dest=d;}
   Controller::walk(out,pos,dest);out.a.push_back(a);out.target.push_back(dest);
   if(a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER)projected_cargo=true;
   else if(c.p.incremental_pickup_repair&&a.op==Op::PICKUP&&int(a.item)>=0&&int(a.item)<12&&o.priv.shed[int(a.item)]>0&&a.quantity>0)projected_cargo=true;
   else if(a.op==Op::DROP)projected_cargo=false;
  }
 }
 return out;
}

inline bool restore_witness(Controller&c,const View&o,Plans&base,const Work&due,const Proof&keep){
 auto&s=c.workflow;std::vector<const Plans*>sources;
 if(s.latest_cover.size()==base.size())sources.push_back(&s.latest_cover);
 if(s.witness.size()==base.size()&&(sources.empty()||&s.witness!=sources.front()))sources.push_back(&s.witness);
  Plans chosen;auto best_rank=std::tuple(100000,100000,100000);
 for(const Plans*source:sources){auto fallback=*source;
  for(size_t u=0;u<fallback.size();u++){int stale=0;fallback[u]=repair_plan_precise(c,o,int(u),fallback[u],stale);}
  s.verify_calls++;auto checked=verify(o,fallback);
  int peak=0;for(const auto&p:fallback)peak=std::max(peak,int(p.a.size()));
  if(!fits(o,fallback))s.witness_no_fit++;
  if(!checked.valid)s.witness_invalid++;
  if(!covers(checked.completed,due))s.witness_not_due++;
   int keep_missing=missing_count(checked.completed,keep.completed)+missing_count(checked.harvested,keep.harvested);
   if(keep_missing)s.witness_not_keep++;
   // The day-start exact certificate is the authoritative execution contract.
   // A later recoordination may syntactically replace one certified service
   // with a newly attractive service even when the shared input cannot fund
   // both (for example five wheat for six feed targets).  Requiring both plans
   // would make a valid recovery impossible.  Restore the certified debt first;
   // among exact debt-covering witnesses retain as much of the newer baseline
   // as resources and the remaining ticks permit.
   if(!checked.valid||!covers(checked.completed,due))continue;
   int cost=0;for(const auto&p:fallback)cost+=int(p.a.size());
   auto rank=std::tuple(keep_missing,peak,cost);if(rank<best_rank){best_rank=rank;chosen=std::move(fallback);}
 }
 if(chosen.empty()){s.certificate_failures++;return false;}
 base=std::move(chosen);c.plans=base;s.witness_used++;return true;
}

inline bool restore_chains(Controller&c,const View&o,Plans&base,const Work&due){
 auto&s=c.workflow;Work covered=work(base);std::set<int>positions;
 for(auto[key,n]:due)if(covered[key]<n)positions.insert(std::get<0>(key));
 bool changed=false;
 for(int pos:positions){
  Work wanted;std::set<Controller::TaskKey>remove;
  for(auto[key,n]:covered)if(std::get<0>(key)==pos&&n>0){wanted[key]=n;remove.insert(key);}
  for(auto[key,n]:due)if(std::get<0>(key)==pos){wanted[key]=std::max(wanted[key],n);remove.insert(key);}
  auto chain=chain_for(c,o,pos,wanted,base);if(chain.actions.empty()){s.chain_unknown++;s.unresolved++;continue;}
  auto stripped=base;for(size_t u=0;u<stripped.size();u++)stripped[u]=c.strip_tasks(o,int(u),stripped[u],remove);
  s.verify_calls++;auto keep=verify(o,base);Plans chosen;int best=100000;
  for(int u=0;u<int(stripped.size());u++){
   int slots=int(Controller::plan_semantics(stripped[u]).size());
   for(int at=0;at<=slots;at++){
    s.repair_trials++;auto trial=stripped;
    trial[u]=supplied(o,u,Controller::insert_service(o,u,stripped[u],chain,at));
    int cost=0;for(const auto&p:trial)cost+=int(p.a.size());
    if(cost>=best)continue;
    if(!fits(o,trial)){s.chain_no_fit++;continue;}
    // `inputs` is only a cheap conservative prefilter.  It can reject a
    // jointly valid route when a worker picks up a spare input after the
    // original certified schedule has already distributed its carried
    // resources.  The exact observed-state replay below is authoritative, so
    // keep auditing the prefilter but do not let it veto an otherwise proven
    // repair.
    if(!inputs(o,trial))s.chain_input_reject++;
    s.verify_calls++;auto checked=verify(o,trial);
    // Recoordination may contain speculative field atoms that the live
    // executor would skip for free.  A repair must preserve every effect the
    // baseline fixed schedule can actually complete (including harvest
    // quantities) and complete the certified debt; it need not turn unrelated
    // pre-existing speculative atoms into valid work.  Because the candidate
    // is built only by retaining the baseline semantics and adding `wanted`,
    // this is a strict differential proof rather than a relaxed success gate.
    if(!covers(checked.completed,keep.completed)||!covers(checked.harvested,keep.harvested)||!covers(checked.completed,wanted)){s.chain_verify_reject++;continue;}
    chosen=std::move(trial);best=cost;
   }
  }
  if(chosen.empty()){s.unresolved++;continue;}
  Work next=work(chosen);for(auto[key,n]:wanted)s.restored+=std::max(0,n-covered[key]);
  base=std::move(chosen);covered=std::move(next);c.plans=base;s.chain_repairs++;changed=true;
 }
 return changed;
}

inline bool schedule_urgent(Controller&c,const View&o,const Work&have,const Work&due){
 auto&s=c.workflow;
 for(auto[key,n]:due)if((have.find(key)==have.end()?0:have.at(key))<n){
  auto[pos,op,item]=key;Action wanted=action(Op(op),item);Counts pickup{},plants{};
  for(size_t u=0;u<o.priv.inventories.size();u++){
   if(cell(u?o.own.hands[u-1]:o.own.farmer)!=pos)continue;
   auto valid=c.valid(o,int(u),wanted,pos,pickup,plants);
   if(valid&&field(*valid)&&valid->op==wanted.op){s.urgent_step=o.step;s.urgent_unit=int(u);s.urgent_action=*valid;s.urgent_scheduled++;return true;}
  }
 }
 return false;
}

inline void inspect(Controller&c,const View&o){
 if constexpr(!enabled)return;
 if(!c.workflow_runtime)return;
 if(c.phase!=3||c.plans.size()!=o.priv.inventories.size())return;
 auto&s=c.workflow;reset_day(s,o.day);auto base=remaining(c);
 if(!s.captured){capture(c,o,base);return;}
 auto due=debt(s);if(due.empty()){s.fast_paths++;return;}
 auto have=work(base);Proof current_proof;bool proof_ready=false;
 if(covers(have,due)){
  // `latest_cover` was already proved from the immediately preceding public
  // state and account() advances it only after the exact scheduled joint unit
  // action succeeds.  An identical suffix therefore needs no replay here.
  if(same_plans(base,s.latest_cover)){s.fast_paths++;return;}
  // A plan can still list every certified task while a recoordination has
  // broken its pickup/harvest-to-service causal chain.  Before accepting a new
  // rolling witness, prove the effects from the current public state.
  s.verify_calls++;current_proof=verify(o,base);proof_ready=true;
  if(covers(current_proof.completed,due)){s.latest_cover=base;s.latest_cover_step=o.step;s.fast_paths++;return;}
  have=current_proof.completed;
 }
 s.mismatches++;
 for(auto[key,n]:due)if(have[key]<n){s.last_mismatch_step=o.step;s.last_mismatch_pos=std::get<0>(key);s.last_mismatch_op=std::get<1>(key);s.last_mismatch_item=std::get<2>(key);break;}
 if(s.first_mismatch_step<0)for(auto[key,n]:due)if(have[key]<n){
  s.first_mismatch_step=o.step;s.first_mismatch_pos=std::get<0>(key);
  s.first_mismatch_op=std::get<1>(key);s.first_mismatch_item=std::get<2>(key);
  s.first_mismatch_due=n;s.first_mismatch_have=have[key];break;
 }
 if(s.first_mismatch_step==o.step){
  s.first_mismatch_capture_step=s.captured_step;s.first_mismatch_hour=o.hour;
  for(const auto&p:base)s.first_plan_max=std::max(s.first_plan_max,int(p.a.size()-std::min(p.index,p.a.size())));
  for(const auto&p:s.witness)s.first_witness_max=std::max(s.first_witness_max,int(p.a.size()-std::min(p.index,p.a.size())));
  s.first_shed_wheat=o.priv.shed[W];s.first_min_worker_distance=100000;s.first_min_loaded_distance=100000;
  for(size_t u=0;u<o.priv.inventories.size();u++){int p=cell(u?o.own.hands[u-1]:o.own.farmer),d=dist(p,s.first_mismatch_pos);
   s.first_min_worker_distance=std::min(s.first_min_worker_distance,d);
   if(o.priv.inventories[u][W]>0)s.first_min_loaded_distance=std::min(s.first_min_loaded_distance,d);
  }
  if(s.first_min_loaded_distance==100000)s.first_min_loaded_distance=-1;
 }
 if constexpr(!P16_AFS_WORKFLOW_REPAIR){schedule_urgent(c,o,have,due);s.unresolved+=missing_count(have,due);return;}
 if(!proof_ready){s.verify_calls++;current_proof=verify(o,base);}Proof keep=current_proof;
 bool fixed=restore_witness(c,o,base,due,keep);
 if(!fixed)fixed=restore_chains(c,o,base,due);
 if(fixed){s.witness=base;s.latest_cover=base;s.latest_cover_step=o.step;s.captured_step=o.step;}
 else if(!schedule_urgent(c,o,have,due))s.unresolved+=missing_count(work(base),due);
}

inline std::vector<int> tick_effects(const View&o,const PlayerAction&out){
 std::vector<int> result(out.units.size());fastkag::ObservedDayScenario scenario(o);
 auto before=scenario.project_units(out.units,0);
 for(size_t u=0;u<out.units.size();u++){
  auto after=scenario.project_units(out.units,int(u)+1);
  if(field(out.units[u]))result[u]=effect(before,after,int(u),out.units[u]);
  before=std::move(after);
 }
 return result;
}

inline int action_result(const fastkag::Simulator&before,const fastkag::Simulator&after,int u,const Action&a){
 int pos0=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer);
 int pos1=cell(u?after.farms()[0].hands[u-1]:after.farms()[0].farmer),i=int(a.item);
 if(a.op==Op::PASS)return 1;
 if(Controller::movement(a.op))return pos0!=pos1;
 if(a.op==Op::PICKUP&&i>=0&&i<12)return std::max(0,after.privates()[0].inventories[u][i]-before.privates()[0].inventories[u][i]);
 if(a.op==Op::DROP)return std::max(0,sum(before.privates()[0].inventories[u])-sum(after.privates()[0].inventories[u]));
 if(a.op==Op::PLACE&&i>=0&&i<9)return std::max(0,before.privates()[0].inventories[u][i]-after.privates()[0].inventories[u][i]);
 return field(a)?effect(before,after,u,a):0;
}
inline std::vector<int> tick_results(const View&o,const PlayerAction&out){
 std::vector<int> result(out.units.size());fastkag::ObservedDayScenario scenario(o);
 auto before=scenario.project_units(out.units,0);
 for(size_t u=0;u<out.units.size();u++){
  auto after=scenario.project_units(out.units,int(u)+1);result[u]=action_result(before,after,int(u),out.units[u]);before=std::move(after);
 }
 return result;
}

inline int required_resource_demand(const Controller&c,int unit,int item){
 if(unit<0||unit>=int(c.plans.size()))return 0;int demand=0;Work counted;
 const auto&pl=c.plans[unit];
 for(size_t k=pl.index;k<pl.a.size();k++){
  const auto&a=pl.a[k];int q=0;
  if(a.op==Op::FEED&&item==W)q=1;
  else if(a.op==Op::FERTILIZE&&item==F)q=1;
  else if(a.op==Op::PLACE&&int(a.item)==item&&item>=9)q=a.quantity;
  if(!q||pl.target[k]<0)continue;
  auto key=Controller::task_key(pl.target[k],a);auto it=c.workflow.required.find(key);
  if(it==c.workflow.required.end())continue;
  auto done=c.workflow.completed.find(key);int completed=done==c.workflow.completed.end()?0:done->second;
  int outstanding=std::max(0,it->second-completed);
  int accepted=std::min(q,std::max(0,outstanding-counted[key]));demand+=accepted;counted[key]+=accepted;
 }
 return demand;
}

// A material pickup is part of the certified work chain, not an optional
// economic action, when this worker's remaining certified tasks cannot be
// supplied by its current backpack without it.  Later helpers are allowed to
// replace movement/PASS outputs, so restore the exact scheduled pickup before
// the shortage turns into a missing FEED/FERTILIZE/PLACE on the next tick.
inline void protect_required_inputs(Controller&c,const View&o,const Acts&scheduled,PlayerAction&out){
 auto&s=c.workflow;
 for(size_t u=0;u<scheduled.size();u++){
  const auto&a=scheduled[u];if(a.op!=Op::PICKUP||Controller::same_action(a,out.units[u]))continue;
  int item=int(a.item);if(item<0||item>=12||a.quantity<=0)continue;
  int demand=required_resource_demand(c,int(u),item),held=o.priv.inventories[u][item];if(demand<=held)continue;
  // Do not discard another successful field/material effect owned by this
  // unit.  Movement and PASS are safe because the compiled route expected the
  // pickup on this tick and will rebuild movement from the next observation.
  if(!Controller::movement(out.units[u].op)&&out.units[u].op!=Op::PASS){s.required_input_restore_rejected++;continue;}
  s.required_input_overrides_detected++;auto before=tick_results(o,out);auto candidate=out;candidate.units[u]=a;auto after=tick_results(o,candidate);
  bool safe=after[u]>=std::min(a.quantity,std::max(0,demand-held));
  for(size_t v=0;v<out.units.size()&&safe;v++)if(v!=u&&before[v]>0&&after[v]<before[v])safe=false;
  if(!safe){s.required_input_restore_rejected++;continue;}
  if constexpr(P16_AFS_WORKFLOW_REPAIR){out=std::move(candidate);s.required_input_restores++;s.output_restores++;}
 }
}

inline bool retain_skipped(Controller&c,const View&o,int unit,const Action&a,int target){
 if constexpr(!enabled)return false;
 auto&s=c.workflow;if(!s.captured||target<0||!field(a))return false;auto key=Controller::task_key(target,a);
 auto it=s.required.find(key);if(it==s.required.end()||s.completed[key]>=it->second)return false;
 s.required_skips_detected++;
 if(s.first_skip_step<0){s.first_skip_step=o.step;s.first_skip_unit=unit;s.first_skip_pos=target;s.first_skip_op=int(a.op);s.first_skip_item=int(a.item);s.first_skip_wheat=o.priv.inventories[unit][W];s.first_skip_fertilizer=o.priv.inventories[unit][F];}
 return false;
}

// Later execution helpers may replace a certified route atom after its plan
// index has already advanced.  Restore only that exact, already-scheduled atom
// and only when an authoritative one-tick projection proves it succeeds and
// preserves every other successful field effect in the final joint action.
inline void protect(Controller&c,const View&o,const Acts&scheduled,PlayerAction&out){
 if constexpr(!enabled)return;
 if(!c.workflow_runtime)return;
 auto&s=c.workflow;if(!s.captured||scheduled.size()!=out.units.size())return;
 protect_required_inputs(c,o,scheduled,out);
 if(s.urgent_step==o.step&&s.urgent_unit>=0&&s.urgent_unit<int(out.units.size())){
  s.output_overrides_detected++;auto before=tick_effects(o,out);auto candidate=out;candidate.units[s.urgent_unit]=s.urgent_action;auto after=tick_effects(o,candidate);
  bool safe=after[s.urgent_unit]>0;
  for(size_t v=0;v<out.units.size()&&safe;v++)if(int(v)!=s.urgent_unit&&before[v]>0&&field(out.units[v])&&after[v]<=0)safe=false;
  if(!safe)s.output_restore_rejected++;
  else if constexpr(P16_AFS_WORKFLOW_REPAIR){out=std::move(candidate);s.output_restores++;}
 }
 for(size_t u=0;u<scheduled.size();u++){
  const auto&planned=scheduled[u];if(!field(planned)||Controller::same_action(planned,out.units[u]))continue;
  int pos=cell(u?o.own.hands[u-1]:o.own.farmer);auto key=Controller::task_key(pos,planned);
  auto need=s.required.find(key);if(need==s.required.end()||s.completed[key]>=need->second)continue;
  s.output_overrides_detected++;auto before=tick_effects(o,out);auto candidate=out;candidate.units[u]=planned;auto after=tick_effects(o,candidate);
  bool safe=after[u]>0;
  for(size_t v=0;v<out.units.size()&&safe;v++)if(v!=u&&before[v]>0&&field(out.units[v])&&after[v]<=0)safe=false;
  if(!safe){s.output_restore_rejected++;continue;}
  if constexpr(P16_AFS_WORKFLOW_REPAIR){out=std::move(candidate);s.output_restores++;}
 }
}

inline void account(Controller&c,const View&o,const PlayerAction&out){
 if constexpr(!enabled)return;
 if(!c.workflow_runtime)return;
 auto&s=c.workflow;reset_day(s,o.day);if(!s.captured||out.units.size()!=o.priv.inventories.size())return;
 fastkag::ObservedDayScenario scenario(o);auto before=scenario.project_units(out.units,0);
 for(size_t u=0;u<out.units.size();u++){
  auto a=out.units[u];auto after=scenario.project_units(out.units,int(u)+1);
  if(field(a)){
   int pos=cell(u?before.farms()[0].hands[u-1]:before.farms()[0].farmer),n=effect(before,after,int(u),a);
   auto key=Controller::task_key(pos,a);auto it=s.required.find(key);
   if(n>0&&it!=s.required.end()&&s.completed[key]<it->second){s.completed[key]++;s.accounted++;}
  }
  before=std::move(after);
 }
 // Roll an exact witness forward only when the action actually issued for every
 // unit is its certified first atom and the observed-state projection proves
 // that atom succeeded.  Otherwise invalidate the shortcut; inspect() will
 // replay-verify or restore from the last day-start witness on the next step.
 if(s.latest_cover.size()==out.units.size()){
  auto results=tick_results(o,out);bool exact=true;Plans next=s.latest_cover;
  for(size_t u=0;u<next.size()&&exact;u++){
   size_t k=std::min(next[u].index,next[u].a.size());
   if(k==next[u].a.size()){exact=out.units[u].op==Op::PASS;continue;}
   const auto&expected=next[u].a[k];
   if(!Controller::same_action(expected,out.units[u])){exact=false;break;}
   int need=expected.op==Op::PICKUP?expected.quantity:1;
   if(results[u]<need){exact=false;break;}
   next[u].index=k+1;
  }
  if(exact){s.latest_cover=std::move(next);s.latest_cover_step=o.step+1;}
  else{s.latest_cover.clear();s.latest_cover_step=-1;}
 }
 if(o.step==718)finalize(s);
}
}
