#pragma once
// First-compile alternatives, separate from the macro crop/animal selector.
namespace dp7::compilechoice {
struct Candidate {Controller ctl;std::string kind;};
inline bool same(const Controller&a,const Controller&b){
 if(a.plant_not_before!=b.plant_not_before||a.plans.size()!=b.plans.size()||a.expected!=b.expected)return false;
 for(size_t u=0;u<a.plans.size();u++)if(!Controller::same_remaining(a.plans[u],b.plans[u]))return false;
 return true;
}
inline std::vector<Candidate> generate(const Controller&source,const View&o){
 std::vector<Candidate>out{{source,"keep"}};
 auto add=[&](int order,bool combine,const std::vector<int>&wait,std::string label){
  auto c=source;c.admission_inspection=nullptr;
  // A reversible project timing intent, not a worker-owned linear task.
  for(int pos:wait)c.plant_not_before[pos]=std::max(c.plant_not_before[pos],o.day+1);
  if(combine)c.p.shared_task_atoms_v2=c.p.split_service_jobs=false;
  c.compile_base(o,order);c.p=source.p;
  for(const auto&x:out)if(same(x.ctl,c))return;
  out.push_back({std::move(c),std::move(label)});
 };
 for(int mode=1;mode<=4;mode++)add(mode,false,{},"ordering_"+std::to_string(mode));
 if(source.p.shared_task_atoms_v2||source.p.split_service_jobs)add(0,true,{},"co_located");
 if(source.p.compile_replant_choices){
  std::array<std::vector<int>,5>groups;std::vector<int>all;
  for(const auto&j:source.jobs(o))if(plant(o.own.tiles[j.pos])&&std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::PLANT;})){
   int k=int(o.own.tiles[j.pos].crop);groups[k].push_back(j.pos);all.push_back(j.pos);
  }
  for(int k=0;k<5;k++)if(!groups[k].empty())add(0,false,groups[k],"defer_replant_"+std::to_string(k));
  if(std::count_if(groups.begin(),groups.end(),[](const auto&x){return !x.empty();})>1)add(0,false,all,"defer_all_replants");
 }
 return out;
}
struct Value {double score=0;bool known=true;std::array<bool,100>lost{};int ticks=0;};
inline Value evaluate(const Controller&source,const View&o,bool residual){
 auto c=source;c.admission_inspection=nullptr;
 // Never recursively enumerate this comparison inside its own scenario or
 // the existing optional next-day scheduling forecast.
 c.p.compile_consequence=c.p.compile_replant_choices=false;
 // Explicit approximate leaf policy: preserve today's rolling coordination
 // and its value veto, but avoid nesting a second macro-planning day inside
 // every simulated recoordination. The real controller keeps its own flags.
 if(c.p.compile_bounded_rollout)c.p.day_value_replan_next_day=false;
 c.p.portfolio_rotation=false;c.last_step=o.step-1;c.resume_compiled_tick=true;
 ObservedDayScenario s(o);while(!s.finished())s.advance(c.act(s.view()));
 Value out;out.ticks=s.ticks();out.score=s.own().money;
 for(int pos=0;pos<100;pos++){
  const auto&a=o.own.tiles[pos];const auto&b=s.own().tiles[pos];
  out.lost[pos]=(animal(a)&&!animal(b))||(plant(a)&&!plant(b)&&b.kind==TileKind::WEED);
 }
 if(residual&&s.step_count()<719){
  View end{s.step_count(),s.step_count()/24,0,s.own(),o.opponent,s.inventory(),s.market(),s.shops()};
  auto v=dayvalue::residual(c,end);out.score=v.score;out.known=v.known;
 }
 return out;
}
inline double static_value(const Controller&c,const View&o){
 // Legacy task-value ablation: count scheduled semantic effects, not the
 // number of groups (which changes when same-site jobs are merged/split).
 std::multiset<Controller::TaskKey>available;
 for(const auto&pl:c.plans)for(size_t k=pl.index;k<pl.a.size();k++)if(pl.target[k]>=0&&!Controller::movement(pl.a[k].op))available.insert(Controller::task_key(pl.target[k],pl.a[k]));
 double score=0;
 for(const auto&j:c.jobs(o)){
  auto remaining=available;bool complete=true;
  for(auto a:j.actions){auto it=remaining.find(Controller::task_key(j.pos,a));if(it==remaining.end()){complete=false;break;}remaining.erase(it);}
  if(complete){available=std::move(remaining);score+=c.job_consequence(o,j);}
 }
 return score;
}
inline void compare(Controller&source,const View&o){
 if((!source.p.compile_consequence&&!source.p.compile_replant_choices)||o.day>=29)return;
 source.compile_choice_calls++;
 // A full day's pack cache may be saturated by prior investment forecasts.
 // This comparison has an independent exact-key working set. It must not
 // evict or modify the parent cache, and cannot change scheduling semantics.
 auto cache=source.p.exact_schedule_cache?std::make_unique<packmemo::Cache>():nullptr;
 packmemo::Scope scope(cache?cache.get():packmemo::active);
 auto candidates=generate(source,o);source.compile_choice_candidates+=int(candidates.size());if(candidates.size()<2)return;
 auto keep=evaluate(candidates[0].ctl,o,source.p.compile_consequence);source.compile_choice_evaluations++;
 if(!keep.known){source.compile_choice_unknown++;return;}
 double best=source.p.compile_consequence?keep.score:static_value(candidates[0].ctl,o);size_t selected=0;
 for(size_t i=1;i<candidates.size();i++){
  auto v=evaluate(candidates[i].ctl,o,source.p.compile_consequence);source.compile_choice_evaluations++;
  if(!v.known){source.compile_choice_unknown++;continue;}
  bool safe=true;for(int pos=0;pos<100;pos++)if(v.lost[pos]&&!keep.lost[pos])safe=false;
  if(!safe){source.compile_choice_unsafe++;continue;}
  double score=source.p.compile_consequence?v.score:static_value(candidates[i].ctl,o);
  if(score>best+1e-6){best=score;selected=i;}
 }
 if(selected){
  auto&c=candidates[selected].ctl;source.plans=std::move(c.plans);source.expected=c.expected;
  source.actual_drop=c.actual_drop;source.resource_degraded=c.resource_degraded;source.unresolved_overflow=c.unresolved_overflow;
  source.plant_not_before=c.plant_not_before;source.compile_choice_changes++;
 }
}
}
