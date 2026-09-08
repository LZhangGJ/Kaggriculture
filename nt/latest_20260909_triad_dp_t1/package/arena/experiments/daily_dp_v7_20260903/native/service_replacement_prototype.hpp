#pragma once
// ISOLATED PROTOTYPE. Not included by production policy.hpp/build_native.py.
// Expand the candidate set; compare conditional cash consequences, not source
// identity, replay coordinates, or a realized future trajectory.
#include "policy.hpp"
namespace dp7::replacement_probe {
struct Candidate {
 int pos=-1,unit=-1,slot=-1;bool pickup=false;
 int removed_care=0,extra_steps=0,peak=0;
 std::vector<Plan>plans;
};
struct Result {
 int eligible=0,trials=0,feasible=0,evaluated=0,unknown=0,selected=-1;
 dayvalue::Value keep;
 std::vector<Candidate>candidates;
 std::vector<dayvalue::Value>values;
};
inline std::vector<Plan> remaining(const Controller&c){
 std::vector<Plan>out;for(int u=0;u<int(c.plans.size());u++)out.push_back(service::remaining(c,u));return out;
}
inline std::vector<Plan> without_care(const View&o,const std::vector<Plan>&base,int pos,int&removed){
 auto result=base;removed=0;
 for(int u=0;u<int(base.size());u++){
  auto groups=exchange::groups(base[u]);
  for(auto&g:groups)if(g.pos==pos){auto before=g.actions.size();
   g.actions.erase(std::remove_if(g.actions.begin(),g.actions.end(),[](auto a){return a.op==Op::CARE;}),g.actions.end());removed+=before-g.actions.size();}
  groups.erase(std::remove_if(groups.begin(),groups.end(),[](auto&g){return g.actions.empty();}),groups.end());
  result[u]=exchange::rebuild(o,u,groups);
 }return result;
}
inline auto key(const Candidate&x){return std::tuple(x.extra_steps,x.peak,x.pos,x.unit,x.slot,x.pickup);}
inline Result generate(const Controller&c,const View&o,int shortlist=4){
 if(shortlist<1||shortlist>32)throw std::invalid_argument("replacement shortlist budget");
 Result out;if(c.phase!=3||o.day>=29||c.plans.size()!=o.priv.inventories.size())return out;
 const auto base=remaining(c);const int old_total=Controller::plan_load(base).second;
 auto reserved=intraday::reserved(c);
 for(auto n:service::missing(c,o)){
  const auto&t=o.own.tiles[n.pos];
  // This is an existing, unfulfilled FEED intent. It is not a rule that every
  // animal must be saved: the score below may reject the investment in feed.
  if(n.op!=Op::FEED||!animal(t)||t.fed_today||t.consecutive_unfed<1)continue;
  int removed=0;auto stripped=without_care(o,base,n.pos,removed);if(!removed)continue;
  out.eligible++;const auto old_order=exchange::order(stripped);
  for(int u=0;u<int(stripped.size());u++)for(int at=0;at<=int(Controller::plan_semantics(stripped[u]).size());at++)for(bool pickup:{false,true}){
   out.trials++;if(pickup&&o.priv.shed[W]<=reserved.shed[W])continue;
   auto trial=liverepair::inserted(o,u,stripped[u],at,n.pos,pickup);
   if(o.hour+int(trial.a.size())>24||!liverepair::wheat_feasible(o,u,trial))continue;
   auto all=stripped;all[u]=std::move(trial);
   if(!exchange::deadlines(o,all))continue;
   auto before_new_feed=all;
   for(size_t k=0;k<before_new_feed[u].a.size();k++)if(before_new_feed[u].target[k]==n.pos&&before_new_feed[u].a[k].op==Op::FEED)before_new_feed[u].target[k]=-1;
   if(exchange::order(before_new_feed)!=old_order)continue;
   out.feasible++;auto load=Controller::plan_load(all);
   Candidate candidate{n.pos,u,at,pickup,removed,load.second-old_total,load.first,std::move(all)};
   // Search budget is a computational bound, never a crop/date policy value.
   out.candidates.push_back(std::move(candidate));
  }
 }
 std::stable_sort(out.candidates.begin(),out.candidates.end(),[](auto&a,auto&b){return key(a)<key(b);});
 std::vector<Candidate>unique;
 for(auto&x:out.candidates){
  bool duplicate=false;
  for(auto&y:unique){bool same=x.plans.size()==y.plans.size();
   for(size_t u=0;same&&u<x.plans.size();u++)same=Controller::same_remaining(x.plans[u],y.plans[u]);
   if(same){duplicate=true;break;}}
  if(!duplicate)unique.push_back(std::move(x));if(int(unique.size())>=shortlist)break;
 }
 out.candidates=std::move(unique);return out;
}
inline int choose(const dayvalue::Value&keep,const std::vector<dayvalue::Value>&values){
 if(!keep.known||!std::isfinite(keep.score))return -1;
 int selected=-1;double best=keep.score;
 for(size_t i=0;i<values.size();i++)if(values[i].known&&std::isfinite(values[i].score)&&values[i].score>best+1e-6){best=values[i].score;selected=int(i);}
 return selected;
}
inline Result score(const Controller&source,const View&o,int shortlist=4){
 auto out=generate(source,o,shortlist);if(out.candidates.empty())return out;
 // New private cache for diagnostics: never mutate the live source's cache.
 auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(cache.get());
 auto c=source;c.schedule_cache.reset();c.schedule_cache_day=-1;c.admission_inspection=nullptr;
 out.keep=dayvalue::schedule_value(c,o,remaining(source));out.evaluated++;
 if(!out.keep.known){out.unknown++;return out;}
 for(size_t i=0;i<out.candidates.size();i++){
  auto value=dayvalue::schedule_value(c,o,out.candidates[i].plans);out.values.push_back(value);out.evaluated++;
  if(!value.known){out.unknown++;continue;}
 }
 out.selected=choose(out.keep,out.values);
 return out;
}
}
