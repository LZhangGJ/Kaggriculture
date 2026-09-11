#pragma once
// Diagnostic copy of the current recoordination checkpoint. Never called by act.
#include "policy.hpp"
namespace dp7audit {
struct RecoordOpportunity {int pos=-1,op=-1,unit=-1,finish=0,extra=0;double nominal_value=0;};
struct RecoordInspection {
 bool eligible=false,legacy_accepts=false;int movable=0,covered=0,assigned=0,remaining=0;
 bool insertion_improves=false;int insertion_assigned=0,peak_saved=0,total_saved=0;
 std::vector<RecoordOpportunity>extra;
};
inline RecoordInspection inspect_recoord(const dp7::Controller&original,const dp7::View&o){
 using namespace dp7;RecoordInspection result;
 if(!original.p.stepwise_recoordination||!original.p.shared_task_atoms_v2||o.day>=29||
    original.day!=o.day||original.phase!=3||o.step<=original.last_step)return result;
 auto c=original;c.last_step=o.step;
 if(c.recover(o)||c.reconcile_seeds(o))return result;
 intraday::complete_pending(c,o);
 if(c.plans.size()!=o.priv.inventories.size())return result;
 result.eligible=true;const int limit=24-o.hour;result.remaining=limit;
 std::vector<Plan>baseline(c.plans.size());int stale=0;
 for(size_t u=0;u<c.plans.size();u++)baseline[u]=c.repair_plan(o,int(u),c.plans[u],stale);
 auto fresh=c.jobs(o);
 std::stable_sort(fresh.begin(),fresh.end(),[](const Job&a,const Job&b){return std::tuple(a.priority,snake(a.pos))<std::tuple(b.priority,snake(b.pos));});
 std::vector<Job>movable;std::set<Controller::TaskKey>keys,covered;
 for(const auto&j:fresh)if(Controller::resource_free_service(j)){
  movable.push_back(j);for(auto a:j.actions)keys.insert(Controller::task_key(j.pos,a));
 }
 for(const auto&pl:baseline)for(size_t k=0;k<pl.a.size();k++)
  if(pl.target[k]>=0&&keys.count(Controller::task_key(pl.target[k],pl.a[k])))covered.insert(Controller::task_key(pl.target[k],pl.a[k]));
 result.movable=int(movable.size());result.covered=int(covered.size());
 auto greedy=baseline;
 for(size_t u=0;u<greedy.size();u++)greedy[u]=c.strip_tasks(o,int(u),greedy[u],keys);
 int assigned=0;
 for(const auto&j:movable){
  int best=-1,total_best=100000;
  for(size_t u=0;u<greedy.size();u++){
   int total=int(greedy[u].a.size())+dist(Controller::plan_end(o,int(u),greedy[u]),j.pos)+int(j.actions.size());
   if(total<=limit&&std::tuple(total,int(u))<std::tuple(total_best,best<0?100000:best)){best=int(u);total_best=total;}
  }
  if(best<0)break;
  int pos=Controller::plan_end(o,best,greedy[best]);Controller::walk(greedy[best],pos,j.pos);
  for(auto a:j.actions){greedy[best].a.push_back(a);greedy[best].target.push_back(j.pos);}assigned++;
 }
 result.assigned=assigned;
 result.legacy_accepts=assigned==int(movable.size())&&
  (covered.size()<keys.size()||Controller::plan_load(greedy)<Controller::plan_load(baseline));
 // Another diagnostic: insert shared jobs at any point while preserving the
 // relative order of every remaining resource-bearing action on each unit.
 auto inserted=c.insert_shared_services(o,baseline,movable,keys,limit,result.insertion_assigned);
 if(result.insertion_assigned==int(movable.size())){
  auto old=Controller::plan_load(baseline),updated=Controller::plan_load(inserted);
  result.insertion_improves=covered.size()<keys.size()||updated<old;
  result.peak_saved=old.first-updated.first;result.total_saved=old.second-updated.second;
 }
 if(result.legacy_accepts)return result;
 // Preserve every original action and append only currently missing work.
 // This is feasibility evidence, not a profit claim or policy candidate.
 auto with_extra=baseline;
 for(const auto&j:movable){
  Acts missing;for(auto a:j.actions)if(!covered.count(Controller::task_key(j.pos,a)))missing.push_back(a);
  if(missing.empty())continue;
  int best=-1,best_total=100000,best_extra=0;
  for(size_t u=0;u<with_extra.size();u++){
   int extra=dist(Controller::plan_end(o,int(u),with_extra[u]),j.pos)+int(missing.size());
   int total=int(with_extra[u].a.size())+extra;
   if(total<=limit&&std::tuple(total,int(u))<std::tuple(best_total,best<0?100000:best)){
    best=int(u);best_total=total;best_extra=extra;
   }
  }
  if(best<0)continue;
  int pos=Controller::plan_end(o,best,with_extra[best]);Controller::walk(with_extra[best],pos,j.pos);
  for(auto a:missing){
   const auto&t=o.own.tiles[j.pos];double value=0;
   if(a.op==Op::HARVEST){int i=animal(t)?product[int(t.animal)-9]:int(t.crop);if(i>=0&&i<9)value=t.yield_units*o.market.prices[i];}
   if(a.op==Op::COLLECT_FERTILIZER)value=o.market.prices[F];
   result.extra.push_back({j.pos,int(a.op),best,best_total,best_extra,value});
   with_extra[best].a.push_back(a);with_extra[best].target.push_back(j.pos);covered.insert(Controller::task_key(j.pos,a));
  }
 }
 return result;
}
}
