// Isolated alternatives, no policy hook and no opponent hidden/future state.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
namespace py=pybind11;
using namespace dp7;
struct Trial{std::string name;Controller c;std::vector<int>deferred;};
static py::dict forecast(Controller c,const View&o){
 // Preserve real per-step recoordination/value veto. Stop only recursive
 // rotation search at an optional following-day boundary.
 c.schedule_cache=std::make_shared<packmemo::Cache>();c.schedule_cache_day=o.day;
 c.p.portfolio_rotation=false;c.admission_inspection=nullptr;
 c.last_step=o.step-1;ObservedDayScenario today(o);
 while(!today.finished())today.advance(c.act(today.view()));
 int missing=0,escaped=0;py::list omitted;
 for(int p=0;p<100;p++)if(animal(o.own.tiles[p])){
  auto&t=today.own().tiles[p];escaped+=!animal(t);
  // Scenario endpoint is after daily fed flags reset; the counter reflects
  // failure to feed, not the reset flag itself.
  if(animal(t)&&t.consecutive_unfed){missing++;omitted.append(p);}
 }
 py::dict out;out["cash_today"]=today.own().money;out["missed_feed_today"]=missing;out["escaped_today"]=escaped;out["missed_feed_positions"]=omitted;
 View end{today.step_count(),today.step_count()/24,0,today.own(),o.opponent,today.inventory(),today.market(),today.shops()};
 auto v=dayvalue::residual(c,end);out["residual_known"]=v.known;out["residual_score"]=v.score;out["funding_gap"]=v.funding_gap;
 return out;
}
PYBIND11_MODULE(_dp7_capacity_probe,m){m.def("inspect",[](const Controller&source,const fastkag::Simulator&env,int seat,bool evaluate){
 auto start=std::chrono::steady_clock::now();View o{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
 if(source.phase!=2||source.day!=o.day)throw std::invalid_argument("phase2 expected");
 auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(cache.get());
 std::vector<Trial>trials;
 auto add=[&](std::string name,const std::vector<int>&defer,bool combine){
  auto c=source;c.schedule_cache=cache;c.schedule_cache_day=o.day;c.admission_inspection=nullptr;
  for(int pos:defer)c.plant_not_before[pos]=std::max(c.plant_not_before[pos],o.day+1);
  if(combine)c.p.shared_task_atoms_v2=c.p.split_service_jobs=false;
  c.compile(o);c.p=source.p;trials.push_back({name,std::move(c),defer});
 };
 add("keep",{},false);add("co_located_tasks",{},true);
 std::array<std::vector<int>,5>groups;std::vector<int>all;
 for(const auto&j:source.jobs(o))if(plant(o.own.tiles[j.pos])&&std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::PLANT;})){
  groups[int(o.own.tiles[j.pos].crop)].push_back(j.pos);all.push_back(j.pos);
 }
 for(int k=0;k<5;k++)if(!groups[k].empty()){
  add("defer_replant_"+std::to_string(k),groups[k],false);
  add("co_located_defer_"+std::to_string(k),groups[k],true);
 }
 if(std::count_if(groups.begin(),groups.end(),[](const auto&x){return !x.empty();})>1){add("defer_all_replants",all,false);add("co_located_defer_all",all,true);}
 py::list result;
 for(auto&t:trials){py::dict r;r["name"]=t.name;r["deferred"]=t.deferred;r["dropped"]=t.c.actual_drop;py::list feed,actions;
  for(size_t u=0;u<t.c.plans.size();u++){py::list route;const auto&p=t.c.plans[u];for(size_t k=p.index;k<p.a.size();k++){
   if(p.a[k].op==Op::FEED)feed.append(p.target[k]);route.append(py::make_tuple(int(p.a[k].op),int(p.a[k].item),p.a[k].quantity,p.target[k]));
  }actions.append(route);}
  r["planned_feed"]=feed;r["plans"]=actions;
  if(evaluate)r["conditional"]=forecast(t.c,o);result.append(r);
 }
 py::dict out;out["trials"]=result;out["step"]=o.step;out["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return out;
 },py::arg("controller"),py::arg("env"),py::arg("seat"),py::arg("evaluate")=true);}
