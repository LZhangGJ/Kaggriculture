#include "policy.hpp"
#include "compile_static_prune_prototype.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
namespace py=pybind11;using namespace dp7;
using Clock=std::chrono::steady_clock;
bool same_player(const PlayerAction&a,const PlayerAction&b){
 if(a.units.size()!=b.units.size()||a.market.size()!=b.market.size())return false;
 for(size_t i=0;i<a.units.size();i++)if(!Controller::same_action(a.units[i],b.units[i]))return false;
 for(size_t i=0;i<a.market.size();i++)if(!Controller::same_action(a.market[i],b.market[i]))return false;return true;
}
PYBIND11_MODULE(_dp7_static_prune_probe,m){
 m.def("inspect",[](const Controller&source,const Simulator&env,int seat){
  View o{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
  py::dict result;result["eligible"]=false;
  if(source.phase!=2||source.day!=o.day||o.day>=29||source.p.compile_consequence||!source.p.compile_replant_choices)return result;
  auto c=source;c.admission_inspection=nullptr;c.last_step=o.step;
  auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(source.p.exact_schedule_cache?cache.get():nullptr);
  // Match the actual entry point. Recovery may postpone compilation entirely.
  if(c.recover(o)||c.reconcile_seeds(o))return result;
  intraday::complete_pending(c,o);if(c.phase==3)c.recoordinate(o);
  service::recover(c,o);exchange::apply(c,o);c.dispatch_idle(o);c.dispatch_midroute(o);
  if(c.phase!=2)return result;
  c.compile_base(o);auto a=c,b=c;
  auto t=Clock::now();compilechoice::compare(a,o);result["original_seconds"]=std::chrono::duration<double>(Clock::now()-t).count();
  t=Clock::now();compileprune::compare(b,o);result["pruned_seconds"]=std::chrono::duration<double>(Clock::now()-t).count();
  if(!compilechoice::same(a,b)||a.actual_drop!=b.actual_drop||a.resource_degraded!=b.resource_degraded||a.unresolved_overflow!=b.unresolved_overflow||a.compile_choice_changes!=b.compile_choice_changes)throw std::logic_error("pruning changed live-state selection");
  result["original_evaluations"]=a.compile_choice_evaluations-c.compile_choice_evaluations;
  result["pruned_evaluations"]=b.compile_choice_evaluations-c.compile_choice_evaluations;
  result["candidate_count"]=a.compile_choice_candidates-c.compile_choice_candidates;
  result["changed"]=a.compile_choice_changes>c.compile_choice_changes;
  // Also verify this hand-written diagnostic entry reproduces the actual
  // current action, including ordered market transactions and unit slots.
  auto ref=source;ref.admission_inspection=nullptr;auto actual=ref.act(o);
  b.resume_compiled_tick=true;b.last_step=o.step-1;auto predicted=b.act(o);
  if(!same_player(actual,predicted))throw std::logic_error("diagnostic entry does not match real first-compile action");
  result["eligible"]=true;result["same_action_and_selection"]=true;return result;
 });
}
