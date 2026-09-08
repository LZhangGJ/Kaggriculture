#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
namespace py=pybind11;using namespace dp7;
using Clock=std::chrono::steady_clock;
double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
PYBIND11_MODULE(_dp7_compile_cost,m){
m.def("newday",[](const Controller&source,const Simulator&env,int seat){
 View o{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
 if(o.hour!=0)throw std::invalid_argument("day boundary expected");
 auto c=source,ref=source;py::dict out;
 {auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(cache.get());auto t=Clock::now();ref.new_day(o);out["reference_seconds"]=seconds(t);}
 auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(cache.get());
 if(c.pending_admission.active)c.intraday_cancelled++;c.pending_admission={};c.intraday_units.clear();c.day=o.day;c.phase=1;c.queue.clear();c.plans.clear();c.resource_degraded=c.actual_drop=0;c.seed_reconcile_checked=false;c.prepared_seed_need={};
 std::optional<Farm>projected;int released=0;if(c.p.plan_zero_expiry){projected=o.own;released=c.project_zero_expiry(*projected,o.step);}c.anticipated_releases+=released;
 View planning{o.step,o.day,o.hour,projected?*projected:o.own,o.opponent,o.priv,o.market,o.shops};
 auto t=Clock::now();c.choose(planning);c.rotate(planning);out["choose_seconds"]=seconds(t);
 if(c.p.portfolio_rotation)for(auto&[pos,kind]:c.target)if(c.deferred_kind[pos]>=0){auto&tile=planning.own.tiles[pos];if(animal(tile)||(plant(tile)&&int(tile.crop)==c.deferred_kind[pos]&&tile.planted_day>=c.deferred_since[pos]&&c.plant_not_before[pos]<=c.day)){c.deferred_kind[pos]=-1;c.plant_not_before[pos]=0;}else kind=c.deferred_kind[pos];}
 t=Clock::now();c.prepare_orders(planning,o,released);out["prepare_seconds"]=seconds(t);
 t=Clock::now();if(c.p.funded_bundle_mode)c.compare_funded_bundles(planning,o,released);out["funded_seconds"]=seconds(t);
 t=Clock::now();if(c.p.joint_investment_portfolio)c.compare_portfolios(planning,o,released);out["portfolio_seconds"]=seconds(t);
 t=Clock::now();if(c.p.portfolio_rotation)c.sync_deferred();if(c.p.portfolio_rotation)c.compare_rotation(planning,o,released);out["rotation_seconds"]=seconds(t);
 if(c.target!=ref.target||c.plant_not_before!=ref.plant_not_before||c.queue.size()!=ref.queue.size()||c.phase!=ref.phase)throw std::logic_error("instrumented newday differs");
 for(size_t i=0;i<c.queue.size();i++)if(!Controller::same_action(c.queue[i],ref.queue[i]))throw std::logic_error("instrumented queue differs");
 out["same_decision"]=true;out["cache_calls"]=cache->calls;out["cache_hits"]=cache->hits;return out;
});
m.def("inspect",[](const Controller&source,const Simulator&env,int seat,bool bounded){
 View o{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
 if(source.phase!=2||source.day!=o.day)throw std::invalid_argument("phase2 expected");
 auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(cache.get());
 auto c=source;c.admission_inspection=nullptr;auto tic=Clock::now();c.compile_base(o);py::dict out;out["base_seconds"]=seconds(tic);
 tic=Clock::now();auto candidates=compilechoice::generate(c,o);out["generation_seconds"]=seconds(tic);py::list rows;
 for(auto&candidate:candidates){auto x=candidate.ctl;x.admission_inspection=nullptr;x.p.compile_consequence=x.p.compile_replant_choices=false;x.p.portfolio_rotation=false;x.last_step=o.step-1;
  // Diagnostic model alternative: keep per-step coordination and today's
  // value veto, but do not recursively expand another macro-planning day.
  if(bounded)x.p.day_value_replan_next_day=false;
  ObservedDayScenario s(o);tic=Clock::now();while(!s.finished())s.advance(x.act(s.view()));
  py::dict r;r["kind"]=candidate.kind;r["rollout_seconds"]=seconds(tic);r["cash_today"]=s.own().money;r["day_value_checks"]=x.day_value_checks-candidate.ctl.day_value_checks;
  r["recoordination_checks"]=x.step_recoord_checks-candidate.ctl.step_recoord_checks;
  View end{s.step_count(),s.step_count()/24,0,s.own(),o.opponent,s.inventory(),s.market(),s.shops()};
  tic=Clock::now();auto v=dayvalue::residual(x,end);r["residual_seconds"]=seconds(tic);r["residual"]=v.score;r["known"]=v.known;rows.append(r);
 }
 out["rows"]=rows;out["cache_calls"]=cache->calls;out["cache_hits"]=cache->hits;out["bounded"]=bounded;return out;
},py::arg("source"),py::arg("env"),py::arg("seat"),py::arg("bounded")=false);}
