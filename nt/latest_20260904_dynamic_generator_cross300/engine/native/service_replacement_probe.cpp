// Isolated observation-only diagnostic. Production Controller remains unchanged.
#include "service_replacement_prototype.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
namespace py=pybind11;
namespace rp=dp7::replacement_probe;
py::dict value(const dp7::dayvalue::Value&v){py::dict r;r["known"]=v.known;r["score"]=v.score;r["wages"]=v.wages;r["trade"]=v.trade;r["funding_gap"]=v.funding_gap;r["horizon_ticks"]=v.horizon_ticks;return r;}
PYBIND11_MODULE(_dp7_replacement_probe,m){m.def("inspect",[](const dp7::Controller&source,const fastkag::Simulator&e,int seat,bool evaluate){
 using namespace dp7;auto start=std::chrono::steady_clock::now();
 View o{e.step_count(),e.day(),e.hour(),e.farms()[seat],e.farms()[1-seat],e.privates()[seat],e.market(),e.shops()};
 auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(cache.get());auto c=source;
 c.schedule_cache.reset();c.schedule_cache_day=-1;c.admission_inspection=nullptr;
 // Inspect the genuine current public-state compile before any first action.
 // No action is issued; live Controller and environment remain unchanged.
 bool compiled=c.phase==2&&c.day==o.day;if(compiled)c.compile(o);
 auto r=evaluate?rp::score(c,o):rp::generate(c,o);py::dict out;py::list candidates;
 for(size_t i=0;i<r.candidates.size();i++){auto&x=r.candidates[i];py::dict q;q["pos"]=x.pos;q["unit"]=x.unit;q["slot"]=x.slot;q["pickup"]=x.pickup;q["removed_care"]=x.removed_care;q["extra_steps"]=x.extra_steps;q["peak"]=x.peak;
  py::list plans;for(auto&p:x.plans){py::list actions;for(size_t k=0;k<p.a.size();k++)actions.append(py::make_tuple(int(p.a[k].op),int(p.a[k].item),p.a[k].quantity,p.target[k]));plans.append(actions);}q["plans"]=plans;
  if(i<r.values.size())q["value"]=value(r.values[i]);candidates.append(q);
 }
 out["step"]=o.step;out["source_phase"]=source.phase;out["compiled_preview"]=compiled;out["eligible"]=r.eligible;out["trials"]=r.trials;out["feasible"]=r.feasible;out["evaluated"]=r.evaluated;out["unknown"]=r.unknown;out["selected"]=r.selected;out["keep"]=value(r.keep);out["candidates"]=candidates;
 out["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return out;
 });}
