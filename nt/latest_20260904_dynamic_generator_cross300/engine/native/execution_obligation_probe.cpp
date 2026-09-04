// Read-only diagnostics, never imported by the policy.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace dp7;
static py::list routes(const Controller&c){
 py::list out;
 for(const auto&p:c.plans){py::list a;
  for(size_t k=p.index;k<p.a.size();k++)a.append(py::make_tuple(int(p.a[k].op),int(p.a[k].item),p.a[k].quantity,p.target[k]));
  out.append(a);
 }return out;
}
static py::list tasks(const std::vector<Job>&js){
 py::list out;for(const auto&j:js){py::dict d;py::list a;
  for(auto x:j.actions)a.append(py::make_tuple(int(x.op),int(x.item),x.quantity));
  d["pos"]=j.pos;d["actions"]=a;d["needs"]=j.needs;d["seeds"]=j.seeds;d["priority"]=j.priority;out.append(d);
 }return out;
}
PYBIND11_MODULE(_dp7_obligation_probe,m){
 m.def("inspect",[](const Controller&source,const fastkag::Simulator&env,int seat,bool preview){
  View o{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
  auto c=source;c.schedule_cache.reset();c.schedule_cache_day=-1;c.admission_inspection=nullptr;
  auto cache=std::make_shared<packmemo::Cache>();packmemo::Scope scope(cache.get());
  py::dict r;r["step"]=o.step;r["day"]=c.day;r["phase"]=c.phase;r["plans"]=routes(c);r["target"]=c.target;
  r["drop"]=c.actual_drop;r["degraded"]=c.resource_degraded;r["day_value_rejected"]=c.day_value_rejected;
  r["feed_stock_target"]=c.feed_stock_target;r["daily_need"]=c.daily_need;r["expected"]=c.expected;
  py::list queue;for(auto x:c.queue)queue.append(py::make_tuple(int(x.op),int(x.item),x.quantity));r["queue"]=queue;
  if(c.day==o.day){auto raw=c.jobs(o);r["jobs"]=tasks(raw);r["reserved"]=tasks(c.reserve(o,raw));}
  if(preview&&source.phase==2&&source.day==o.day){c=source;c.schedule_cache=cache;c.schedule_cache_day=o.day;c.admission_inspection=nullptr;c.compile(o);
   r["compile_plans"]=routes(c);r["compile_drop"]=c.actual_drop;r["compile_degraded"]=c.resource_degraded;
  }
  return r;
 },py::arg("controller"),py::arg("env"),py::arg("seat"),py::arg("preview")=false);
}
