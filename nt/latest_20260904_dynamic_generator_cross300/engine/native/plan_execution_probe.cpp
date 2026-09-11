// Read-only diagnostics. Not imported by the submitted policy.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
PYBIND11_MODULE(_dp7_plan_probe,m){
 m.def("inspect",[](const dp7::Controller&c){
  py::dict result;py::list plans;
  for(const auto&p:c.plans){py::list actions;for(size_t k=p.index;k<p.a.size();k++)
   actions.append(py::make_tuple(int(p.a[k].op),int(p.a[k].item),p.a[k].quantity,p.target[k]));plans.append(actions);}
  result["plans"]=plans;result["actual_drop"]=c.actual_drop;result["resource_degraded"]=c.resource_degraded;return result;
 });
}
