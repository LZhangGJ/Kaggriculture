// Read-only probe of the current public-state market recommendation.
// No counterfactual continuation, no opponent-private access.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
PYBIND11_MODULE(_dp7_market_probe,m){
 m.def("inspect",[](const dp7::Controller&c,const fastkag::Simulator&e,int seat){
  if(seat<0||seat>1)throw std::invalid_argument("seat");
  dp7::View v{e.step_count(),e.day(),e.hour(),e.farms()[seat],e.farms()[1-seat],e.privates()[seat],e.market(),e.shops()};
  py::dict d;d["phase"]=c.phase;d["tactical"]=c.competition_sales(v);
  d["holding_demand"]=c.holding_demand(v);d["projected_opponent_supply"]=c.opponent_supply(v);
  return d;
 });
}
