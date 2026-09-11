// Exclusion reasons for live candidates; diagnostic only.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
PYBIND11_MODULE(_dp7_feed_probe,m){m.def("inspect",[](const dp7::Controller&c,const fastkag::Simulator&e,int seat){
 using namespace dp7;View o{e.step_count(),e.day(),e.hour(),e.farms()[seat],e.farms()[1-seat],e.privates()[seat],e.market(),e.shops()};
 py::list rows;std::vector<Plan>base;for(int u=0;u<int(c.plans.size());u++)base.push_back(service::remaining(c,u));
 auto order=exchange::order(base);auto reserved=intraday::reserved(c);
 for(auto n:service::missing(c,o))if(n.op==Op::FEED){
  int examined=0,stock=0,late=0,wheat=0,deadline=0,causal=0,accepted=0;py::list feasible;
  for(int u=0;u<int(base.size());u++)for(int at=0;at<=int(Controller::plan_semantics(base[u]).size());at++)for(bool pickup:{false,true}){
   examined++;if(pickup&&o.priv.shed[W]<=reserved.shed[W]){stock++;continue;}
   auto trial=liverepair::inserted(o,u,base[u],at,n.pos,pickup);
   if(o.hour+int(trial.a.size())>24){late++;continue;}
   if(!liverepair::wheat_feasible(o,u,trial)){wheat++;continue;}
   auto all=base;all[u]=trial;if(!exchange::deadlines(o,all)){deadline++;continue;}
   for(size_t k=0;k<all[u].a.size();k++)if(all[u].target[k]==n.pos&&all[u].a[k].op==Op::FEED)all[u].target[k]=-1;
   if(exchange::order(all)!=order){causal++;continue;}
   accepted++;feasible.append(py::make_tuple(u,at,pickup,trial.a.size()));
  }
  py::dict r;r["pos"]=n.pos;r["trials"]=examined;r["stock"]=stock;r["late"]=late;r["wheat"]=wheat;r["other_deadline"]=deadline;r["causal"]=causal;r["accepted"]=accepted;r["feasible"]=feasible;rows.append(r);
 }
 py::dict result;result["step"]=o.step;result["shed_wheat"]=o.priv.shed[W];result["reserved_wheat"]=reserved.shed[W];result["missing"]=rows;return result;
});}
