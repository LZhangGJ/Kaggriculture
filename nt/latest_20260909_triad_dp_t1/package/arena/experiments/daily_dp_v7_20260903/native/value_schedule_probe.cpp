// Read-only diagnostic. Never called by the production controller.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace dp7;
namespace {
View public_view(const Simulator&e,int seat){return {e.step_count(),e.day(),e.hour(),e.farms()[seat],e.farms()[1-seat],e.privates()[seat],e.market(),e.shops()};}
py::dict value(const dayvalue::Value&v){py::dict d;d["score"]=v.score;d["known"]=v.known;d["trade"]=v.trade;d["seeds"]=v.seeds;d["wages"]=v.wages;d["funding_gap"]=v.funding_gap;d["calendars"]=v.calendars;return d;}
py::dict assets(const Farm&f,const PrivateState&p,const Market&m){
 py::dict d;std::array<int,8>scale{};Counts field{};
 for(auto&t:f.tiles){if(plant(t)){scale[int(t.crop)]++;field[int(t.crop)]+=t.yield_units;}else if(animal(t)){scale[5+int(t.animal)-9]++;field[product[int(t.animal)-9]]+=t.yield_units;field[F]+=t.fertilizer_available;}}
 d["cash"]=f.money;d["scale"]=scale;d["field"]=field;d["shed"]=p.shed;d["seeds"]=p.seeds;d["market"]=m.inventory;return d;
}
py::dict endpoint(const Controller&source,const View&o,const std::vector<Plan>&plans){
 ObservedDayScenario model(o);auto c=source;c.admission_inspection=nullptr;c.plans=plans;c.phase=3;c.day=o.day;c.last_step=o.step-1;
 c.p.day_consequence_compare=false;c.p.stepwise_recoordination=false;c.p.resource_aware_exchange=false;
 while(!model.finished())model.advance(c.act(model.view()));
 auto r=assets(model.own(),model.inventory(),model.market());r["step"]=model.step_count();return r;
}
}
PYBIND11_MODULE(_dp7_valueprobe,m){
 m.def("assets",[](const Simulator&e,int seat){return assets(e.farms()[seat],e.privates()[seat],e.market());});
 m.def("inspect",[](const Controller&original,const Simulator&e,int seat){
  auto o=public_view(e,seat);py::dict r;r["checked"]=false;
  if(!original.p.day_consequence_compare||!original.p.stepwise_recoordination||o.day>=29||original.day!=o.day||original.phase!=3||o.step<=original.last_step||original.plans.size()!=o.priv.inventories.size())return r;
  auto initial=original;initial.admission_inspection=nullptr;initial.last_step=o.step;
  if(initial.recover(o)||initial.reconcile_seeds(o))return r;
  intraday::complete_pending(initial,o);
  std::vector<Plan>keep;int stale=0;
  for(size_t u=0;u<initial.plans.size();u++)keep.push_back(initial.repair_plan(o,u,initial.plans[u],stale));
  auto next=initial;next.p.day_consequence_compare=false;next.recoordinate(o);
  auto chosen=initial;chosen.recoordinate(o);
  if(chosen.day_value_checks==initial.day_value_checks)return r;
  auto a=dayvalue::schedule_value(initial,o,keep),b=dayvalue::schedule_value(initial,o,next.plans);
  bool reject=a.known&&b.known&&b.score<a.score-1e-6;
  if((chosen.day_value_rejected>initial.day_value_rejected)!=reject)throw std::logic_error("diagnostic differs from live selector");
  r["checked"]=true;r["step"]=o.step;r["day"]=o.day;r["hour"]=o.hour;r["rejected"]=reject;r["unknown"]=!a.known||!b.known;
  r["keep"]=value(a);r["proposal"]=value(b);r["keep_endpoint"]=endpoint(initial,o,keep);r["proposal_endpoint"]=endpoint(initial,o,next.plans);
  return r;
 });
}
