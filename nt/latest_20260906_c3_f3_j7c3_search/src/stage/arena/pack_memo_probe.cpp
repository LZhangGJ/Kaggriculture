// Prototype against the frozen policy layout; no production source is edited.
#include "policy.hpp"
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace dp7;
namespace {
constexpr const char*ops[]{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
py::list action_json(const Action&a){py::list x;x.append(ops[int(a.op)]);if(a.item!=Item::NONE){x.append(item_name(int(a.item)));if(a.op==Op::PLACE||a.op==Op::PICKUP||a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL||a.op==Op::BUY_PRODUCT||a.op==Op::SELL)x.append(a.quantity);}else if(a.quantity!=1)x.append(a.quantity);return x;}
py::dict actions(const PlayerAction&a){py::dict out;out["farmer"]=action_json(a.units.at(0));py::list hands,market;for(size_t u=1;u<a.units.size();u++)hands.append(action_json(a.units[u]));for(auto m:a.market)market.append(action_json(m));out["hands"]=hands;out["market"]=market;return out;}
}
PYBIND11_MODULE(_dp7_packmemo,m){
 m.def("act",[](Controller&c,const Simulator&env,int seat,bool enabled){
  const View o{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
  packmemo::Cache cache;PlayerAction result;auto start=std::chrono::steady_clock::now();
  {packmemo::Scope scope(enabled?&cache:nullptr);result=c.act(o);}
  py::dict out;out["seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  out["action"]=actions(result);out["calls"]=cache.calls;out["hits"]=cache.hits;out["misses"]=cache.misses;
  out["distinct"]=cache.entries.size();out["not_stored"]=cache.not_stored;out["accounted_bytes"]=cache.accounted_bytes;
  out["logical_regret_trials"]=c.regret_trials;out["logical_regret_improvements"]=c.regret_improvements;return out;
 },py::arg("controller"),py::arg("env"),py::arg("seat"),py::arg("enabled")=true);
}
