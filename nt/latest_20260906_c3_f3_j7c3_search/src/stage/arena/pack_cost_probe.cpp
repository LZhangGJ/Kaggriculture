// Read-only profile of a copied controller at a REAL decision, not an oracle.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace dp7;
namespace {
constexpr const char*ops[]{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
py::list action_json(const Action&a){py::list out;out.append(ops[int(a.op)]);if(a.item!=Item::NONE)out.append(item_name(int(a.item)));if(a.op==Op::PICKUP||a.op==Op::PLACE||a.op==Op::BUY_SEED||a.op==Op::BUY_PRODUCT||a.op==Op::BUY_ANIMAL||a.op==Op::SELL)out.append(a.quantity);return out;}
py::dict actions(const PlayerAction&a){py::dict out;out["farmer"]=action_json(a.units.at(0));py::list hands,market;for(size_t u=1;u<a.units.size();u++)hands.append(action_json(a.units[u]));for(auto m:a.market)market.append(action_json(m));out["hands"]=hands;out["market"]=market;return out;}
}
PYBIND11_MODULE(_dp7_packcost,m){
 m.def("inspect",[](const Controller&source,const Simulator&env,int seat){
  Controller c=source;c.admission_inspection=nullptr;
  const View o{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],env.market(),env.shops()};
  packprofile::Stats stats;PlayerAction result;auto start=std::chrono::steady_clock::now();
  {packprofile::Active profile(stats);result=c.act(o);}
  double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
  py::dict out,groups;for(const auto&[name,g]:stats.groups){py::dict d;d["calls"]=g.calls;d["repeated_inputs"]=g.repeated;d["pack_compute_seconds"]=g.compute_seconds;groups[py::str(name)]=d;}
  out["groups"]=groups;out["distinct_inputs"]=stats.seen.size();out["key_ints"]=stats.key_ints;out["uncatalogued"]=stats.uncatalogued;
  out["instrumented_seconds"]=elapsed;out["action"]=actions(result);return out;
 });
}
