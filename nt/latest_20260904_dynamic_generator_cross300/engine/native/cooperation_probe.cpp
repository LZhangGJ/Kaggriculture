// Offline capability census; never changes the live Controller or Simulator.
#include "policy.hpp"
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace dp7;
namespace {
constexpr const char*names[]{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
Action parse(py::handle h){auto s=py::reinterpret_borrow<py::sequence>(h);Action a;std::string op=py::cast<std::string>(s[0]);for(int i=0;i<24;i++)if(op==names[i])a.op=Op(i);if(s.size()>1&&py::isinstance<py::str>(s[1]))for(int i=0;i<12;i++)if(py::cast<std::string>(s[1])==item_name(i))a.item=Item(i);if(s.size()>2)a.quantity=py::cast<int>(s[2]);return a;}
View view(const Simulator&e,int s){return {e.step_count(),e.day(),e.hour(),e.farms()[s],e.farms()[1-s],e.privates()[s],e.market(),e.shops()};}
int pos(const Simulator&e,int s,int u){return cell(u?e.farms()[s].hands[u-1]:e.farms()[s].farmer);}
bool supported(Op op){return op==Op::PLANT||op==Op::WATER||op==Op::HARVEST||op==Op::FERTILIZE||op==Op::FEED||op==Op::CARE||op==Op::COLLECT_FERTILIZER;}
bool effective(const Simulator&a,const Simulator&b,int s,int u,Action op){
 auto&t=a.farms()[s].tiles[pos(a,s,u)];auto&n=b.farms()[s].tiles[pos(a,s,u)];auto&iv=a.privates()[s].inventories[u];auto&nv=b.privates()[s].inventories[u];
 switch(op.op){case Op::WATER:return plant(t)&&!t.watered_today&&n.watered_today;
 case Op::PLANT:return a.privates()[s].seeds[int(op.item)]>b.privates()[s].seeds[int(op.item)];
 case Op::CARE:return animal(t)&&!t.cared_today&&n.cared_today;
 case Op::FEED:return iv[W]>nv[W]&&n.fed_today;
 case Op::FERTILIZE:return iv[F]>nv[F];
 case Op::HARVEST:case Op::COLLECT_FERTILIZER:return sum(nv)>sum(iv);
 default:return false;}
}
}
PYBIND11_MODULE(_dp7_coopprobe,m){
 m.def("inspect",[](const Controller&c,const Simulator&e,int seat,py::dict packed){
  py::dict result;py::list opportunities;result["opportunities"]=opportunities;
  if(c.phase!=3||c.plans.size()!=e.farms()[seat].hands.size()+1)return result;
  Acts actual{parse(packed["farmer"])};for(auto h:packed["hands"])actual.push_back(parse(h));
  const auto original=view(e,seat);auto jobs=c.jobs(original);
  struct Claim{int owner,delay,pos;Action a;};std::map<Controller::TaskKey,Claim>claims;
  for(size_t u=0;u<c.plans.size();u++){const auto&pl=c.plans[u];for(size_t k=pl.index;k<pl.a.size();k++)if(pl.target[k]>=0&&supported(pl.a[k].op)){
    auto key=Controller::task_key(pl.target[k],pl.a[k]);auto it=claims.find(key);int delay=int(k-pl.index)+1;
    if(it==claims.end()||delay<it->second.delay)claims[key]={int(u),delay,pl.target[k],pl.a[k]};
  }}
  for(const auto&j:jobs)for(auto a:j.actions)if(supported(a.op))claims.try_emplace(Controller::task_key(j.pos,a),Claim{-1,-1,j.pos,a});
  int locked=0;for(const auto&j:jobs)if(j.actions.size()>1&&std::any_of(j.actions.begin(),j.actions.end(),[](Action a){return a.op==Op::PLANT;}))locked++;
  result["locked_crop_jobs"]=locked;
  for(size_t u=0;u<actual.size();u++)if(actual[u].op==Op::PASS){
   auto prefix=e.project_unit_phase(seat,actual,int(u));auto pv=view(prefix,seat);int tile=pos(e,seat,int(u));
   for(auto&[key,q]:claims){if(q.pos!=tile||q.owner==int(u))continue;
    bool duplicate=false;for(size_t v=0;v<actual.size();v++)if(pos(e,seat,int(v))==tile&&Controller::same_action(actual[v],q.a))duplicate=true;if(duplicate)continue;
    Counts pickups{},seeds{};auto valid=c.valid(pv,int(u),q.a,tile,pickups,seeds);if(!valid||!Controller::same_action(*valid,q.a))continue;
    // The official all-or-none PLANT reservation is evaluated for the whole turn.
    if(q.a.op==Op::PLANT){int demand=1;for(auto a:actual)demand+=a.op==Op::PLANT&&a.item==q.a.item;if(demand>original.priv.seeds[int(q.a.item)])continue;}
    auto alternate=actual;alternate[u]=q.a;auto applied=e.project_unit_phase(seat,alternate,int(u+1));if(!effective(prefix,applied,seat,int(u),q.a))continue;
    pickups={};seeds={};auto initial=c.valid(original,int(u),q.a,tile,pickups,seeds);
    bool prefix_enabled=!initial||!Controller::same_action(*initial,q.a);
    int prerequisite=-1;for(size_t v=0;v<u;v++)if(pos(e,seat,int(v))==tile&&!Controller::movement(actual[v].op)&&actual[v].op!=Op::PASS)prerequisite=int(v);
    bool earlier_planned_same_plot=false;if(q.owner>=0){const auto&pl=c.plans[q.owner];for(size_t k=pl.index;k<pl.a.size();k++){
     if(pl.target[k]==tile&&!Controller::movement(pl.a[k].op)){if(Controller::same_action(pl.a[k],q.a))break;earlier_planned_same_plot=true;}
    }}
    py::dict r;r["unit"]=u;r["owner"]=q.owner;r["position"]=tile;r["operation"]=names[int(q.a.op)];r["item"]=int(q.a.item);r["planned_delay"]=q.delay;
    r["prefix_enabled"]=prefix_enabled;r["prerequisite_unit"]=prerequisite;r["earlier_planned_same_plot"]=earlier_planned_same_plot;
    r["own_held_before"]=original.priv.inventories[u];r["own_held_after"]=applied.privates()[seat].inventories[u];opportunities.append(r);
   }
  }return result;
 });
}
