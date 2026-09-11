// Read-only observations of latent harvest and the EXISTING short sell model.
// No real-future continuation, hidden opponent inventory or policy mutation.
#include "policy.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
PYBIND11_MODULE(_dp7_cash_window_probe,m){m.def("inspect",[](const dp7::Controller&c,const fastkag::Simulator&e,int seat){
 using namespace dp7;View o{e.step_count(),e.day(),e.hour(),e.farms()[seat],e.farms()[1-seat],e.privates()[seat],e.market(),e.shops()};
 py::dict result;result["same_day_controller"]=c.day==o.day;result["phase"]=c.phase;result["inventory"]=o.market.inventory;result["prices"]=o.market.prices;result["shed"]=o.priv.shed;
 std::set<int>generated,scheduled;for(auto&j:c.jobs(o))for(auto a:j.actions)if(a.op==Op::HARVEST)generated.insert(j.pos);
 for(auto&p:c.plans)for(size_t k=p.index;k<p.a.size();k++)if(p.a[k].op==Op::HARVEST)scheduled.insert(p.target[k]);
 py::list mature;
 for(int pos=0;pos<100;pos++){
  auto&t=o.own.tiles[pos];int i=animal(t)?product[int(t.animal)-9]:plant(t)?int(t.crop):-1;
  if(i<0||t.yield_units<=0||(plant(t)&&o.day-t.planted_day<first[i]))continue;
  int distance=100,owner=-1;for(size_t u=0;u<o.priv.inventories.size();u++){int d=dist(cell(u?o.own.hands[u-1]:o.own.farmer),pos);if(d<distance){distance=d;owner=int(u);}}
  py::dict row;row["pos"]=pos;row["item"]=i;row["quantity"]=t.yield_units;row["generated"]=generated.count(pos)>0;row["scheduled"]=scheduled.count(pos)>0;row["nearest_unit"]=owner;
  row["optimistic_harvest_drop_steps"]=distance+1+near(pos)+1;
  row["current_warehouse_quote"]=revenue(i,o.market.inventory[i],t.yield_units);mature.append(row);
 }
 result["mature"]=mature;result["tactical"]=c.competition_sales(o);
 // Reconstruct the exact formula used by competition_sales, then assert its
 // proposed quantities agree. This is an audit, not a new predictor.
 int end=std::min(719,o.step+24*c.p.sell_horizon_days);Quantities dem{},supply{},forecast{};
 static const std::array<std::vector<int>,8>items{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C},{MI,T,W},{S,MI},{WO}}};
 for(int s=o.step;s<end;s++){if(s%24==0)for(int i=0;i<8;i++)dem[i]++;if(s%4==0)for(int sh:o.shops)for(int i:items.at(sh))dem[i]+=items.at(sh).size()==1?2:1;}
 for(int side=0;side<2;side++)for(auto&t:(side?o.opponent:o.own).tiles){
  if(animal(t)){int k=int(t.animal)-9;supply[product[k]]+=t.yield_units;for(int d=c.day+1;d<=std::min(29,(end-1)/24);d++){int ds=d-t.placed_day-afirst[k];if(ds>=0&&ds%ainterval[k]==0)supply[product[k]]+=1+ainterval[k];}}
  else if(plant(t)){int i=int(t.crop);supply[i]+=t.yield_units;
   if(ongoing(i)){for(int d=c.day+1;d<=std::min(29,(end-1)/24);d++){int ds=d-t.planted_day-first[i];if(ds>=0&&ds%interval[i]==0&&ds/interval[i]<4)supply[i]+=2;}}
   else if(t.planted_day+c.p.crop_harvest_age[i]<=std::min(29,(end-1)/24))supply[i]+=std::max(0,c.finite_yield(i,c.p.crop_harvest_age[i])-t.yield_units);
  }
 }
 auto tactical=c.competition_sales(o);for(int i=0;i<9;i++)forecast[i]=o.market.inventory[i]-dem[i]+c.p.competitive_sell_weight*supply[i];
 if(c.day==o.day&&c.p.competitive_sell_weight>0&&c.day<29)for(int i=1;i<8;i++){
  int n=o.priv.shed[i];int q=n>0&&revenue(i,o.market.inventory[i],n)-revenue(i,forecast[i],n)>c.p.sell_advantage*n?n:0;
  if(q!=tactical[i])throw std::runtime_error("cash-window predictor audit mismatch");
 }
 result["forecast_inventory"]=forecast;result["forecast_end"]=end;return result;
 });m.def("fills",[](const fastkag::Simulator&e,int seat){return e.last_market_fills()[seat];});}
