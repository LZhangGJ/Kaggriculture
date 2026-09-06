// Diagnostic binding only. No call from live act(), no policy promotion.
#include "policy.hpp"
#include "observed_day_scenario.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <chrono>
namespace py=pybind11;
using namespace dp7;
namespace {
View view(const Simulator&e,int seat){return {e.step_count(),e.day(),e.hour(),e.farms().at(seat),e.farms().at(1-seat),e.privates().at(seat),e.market(),e.shops()};}
struct Outcome {
 double cash=0,quote=0,milliseconds=0;
 Counts shed{},sold{},bought{},field{};
 std::array<int,5>seeds{};
 std::array<int,8>scale{};
 int overflow=0,ticks=0,projects=0,remaining=0;
 Farm farm;Market market;
};
Outcome simulate(const Controller&original,const View&v,const std::vector<Plan>&plans){
 ObservedDayScenario scenario(v);Controller c=original;c.admission_inspection=nullptr;
 c.plans=plans;c.phase=3;c.day=v.day;c.last_step=v.step-1;
 // Do not instantly erase the schedule being tested with the same optimizer.
 // All OTHER live production, market and failure-recovery policies remain.
 c.p.stepwise_recoordination=false;c.p.resource_aware_exchange=false;
 int projects=c.intraday_activated;Outcome out;auto begin=std::chrono::steady_clock::now();
 while(!scenario.finished()){
  auto action=c.act(scenario.view());scenario.advance(action);
  const auto&fill=scenario.fills();
  for(size_t j=0;j<fill.size();j++){
   const auto&a=action.market[j];int i=int(a.item);
   if(a.op==Op::SELL&&i>=0&&i<9)out.sold[i]+=fill[j];
   if((a.op==Op::BUY_PRODUCT||a.op==Op::BUY_ANIMAL)&&i>=0&&i<12)out.bought[i]+=fill[j];
  }
  out.overflow+=scenario.overflow();
 }
 out.cash=scenario.own().money;out.quote=scenario.liquidation_quote();out.shed=scenario.inventory().shed;out.seeds=scenario.inventory().seeds;
 out.farm=scenario.own();out.market=scenario.market();out.projects=c.intraday_activated-projects;out.ticks=scenario.ticks();
 for(const auto&pl:c.plans)out.remaining+=int(pl.a.size()-std::min(pl.index,pl.a.size()));
 for(const auto&t:scenario.own().tiles){
  if(plant(t)){out.scale[int(t.crop)]++;out.field[int(t.crop)]+=t.yield_units;}
  else if(animal(t)){out.scale[5+int(t.animal)-9]++;out.field[product[int(t.animal)-9]]+=t.yield_units;out.field[F]+=t.fertilizer_available;}
 }
 out.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();return out;
}
bool same_productive(const Farm&a,const Farm&b){
 if(a.unlocked_mask!=b.unlocked_mask||a.hires_today!=b.hires_today||a.hands.size()!=b.hands.size()||a.farmer.x!=b.farmer.x||a.farmer.y!=b.farmer.y)return false;
 for(size_t u=0;u<a.hands.size();u++)if(a.hands[u].x!=b.hands[u].x||a.hands[u].y!=b.hands[u].y)return false;
 auto key=[](const Tile&t){return std::tuple(int(t.kind),int(t.crop),int(t.animal),t.planted_day,t.placed_day,t.yield_units,t.consecutive_unwatered,t.consecutive_unfed,t.fertilized_until_day,t.pending_care_bonus,t.max_lifespan_step,t.watered_today,t.fed_today,t.cared_today,t.fertilizer_available);};
 for(int i=0;i<100;i++)if(key(a.tiles[i])!=key(b.tiles[i]))return false;return true;
}
py::dict pack(const Outcome&r){py::dict d;d["cash"]=r.cash;d["liquidation_quote_not_bank"]=r.quote;d["cash_plus_quote_not_objective"]=r.cash+r.quote;d["shed"]=r.shed;d["seeds"]=r.seeds;d["sold"]=r.sold;d["bought"]=r.bought;d["field"]=r.field;d["scale"]=r.scale;d["overflow"]=r.overflow;d["ticks"]=r.ticks;d["intraday_projects"]=r.projects;d["remaining_plan_actions"]=r.remaining;d["milliseconds"]=r.milliseconds;return d;}
}
PYBIND11_MODULE(_dp7_dayprobe,m){
 m.def("inspect",[](const Controller&original,const Simulator&live,int seat){
  auto o=view(live,seat);py::dict result;result["eligible"]=false;
  if(o.day>=29||original.day!=o.day||original.phase!=3||o.step<=original.last_step||original.plans.size()!=o.priv.inventories.size())return result;
  auto initial=original;initial.admission_inspection=nullptr;initial.last_step=o.step;
  if(initial.recover(o)||initial.reconcile_seeds(o))return result;
  intraday::complete_pending(initial,o);
  auto base=initial,alternate=initial;
  base.p.stepwise_recoordination=alternate.p.stepwise_recoordination=true;
  base.p.shared_task_atoms_v2=alternate.p.shared_task_atoms_v2=true;
  base.p.shared_service_insertions=false;alternate.p.shared_service_insertions=true;
  base.recoordinate(o);alternate.recoordinate(o);
  bool changed=false;for(size_t u=0;u<base.plans.size();u++){
   const auto&a=base.plans[u];const auto&b=alternate.plans[u];
   const size_t na=a.a.size()-a.index,nb=b.a.size()-b.index;
   if(na!=nb){changed=true;continue;}
   for(size_t k=0;k<na;k++)if(!Controller::same_action(a.a[a.index+k],b.a[b.index+k])||a.target[a.index+k]!=b.target[b.index+k])changed=true;
  }
  result["eligible"]=true;result["changed"]=changed;result["day"]=o.day;result["hour"]=o.hour;result["step"]=o.step;
  if(!changed)return result;
  auto a=simulate(base,o,base.plans),b=simulate(alternate,o,alternate.plans);
  result["base"]=pack(a);result["alternate"]=pack(b);
  result["bank_delta"]=b.cash-a.cash;result["cash_quote_delta_not_profit"]=b.cash+b.quote-a.cash-a.quote;
  bool same=same_productive(a.farm,b.farm)&&a.shed==b.shed&&a.seeds==b.seeds&&a.market.inventory==b.market.inventory;
  result["same_endpoint_except_cash"]=same;
  result["strict_cash_improvement_in_PASS_scenario"]=same&&b.cash>a.cash;
  result["strict_cash_loss_in_PASS_scenario"]=same&&b.cash<a.cash;
  result["scenario"]= "opponent PASS; own live execution except further joint reassignment; current day only";
  return result;
 });
}
