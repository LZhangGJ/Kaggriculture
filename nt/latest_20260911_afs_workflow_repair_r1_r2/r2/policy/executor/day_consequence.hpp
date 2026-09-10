#pragma once
#include "observed_day_scenario.hpp"
#include <map>
namespace dp7::dayvalue {
using Cal=portfolio::DeliveryCalendar;
struct Work {int pos,kind;Cal cal;};
struct Value {
 double score=0,trade=0,seeds=0,wages=0,funding_gap=0;
 bool known=true;
 int calendars=0;
 int horizon_ticks=0,next_day_proposed=0;
};
// Conditional single-animal lifecycle, initialized from its actual state.
// One logical work group, NOT a persistent assignment to one real worker.
inline Cal animal_calendar(const Controller&source,const View&o,Tile initial){
 if(!animal(initial))throw std::invalid_argument("animal calendar initial state");
 const int kind=int(initial.animal),item=product[kind-9];Cal result;
 Controller ctl(source.p);ctl.p.shared_task_atoms_v2=ctl.p.split_service_jobs=false;
 ctl.target={{44,kind}};Farm f;f.tiles.resize(100);f.tiles[44]=initial;f.farmer={4,4};
 for(int d=o.day;d<30&&animal(f.tiles[44]);d++){
  ctl.day=d;PrivateState pr;pr.shed[W]=1;pr.inventories.resize(1);pr.inventory_order.resize(1);
  int hour=d==o.day?o.hour:0;
  View v{24*d+hour,d,hour,f,o.opponent,pr,o.market,o.shops};
  auto jobs=ctl.jobs(v);Acts planned;int need=0;
  for(const auto&j:jobs)need+=j.needs[W];
  if(need)planned.push_back(action(Op::PICKUP,W,need));
  for(const auto&j:jobs)planned.insert(planned.end(),j.actions.begin(),j.actions.end());
  ObservedDayScenario model(v);size_t index=0;int output_quantity=0,fertilizer=0,last_output=-1,last_fert=-1;
  while(!model.finished()){
   int h=model.view().hour;Action a=index<planned.size()?planned[index++]:action(Op::PASS);
   auto before=model.inventory().inventories[0];bool cared=model.own().tiles[44].cared_today;
   // Record unit effects BEFORE automatic night deposits clear the bag.
   PlayerAction own{{a},{}};
   model.advance(own);
   if(a.op==Op::FEED&&before[W]>0)result.feed[d]++;
   if(a.op==Op::CARE&&!cared)result.care[d]++;
   // jobs() supplies the currently visible quantity. HARVEST/COLLECT need no
   // input and at this one-cell site cannot be displaced by another worker.
   if(a.op==Op::HARVEST){
    int got=0;for(const auto&j:jobs)got+=j.out[item];
    output_quantity+=got;result.harvest[d]+=got>0;last_output=h;
   }
   if(a.op==Op::COLLECT_FERTILIZER){fertilizer++;result.fert[d]++;last_fert=h;}
  }
  result.cash.quantity[d][W]-=result.feed[d];
  if(d<29){result.cash.quantity[d+1][item]+=output_quantity;result.cash.quantity[d+1][F]+=fertilizer;}
  else{
   if(last_output<22)result.cash.quantity[d][item]+=output_quantity;
   if(last_fert<22)result.cash.quantity[d][F]+=fertilizer;
  }
  f.tiles[44]=model.own().tiles[44];
 }
 return result;
}
inline auto tile_key(const Tile&t){return std::tuple(int(t.kind),int(t.crop),int(t.animal),t.planted_day,t.placed_day,t.yield_units,t.consecutive_unwatered,t.consecutive_unfed,t.fertilized_until_day,t.pending_care_bonus,t.max_lifespan_step,t.watered_today,t.fed_today,t.cared_today,t.fertilizer_available);}
inline std::vector<Work> work_calendar(const Controller&c,const View&o){
 std::vector<Work>out;std::map<decltype(tile_key(Tile{})),Cal>cache;
 for(int pos=0;pos<100;pos++){
  auto&t=o.own.tiles[pos];int kind=animal(t)?int(t.animal):plant(t)?int(t.crop):-1;if(kind<0)continue;
  auto key=tile_key(t);auto it=cache.find(key);
  if(it==cache.end())it=cache.emplace(key,kind>=9?animal_calendar(c,o,t):portfolio::crop_delivery(c,o,kind,t)).first;
  out.push_back({pos,kind,it->second});
 }return out;
}
inline double wages_on(const Controller&c,const std::vector<Work>&works,int day){
 std::vector<Job>jobs;
 for(const auto&w:works){const auto&cal=w.cal;Job j;j.pos=w.pos;j.priority=0;
  if(cal.harvest[day]){j.actions.push_back(action(Op::HARVEST));j.out[w.kind>=9?product[w.kind-9]:w.kind]=1;}
  if(cal.fert[day]){
   if(w.kind>=9){j.actions.push_back(action(Op::COLLECT_FERTILIZER));j.out[F]=1;}
   else{j.actions.push_back(action(Op::FERTILIZE));j.needs[F]=cal.fert[day];}
  }
  if(cal.dig[day])j.actions.push_back(action(Op::DIG));
  if(cal.planting[day])j.actions.push_back(action(Op::PLANT,w.kind));
  if(cal.feed[day]){j.actions.push_back(action(Op::FEED));j.needs[W]=cal.feed[day];}
  if(cal.care[day])j.actions.push_back(action(Op::CARE));
  if(cal.water[day])j.actions.push_back(action(Op::WATER));
  if(!j.actions.empty())jobs.push_back(j);
 }
 for(int hands=0;hands<=c.p.max_hands;hands++)if(c.pack(jobs,c.starts(hands),day==29?22:23,day==29,false,true).second==0)return Controller::hirecost(hands);
 return INFINITY;
}
// Minimal inventory needed NOW to cover future net withdrawals before the
// next deposits arrive. Prevents selling feed and then charging its repurchase.
inline int reserve_after(const Controller::Calendar&flow,int day,int item){
 int balance=0,low=0;for(int d=day+1;d<30;d++){balance+=int(std::nearbyint(flow[d][item]));low=std::min(low,balance);}return -low;
}
inline Value residual(const Controller&source,const View&o){
 Value result;result.score=o.own.money;
 if(o.step>=719)return result;
 if(o.hour!=0)throw std::invalid_argument("residual must start at a day boundary");
 auto c=source;c.day=o.day;c.p.day_consequence_compare=false;
 const auto works=work_calendar(c,o);result.calendars=works.size();
 Controller::Calendar flow{};std::array<std::array<int,5>,30>seed_need{};
 for(const auto&w:works)for(int d=o.day;d<30;d++){
  for(int i=0;i<9;i++)flow[d][i]+=w.cal.cash.quantity[d][i];
  if(w.kind<5)seed_need[d][w.kind]+=w.cal.planting[d];
 }
 c.p.opponent_supply_weight=c.p.day_value_public_supply?1.:0.;
 const auto context=c.portfolio_context(o);Quantities market{};Counts stock=o.priv.shed;
 for(auto&bag:o.priv.inventories)add(stock,bag);
 auto seeds=o.priv.seeds;for(int i=0;i<9;i++)market[i]=o.market.inventory[i];
 double bank=o.own.money;
 for(int d=o.day;d<30;d++){
  const double wage=wages_on(c,works,d);
  if(!std::isfinite(wage)){result.known=false;return result;}
  result.wages+=wage;bank-=wage;
  for(int i=0;i<5;i++){
   int reused=std::min(seeds[i],seed_need[d][i]);seeds[i]-=reused;
   double expense=(seed_need[d][i]-reused)*seed_price[i];result.seeds+=expense;bank-=expense;
  }
  for(int i=0;i<9;i++){
   market[i]+=context.rival[d][i]-context.dem[d][i];
   int q=int(std::nearbyint(flow[d][i]));stock[i]+=q;
   if(stock[i]<0){double paid=Controller::projected_trade(i,market[i],stock[i]);bank+=paid;result.trade+=paid;stock[i]=0;}
   int hold=(i==W||i==F)?reserve_after(flow,d,i):0;
   int sell=std::max(0,stock[i]-hold);stock[i]-=sell;
   double earned=Controller::projected_trade(i,market[i],sell);bank+=earned;result.trade+=earned;
  }
  result.funding_gap=std::max(result.funding_gap,std::max(0.,-bank));
 }
 result.score=bank;result.known=std::isfinite(bank)&&result.funding_gap==0;
 return result;
}
inline Value schedule_value(const Controller&source,const View&o,const std::vector<Plan>&plans){
 ObservedDayScenario scenario(o);auto c=source;c.admission_inspection=nullptr;c.plans=plans;
 c.workflow_runtime=false;
 c.phase=3;c.day=o.day;c.last_step=o.step-1;
 c.p.day_consequence_compare=false;c.p.stepwise_recoordination=false;c.p.resource_aware_exchange=false;
 // A conditional schedule probe must not recursively search another set of
 // rotation scenarios at its optional next-day boundary. Wait intents remain.
 c.p.portfolio_rotation=false;
 while(!scenario.finished())scenario.advance(c.act(scenario.view()));
 if(scenario.step_count()>=719){Value v;v.score=scenario.own().money;return v;}
 View endpoint{scenario.step_count(),scenario.step_count()/24,0,scenario.own(),o.opponent,scenario.inventory(),scenario.market(),scenario.shops()};
 if(!source.p.day_value_replan_next_day){auto value=residual(c,endpoint);value.horizon_ticks=scenario.ticks();return value;}
 // A bounded CONDITIONAL day, not a clone of the real game's future. Let the
 // existing controller revise investments, staff and orders using only this
 // scenario's actual funds and assets. No oracle or alternate terminal search.
 ObservedDayScenario following(endpoint);
 c.p.stepwise_recoordination=source.p.stepwise_recoordination;
 c.p.resource_aware_exchange=source.p.resource_aware_exchange;
 int proposed=0;
 while(!following.finished()){
  bool first=following.view().hour==0;
  auto act=c.act(following.view());
  if(first)for(auto[pos,k]:c.target)proposed+=k>=0;
  following.advance(act);
 }
 Value value;
 if(following.step_count()>=719)value.score=following.own().money;
 else{
  View end{following.step_count(),following.step_count()/24,0,following.own(),o.opponent,following.inventory(),following.market(),following.shops()};
  value=residual(c,end);
 }
 value.horizon_ticks=scenario.ticks()+following.ticks();value.next_day_proposed=proposed;
 return value;
}
inline bool accept(Controller&c,const View&o,const std::vector<Plan>&keep,const std::vector<Plan>&next){
 c.day_value_checks++;
 const auto a=schedule_value(c,o,keep),b=schedule_value(c,o,next);
 if(!a.known||!b.known){c.day_value_unknown++;return true;} // Original decision, not a new hard mask.
 if(b.score<a.score-1e-6){c.day_value_rejected++;return false;}
 return true; // Equal estimates retain the prior scheduling decision.
}
} // namespace dp7::dayvalue
