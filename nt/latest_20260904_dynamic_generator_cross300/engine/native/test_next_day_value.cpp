#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;
void check(bool yes,const char*why){checks++;if(!yes)throw std::runtime_error(why);}
View view(const Simulator&e){return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
Simulator at(int d,int h=0){Simulator e(Config{},4839);while(e.step_count()<d*24+h)e.step({});return e;}
int main(){try{
 auto e=at(12,23);auto&f=const_cast<Farm&>(e.farms()[0]);f.tiles.assign(100,Tile{});f.money=5000;f.farmer={4,4};
 auto&pr=const_cast<PrivateState&>(e.privates()[0]);pr=PrivateState{};pr.inventories.resize(1);pr.inventory_order.resize(1);
 Controller c;c.day=12;c.phase=3;c.p.max_land=1;c.p.economic_land=true;c.target={{44,W}};
 auto old=dayvalue::schedule_value(c,view(e),{Plan{}});
 c.p.day_value_replan_next_day=true;auto fresh=dayvalue::schedule_value(c,view(e),{Plan{}});
 check(old.known&&old.calendars==0&&old.horizon_ticks==1,"old empty-state coverage changed");
 check(fresh.horizon_ticks==25&&fresh.next_day_proposed>0,"next-day investment planning not executed");
 check(fresh.calendars>0&&fresh.score!=old.score,"next-day realizable capacity not included");
 check(f.money==5000&&e.step_count()==311&&pr.shed==Counts{},"scenario changed real state");
 // The flag must not invoke value selection by itself.
 Params pp;pp.day_value_replan_next_day=true;Controller isolated(pp);Simulator off(Config{},392);isolated.act(view(off));
 check(isolated.day_value_checks==0,"flag enabled selector implicitly");
 // Differential: manual one-day replay through the unchanged original kernel
 // helper, preserving the forecast assumptions and exact controller switches.
 int cases=0;
 for(int d:{5,12,20,28})for(bool stored:{false,true})for(bool cash:{false,true}){
  auto actual=at(d,23);auto&farm=const_cast<Farm&>(actual.farms()[0]);farm.tiles.assign(100,Tile{});farm.money=cash?5000:0;
  auto&priv=const_cast<PrivateState&>(actual.privates()[0]);priv=PrivateState{};priv.inventories.resize(1);priv.inventory_order.resize(1);
  if(stored)priv.shed[MI]=6;
  Controller trial(c.p);trial.day=d;trial.phase=3;trial.target={{44,W}};
  auto predicted=dayvalue::schedule_value(trial,view(actual),{Plan{}});
  ObservedDayScenario day1(view(actual));auto ctl=trial;ctl.plans={Plan{}};ctl.last_step=actual.step_count()-1;ctl.p.day_consequence_compare=false;ctl.p.stepwise_recoordination=false;ctl.p.resource_aware_exchange=false;
  while(!day1.finished())day1.advance(ctl.act(day1.view()));
  View ep{day1.step_count(),d+1,0,day1.own(),actual.farms()[1],day1.inventory(),day1.market(),day1.shops()};
  ObservedDayScenario day2(ep);ctl.p.stepwise_recoordination=trial.p.stepwise_recoordination;ctl.p.resource_aware_exchange=trial.p.resource_aware_exchange;
  while(!day2.finished())day2.advance(ctl.act(day2.view()));
  dayvalue::Value expected;
  if(day2.step_count()==719)expected.score=day2.own().money;
  else{View end{day2.step_count(),d+2,0,day2.own(),actual.farms()[1],day2.inventory(),day2.market(),day2.shops()};expected=dayvalue::residual(ctl,end);}
  check(predicted.score==expected.score&&predicted.known==expected.known,"forecast differs from manual public-state continuation");
  check(predicted.horizon_ticks==(d==28?24:25),"final horizon exceeded or shortened");cases++;
 }
 // Final-day valuation cannot invent actions after the game ends.
 auto end=at(29,22);Controller terminal(c.p);terminal.day=29;terminal.phase=3;
 auto tv=dayvalue::schedule_value(terminal,view(end),{Plan{}});check(tv.known&&tv.next_day_proposed==0,"post-terminal investment");
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<",\"continuation_cases\":"<<cases<<",\"old_empty_value\":"<<old.score<<",\"new_value\":"<<fresh.score<<",\"new_value_known\":"<<(fresh.known?"true":"false")<<",\"next_day_projects\":"<<fresh.next_day_proposed<<",\"next_day_actual_assets\":"<<fresh.calendars<<"}\n";
}catch(const std::exception&e){std::cerr<<"check "<<checks<<": "<<e.what()<<"\n";return 1;}}
