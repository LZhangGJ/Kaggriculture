#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;void check(bool x,const char*why){checks++;if(!x)throw std::runtime_error(why);}
struct Fixture {
 Farm f,r;PrivateState pr;Market market;std::vector<int8_t>shops;Controller c;int day=8,hour=2;
 Fixture(int hands=3){
  f.tiles.resize(100);r.tiles.resize(100);f.farmer=r.farmer={4,4};f.money=20000;f.unlocked_mask=7;
  for(int i=0;i<hands;i++){int d=depot[(i+1)%4];f.hands.push_back({int16_t(d%10),int16_t(d/10)});}
  pr.inventories.resize(hands+1);pr.inventory_order.resize(hands+1);pr.shed[W]=20;pr.seeds[W]=20;
  market.inventory.fill(10000);for(int i=0;i<9;i++)market.prices[i]=price(i,10000);
  c.day=day;c.phase=2;c.last_step=day*24+hour-1;c.p.shared_task_atoms_v2=true;c.p.fix_finite_projection=true;c.p.fix_logistics=true;
  c.p.stepwise_recoordination=true;c.p.shared_service_insertions=true;c.p.incremental_pickup_repair=true;c.p.max_hands=6;
  for(int pos:{42,43,53,63}){auto&t=f.tiles[pos];t.kind=TileKind::PLANT;t.crop=Item(W);t.planted_day=4;t.yield_units=3;t.max_lifespan_step=9*24;c.target.emplace_back(pos,W);}
  for(int pos:{34,24}){auto&t=f.tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item(SH);t.placed_day=2;t.fertilizer_available=true;t.yield_units=2;c.target.emplace_back(pos,SH);}
 }
 View view(){return {day*24+hour,day,hour,f,r,pr,market,shops};}
};
int main(){try{
 {Fixture f;auto base=f.c,wrapped=f.c;base.compile_base(f.view());wrapped.compile(f.view());check(compilechoice::same(base,wrapped),"off switch changed compiled plan");check(wrapped.compile_choice_calls==0,"off switch searched");}
 {Fixture f;auto real=f.c,conditional=f.c;conditional.compile_base(f.view());conditional.resume_compiled_tick=true;
  auto a=real.act(f.view()),b=conditional.act(f.view());check(a.units.size()==b.units.size()&&a.market.size()==b.market.size(),"first compiled action shape mismatch");
  for(size_t u=0;u<a.units.size();u++)check(Controller::same_action(a.units[u],b.units[u]),"first compiled unit action mismatch");
  for(size_t i=0;i<a.market.size();i++)check(Controller::same_action(a.market[i],b.market[i]),"first compiled market order mismatch");
  check(conditional.step_recoord_checks==real.step_recoord_checks,"recoordinated before first compiled action");check(!conditional.resume_compiled_tick,"first-tick continuation leaked");
 }
 {Fixture f;f.c.p.compile_replant_choices=true;f.c.compile_base(f.view());auto before=f.c;
  auto trials=compilechoice::generate(f.c,f.view());check(compilechoice::same(before,f.c),"generation changed source");
  int waits=0;
  for(auto&t:trials){
   // Effects belong to crop cohorts, not just a coordinate. Watering the old
   // crop before harvest and a new crop after planting is not a duplicate.
   std::multiset<Controller::TaskKey>declared;
   for(const auto&j:t.ctl.jobs(f.view()))for(auto a:j.actions)declared.insert(Controller::task_key(j.pos,a));
   std::map<int,std::pair<int,size_t>>last_water;
   for(size_t u=0;u<t.ctl.plans.size();u++){auto&pl=t.ctl.plans[u];
    for(size_t k=pl.index;k<pl.a.size();k++){
     auto a=pl.a[k];int pos=pl.target[k];
     if(Controller::movement(a.op)||a.op==Op::PICKUP||a.op==Op::DROP)continue;
     auto it=declared.find(Controller::task_key(pos,a));check(it!=declared.end(),"compiled duplicate or undeclared effect");declared.erase(it);
     if(a.op==Op::WATER){auto prior=last_water.find(pos);if(prior!=last_water.end()){
       check(prior->second.first==int(u),"dependent crop cohorts assigned without ordering");
       bool replanted=false;for(size_t z=prior->second.second+1;z<k;z++)replanted|=pl.target[z]==pos&&pl.a[z].op==Op::PLANT;
       check(replanted,"water repeated for the same crop cohort");
      }last_water[pos]={int(u),k};}
    }
   }
   if(t.kind=="defer_replant_0"){
    waits++;for(auto&j:t.ctl.jobs(f.view()))if(plant(f.f.tiles[j.pos])){bool harvest=false;for(auto a:j.actions){harvest|=a.op==Op::HARVEST;check(a.op!=Op::PLANT,"defer still plants");}check(harvest,"defer erased old harvest");check(sum(j.seeds)==0,"defer reserved seeds");}
    auto next=t.ctl;next.day=9;auto farm=f.f;for(int pos:{42,43,53,63})farm.tiles[pos]=Tile{};
    View v{9*24,9,0,farm,f.r,f.pr,f.market,f.shops};int n=0;for(auto&j:next.jobs(v))for(auto a:j.actions)n+=a.op==Op::PLANT;
    check(n==4,"one-day defer permanently locked plots");
    auto state=t.ctl;auto fv=compilechoice::evaluate(t.ctl,f.view(),false);check(fv.ticks==22,"scenario not bounded to this day");check(compilechoice::same(state,t.ctl),"evaluation mutated source");
    auto live=t.ctl;live.last_step=f.view().step-1;live.p.compile_consequence=live.p.compile_replant_choices=false;
    ObservedDayScenario scenario(f.view());while(!scenario.finished()){
     auto obs=scenario.view();auto act=live.act(obs);
     for(size_t u=0;u<act.units.size();u++)if(act.units[u].op==Op::PLANT){int pos=cell(u?obs.own.hands[u-1]:obs.own.farmer);check(live.plant_not_before[pos]<=obs.day,"intraday execution bypassed defer");}
     scenario.advance(act);
    }
   }
  }check(waits==1,"missing reversible replant candidate");
 }
 {Fixture f;f.c.p.compile_replant_choices=true;f.c.plant_not_before[42]=12;f.c.compile_base(f.view());
  for(auto&t:compilechoice::generate(f.c,f.view()))check(t.ctl.plant_not_before[42]>=12,"existing wait brought forward");
 }
 {Fixture f;f.c.p.compile_replant_choices=true;f.c.p.compile_consequence=true;f.c.compile(f.view());check(f.c.compile_choice_calls==1,"first compiler bypassed new selector");check(f.c.compile_choice_evaluations<=f.c.compile_choice_candidates,"recursive candidate explosion");check(f.c.compile_choice_evaluations>0,"no actual consequence comparisons");}
 for(bool bounded:{false,true})for(int hands=0;hands<=5;hands++)for(bool money:{false,true}){
  Fixture f(hands);f.f.money=money?20000:0;f.c.p.compile_consequence=true;f.c.p.compile_replant_choices=true;
  f.c.p.compile_bounded_rollout=bounded;f.c.p.day_value_replan_next_day=true;
  auto a=f.c,b=f.c;a.p.exact_schedule_cache=false;b.p.exact_schedule_cache=true;
  a.compile(f.view());auto cache=std::make_shared<packmemo::Cache>();{packmemo::Scope scope(cache.get());b.compile(f.view());}
  check(compilechoice::same(a,b),"cache changed chosen semantics");check(a.plans.size()==f.pr.inventories.size(),"wrong unit count");
  check(a.p.day_value_replan_next_day&&b.p.day_value_replan_next_day,"approximate leaf flags leaked to actual policy");
 }
 {Fixture f;f.c.day=29;f.day=29;f.c.p.compile_consequence=f.c.p.compile_replant_choices=true;f.c.compile(f.view());check(f.c.compile_choice_calls==0,"terminal phase admitted investments");}
 {Fixture f;f.c.p.compile_consequence=f.c.p.compile_replant_choices=true;f.c.last_step=719;f.c.plant_not_before[42]=20;f.c.compile_choice_calls=10;
  Simulator env(Config{},1);View v{0,0,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  auto action=f.c.act(v);Controller fresh(f.c.p);auto other=fresh.act(v);
  check(f.c.plant_not_before==fresh.plant_not_before&&f.c.compile_choice_calls==fresh.compile_choice_calls,"cross-game deferred state leaked");check(action.units.size()==other.units.size()&&action.market.size()==other.market.size(),"reset action shape changed");
 }
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
