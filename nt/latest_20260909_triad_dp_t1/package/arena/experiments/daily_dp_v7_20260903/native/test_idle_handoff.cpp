#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;void check(bool ok,const char*why){checks++;if(!ok)throw std::runtime_error(why);}
Plan plan(std::initializer_list<Action>actions){Plan p;p.a=actions;p.target.assign(p.a.size(),44);return p;}
struct Fixture{
 Simulator e{Config{},777};Controller c;PlayerAction out;
 Fixture(int step=104,int n=2){while(e.step_count()<step)e.step({});auto&f=const_cast<Farm&>(e.farms()[0]);auto&p=const_cast<PrivateState&>(e.privates()[0]);f.farmer={4,4};f.hands.assign(n-1,{4,4});f.tiles[44]=Tile{};p.inventories.assign(n,{});p.inventory_order.assign(n,{});p.seeds[W]=2;c.day=e.day();c.last_step=e.step_count();c.phase=3;c.p.idle_task_handoff=true;c.plans.resize(n);out.units.resize(n);}
 View view()const{return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
 Tile&tile(){return const_cast<Farm&>(e.farms()[0]).tiles[44];}
 void cow(){auto&t=tile();t.kind=TileKind::ANIMAL;t.animal=Item(CO);t.placed_day=0;t.yield_units=3;t.fertilizer_available=true;}
 void crop(int crop=W){auto&t=tile();t.kind=TileKind::PLANT;t.crop=Item(crop);t.planted_day=0;t.yield_units=2;}
 void apply(){handoff::apply(c,view(),out);}
};
int main(){
 {Fixture f;f.c.p.idle_task_handoff=false;f.c.plans[0]=plan({action(Op::WATER)});f.out.units[0]=action(Op::PLANT,W);f.apply();check(f.out.units[1].op==Op::PASS&&f.c.plans[0].a.size()==1&&f.c.idle_handoff_checks==0,"disabled changes behavior");}
 {Fixture f;f.c.plans[0]=plan({action(Op::WATER)});f.out.units[0]=action(Op::PLANT,W);f.apply();check(f.out.units[1].op==Op::WATER&&f.c.plans[0].a.empty(),"same turn plant water missing");check(f.c.idle_handoff_prefix==1&&f.c.idle_handoff_applied==1,"prefix counters");auto after=f.e.project_unit_phase(0,f.out.units);check(after.farms()[0].tiles[44].watered_today,"water effect missing");}
 {Fixture f;f.c.plans[1]=plan({action(Op::WATER)});f.out.units[1]=action(Op::PLANT,W);f.apply();check(f.out.units[0].op==Op::PASS,"water scheduled before plant");}
 {Fixture f(119,3);f.c.plans[0]=plan({action(Op::WATER)});f.out.units[0]=action(Op::PLANT,W);f.apply();check(f.c.idle_handoff_applied==1&&f.out.units[2].op==Op::PASS,"duplicate water");f.e.step({f.out,{}});check(plant(f.e.farms()[0].tiles[44])&&f.e.farms()[0].tiles[44].consecutive_unwatered==0,"day deadline failed");}
 {Fixture f;f.crop();f.c.plans[0]=plan({action(Op::WATER)});f.out.units[0]=action(Op::WATER);f.apply();check(f.out.units[1].op==Op::PASS,"duplicate actual task");}
 {Fixture f;f.crop(S);f.c.plans[0]=plan({action(Op::FERTILIZE),action(Op::WATER)});f.apply();check(f.c.idle_handoff_applied==0,"future fertilize precedence lost");}
 {Fixture f(104,3);f.crop(S);f.c.plans[1]=plan({action(Op::WATER)});f.out.units[2]=action(Op::FERTILIZE);f.apply();check(f.out.units[0].op==Op::PASS,"actual later fertilizer precedence lost");}
 {Fixture f;f.crop();f.c.plans[0]=plan({action(Op::WATER)});f.out.units[1]=action(Op::NORTH);f.apply();check(f.out.units[1].op==Op::NORTH&&f.c.plans[0].a.size()==1,"busy worker preempted");}
 {Fixture f;f.cow();f.c.plans[0]=plan({action(Op::HARVEST),action(Op::DROP)});f.apply();check(f.c.idle_handoff_applied==0,"promised delivery lost");}
 {Fixture f;f.cow();f.c.plans[0]=plan({action(Op::HARVEST),action(Op::DROP)});f.c.plans[1]=plan({action(Op::DROP)});f.apply();check(f.out.units[1].op==Op::HARVEST&&f.c.plans[0].a.size()==1,"safe output handoff missing");auto after=f.e.project_unit_phase(0,f.out.units);check(after.privates()[0].inventories[1][MI]==3,"recipient products mismatch");}
 {Fixture f;f.cow();f.c.plans[0]=plan({action(Op::COLLECT_FERTILIZER)});f.apply();check(f.out.units[1].op==Op::COLLECT_FERTILIZER,"unpromised eod collection rejected");}
 {Fixture f;f.cow();f.c.plans[0]=plan({action(Op::FEED),action(Op::COLLECT_FERTILIZER)});f.apply();check(f.c.idle_handoff_applied==0,"earlier same plot task discarded");}
 {Fixture f;f.cow();f.c.phase=1;f.c.plans[0]=plan({action(Op::COLLECT_FERTILIZER)});f.apply();check(f.c.idle_handoff_checks==0,"preparation mutated");}
 {Fixture f(700);f.cow();f.c.plans[0]=plan({action(Op::HARVEST)});f.apply();check(f.c.idle_handoff_applied==0,"terminal unbanked harvest");}
 {Fixture f;f.crop();f.c.plans[0]=plan({action(Op::HARVEST)});f.c.plans[1]=plan({action(Op::WATER)});f.apply();check(f.out.units[0].op==Op::WATER&&f.out.units[1].op==Op::HARVEST,"finite harvest did not preserve preceding water");auto after=f.e.project_unit_phase(0,f.out.units);check(after.privates()[0].inventories[1][W]==3,"same-turn water yield lost");}
 {Fixture f;f.cow();f.c.plans[0]=plan({action(Op::PASS),action(Op::COLLECT_FERTILIZER),action(Op::CARE)});f.c.plans[0].index=1;f.apply();check(f.out.units[1].op==Op::COLLECT_FERTILIZER&&f.c.plans[0].index==1&&f.c.plans[0].a[1].op==Op::CARE,"nonzero index handoff corrupts suffix");}
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";
}
