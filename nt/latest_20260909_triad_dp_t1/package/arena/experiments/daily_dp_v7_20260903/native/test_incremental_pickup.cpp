#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;void check(bool b,const char*m){checks++;if(!b)throw std::runtime_error(m);}
int pickups(const Plan&p){return std::count_if(p.a.begin(),p.a.end(),[](auto a){return a.op==Op::PICKUP;});}
View view(const Simulator&e){return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
Simulator execute(Simulator e,const Plan&p){for(size_t k=p.index;k<p.a.size();k++){PlayerAction a;a.units={p.a[k]};e.step({a,PlayerAction{}});}return e;}
int main(){
 for(int input:{W,F})for(int initial:{1,2,3}){
  Simulator env(Config{},781223);auto&farm=const_cast<Farm&>(env.farms()[0]);farm.farmer={3,3};
  auto&pr=const_cast<PrivateState&>(env.privates()[0]);pr.inventories[0][input]=initial;pr.inventory_order[0]={input};pr.shed[input]=1;
  for(int pos:{33,34}){auto&t=farm.tiles[pos];if(input==W){t.kind=TileKind::ANIMAL;t.animal=Item(CO);}else{t.kind=TileKind::PLANT;t.crop=Item(S);t.planted_day=-9;t.fertilized_until_day=-1;}}
  Plan original;int pos=33;for(int i=0;i<initial;i++){// FERTILIZE can legally consume repeated fertilizer; FEED cannot.
   if(input==W&&i>0){pr.inventories[0][input]=1;break;}
   original.a.push_back(action(input==W?Op::FEED:Op::FERTILIZE));original.target.push_back(33);
  }
  Controller::walk(original,pos,44);original.a.push_back(action(Op::PICKUP,input));original.target.push_back(44);
  Controller::walk(original,pos,34);original.a.push_back(action(input==W?Op::FEED:Op::FERTILIZE));original.target.push_back(34);
  Controller c;c.day=0;int stale=0;auto old=c.repair_plan(view(env),0,original,stale);check(pickups(old)==0,"legacy failure not reproduced");
  c.p.incremental_pickup_repair=true;stale=0;auto fixed=c.repair_plan(view(env),0,original,stale);check(pickups(fixed)==1,"incremental refill removed");
  auto e=execute(env,fixed);check(input==W?e.farms()[0].tiles[34].fed_today:e.farms()[0].tiles[34].fertilized_until_day>=0,"second service did not execute");
  check(e.privates()[0].shed[input]==0,"refill was not consumed from shed");
 }
 for(int item:{W,F,MI}){
  Simulator e(Config{},82715);auto&pr=const_cast<PrivateState&>(e.privates()[0]);pr.shed[item]=3;
  Controller c;c.day=0;c.p.incremental_pickup_repair=true;
  Plan p{{action(Op::PICKUP,item,3),action(Op::EAST),action(Op::DROP)},{44,-1,45},0};int stale=0;
  auto repaired=c.repair_plan(view(e),0,p,stale);check(std::any_of(repaired.a.begin(),repaired.a.end(),[](auto a){return a.op==Op::DROP;}),"pickup then delivery was discarded");
  auto done=execute(e,repaired);check(done.privates()[0].inventories[0][item]==0&&done.privates()[0].shed[item]==3,"pickup delivery quantity changed");
  check(!c.semantic_needed(view(e),0,action(Op::PICKUP,item,0),44),"zero quantity admitted");
  check(!c.semantic_needed(view(e),0,action(Op::PICKUP,item,-1),44),"negative quantity admitted");
  Plan partial{{action(Op::PICKUP,item,5)},{44},0};auto pa=c.repair_plan(view(e),0,partial,stale);auto pe=execute(e,pa);
  check(pe.privates()[0].inventories[0][item]==3,"partial real pickup invented resources");
  partial.index=1;auto consumed=c.repair_plan(view(pe),0,partial,stale);check(consumed.a.empty(),"executed pickup was replayed");
 }
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";
}
