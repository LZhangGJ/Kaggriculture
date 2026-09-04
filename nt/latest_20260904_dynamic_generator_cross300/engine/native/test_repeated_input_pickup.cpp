// Read-only reproduction of a remaining route-normalisation limitation.
// Synthetic fixture only; never injected into a live evaluation game.
#include "policy.hpp"
#include <iostream>
using namespace dp7;
int main(){
 Simulator env(Config{},987611);auto&farm=const_cast<Farm&>(env.farms()[0]);farm.farmer={3,3};
 auto&priv=const_cast<PrivateState&>(env.privates()[0]);priv.inventories[0][W]=1;priv.inventory_order[0]={W};priv.shed[W]=1;
 for(int pos:{33,34}){auto&t=farm.tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item(CO);t.placed_day=0;}
 View v{0,0,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
 Controller c;c.day=0;c.phase=3;c.target={{33,CO},{34,CO}};
 Plan original;int pos=33;original.a.push_back(action(Op::FEED));original.target.push_back(pos);
 Controller::walk(original,pos,44);original.a.push_back(action(Op::PICKUP,W,1));original.target.push_back(44);
 Controller::walk(original,pos,34);original.a.push_back(action(Op::FEED));original.target.push_back(34);
 int stale=0;Plan repaired=c.repair_plan(v,0,original,stale);
 auto run=[&](const Plan&p){auto e=env;for(auto x:p.a){PlayerAction a;a.units={x};e.step({a,PlayerAction{}});}return int(e.farms()[0].tiles[33].fed_today)+int(e.farms()[0].tiles[34].fed_today);};
 auto pickups=[](const Plan&p){return std::count_if(p.a.begin(),p.a.end(),[](auto a){return a.op==Op::PICKUP;});};
 int a=run(original),b=run(repaired);bool ok=a==2&&b==2&&pickups(repaired)==1;
 std::cout<<"{\"status\":\""<<(ok?"PASS":"FAIL")<<"\",\"original_fed\":"<<a<<",\"repaired_fed\":"<<b<<",\"original_pickups\":"<<pickups(original)<<",\"repaired_pickups\":"<<pickups(repaired)<<",\"stale_groups\":"<<stale<<"}\n";
 return ok?0:1;
}
