#include "policy/triad.hpp"
#include <cassert>
#include <iostream>
using namespace triad;
int main(){
 Farm own,other;own.tiles.resize(100);other.tiles.resize(100);
 own.farmer={4,4};other.farmer={4,4};own.money=1000;
 for(int p=0;p<100;p++)if(quad(p)!=0){own.tiles[p].kind=TileKind::LOCKED;other.tiles[p].kind=TileKind::LOCKED;}
 PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);priv.shed[W]=1;priv.shed[CO]=1;
 Market m;for(int i=0;i<9;i++){m.inventory[i]=10000;m.prices[i]=price(i,10000);}std::vector<int8_t> shops;
 View v{24,1,0,own,other,priv,m,shops};
 auto run=[&](int mode){
  Settings s;s.a06_reinvest=mode;s.scenario=0;
  triad::Controller c(s);c.core.day=1;c.model.day=1;c.core.phase=3;c.core.plans.resize(1);
  c.core.pending_admission={true,1,24,0,34,CO};
  auto job=dp7::intraday::project_job(c.core,v,34,CO);
  assert(job.needs[W]==1&&job.needs[CO]==1);
  PlayerAction a;a.units.resize(1);c.settle_market(v,a,false);
  int sold=0;for(auto x:a.market)if(x.op==Op::SELL&&x.item==Item::WHEAT)sold+=x.quantity;
  std::cout<<"mode="<<mode<<" pending_wheat="<<job.needs[W]<<" sell_wheat="<<sold<<"\n";
  return sold;
 };
 assert(run(0)==1);assert(run(1)==0);assert(run(2)==0);assert(run(3)==0);assert(run(4)==0);assert(run(5)==0);
 std::cout<<"PASS pending paid feed preserved through final SELL regeneration\n";
 // Synthetic observation-unit tests for the receipt gate, not benchmark games.
 {Settings s;s.a06_reinvest=4;s.scenario=0;triad::Controller c(s);
  own.money=3000;View a{0,0,0,own,other,priv,m,shops};c.act(a);assert(c.a06_received_today==0);
  own.money=2800;View b{1,0,1,own,other,priv,m,shops};c.act(b);assert(c.a06_received_today==0);
  own.money=3100;View d{2,0,2,own,other,priv,m,shops};c.act(d);assert(c.a06_received_today==300);
  own.money=4000;View e{24,1,0,own,other,priv,m,shops};c.act(e);assert(c.a06_received_today==0);
  std::cout<<"PASS receipt gate: opening0, purchase0, actual increase300, newday reset0\n";
 }
}
