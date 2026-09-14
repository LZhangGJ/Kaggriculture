#include "../policy/triad.hpp"
#include <iostream>
#include <cstdlib>
using namespace dp7;
static int checks=0;
static void ck(bool x,const char*m){++checks;if(!x){std::cerr<<"FAIL "<<m<<"\n";std::exit(1);}}
struct Fixture {
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;Controller c;
 Fixture(int hands=0){own.tiles.resize(100);rival.tiles.resize(100);own.farmer={4,4};own.unlocked_mask=15;own.money=2000;
  for(int u=0;u<hands;u++)own.hands.push_back({4,4});priv.inventories.resize(hands+1);priv.inventory_order.resize(hands+1);
  market.inventory.fill(10000);for(int k=0;k<9;k++)market.prices[k]=price(k,10000);
  c.day=29;c.phase=3;c.p.stepwise_recoordination=true;c.p.fix_logistics=true;c.plans.resize(hands+1);
 }
 View view(int step=708){return{step,step/24,step%24,own,rival,priv,market,shops};}
 void crop(int pos,int yield=3){auto&t=own.tiles[pos];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=26;t.yield_units=yield;t.fertilized_until_day=-1;t.max_lifespan_step=744;}
 void bag(int u,int item,int n){priv.inventories[u][item]=n;priv.inventory_order[u].push_back(item);}
};
static Plan route(int start,std::initializer_list<std::pair<int,Op>>actions){Plan p;int pos=start;
 for(auto[d,op]:actions){Controller::walk(p,pos,d);p.a.push_back(action(op));p.target.push_back(d);}return p;}
int main(){
 // Exhaustive actual last-action boundary, independent of replay coordinates.
 for(int pos=0;pos<100;pos++)for(int limit=1;limit<=23;limit++){
  Fixture f;f.own.farmer={int16_t(pos%10),int16_t(pos/10)};auto o=f.view(719-limit);
  auto p=route(pos,{{pos,Op::HARVEST}});bool fits=terminalroll::close_delivery(o,0,p,limit);
  ck(fits==(near(pos)+2<=limit),"harvest plus return plus DROP must fit last action718");
  ck(int(p.a.size())==near(pos)+2,"exact added depot path and DROP cost");
  ck(p.a.back().op==Op::DROP&&at_depot(p.target.back()),"generated output chain ends at an actual depot");
 }
 {Fixture f;f.bag(0,W,2);auto o=f.view();auto p=route(44,{{44,Op::DROP},{34,Op::HARVEST}});
  ck(terminalroll::close_delivery(o,0,p,11),"post-DROP harvest closed");
  ck(std::count_if(p.a.begin(),p.a.end(),[](auto a){return a.op==Op::DROP;})==2,"second output needs second DROP");}
 {Fixture f;f.crop(34);auto o=f.view();auto p=route(44,{{34,Op::WATER},{34,Op::HARVEST},{44,Op::DROP}});auto before=p.a.size();
  ck(terminalroll::close_delivery(o,0,p,11)&&p.a.size()==before,"existing post-harvest DROP reused without redundant depot");}
 {Fixture f;auto o=f.view();auto p=route(44,{{34,Op::HARVEST},{44,Op::DROP}});Job j{85,{action(Op::HARVEST)}};
  auto t=terminalroll::insert_chain(o,0,p,j,1);ck(t.a.back().op==Op::DROP&&t.target.back()==55,"new final crop chooses its actual nearest shared depot");
  ck(int(t.a.size())==1+1+6+1+3+1,"rebased DROP travel is charged exactly");}
 {Fixture f(1);f.crop(34);auto o=f.view();std::vector<Plan>p{route(44,{{34,Op::FERTILIZE}}),route(44,{{34,Op::WATER},{34,Op::HARVEST},{44,Op::DROP}})};
  ck(terminalroll::chain_order(o,p),"same-tick lower-unit fertilizer precedes water");std::swap(p[0],p[1]);
  ck(!terminalroll::chain_order(o,p),"higher-unit fertilizer after water rejected");
  p={route(44,{{34,Op::HARVEST},{44,Op::DROP}}),route(44,{{34,Op::WATER}})};ck(!terminalroll::chain_order(o,p),"cross-worker finite harvest cannot precede its water");}
 {Fixture f;f.crop(34);auto o=f.view();std::vector<Plan>base(1);auto j=Job{34,{action(Op::WATER),action(Op::HARVEST)}};j.out[W]=4;
  int n=terminalroll::augment(f.c,o,base,{j},11);ck(n==1,"valuable complete chain admitted");auto q=terminalroll::quote(f.c,o,base);
  ck(q.known&&q.harvest[34]==4&&q.cash>o.own.money,"cash requires successful harvest and actual market kernel sale");
  ck(sum(q.terminal_stock)==0&&base[0].a.back().op==Op::DROP,"new output is sold, not stranded");
  ck(f.own.money==2000&&f.own.tiles[34].yield_units==3,"quote never mutates observation");}
 {Fixture f;f.crop(34);auto o=f.view(717);std::vector<Plan>base(1);Job j{34,{action(Op::HARVEST)}};j.out[W]=3;
  ck(terminalroll::augment(f.c,o,base,{j},2)==0&&base[0].a.empty(),"impossible last-ticks harvest not installed");}
 {Fixture f;f.crop(34);f.own.tiles[34].crop=Item::STRAWBERRY;f.own.tiles[34].planted_day=10;f.market.inventory[S]=1000000;f.market.prices[S]=price(S,1000000);auto o=f.view();std::vector<Plan>base(1);Job j{34,{action(Op::HARVEST)}};j.out[S]=3;ck(f.market.prices[S]==1,"fixture actually reaches the strawberry price floor");
  ck(terminalroll::augment(f.c,o,base,{j},11)==1,"floor-price harvest still earns actual receipts with sunk labor");auto q=terminalroll::quote(f.c,o,base);ck(q.cash==2003,"floor sale pays one per unit, not zero");}
 {Fixture f;f.crop(34,6);auto o=f.view();std::vector<Plan>base{route(44,{{34,Op::HARVEST},{44,Op::DROP}})};Job j{34,{action(Op::WATER)}};
  ck(terminalroll::augment(f.c,o,base,{j},11)==0,"zero marginal terminal benefit water on max-yield crop is rejected");}
 {Fixture f;f.crop(34);f.priv.shed[G]=100;auto o=f.view();std::vector<Plan>base(1);Job j{34,{action(Op::HARVEST)}};j.out[W]=3;
  ck(terminalroll::augment(f.c,o,base,{j},11)==0,"unsellable full shed blocks final deposit despite geometric fit");}
 {Fixture f;f.crop(34);f.bag(0,F,1);auto o=f.view();std::vector<Plan>base{route(44,{{34,Op::FERTILIZE}})};Job j{34,{action(Op::WATER),action(Op::HARVEST)}};j.out[W]=5;
  ck(terminalroll::augment(f.c,o,base,{j},11)==1,"already-carried fertilizer task retained ahead of new harvest chain");auto q=terminalroll::quote(f.c,o,base);
  ck(q.fert[34]==1&&q.water[34]==1&&q.harvest[34]==5,"resource consumption and real production bonus preserved");}
 {Fixture f(1);f.crop(44);f.own.hands[0]={4,4};auto o=f.view();std::vector<Plan>p{route(44,{{44,Op::WATER}}),route(44,{{44,Op::HARVEST},{44,Op::DROP}})};
  auto q=terminalroll::quote(f.c,o,p);ck(q.water[44]==1&&q.harvest[44]==4,"ordered effects count same-tick water and harvest exactly");}
 {Fixture f;f.crop(34);auto o=f.view();std::vector<Plan>base{route(44,{{34,Op::HARVEST},{44,Op::DROP}})};Job j{34,{action(Op::HARVEST)}};j.out[W]=3;
  ck(terminalroll::missing_jobs(base,{j}).empty(),"covered harvest not duplicated");ck(terminalroll::augment(f.c,o,base,{j},11)==0,"complete covered route unchanged");}
 {Fixture f;f.crop(34);auto o=f.view();Job j{34,{action(Op::FERTILIZE),action(Op::WATER),action(Op::HARVEST)}};j.needs[F]=1;j.out[W]=5;
  ck(terminalroll::missing_jobs(std::vector<Plan>(1),{j}).empty(),"unfunded material task not erased into an apparently free service");}
 {Fixture f;f.bag(0,W,7);auto o=f.view();std::vector<Plan>base(1);ck(terminalroll::augment(f.c,o,base,{},11)==1,"idle cargo receives valid return and sale chain");auto q=terminalroll::quote(f.c,o,base);ck(sum(q.terminal_stock)==0&&q.cash>o.own.money,"idle cargo realized rather than marked to market");}
 {Fixture f;f.crop(34);f.priv.inventory_order.clear();auto o=f.view();std::vector<Plan>base(1);Job j{34,{action(Op::HARVEST)}};
  ck(terminalroll::augment(f.c,o,base,{j},11)==0&&f.c.r4_route_errors==1&&base[0].a.empty(),"invalid conditional input fails closed to unchanged repaired route");
  ck(f.c.r4_route_conditional_gain==0.&&f.c.r4_route_insertions==0,"failed quote commits neither route nor gain telemetry");}
 {Fixture f;auto o=f.view(695);std::vector<Plan>base(1);ck(terminalroll::augment(f.c,o,base,{},24)==0&&f.c.r4_route_checks==0,"ordinary-day scheduling untouched");}
 std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"scope\":\"terminal complete chains, resource/precedence/capacity/cash and error negatives\"}\n";
 return 0;
}
