#include "policy/search.hpp"
#include <cassert>
#include <iostream>
using namespace dp7;
struct State {
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;
 int step=48;
 State(int workers=1){
  own.tiles.resize(100);rival.tiles.resize(100);own.farmer={4,4};rival.farmer={4,4};
  own.unlocked_mask=rival.unlocked_mask=15;
  for(int i=1;i<workers;i++)own.hands.push_back({4,4});
  priv.inventories.resize(workers);priv.inventory_order.resize(workers);
  for(int i=0;i<9;i++)market.prices[i]=price(i,0);
 }
 View view()const{return {step,step/24,step%24,own,rival,priv,market,shops};}
 void bag(int u,int item,int q){priv.inventories[u][item]=q;priv.inventory_order[u].push_back(item);}
};
triad::Controller controller(const State&s){triad::Controller c;c.core.day=s.step/24;c.core.phase=3;c.core.plans.resize(s.priv.inventories.size());return c;}
int main(){
 using namespace triad::route_detail;
 int tests=0;
 {
  State s(2);s.priv.shed[E]=90;s.bag(0,E,10);s.bag(1,E,10);auto c=controller(s);
  PlayerAction a{{action(Op::DROP),action(Op::DROP)},{action(Op::SELL,E,110)}};
  fastkag::ObservedDayScenario before(s.view());auto lossy=before.project_units(a.units,-1);
  assert(sum(lossy.privates()[0].shed)==100&&sum(lossy.privates()[0].inventories[1])==0);
  c.route_capacity(s.view(),a);auto safe=before.project_units(a.units,-1);
  assert(sum(safe.privates()[0].shed)==100&&safe.privates()[0].inventories[1][E]==10);
  assert(a.units[1].op==Op::PASS);tests++;
 }
 {
  State s;s.priv.shed[E]=95;s.bag(0,E,10);auto c=controller(s);
  PlayerAction a{{action(Op::DROP)},{}};c.route_capacity(s.view(),a);
  fastkag::ObservedDayScenario w(s.view());auto safe=w.project_units(a.units,-1);
  assert(a.units[0].op==Op::PLACE&&safe.privates()[0].inventories[0][E]==5);tests++;
 }
 {
  State s;s.bag(0,E,8);s.bag(0,W,1);s.own.tiles[43].kind=TileKind::ANIMAL;s.own.tiles[43].animal=Item::GOOSE;
  auto c=controller(s);c.core.plans[0]={{action(Op::WEST),action(Op::FEED)},{43,43},0};
  auto choices=deliveries(s.view(),c.core.plans);assert(!choices.empty());bool selective=false;
  for(auto&d:choices)for(auto a:d.plan.a)if(a.op==Op::PLACE&&int(a.item)==E)selective=true;
  assert(selective);tests++;
 }
 {
  State s;s.own.money=0;s.priv.shed[E]=10;auto c=controller(s);c.core.phase=2;
  PlayerAction a{{action(Op::PASS)},{action(Op::HIRE)}};c.route_finance(s.view(),a);
  fastkag::ObservedDayScenario w(s.view());w.advance(a);
  assert(w.own().hands.size()==1&&a.market.front().op==Op::SELL);tests++;
 }
 {
  State s;s.own.money=0;s.bag(0,E,10);auto c=controller(s);c.core.phase=2;
  PlayerAction a{{action(Op::PASS)},{action(Op::HIRE)}};c.route_finance(s.view(),a);
  fastkag::ObservedDayScenario w(s.view());w.advance(a);
  assert(w.own().hands.empty()); // Carried inventory is not cash.
  assert(unfilled(a.market,w.fills())==1);tests++;
 }
 {
  State s;s.own.money=1;auto c=controller(s);c.core.phase=2;
  PlayerAction a{{action(Op::PASS)},{action(Op::BUY_ANIMAL,CO),action(Op::HIRE)}};
  c.route_finance(s.view(),a);fastkag::ObservedDayScenario w(s.view());w.advance(a);
  assert(w.own().hands.size()==1);tests++;
 }
 {
  triad::Controller::RouteValue a,b;auto k=dp7::Controller::task_key(43,action(Op::WATER));
  a.completed[k]=1;a.moves=10;a.assets=100;b.moves=5;b.assets=100;
  assert(!preserves(b,a));b.completed[k]=1;b.overflow=1;assert(!preserves(b,a));tests++;
 }
 {
  State s;auto c=controller(s);PlayerAction a{{action(Op::PASS)},{action(Op::HIRE)}};
  c.route_finance(s.view(),a);s.step++;c.route_reconcile(s.view());
  assert(c.logistics.fill_mismatches==1&&c.logistics.dirty);tests++;
 }
 {
  State s;s.step=718;s.priv.shed[E]=10;fastkag::PublicFlowScenario w(s.view(),{},0);
  w.advance({});assert(w.done()&&w.liquidation_quote()==0);tests++;
 }
 {
  State s;s.step=71;auto c=controller(s);c.logistics.probe=true;
  auto v=c.route_evaluate(s.view(),true);assert(v.ticks==1&&std::isfinite(v.score));tests++;
 }
 {
  State s;s.own.tiles[43].kind=TileKind::ANIMAL;s.own.tiles[43].animal=Item::GOOSE;s.own.tiles[43].yield_units=2;
  auto c=controller(s);Plan p{{action(Op::WEST),action(Op::HARVEST),action(Op::EAST),action(Op::PLACE,E,2)},{43,43,44,44},0};
  int stale=0;auto fixed=c.core.repair_plan(s.view(),0,p,stale);
  assert(std::any_of(fixed.a.begin(),fixed.a.end(),[](auto a){return a.op==Op::PLACE&&int(a.item)==E;}));tests++;
 }
 {
  State s(2);s.priv.seeds[T]=1;
  PlayerAction a{{action(Op::PLANT,T),action(Op::PLANT,T)},{}};
  fastkag::ObservedDayScenario w(s.view());auto before=w.project_units(a.units,0),after=w.project_units(a.units,1);
  assert(effect(before,after,0,a.units[0])==0);tests++;
 }
 {
  State s;s.own.tiles[43].kind=TileKind::ANIMAL;s.own.tiles[43].animal=Item::GOOSE;s.own.tiles[43].fertilizer_available=true;
  auto c=controller(s);auto key=dp7::Controller::task_key(43,action(Op::COLLECT_FERTILIZER));
  c.logistics.required[key]=1;c.route_restore(s.view());
  assert(c.logistics.restored==1);auto future=c.route_evaluate(s.view(),false);
  assert(future.completed[key]==1&&outstanding(c,future));tests++;
 }
 {
  State s;s.own.tiles[43].kind=TileKind::PLANT;s.own.tiles[43].crop=Item::STRAWBERRY;s.bag(0,F,1);
  auto c=controller(s);auto key=dp7::Controller::task_key(43,action(Op::FERTILIZE));
  c.logistics.required[key]=1;c.route_restore(s.view());
  assert(c.logistics.restored==1&&inputs(s.view(),c.core.plans));tests++;
 }
 {
  triad::Controller::RouteValue v;auto invented=dp7::Controller::task_key(81,action(Op::PLANT,S));
  auto committed_key=dp7::Controller::task_key(43,action(Op::WATER));v.completed[invented]=v.completed[committed_key]=1;
  restrict_work(v,{{committed_key,1}});assert(v.completed.size()==1&&v.completed.count(committed_key));tests++;
 }
 {
  State s;s.step=71;s.priv.shed[E]=80;s.bag(0,E,40);auto c=controller(s);
  PlayerAction a{{action(Op::PASS)},{}};c.route_finance(s.view(),a);
  fastkag::ObservedDayScenario w(s.view());w.advance(a);assert(w.overflow()==0);tests++;
 }
 {
  State s;s.own.money=0;s.priv.shed[E]=100;auto c=controller(s);c.core.phase=2;
  PlayerAction a{{action(Op::PASS)},Acts(10,action(Op::HIRE))};c.route_finance(s.view(),a);
  assert(a.market.size()<=10&&c.core.queue.size()==1&&c.core.phase==1);
  fastkag::ObservedDayScenario w(s.view());w.advance(a);assert(w.own().hands.size()==9);
  PlayerAction b{Acts(10,action(Op::PASS)),c.core.queue};c.route_finance(w.view(),b);w.advance(b);
  assert(w.own().hands.size()==10);tests++;
 }
 {
  State s;auto c=controller(s);c.core.phase=1;
  for(int pos=0;pos<20;pos++){
   s.own.tiles[pos].kind=TileKind::PLANT;s.own.tiles[pos].crop=Item::TOMATO;
   s.own.tiles[pos].fertilized_until_day=4;s.own.tiles[pos].consecutive_unwatered=1;c.core.target.emplace_back(pos,T);
  }
  assert(!prepared_complete(c,s.view()));tests++;
 }
 {
  State s;auto c=controller(s);PlayerAction a{{action(Op::PASS)},{action(Op::HIRE)}};
  c.route_finance(s.view(),a);fastkag::ObservedDayScenario w(s.view());w.advance(a);
  c.route_reconcile(w.view());assert(c.logistics.fill_mismatches==0&&c.core.plans.size()==2);tests++;
 }
 {
  // Match the official first-yield boundary for every crop, including finite
  // crops whose yield_units is already positive one day before maturity.
  for(int crop=0;crop<5;crop++)for(int offset:{-1,0}){
   State s;s.step=12*24+1;auto&t=s.own.tiles[44];t.kind=TileKind::PLANT;
   t.crop=Item(crop);t.planted_day=12-first[crop]-offset;t.yield_units=1;
   auto c=controller(s);auto a=action(Op::HARVEST);Counts pickups{},seeds{};
   bool ready=offset==0;
   assert(c.core.semantic_needed(s.view(),0,a,44)==ready);
   assert(bool(c.core.valid(s.view(),0,a,44,pickups,seeds))==ready);
   fastkag::ObservedDayScenario w(s.view());auto before=w.project_units({a},0),after=w.project_units({a},1);
   assert((effect(before,after,0,a)>0)==ready);
  }
  tests++;
 }
 std::cout<<"PASS "<<tests<<" route economics checks\n";
}
