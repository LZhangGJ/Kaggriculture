#include "policy/search.hpp"
#include <cassert>
#include <iostream>
using namespace triad;
static bool acts_equal(const PlayerAction&a,const PlayerAction&b){
 auto eq=[](const auto&x,const auto&y){if(x.size()!=y.size())return false;for(size_t i=0;i<x.size();i++)if(x[i].op!=y[i].op||x[i].item!=y[i].item||x[i].quantity!=y[i].quantity)return false;return true;};
 return eq(a.units,b.units)&&eq(a.market,b.market);
}
int main(){
 Farm own,other;own.tiles.resize(100);other.tiles.resize(100);own.unlocked_mask=other.unlocked_mask=1;own.farmer=other.farmer={4,4};own.money=3000;other.money=3000;
 for(int p=0;p<100;p++)if(quad(p)!=0){own.tiles[p].kind=TileKind::LOCKED;other.tiles[p].kind=TileKind::LOCKED;}
 PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);priv.seeds[W]=2;priv.shed[CO]=1;priv.shed[W]=2;
 Market market;for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=price(i,10000);}std::vector<int8_t>shops;
 View v{24,1,0,own,other,priv,market,shops};
 Settings s;s.scenario=1;s.a06_reinvest=4;s.a06_roll_scope=2;s.a06_execution_candidates=1;s.a06_feed_contract=0;s.a06_state_contract=0;s.portfolio_passes=6;s.candidate_extra=1;
 SearchController joint(s);auto prepared=joint.prepare(v);Settings old=s;old.a06_execution_candidates=0;SearchController control(old);auto original=control.prepare(v);
 assert(!original.empty()&&prepared.size()==2*original.size());size_t prefix_steps=0;
 for(size_t i=0;i<original.size();i++){
  for(int k=0;k<2;k++){
   auto&p=prepared[2*i+k];assert(p.id==original[i].id+16*k);assert(proposal_key(p.policy)==proposal_key(original[i].policy));assert(p.policy.s.a06_roll_scope==0&&p.policy.s.a06_reinvest==(k?1:4));
   joint.install(p,v.day);assert(joint.live.s.a06_reinvest==p.policy.s.a06_reinvest&&joint.live.s.a06_roll_scope==0);assert(joint.restore_day==2);
   auto live=p.policy,roll=p.policy;roll.call_origin=triad::Controller::CallOrigin::PublicPrediction;
   fastkag::ObservedDayScenario world_live(v),world_roll(v);
   for(int t=0;t<12;t++){
    auto a=live.act(world_live.view()),b=roll.act(world_roll.view());assert(acts_equal(a,b));world_live.advance(a);world_roll.advance(b);
    assert(world_live.own().money==world_roll.own().money);assert(world_live.inventory().shed==world_roll.inventory().shed);assert(world_live.inventory().seeds==world_roll.inventory().seeds);assert(world_live.inventory().inventories==world_roll.inventory().inventories);assert(world_live.fills()==world_roll.fills());prefix_steps++;
   }
  }
 }
 // The first issued unit phase consumes a prepaid seed exactly once, does not
 // advance time, borrow unexecuted market receipts, or touch the input fixture.
 PlayerAction issued;issued.units={action(Op::PLANT,W)};issued.market={action(Op::SELL,W,2)};
 fastkag::ObservedDayScenario before(v);auto projected=before.project_units(issued.units,-1);
 assert(projected.privates()[0].seeds[W]==1);assert(priv.seeds[W]==2);assert(projected.farms()[0].money==own.money);assert(projected.privates()[0].shed[W]==priv.shed[W]);assert(plant(projected.farms()[0].tiles[44]));assert(projected.step_count()==v.step);
 s.a06_state_contract=1;triad::Controller c(s);c.plan(v);c.core.plans.resize(1);c.core.plans[0].a.clear();c.core.plans[0].target.clear();c.core.plans[0].index=0;
 auto real=c,predict=c;predict.call_origin=triad::Controller::CallOrigin::PublicPrediction;
 auto vr=real.a06_live_intraday_values(real.core,v,issued),vp=predict.a06_live_intraday_values(predict.core,v,issued);assert(vr==vp);assert(real.a06_contract_calls==1&&predict.a06_contract_calls==1);assert(priv.seeds[W]==2&&own.money==3000);
 std::cout<<"PASS "<<prepared.size()<<" economic execution candidates preserve original plan keys, paid commitments and install mode\n";
 std::cout<<"PASS "<<prefix_steps<<" matched known-day prefix transitions: same mechanism/action/cash/fills/stock across origins\n";
 std::cout<<"PASS exact issued PLANT seed consumption, no sale credit, input immutability, phase-contract origin equality\n";
}
