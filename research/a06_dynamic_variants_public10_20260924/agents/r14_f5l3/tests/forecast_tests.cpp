#include "policy/public_flow_scenario.hpp"
#include "policy/r12_calendar.hpp"
#include <iostream>
#include <stdexcept>
#define REQUIRE(x) do { if(!(x)) throw std::runtime_error(#x); } while(0)
int main(){
 using namespace fastkag;
 triad::SaleClock c;
 for(int item=1;item<=7;item++){
  auto p=c.rival[item].density();REQUIRE(p[1]==.5&&p[17]==.5);
  c.rival[item].add(2,4,5);c.rival[item].add(3,9,7);c.rival[item].add(4,20,11);
  auto d=c.rival[item].density();REQUIRE(std::abs(std::accumulate(d.begin(),d.end(),0.)-1)<1e-12);
 }
 for(int item=0;item<9;item++)for(double amount:{-11.8,-2.,0.,.3,1.,3.6,17.3,65.8}){
  int total=0;for(int hour=0;hour<24;hour++)total+=c.quantity(item,amount,hour);
  REQUIRE(total==int(std::round(amount)));
 }
 std::cout<<"PASS: observed-hour density normalization and daily quantity conservation\n";
 Simulator env(Config{},123);
 auto own=env.farms()[0],rival=env.farms()[1];auto priv=env.privates()[0];auto market=env.market();std::vector<int8_t>shops;
 auto &t=rival.tiles[22];t=Tile{};t.kind=TileKind::PLANT;t.crop=Item::MELON;t.planted_day=0;t.watered_today=true;t.yield_units=6;
 dp7::View v{257,10,17,own,rival,priv,market,shops};competitive::Planner planner;competitive::Flow f{};
 PublicFlowScenario keep(v,f,.85,nullptr,&planner,0),retire(v,f,.85,nullptr,&planner,4);
 PlayerAction pass;keep.advance(pass);retire.advance(pass);
 REQUIRE(keep.view().opponent.tiles[22].kind==TileKind::PLANT);
 REQUIRE(retire.view().opponent.tiles[22].kind!=TileKind::PLANT);
 REQUIRE(retire.view().own.money==keep.view().own.money);
 REQUIRE(rival.tiles[22].kind==TileKind::PLANT); // input and real referee untouched
 REQUIRE(retire.view().step==258);
 std::cout<<"PASS: retirement changes only synthetic rival crop, not input state\n";
 planner.day=10;planner.cfg.r14_forecast=5;planner.cfg.competition=2;planner.cfg.supply=.85;
 planner.rival_early_share.fill(.95);planner.demand(v);planner.public_rival(v);
 competitive::Asset a; a.f[12][dp7::M]=30;a.f[17][dp7::WO]=7;a.labor[11]=12;
 auto value=planner.value(v,a);auto parts=triad::r12::decompose(planner,v,a);
 REQUIRE(std::abs(value-parts.score)<1e-8);
 std::cout<<"PASS: diagnostic score decomposition matches active primary objective\n";
 return 0;
}
