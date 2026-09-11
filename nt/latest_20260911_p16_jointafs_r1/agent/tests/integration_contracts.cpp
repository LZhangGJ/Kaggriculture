#include "policy/search.hpp"
#include "policy/observation_codec.hpp"
#include <fstream>
#include <cstring>
#include <iostream>
static int checks=0;
static void require(bool ok,const char*why){++checks;if(!ok)throw std::runtime_error(why);}
static std::vector<double> load(const char*path){std::ifstream f(path);if(!f)throw std::runtime_error("missing fixture");std::vector<double>v;double x;while(f>>x)v.push_back(x);return v;}
int main(int argc,char**argv){try{
 if(argc!=3)throw std::runtime_error("handoff and settings paths required");
 auto packed=load(argv[1]),config=load(argv[2]);require(config.size()==38,"complete ABI38 settings");triad::Settings settings;std::memcpy(&settings,config.data(),sizeof settings);
 Cursor r{packed.data(),packed.size()};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();auto fa=r.farm(),fb=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int count=r.count();for(int i=0;i<count;i++)shops.push_back(r.integer());require(r.i==packed.size(),"fixture completely decoded");
 auto own=seat?fb:fa,other=seat?fa:fb;dp7::View v{step,day,hour,own,other,priv,market,shops};
 require(T3_OBLIGATION_REPAIR&&T3_CAPACITY_REPAIR&&T3_RECEIPT_REPAIR,"all three T3 repair families retained");
 require(P16_WORKING_CAPITAL_GATE&&P16_LIVE_REMAINING_VALUE,"both transferred active repairs enabled");
 triad::Controller controller(settings);auto a=controller.act(v);
 require(bool(controller.core.t3.project_sales),"sale snapshot installed even on funding early return");
 require(controller.capital_pending,"funding continuation survives the early return");
 require(controller.core.t3.receipt_day==day,"preparation receipt belongs to current day");
 require(own.money==26&&priv.shed[0]==3,"real collateral operation did not mutate observation");
 auto detached=controller;detached.core.t3.suspended=true;detached.capital_pending=false;
 require(controller.capital_pending&&!controller.core.t3.suspended,"copied simulation controls independent of live instance");
 auto deadlines=controller.local_sale.deadline;dp7::Counts sells{};sells[7]=10;auto snapshot=controller.core.t3.project_sales;snapshot(v,priv,{},sells);
 require(controller.local_sale.deadline==deadlines,"simulated sale cannot mutate live sale deadlines");
 fastkag::ObservedDayScenario world(v);world.advance(a);int feeds=0,hires=0,underfilled=0;
 while(!world.finished()){
  auto current=world.view();auto out=controller.act(current);
  require(out.market.size()<=10,"joint market slot limit preserved");
  for(auto u:out.units)feeds+=u.op==fastkag::Op::FEED;
  world.advance(out);const auto&fills=world.fills();
  for(size_t i=0;i<out.market.size();i++){auto order=out.market[i];if(order.op==fastkag::Op::HIRE){hires++;underfilled+=fills[i]<1;}}
 }
 int preserved=0;for(int pos=0;pos<100;pos++)if(dp7::animal(own.tiles[pos])){auto&t=world.own().tiles[pos];preserved+=dp7::animal(t)&&t.animal==own.tiles[pos].animal;}
 require(preserved==4,"all four original animals survive funding + T3 execution");
 require(feeds>=4,"selected four feed commitments reach actions");
 require(underfilled==0,"no duplicate/unfunded hire after capital reconciliation in fixture");
 require(world.own().money>=0,"no fabricated credit");
 require(controller.capital_financings>=1,"true collateral financing path exercised");
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<",\"feed_actions\":"<<feeds<<",\"hire_actions\":"<<hires<<",\"hire_shortfalls\":"<<underfilled<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"INTEGRATION FAIL after "<<checks<<": "<<e.what()<<"\n";return 1;}}
