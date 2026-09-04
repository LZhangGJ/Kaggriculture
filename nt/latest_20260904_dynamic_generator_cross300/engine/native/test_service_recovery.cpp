#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;
void check(bool x,const char*m){++checks;if(!x)throw std::runtime_error(m);}
struct Fixture {
 Simulator env{Config{},924713};Farm own=env.farms()[0],rival=env.farms()[1];PrivateState priv=env.privates()[0];Market market=env.market();std::vector<int8_t>shops;
 Controller c;int day=9,hour=8;
 Fixture(){c.day=day;c.phase=3;c.p.recover_service_inputs=true;own.hands.push_back({4,3});priv.inventories.resize(2);priv.inventory_order.resize(2);c.plans.resize(2);
  auto&t=own.tiles[33];t.kind=TileKind::ANIMAL;t.animal=Item(CO);t.placed_day=0;c.target={{33,CO}};}
 View view(){return {day*24+hour,day,hour,own,rival,priv,market,shops};}
 PlayerAction idle(){PlayerAction a;a.units.resize(2);return a;}
};
int main(){
 {Fixture f;f.priv.shed[W]=1;service::recover(f.c,f.view());check(f.c.service_recovered_feed==1,"owned shed not recovered");int n=f.c.service_recovered_feed;service::recover(f.c,f.view());check(f.c.service_recovered_feed==n,"duplicate recovery");
  auto a=f.idle();a.market={action(Op::SELL,W,1)};service::protect(f.c,f.view(),a);check(a.market.empty(),"reserved wheat sold");}
 {Fixture f;f.priv.shed[W]=1;f.c.plans[0]={{action(Op::PICKUP,W,1)},{44},0};service::recover(f.c,f.view());check(f.c.service_recovered_feed==0,"stole another pickup reservation");}
 {Fixture f;auto&t=f.own.tiles[33];t=Tile{};t.kind=TileKind::PLANT;t.crop=Item(S);t.planted_day=0;t.watered_today=true;f.c.target={{33,S}};f.priv.inventories[1][F]=1;
  service::recover(f.c,f.view());check(f.c.service_recovered_fertilize==1,"fertilizer on other worker not used");check(f.c.plans[0].a.empty(),"empty farmer chosen instead of actual carrier");
  f.c.plans={Plan{},Plan{}};f.c.plans[0]={{action(Op::DIG)},{33},0};service::recover(f.c,f.view());check(f.c.service_recovered_fertilize==1,"fertilized a plot scheduled for replacement");}
 {Fixture f;f.hour=23;f.priv.shed[W]=1;service::recover(f.c,f.view());check(f.c.service_recovered_feed==0,"late service admitted");f.day=f.c.day=29;f.hour=0;service::recover(f.c,f.view());check(f.c.service_recovered_feed==0,"terminal feed admitted");}
 {Fixture f;f.priv.shed[W]=1;f.c.p.recover_service_inputs=false;service::recover(f.c,f.view());check(f.c.service_checks==0&&f.c.plans[0].a.empty(),"disabled recovery changed route");}
 {Fixture f;f.c.p.procure_service_inputs=true;auto a=f.idle();service::procure(f.c,f.view(),a);check(a.market.size()==1&&a.market[0].op==Op::BUY_PRODUCT,"affordable buy missing");check(f.c.plans[0].a.empty()&&f.c.plans[1].a.empty(),"unfilled purchase prematurely executed");service::recover(f.c,f.view());check(f.c.service_recovered_feed==0,"failed buy invented inventory");f.priv.shed[W]=1;service::recover(f.c,f.view());check(f.c.service_recovered_feed==1,"actual fill not bound");}
 {Fixture f;f.c.p.procure_service_inputs=true;f.own.money=0;auto a=f.idle();service::procure(f.c,f.view(),a);check(a.market.empty(),"unaffordable buy without financing");f.c.p.finance_service_inputs=true;f.priv.shed[MI]=1;service::procure(f.c,f.view(),a);check(a.market.size()==2&&a.market[0].op==Op::SELL&&a.market[1].op==Op::BUY_PRODUCT,"shed finance order invalid");}
 {Fixture f;f.c.p.procure_service_inputs=f.c.p.finance_service_inputs=true;f.own.money=0;f.priv.inventories[0][F]=1;f.priv.inventory_order[0]={F};auto a=f.idle();service::procure(f.c,f.view(),a);
  check(a.units[0].op==Op::PLACE&&a.market.size()==2,"carrier finance missing");service::protect(f.c,f.view(),a);check(a.market.size()==2,"financing sale removed");
  Simulator env(Config{},812671);for(int s=0;s<f.day*24+f.hour;s++)env.step({PlayerAction{},PlayerAction{}});
  const_cast<Farm&>(env.farms()[0])=f.own;const_cast<PrivateState&>(env.privates()[0])=f.priv;const_cast<Market&>(env.market())=f.market;
  env.step({a,PlayerAction{}});check(env.privates()[0].shed[W]==1&&env.privates()[0].inventories[0][F]==0,"frozen official PLACE SELL BUY failed");
  auto v=View{env.step_count(),env.day(),env.hour(),env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};service::recover(f.c,v);check(f.c.service_recovered_feed==1,"financing actual fill not assigned");}
 {Fixture f;f.c.p.procure_service_inputs=f.c.p.finance_service_inputs=true;f.own.money=0;f.priv.inventories[0][F]=1;f.own.farmer={0,0};auto a=f.idle();service::procure(f.c,f.view(),a);check(a.market.empty(),"remote inventory counted as instant shed collateral");}
 {Fixture f;f.c.p.procure_service_inputs=true;auto a=f.idle();a.units[1]=action(Op::FEED);f.own.hands[0]={3,3};f.priv.inventories[1][W]=1;service::procure(f.c,f.view(),a);check(a.market.empty(),"issued FEED bought duplicate resource");}
 {Fixture f;f.c.p.procure_service_inputs=true;auto a=f.idle();for(int i=0;i<10;i++)a.market.push_back(action(Op::SELL,S,1));service::procure(f.c,f.view(),a);check(a.market.size()==10,"market slot overflow");}
 {Fixture f;f.day=f.c.day=29;f.hour=22;f.priv.inventories[0][MI]=3;auto a=f.idle();a.units[0]=action(Op::DROP);a.market={action(Op::SELL,MI,3)};
  service::protect(f.c,f.view(),a);check(a.market.size()==1&&a.market[0].quantity==3,"unreserved terminal DROP SELL was deleted");}
 {Fixture f;f.priv.shed[W]=1;f.c.plans[1]={{action(Op::PICKUP,W,1)},{44},0};f.priv.inventories[0][MI]=2;auto a=f.idle();a.units[0]=action(Op::DROP);a.market={action(Op::SELL,MI,2),action(Op::SELL,W,1)};
  service::protect(f.c,f.view(),a);check(a.market.size()==1&&int(a.market[0].item)==MI&&a.market[0].quantity==2,"wheat reservation changed unrelated milk sale");}
 {Fixture f;f.day=f.c.day=29;f.priv.inventories[0][W]=2;auto a=f.idle();a.units[0]=action(Op::DROP);a.market={action(Op::SELL,W,2)};
  service::protect(f.c,f.view(),a);check(a.market.size()==1&&a.market[0].quantity==2,"terminal wheat wrongly protected without reservation");}
 {Fixture f;f.priv.shed[W]=3;f.c.plans[1]={{action(Op::PICKUP,W,1)},{44},0};auto a=f.idle();a.market={action(Op::SELL,W,2),action(Op::SELL,W,2)};
  service::protect(f.c,f.view(),a);int sold=0;for(auto x:a.market)sold+=x.quantity;check(sold==2,"multiple sales exceeded one shared resource reservation");}
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";
}
