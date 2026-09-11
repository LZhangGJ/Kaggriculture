#include "rotation_portfolio_prototype.hpp"
#include <iostream>
using namespace dp7;
int checks=0;void check(bool b,const char*s){checks++;if(!b)throw std::runtime_error(s);}
struct Fixture{
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;Controller c;int day=8,hour=0;
 Fixture(){own.tiles.resize(100);rival.tiles.resize(100);own.farmer=rival.farmer={4,4};own.unlocked_mask=1;own.money=100000;
  priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(1000);for(int i=0;i<9;i++)market.prices[i]=price(i,1000);
  c.day=day;c.p.fix_finite_projection=true;c.p.max_hands=15;
 }
 View view(){return{24*day+hour,day,hour,own,rival,priv,market,shops};}
 Tile make_crop(int kind,int age){Tile t;t.kind=TileKind::PLANT;t.crop=Item(kind);t.planted_day=day-age;t.yield_units=ongoing(kind)?2:3;t.max_lifespan_step=ongoing(kind)?-1:(t.planted_day+(kind==W?4:kind==C?3:12)+1)*24;return t;}
};
bool same(const portfolio::DeliveryCalendar&a,const portfolio::DeliveryCalendar&b){return a.cash.quantity==b.cash.quantity&&a.cash.fixed==b.cash.fixed&&a.harvest==b.harvest&&a.water==b.water&&a.feed==b.feed&&a.fert==b.fert&&a.care==b.care&&a.planting==b.planting&&a.dig==b.dig;}
int main(){try{
 // The new contract must preserve every tested same-kind forecast, including
 // late clocks, terminal liquidation and already-watered crops.
 for(int day:{0,4,8,16,25,28,29})for(int hour:{0,12,21,22})for(int kind=0;kind<5;kind++)for(int age:{0,2,4,10,16})for(bool watered:{false,true}){
  if(age>day)continue;Fixture x;x.day=x.c.day=day;x.hour=hour;auto t=x.make_crop(kind,age);t.watered_today=watered;
  auto a=portfolio::crop_delivery(x.c,x.view(),kind,t);auto b=rotationproto::crop(x.c,x.view(),kind,t);
  check(same(a,b.cal),"same-kind forecast regression");
  for(int d=day;d<30;d++){int n=0;for(int k=0;k<5;k++)n+=b.seeds[d][k];check(n==b.cal.planting[d],"typed seed counts inconsistent");}
 }
 // Old harvest comes from the actual old tile, not desired replacement crop.
 for(int old:{W,C,M})for(int next=0;next<5;next++){
  Fixture x;x.day=x.c.day=14;int age=x.c.h_age(old);x.own.tiles[44]=x.make_crop(old,age);x.c.target={{44,next}};
  auto expected=rotationproto::crop(x.c,x.view(),next,x.own.tiles[44]);auto jobs=x.c.jobs(x.view());
  for(auto&j:jobs)for(int i=0;i<5;i++)x.priv.seeds[i]+=j.seeds[i];
  ObservedDayScenario model(x.view());for(auto&j:jobs)for(auto a:j.actions)model.advance(PlayerAction{{a},{}});
  while(!model.finished())model.advance(PlayerAction{});
  for(int i=0;i<5;i++)check(model.inventory().shed[i]==expected.cal.cash.quantity[15][i],"handoff crop credited to wrong product");
  check(model.own().tiles[44].crop==Item(next),"native replacement not executed");
  check(expected.seeds[14][next]==1,"new seed type not used");
 }
 {
  Fixture x;x.own.tiles[44]=x.make_crop(W,4);auto original=rotationproto::physical_works(x.c,x.view());
  auto next=rotationproto::replace(x.c,x.view(),original,44,S);
  check(original.size()==1&&next.size()==1,"replacement duplicates plot");check(original[0].kind==W&&next[0].kind==S,"replacement mutated original");
  check(next[0].cal.cash.quantity[9][W]==4&&next[0].cal.cash.quantity[9][S]==0,"wheat became strawberry on day one");
  check(next[0].seeds[8][W]==0&&next[0].seeds[8][S]==1,"charged old and new seed together");
  bool threw=false;try{next.push_back(next[0]);rotationproto::value(x.c,x.view(),next);}catch(const std::invalid_argument&){threw=true;}check(threw,"duplicate portfolio accepted");
 }
 {
  Fixture x;rotationproto::Work cow{44,CO,{},{}},wheat{45,W,{},{}},strawberry{45,S,{},{}};
  cow.cal.cash.quantity[10][W]=-6;wheat.cal.cash.quantity[9][W]=6;strawberry.cal.cash.quantity[18][S]=8;
  auto alone=rotationproto::value(x.c,x.view(),{cow},false),self=rotationproto::value(x.c,x.view(),{cow,wheat},false);
  auto change=rotationproto::value(x.c,x.view(),{cow,strawberry},false);
  check(alone.buy_by_item[W]>0&&self.buy_by_item[W]==0,"self-produced feed was not netted");
  check(change.buy_by_item[W]>0,"replacement hides feed purchase obligation");
  check(change.cash==x.own.money+change.sales-change.purchases-change.seeds-change.wages,"cash ledger not conserved");
  auto reversed=rotationproto::value(x.c,x.view(),{wheat,cow},false);check(reversed.cash==self.cash,"work order changes economic valuation");
  x.priv.shed[W]=6;auto owned=rotationproto::value(x.c,x.view(),{cow},false);check(owned.buy_by_item[W]==0,"owned feed charged twice");
 }
 {
  Fixture x;rotationproto::Work a{44,W,{},{}},b{45,W,{},{}};a.cal.water[9]=b.cal.water[9]=1;
  auto v=rotationproto::value(x.c,x.view(),{a,b});check(v.wages==0,"two small projects each charged a worker");
  a.seeds[9][S]=1;auto paid=rotationproto::value(x.c,x.view(),{a,b},false);x.priv.seeds[S]=1;
  auto reuse=rotationproto::value(x.c,x.view(),{a,b},false);check(paid.seeds-reuse.seeds==seed_price[S],"owned seed charged again");
 }
 {
  Fixture x;x.own.tiles[44]=x.make_crop(W,4);
  auto now=rotationproto::crop(x.c,x.view(),S,x.own.tiles[44]);
  auto wait=rotationproto::crop(x.c,x.view(),S,x.own.tiles[44],10);
  auto stop=rotationproto::crop(x.c,x.view(),S,x.own.tiles[44],30);
  check(now.cal.cash.quantity[9][W]==4&&wait.cal.cash.quantity[9][W]==4&&stop.cal.cash.quantity[9][W]==4,"timing choice erased old harvest");
  check(now.seeds[8][S]==1&&wait.seeds[8][S]==0&&wait.seeds[9][S]==0&&wait.seeds[10][S]==1,"investment delay not respected");
  check(wait.cal.water[9]==0&&wait.cal.fert[9]==0,"waiting empty plot charged maintenance");
  int seed_sum=0;for(auto&day:stop.seeds)for(int n:day)seed_sum+=n;check(seed_sum==0,"stop-new-investment bought seed");
 }
 {
  Fixture x;x.day=x.c.day=19;x.own.tiles[44]=x.make_crop(W,4);
  auto now=rotationproto::crop(x.c,x.view(),S,x.own.tiles[44]);
  auto late=rotationproto::crop(x.c,x.view(),S,x.own.tiles[44],20);
  double cash_now=0,cash_late=0;for(int d=19;d<30;d++){cash_now+=now.cal.cash.quantity[d][S];cash_late+=late.cal.cash.quantity[d][S];}
  check(cash_now>0&&cash_late==0,"one day later should miss last strawberry harvest");
 }
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
