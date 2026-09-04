#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;void check(bool b,const char*s){checks++;if(!b)throw std::runtime_error(s);}
struct Fxt{
 Farm f,other;PrivateState priv;Market market;std::vector<int8_t>shops;Controller c;int day=8;
 Fxt(){f.tiles.resize(100);other.tiles.resize(100);f.farmer=other.farmer={4,4};f.unlocked_mask=1;f.money=10000;priv.inventories.resize(1);priv.inventory_order.resize(1);
  market.inventory.fill(10000);for(int i=0;i<9;i++)market.prices[i]=price(i,10000);
  c.day=day;c.phase=2;c.p.rotate_finite=true;c.p.bias.fill(0.);c.p.bias[S]=1.;c.p.fix_finite_projection=true;c.target={{44,W}};
  auto&t=f.tiles[44];t.kind=TileKind::PLANT;t.crop=Item(W);t.planted_day=day-4;t.yield_units=3;t.max_lifespan_step=(day+1)*24;}
 View view(){return{24*day,day,0,f,other,priv,market,shops};}
};
int main(){try{
 {Fxt x;x.c.p.rotate_finite=false;x.c.rotate(x.view());check(x.c.target[0].second==W,"off flag changed target");}
 {Fxt x;x.c.rotate(x.view());check(x.c.target[0].second==S,"mature finite plot cannot change crop");
  auto jobs=x.c.jobs(x.view());check(jobs.size()==1,"missing transaction job");auto&a=jobs[0].actions;
  check(a.size()==4&&a[0].op==Op::WATER&&a[1].op==Op::HARVEST&&a[2].op==Op::PLANT&&a[2].item==Item(S)&&a[3].op==Op::WATER,"harvest then replant chain/order");
  check(jobs[0].seeds[S]==1&&jobs[0].seeds[W]==0,"replacement seed reservation");
  x.priv.seeds[S]=1;ObservedDayScenario scenario(x.view());for(auto action:a)scenario.advance(PlayerAction{{action},{}});
  check(plant(scenario.own().tiles[44])&&scenario.own().tiles[44].crop==Item(S),"native execution failed replant");
  check(scenario.own().tiles[44].watered_today,"new crop not watered");check(scenario.inventory().seeds[S]==0,"seed use not written back");check(scenario.inventory().inventories[0][W]==4,"prior harvest discarded");}
 {Fxt x;x.f.tiles[44].planted_day=x.day-1;x.f.tiles[44].max_lifespan_step=(x.day+4)*24;x.c.rotate(x.view());check(x.c.target[0].second==W,"premature crop overwritten");}
 {Fxt x;x.f.tiles[44].crop=Item(S);x.c.target={{44,S}};x.c.rotate(x.view());check(x.c.target[0].second==S,"ongoing crop overwritten");}
 {Fxt x;x.f.tiles[44].kind=TileKind::ANIMAL;x.f.tiles[44].animal=Item(CO);x.c.target={{44,CO}};x.c.rotate(x.view());check(x.c.target[0].second==CO,"animal overwritten");}
 {Fxt x;x.f.money=0;x.c.rotate(x.view());check(x.c.target[0].second==W,"unfunded replacement admitted");}
 {Fxt x;x.f.money=0;x.priv.seeds[S]=1;x.c.rotate(x.view());check(x.c.target[0].second==S,"owned seed ignored by rotation");}
 {Fxt x;x.c.p.max_strawberry=0;x.c.rotate(x.view());check(x.c.target[0].second==W,"configured capacity ignored");}
 {Fxt x;x.day=20;x.c.day=20;x.f.tiles[44].planted_day=16;x.f.tiles[44].max_lifespan_step=21*24;x.c.rotate(x.view());check(x.c.target[0].second==W,"crop with no remaining harvest proposed");}
 {Fxt x;x.day=29;x.c.day=29;x.f.tiles[44].planted_day=25;x.c.rotate(x.view());check(x.c.target[0].second==W,"terminal reinvestment");}
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
