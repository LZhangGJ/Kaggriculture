#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;void check(bool ok,const char*why){checks++;if(!ok)throw std::runtime_error(why);}
struct Fixture{
 Simulator e{Config{},123};Controller c;
 Fixture(int hour=12,int n=1){while(e.step_count()<4*24+hour)e.step({});auto&f=const_cast<Farm&>(e.farms()[0]);auto&pr=const_cast<PrivateState&>(e.privates()[0]);
 f.farmer={4,4};f.hands.assign(n-1,{4,4});f.unlocked_mask=15;for(auto&t:f.tiles)t=Tile{};pr.inventories.assign(n,{});pr.inventory_order.assign(n,{});pr.shed={};c.day=e.day();c.phase=3;c.plans.resize(n);}
 View v()const{return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
 PrivateState&pr(){return const_cast<PrivateState&>(e.privates()[0]);}
 void sheep(int pos){auto&t=const_cast<Farm&>(e.farms()[0]).tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item(SH);t.placed_day=0;t.consecutive_unfed=1;c.target.push_back({pos,SH});}
 void route(int u,int pos){auto&p=c.plans[u];int start=44;Controller::walk(p,start,pos);p.a.push_back(action(Op::CARE));p.target.push_back(pos);Controller::walk(p,start,44);p.a.push_back(action(Op::DROP));p.target.push_back(44);}
 void market_ready(){c.p.continuous_market_execution=true;c.p.competitive_sell_weight=1;c.p.sell_horizon_days=1;pr().shed[MI]=6;
 auto&f=const_cast<Farm&>(e.farms()[1]);auto&t=f.tiles[44];t.kind=TileKind::ANIMAL;t.animal=Item(CO);t.yield_units=100;t.placed_day=0;}
};
int count(const Controller&c,Op op){int n=0;for(const auto&p:c.plans)for(size_t k=p.index;k<p.a.size();k++)n+=p.a[k].op==op;return n;}
int main(){
 {Fixture f;f.sheep(84);f.route(0,84);f.pr().inventories[0][W]=1;auto before=f.c.plans;liverepair::feed(f.c,f.v());check(Controller::same_remaining(f.c.plans[0],before[0]),"off changed plan");}
 {Fixture f;f.sheep(84);f.route(0,84);f.pr().inventories[0][W]=1;f.c.p.insert_missing_feed=true;
  check(f.v().hour+int(service::append(f.c,f.v(),0,{84,W,Op::FEED},false).a.size())>24,"fixture append was feasible");
  liverepair::feed(f.c,f.v());check(f.c.feed_insert_applied==1&&count(f.c,Op::FEED)==1,"middle insertion missing");
  check(liverepair::wheat_feasible(f.v(),0,f.c.plans[0])&&f.c.plans[0].a.size()<=12,"insert overspends");
  liverepair::feed(f.c,f.v());check(count(f.c,Op::FEED)==1,"duplicate feed");}
 {Fixture f;f.sheep(84);f.route(0,84);f.pr().shed[W]=1;f.c.p.insert_missing_feed=true;liverepair::feed(f.c,f.v());check(f.c.feed_insert_applied==1&&count(f.c,Op::PICKUP)==1,"pickup insertion missing");}
 {Fixture f;f.sheep(84);f.route(0,84);f.c.p.insert_missing_feed=true;liverepair::feed(f.c,f.v());check(f.c.feed_insert_applied==0,"unowned wheat used");}
 {Fixture f(23);f.sheep(84);f.pr().inventories[0][W]=1;f.c.p.insert_missing_feed=true;liverepair::feed(f.c,f.v());check(count(f.c,Op::FEED)==0,"late feed accepted");}
 {Fixture f(12,2);f.sheep(84);f.sheep(83);f.pr().shed[W]=1;f.c.p.insert_missing_feed=true;liverepair::feed(f.c,f.v());check(count(f.c,Op::FEED)==1&&intraday::reserved(f.c).shed[W]==1,"double reservation");}
 {Fixture f;f.sheep(84);f.c.target.clear();f.pr().inventories[0][W]=1;f.c.p.insert_missing_feed=true;liverepair::feed(f.c,f.v());check(count(f.c,Op::FEED)==0,"invented maintenance commitment");}
 {Fixture f;f.market_ready();check(f.c.competition_sales(f.v())[MI]==6,"no tactical opportunity fixture");PlayerAction a;a.units.resize(1);a.market={action(Op::HIRE),action(Op::HIRE)};auto before=a.market;liverepair::market(f.c,f.v(),a);check(a.market.size()==3&&a.market[0].op==Op::HIRE&&a.market[1].op==Op::HIRE&&a.market[2].op==Op::SELL,"old orders moved");}
 {Fixture f;f.market_ready();PlayerAction a;a.units.resize(1);a.market.assign(10,action(Op::HIRE));liverepair::market(f.c,f.v(),a);check(a.market.size()==10&&std::all_of(a.market.begin(),a.market.end(),[](auto x){return x.op==Op::HIRE;}),"full slots displaced");}
 {Fixture f;f.market_ready();PlayerAction a;a.units.resize(1);a.market={action(Op::SELL,MI,2),action(Op::BUY_PRODUCT,W,1)};liverepair::market(f.c,f.v(),a);check(a.market.size()==2&&a.market[0].quantity==6&&a.market[1].op==Op::BUY_PRODUCT,"duplicate sell or financing reordered");liverepair::market(f.c,f.v(),a);check(a.market[0].quantity==6,"repeated merge adds twice");}
 {Fixture f;f.market_ready();PlayerAction a;a.units={action(Op::PICKUP,MI,2)};liverepair::market(f.c,f.v(),a);check(a.market.size()==1&&a.market[0].quantity==4,"same-step pickup consumed");}
 {Fixture f;f.market_ready();f.c.plans[0].a={action(Op::PICKUP,MI,3)};f.c.plans[0].target={44};PlayerAction a;a.units.resize(1);liverepair::market(f.c,f.v(),a);check(a.market[0].quantity==3,"future pickup reservation consumed");}
 {Fixture f;f.market_ready();f.c.queue={action(Op::SELL,MI,6)};PlayerAction a;a.units.resize(1);liverepair::market(f.c,f.v(),a);check(a.market.empty(),"queued sales duplicated");}
 {Fixture f;f.market_ready();f.c.p.continuous_market_execution=false;PlayerAction a;a.units.resize(1);liverepair::market(f.c,f.v(),a);check(a.market.empty()&&f.c.live_market_checks==0,"off changed orders");}
 {Fixture f(21,2);f.sheep(45);f.c.p.insert_missing_feed=true;f.pr().inventories[0][W]=1;f.pr().inventories[0][F]=1;
  auto&t=const_cast<Farm&>(f.e.farms()[0]).tiles[44];t.kind=TileKind::PLANT;t.crop=Item(S);t.planted_day=0;
  f.c.plans[0].a={action(Op::FERTILIZE)};f.c.plans[0].target={44};f.c.plans[1].a={action(Op::WATER)};f.c.plans[1].target={44};
  liverepair::feed(f.c,f.v());check(count(f.c,Op::FEED)==1&&f.c.plans[0].a[0].op==Op::FERTILIZE,"cross-worker fertilizer water order lost");}
 {Fixture f;f.sheep(84);f.route(0,84);f.pr().inventories[0][W]=1;f.c.p.insert_missing_feed=true;liverepair::feed(f.c,f.v());
  auto plan=f.c.plans[0];for(auto a:plan.a){PlayerAction out;out.units={a};f.e.step({out,{}});}while(f.e.day()==4)f.e.step({});
  const auto&t=f.e.farms()[0].tiles[84];check(animal(t)&&t.consecutive_unfed==0,"planned feed not executed in engine");}
 {Fixture f;f.market_ready();f.pr().shed[MI]=0;f.pr().inventories[0][MI]=20;PlayerAction a;a.units.resize(1);liverepair::market(f.c,f.v(),a);check(a.market.empty(),"sold unbanked future cargo");}
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";
}
