#include "service_replacement_prototype.hpp"
#include <iostream>
using namespace dp7;
namespace rp=dp7::replacement_probe;
int checks=0;void check(bool ok,const char*why){checks++;if(!ok)throw std::runtime_error(why);}
struct Fixture{
 Farm own,opponent;PrivateState priv;Market market;std::vector<int8_t>shops;Controller c;int day=27,hour=23;
 Fixture(int n=1){own.tiles.resize(100);opponent.tiles.resize(100);own.farmer=opponent.farmer={4,4};own.hands.assign(n-1,{4,4});own.unlocked_mask=15;own.money=10000;
  priv.inventories.resize(n);priv.inventory_order.resize(n);market.inventory.fill(10000);for(int i=0;i<9;i++)market.prices[i]=price(i,10000);c.day=day;c.phase=3;c.plans.resize(n);}
 View view(){return {day*24+hour,day,hour,own,opponent,priv,market,shops};}
 void sheep(int pos){auto&t=own.tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item(SH);t.placed_day=0;t.consecutive_unfed=1;t.pending_care_bonus=4;c.target.push_back({pos,SH});}
 void cargo(int u,int item,int q){priv.inventories[u][item]=q;if(q)priv.inventory_order[u].push_back(item);}
 void care(int u,int pos){int start=cell(u?own.hands[u-1]:own.farmer);Controller::walk(c.plans[u],start,pos);c.plans[u].a.push_back(action(Op::CARE));c.plans[u].target.push_back(pos);}
};
int count(const std::vector<Plan>&ps,Op op){int n=0;for(auto&p:ps)for(auto a:p.a)n+=a.op==op;return n;}
int main(){try{
 {Fixture f;f.sheep(44);f.care(0,44);f.cargo(0,W,1);f.c.p.insert_missing_feed=true;liverepair::feed(f.c,f.view());check(count(f.c.plans,Op::FEED)==0,"insertion should exceed deadline");
  auto before=f.c.plans;auto r=rp::generate(f.c,f.view());check(r.feasible>0&&!r.candidates.empty(),"replacement not generated");
  check(count(r.candidates[0].plans,Op::FEED)==1&&count(r.candidates[0].plans,Op::CARE)==0,"replacement semantic diff");
  check(Controller::same_remaining(f.c.plans[0],before[0]),"generation mutated source");
  check(exchange::deadlines(f.view(),r.candidates[0].plans),"replacement late");
  ObservedDayScenario a(f.view()),b(f.view());a.advance(PlayerAction{{action(Op::CARE)},{}});b.advance(PlayerAction{{action(Op::FEED)},{}});
  check(!animal(a.own().tiles[44])&&animal(b.own().tiles[44]),"official kernel maintenance effect");
  auto value=rp::score(f.c,f.view());check(value.keep.known&&value.selected>=0,"profitable pending production rejected");
  check(Controller::same_remaining(f.c.plans[0],before[0]),"valuation mutated source");}
 {Fixture f;f.sheep(44);f.care(0,44);check(rp::generate(f.c,f.view()).candidates.empty(),"unowned feed credited");}
 {Fixture f;f.sheep(44);f.care(0,44);f.cargo(0,W,1);f.c.target.clear();check(rp::generate(f.c,f.view()).candidates.empty(),"invented animal service commitment");}
 {Fixture f;f.sheep(44);f.care(0,44);f.cargo(0,W,1);f.own.tiles[44].fed_today=true;check(rp::generate(f.c,f.view()).candidates.empty(),"already fed duplicate");}
 {Fixture f;f.sheep(44);f.cargo(0,W,1);check(rp::generate(f.c,f.view()).candidates.empty(),"unrelated task removed without CARE");}
 {Fixture f(2);f.own.farmer={5,4};f.sheep(44);f.care(0,44);f.cargo(1,W,1);auto r=rp::generate(f.c,f.view());
  check(!r.candidates.empty()&&r.candidates[0].unit==1,"worker ownership remained locked");check(r.candidates[0].plans[0].a.empty(),"cancelled owner's obsolete movement retained");}
 {Fixture f;f.hour=22;f.sheep(44);f.care(0,44);f.priv.shed[W]=1;auto r=rp::generate(f.c,f.view());check(!r.candidates.empty()&&r.candidates[0].pickup,"explicit pickup absent");
  auto c=f.c;c.plans=r.candidates[0].plans;check(intraday::reserved(c).shed[W]==1,"replacement grain reservation wrong");}
 {Fixture f(2);f.hour=22;f.sheep(44);f.care(0,44);f.priv.shed[W]=1;f.c.plans[1].a={action(Op::PICKUP,W)};f.c.plans[1].target={44};
  auto r=rp::generate(f.c,f.view());check(!r.candidates.empty()&&r.candidates[0].unit==1&&!r.candidates[0].pickup,"owner's already planned uncommitted pickup wrongly forbidden");
  check(count(r.candidates[0].plans,Op::PICKUP)==1,"duplicated existing pickup");}
 {Fixture f(2);f.hour=21;f.sheep(44);f.sheep(45);f.care(0,44);f.priv.shed[W]=1;
  f.c.plans[1].a={action(Op::PICKUP,W),action(Op::EAST),action(Op::FEED)};f.c.plans[1].target={44,-1,45};
  check(rp::generate(f.c,f.view()).candidates.empty(),"same grain spent on two animals");}
 {Fixture f;f.day=28;f.c.day=28;f.sheep(44);f.own.tiles[44].placed_day=28;f.own.tiles[44].pending_care_bonus=0;f.care(0,44);f.cargo(0,W,1);
  auto r=rp::score(f.c,f.view());check(r.keep.known&&r.evaluated>=2&&r.selected>=0,"profitable fertilizer byproduct ignored");}
 {Fixture f;f.day=28;f.c.day=28;f.sheep(44);f.own.tiles[44].placed_day=28;f.own.tiles[44].pending_care_bonus=0;f.care(0,44);f.cargo(0,W,1);
  // No wool can mature before the end, but feed creates fertilizer. Only when
  // its value is BELOW the sale opportunity of retained grain is KEEP better.
  f.market.inventory[W]=1000;f.market.prices[W]=price(W,1000);
  auto r=rp::score(f.c,f.view());check(r.keep.known&&r.evaluated>=2&&r.selected==-1,"feed with negative net byproduct value forced");}
 {Fixture f;f.sheep(44);f.care(0,44);f.cargo(0,W,1);f.c.plans[0].a.push_back(action(Op::FEED));f.c.plans[0].target.push_back(44);
  check(rp::generate(f.c,f.view()).candidates.empty(),"already planned FEED duplicated");}
 {dayvalue::Value k,a,b;k.score=100;a.score=99;b.score=101;check(rp::choose(k,{a,b})==1,"net-cash choice");b.known=false;check(rp::choose(k,{a,b})==-1,"unknown candidate selected");k.known=false;b.known=true;check(rp::choose(k,{b})==-1,"unknown KEEP forced change");k.known=true;b.score=100;check(rp::choose(k,{b})==-1,"equal value churn");b.score=NAN;check(rp::choose(k,{b})==-1,"NaN candidate selected");}
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
