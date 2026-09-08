#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;void check(bool v,const char*s){checks++;if(!v)throw std::runtime_error(s);}
struct Fxt{Farm f,r;PrivateState pr;Market m;std::vector<int8_t>shops;Controller c;int day=8;
 Fxt(){f.tiles.resize(100);r.tiles.resize(100);f.farmer=r.farmer={4,4};f.unlocked_mask=1;f.money=20000;
  pr.inventories.resize(1);pr.inventory_order.resize(1);m.inventory.fill(10000);for(int i=0;i<9;i++)m.prices[i]=price(i,10000);
  c.day=day;c.p.portfolio_rotation=true;c.p.rotation_timing=true;c.p.fix_finite_projection=true;c.p.fix_logistics=true;c.p.max_hands=12;
  c.target={{44,W}};auto&t=f.tiles[44];t.kind=TileKind::PLANT;t.crop=Item(W);t.planted_day=4;t.yield_units=3;t.max_lifespan_step=9*24;
 }
 View view(){return{day*24,day,0,f,r,pr,m,shops};}
};
int main(){try{
 {Fxt x;x.c.p.portfolio_rotation=false;auto before=x.c.target;x.c.compare_rotation(x.view(),x.view(),0);check(x.c.target==before&&x.c.rotation_evaluated==0,"off switch entered search");}
 {Fxt x;rotationcash::edit(x.c,{{44},S,10});auto jobs=x.c.jobs(x.view());check(jobs.size()==1,"waiting erased old work");
  bool harvest=false;for(auto a:jobs[0].actions){harvest|=a.op==Op::HARVEST;check(a.op!=Op::PLANT,"wait still plants");}
  check(harvest&&jobs[0].seeds[S]==0,"waiting lost harvest or reserves seed");x.c.prepare_orders(x.view(),x.view(),0);
  for(auto a:x.c.queue)check(!(a.op==Op::BUY_SEED&&a.item==Item(S)),"deferred seed purchased prematurely");
  auto next=rotationcash::consequence(x.c,x.view());check(next.ticks==24&&std::isfinite(next.score),"conditional current-day execution failed");
  check(x.c.plant_not_before[44]==10&&x.c.deferred_since[44]==8,"scenario mutated real plan");
 }
 {Fxt x;rotationcash::edit(x.c,{{44},S,8});x.c.prepare_orders(x.view(),x.view(),0);ObservedDayScenario s(x.view());auto c=x.c;c.p.portfolio_rotation=false;c.p.day_consequence_compare=false;
  while(!s.finished())s.advance(c.act(s.view()));
  check(plant(s.own().tiles[44])&&s.own().tiles[44].crop==Item(S),"funded replacement not planted");
  check(s.inventory().shed[W]==4,"old harvest disappeared during actual preparation/worker execution");
 }
 {Fxt x;x.f.tiles[44]=Tile{};x.c.target={{44,S}};rotationcash::edit(x.c,{{44},S,10});x.c.phase=3;x.c.plans.resize(1);x.c.p.intraday_admission=true;x.c.p.intraday_procurement=true;
  // All other plots are unavailable, so the intraday path cannot hide a wait
  // violation by choosing a different equivalent vacant plot.
  for(int p=0;p<100;p++)if(p!=44)x.f.tiles[p].kind=TileKind::LOCKED;
  auto candidate=intraday::propose(x.c,x.view(),PlayerAction{{Action{}},{}});check(!candidate.valid,"intraday admission filled deferred plot");
  x.c.pending_admission={true,8,191,0,44,S};intraday::complete_pending(x.c,x.view());check(x.c.intraday_activated==0&&!x.c.pending_admission.active,"pending admission bypassed wait");
 }
 {Fxt x;x.c.p.rotation_timing=false;auto immediate=rotationcash::proposals(x.c,x.view());for(auto&e:immediate)check(e.when==8,"disabled timing emitted waiting");
  x.c.p.rotation_timing=true;auto timed=rotationcash::proposals(x.c,x.view());bool now=false,later=false,stop=false;
  for(auto&e:timed){now|=e.when==8;later|=e.when>8&&e.when<30;stop|=e.when==30;}
  check(now&&later&&stop,"missing timing family");
  auto initial=x.c.target;x.c.prepare_orders(x.view(),x.view(),0);x.c.compare_rotation(x.view(),x.view(),0);
  check(x.c.rotation_generated>0&&x.c.rotation_evaluated>=2,"production selection path not executed");
  check(x.c.rotation_evaluated<=5,"unbounded conditional shortlist");
 }
 {Fxt x;x.f.tiles[44].planted_day=7;x.f.tiles[44].max_lifespan_step=12*24;check(rotationcash::proposals(x.c,x.view()).empty(),"immature crop proposed for destruction");
  x.f.tiles[44].kind=TileKind::ANIMAL;x.f.tiles[44].animal=Item(SH);x.c.target={{44,SH}};check(rotationcash::proposals(x.c,x.view()).empty(),"animal proposed as liquid crop");}
 {Fxt x;rotationcash::edit(x.c,{{44},S,10});x.c.last_step=719;
  Simulator env(Config{},100);View v{env.step_count(),env.day(),env.hour(),env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  auto actual=x.c.act(v);Controller fresh(x.c.p);auto expected=fresh.act(v);
  check(x.c.target==fresh.target&&x.c.plant_not_before==fresh.plant_not_before&&x.c.deferred_kind==fresh.deferred_kind,"wait state contaminated next game");
  check(actual.units.size()==expected.units.size()&&actual.market.size()==expected.market.size(),"next game action shape differs");
 }
 {Fxt x;x.f.tiles[44]=Tile{};rotationcash::edit(x.c,{{44},S,10});x.c.prepare_orders(x.view(),x.view(),0);
  auto p=x.c.preview_bundle(x.view(),false);check(p.proposed==0,"future wait counted as failed current start");
  x.c.target={{44,-1}};x.c.sync_deferred();check(x.c.deferred_kind[44]<0&&x.c.plant_not_before[44]==0,"cancelled investment retained ghost calendar");
  check(rotationcash::proposals(x.c,x.view()).empty(),"cancelled pending plot remains a replacement");
  rotationcash::edit(x.c,{{44},S,10});x.c.target={{44,SH}};x.c.sync_deferred();check(x.c.deferred_kind[44]<0,"animal plan retained obsolete crop commitment");
 }
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
