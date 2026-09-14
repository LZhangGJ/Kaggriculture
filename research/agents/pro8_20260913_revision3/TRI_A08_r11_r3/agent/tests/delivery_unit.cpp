#include "../policy/executor/policy.hpp"
#include <iostream>
#include <stdexcept>
#include <random>
using namespace dp7;
static long checks=0,selected=0;
void require(bool x,const char*what){++checks;if(!x)throw std::runtime_error(what);}
int mirror(int c,int r){int x=c%10,y=c/10;if(r&1)x=9-x;if(r&2)y=9-y;return y*10+x;}
struct Fixture {
 fastkag::Farm farm,other;fastkag::PrivateState priv;fastkag::Market market;std::vector<int8_t>shops;Controller ctl;int h;
 Fixture(int reflection,int hour,int cargo,int background,int cost):h(hour){
  auto pos=[&](int c){c=mirror(c,reflection);return fastkag::Position{int16_t(c%10),int16_t(c/10)};};
  farm.tiles.resize(100);other.tiles.resize(100);farm.money=50000;farm.farmer=pos(46);farm.hands={pos(1)};farm.unlocked_mask=15;other=farm;
  for(int c:{1,49}){auto&t=farm.tiles[mirror(c,reflection)];t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.placed_day=10;}
  for(int c:{0,9}){auto&t=farm.tiles[mirror(c,reflection)];t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=0;t.max_lifespan_step=10000;}
  priv.inventories.resize(2);priv.inventory_order.resize(2);priv.inventories[0][MI]=cargo;priv.inventories[1][E]=background;priv.inventories[1][W]=1;
  if(cargo)priv.inventory_order[0]={MI};if(background)priv.inventory_order[1]={E,W};else priv.inventory_order[1]={W};
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=price(i,10000);}
  ctl.day=20;ctl.phase=3;ctl.p.midroute_delivery=true;ctl.p.action_shadow=cost;ctl.delivery_observation_step=20*24+h;ctl.plans.resize(2);
  int cur=mirror(46,reflection);for(auto[c,op]:std::vector<std::pair<int,Op>>{{49,Op::CARE},{9,Op::WATER},{0,Op::WATER}}){
   int dest=mirror(c,reflection);Controller::walk(ctl.plans[0],cur,dest);ctl.plans[0].a.push_back(action(op));ctl.plans[0].target.push_back(dest);
  }ctl.plans[1].a={action(Op::FEED)};ctl.plans[1].target={mirror(1,reflection)};
 }
 View view()const{return {20*24+h,20,h,farm,other,priv,market,shops};}
};
int main(){try{
 for(int r=0;r<4;r++)for(int h=0;h<24;h++)for(int q:{0,1,6,10,25,100})for(int bg:{0,50,90,95,101})for(int cost:{0,4,1000}){
  Fixture f(r,h,q,bg,cost);auto v=f.view();auto c=deliveryhandoff::propose(f.ctl,v);
  require(!c.selected||!c.best.empty(),"selection consistency");
  if(c.selected){selected++;require(q>0,"empty cargo selected");require(c.donor==0&&c.receiver==1,"material donor selected");
   require(exchange::order(c.best)==exchange::order(c.base),"field task order changed");require(exchange::deadlines(v,c.best),"late task");
   int extra=std::max(1,int(c.best[0].a.size()+c.best[1].a.size())-int(c.base[0].a.size()+c.base[1].a.size()));
   double quote=revenue(MI,10000,q)*std::min(q,c.overflow_before)/q;
   require(c.extra_actions==extra,"choice action cost mismatch");require(std::abs(c.conditional_quote-quote)<1e-8,"choice quote mismatch");
   require(std::abs(c.score-(quote-cost*extra)/extra)<1e-8,"choice score mismatch");require(quote>cost*extra,"negative return admitted");
   int drops=0;for(auto a:c.best[0].a)drops+=a.op==Op::DROP;require(drops==1,"missing or duplicated DROP");
  }
 }
 // Exercise material guards directly; these are not deadline rejections.
 Counts cargo{};cargo[MI]=10;Plan p;p.a={action(Op::CARE)};p.target={49};
 require(deliveryhandoff::donor_safe(p,cargo),"safe donor rejected");
 for(Op op:{Op::DROP,Op::PICKUP,Op::FEED,Op::FERTILIZE,Op::PLACE,Op::PLANT}){p.a={action(op)};require(!deliveryhandoff::donor_safe(p,cargo),"material commitment unprotected");}
 p.a={action(Op::PLANT,W)};require(deliveryhandoff::donor_safe(p,cargo,true),"funded planting capability lost");
 p.a={action(Op::CARE)};for(int i=9;i<12;i++){cargo[i]=1;require(!deliveryhandoff::donor_safe(p,cargo),"held animal unprotected");cargo[i]=0;}
 for(Op op:{Op::PICKUP,Op::DROP,Op::PLACE,Op::FEED,Op::FERTILIZE,Op::PLANT,Op::DIG}){
  exchange::Group g{0,{action(Op::HARVEST),action(op)}};require(!deliveryhandoff::free_group(g),"resource job transferred");
 }
 require(deliveryhandoff::free_group({0,{action(Op::WATER),action(Op::HARVEST),action(Op::CARE),action(Op::COLLECT_FERTILIZER)}}),"free service group lost");
 Fixture seed_test(0,5,10,95,4);auto seed_plans=seed_test.ctl.plans;
 seed_plans[0].a.push_back(action(Op::PLANT,W));seed_plans[0].target.push_back(2);
 require(!deliveryhandoff::seeds_funded(seed_test.view(),seed_plans),"unfunded seeds credited");
 seed_test.priv.seeds[W]=1;require(deliveryhandoff::seeds_funded(seed_test.view(),seed_plans),"current seeds ignored");
 Fixture positive(0,5,10,95,4);require(deliveryhandoff::propose(positive.ctl,positive.view()).selected,"positive transport lost");
 positive.ctl.delivery_observation_step=-1;require(!deliveryhandoff::propose(positive.ctl,positive.view()).selected,"hypothetical MPC authorized");
 require(selected>0,"no positive cases exercised");
 std::cout<<"PASS checks="<<checks<<" states="<<(4*24*6*5*3)<<" selected="<<selected<<"\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL after "<<checks<<" checks: "<<e.what()<<"\n";return 1;}}
