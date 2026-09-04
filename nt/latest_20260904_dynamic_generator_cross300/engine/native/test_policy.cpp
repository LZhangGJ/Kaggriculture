// Focused native policy/rule fixtures. Does not replace full-game evaluation.
#include "policy.hpp"
#include "investment_audit.hpp"
#include "investment_candidates.hpp"
#include "resource_handoff_audit.hpp"
#include <iostream>
#include <stdexcept>
#include <random>
using namespace fastkag;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
struct CropResult{int quantity=0,water=0,fert=0;};
CropResult single_crop(int crop,int harvest,bool efficient,bool fertilize){
 Simulator env(Config{},61791);dp7::Params p;p.efficient_water=efficient;dp7::Controller ctl(p);CropResult out;
 while(!env.done()){
  std::array<PlayerAction,2>a;a[0].units.resize(1);auto&t=env.farms()[0].tiles[44];ctl.day=env.day();
  if(env.step_count()==0){a[0].market={dp7::action(Op::BUY_SEED,crop,1),dp7::action(Op::BUY_PRODUCT,8,10)};}
  else if(env.step_count()==1)a[0].units[0]=dp7::action(Op::PICKUP,8,10);
  else if(env.step_count()==2)a[0].units[0]=dp7::action(Op::PLANT,crop);
  else if(dp7::plant(t)){
   int age=env.day()-t.planted_day;bool ready=dp7::ongoing(crop)?t.yield_units>=4||(t.yield_units>0&&t.max_lifespan_step>=0&&t.max_lifespan_step<=(env.day()+1)*24):age>=harvest;
   bool due=dp7::ongoing(crop)?ctl.fertilize_due(t):age>=(crop==0?2:crop==1?2:6)&&age<=harvest&&t.fertilized_until_day<env.day();
   if(fertilize&&due&&!t.watered_today){if(env.privates()[0].inventories[0][8]<=0)a[0].units[0]=dp7::action(Op::PICKUP,8,1);else{a[0].units[0]=dp7::action(Op::FERTILIZE);out.fert++;}}
   else if(ctl.water_due(t)){a[0].units[0]=dp7::action(Op::WATER);out.water++;}
   else if(ready){a[0].units[0]=dp7::action(Op::HARVEST);out.quantity+=t.yield_units;}
  }
  bool harvested=a[0].units[0].op==Op::HARVEST;env.step(a);
  if(harvested&&!dp7::ongoing(crop))break;
 }
 return out;
}
int repeated_finite_output(int crop,int harvest){
 Simulator env(Config{},61791);dp7::Params p;p.efficient_water=true;p.crop_harvest_age[crop]=harvest;
 dp7::Controller ctl(p);int harvested=0,first_estimate=-1;
 while(!env.done()){
  auto&t=env.farms()[0].tiles[44];ctl.day=env.day();std::array<PlayerAction,2>a;a[0].units.resize(1);
  if(env.step_count()==0)a[0].market={dp7::action(Op::BUY_SEED,crop,16)};
  else if(t.kind==TileKind::EMPTY&&env.day()+harvest<=29)a[0].units[0]=dp7::action(Op::PLANT,crop);
  else if(dp7::plant(t)){
   if(first_estimate<0){dp7::View v{env.step_count(),env.day(),env.hour(),env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};first_estimate=ctl.replant_supply(v)[crop]+ctl.finite_yield(crop,harvest);}
   if(ctl.water_due(t))a[0].units[0]=dp7::action(Op::WATER);
   else if(env.day()-t.planted_day>=harvest){a[0].units[0]=dp7::action(Op::HARVEST);harvested+=t.yield_units;}
  }
  env.step(a);
 }
 require(harvested==first_estimate,"conditional finite replant supply must equal single-tile execution");
 return harvested;
}
void zero_expiry_rule_fixture(int crop){
 Simulator env(Config{},71518);dp7::Controller ctl;bool observed=false;
 while(!env.done()){
  auto before=env.farms()[0];auto projected=before;int n=dp7::Controller::project_zero_expiry(projected,env.step_count());
  std::array<PlayerAction,2>a;a[0].units.resize(1);auto&t=env.farms()[0].tiles[44];ctl.day=env.day();
  if(n){require(env.farms()[0].tiles[44].kind==TileKind::PLANT,"projection mutated official farm");env.step(a);require(projected.tiles[44].kind==TileKind::WEED&&env.farms()[0].tiles[44].kind==TileKind::WEED,"zero expiry differs from actual PASS settlement");observed=true;break;}
  if(env.step_count()==0)a[0].market={dp7::action(Op::BUY_SEED,crop,1)};
  else if(env.step_count()==1)a[0].units[0]=dp7::action(Op::PLANT,crop);
  else if(dp7::plant(t)&&ctl.water_due(t))a[0].units[0]=dp7::action(Op::WATER);
  else if(dp7::plant(t)&&t.yield_units>0)a[0].units[0]=dp7::action(Op::HARVEST);
  env.step(a);
 }
 require(observed,"natural crop lifecycle did not reach zero-yield expiry");
}
int main(){
 int cases=0;
 {
  using namespace dp7;
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);for(auto&t:own.tiles)t.kind=TileKind::LOCKED;
  own.farmer={4,4};own.hands={{2,1}};own.money=5000;own.tiles[44]=Tile{};
  own.tiles[12].kind=TileKind::PLANT;own.tiles[12].crop=Item::STRAWBERRY;
  PrivateState priv{};priv.inventories.resize(2);priv.inventory_order.resize(2);Market market{};std::vector<int8_t>shops;
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=price(i,10000);}
  View v{120,5,0,own,rival,priv,market,shops};Params p;p.intraday_admission=true;
  PlayerAction a;a.units.resize(2);a.units[1]=action(Op::WATER);
  for(int kind:{W,C,T,S,M,G,CO,SH}){
   p.bias.fill(0);p.bias[kind]=1;Controller ctl(p);ctl.day=5;ctl.phase=3;ctl.plans.resize(2);
   priv.seeds.fill(1);priv.shed={};priv.shed[W]=1;for(int k:{G,CO,SH})priv.shed[k]=1;
   auto chosen=intraday::propose(ctl,v,a);require(chosen.valid&&chosen.kind==kind&&chosen.unit==0,"intraday eight-industry representation");cases++;
   auto before=a;intraday::consider(ctl,v,a);require(ctl.pending_admission.active&&ctl.pending_admission.kind==kind,"intraday proposal not persisted");cases++;
   require(Controller::same_action(a.units[1],before.units[1])&&a.units[0].op==Op::PASS,"intraday procurement interrupted existing production");cases++;
   auto next=v;next.step++;next.hour++;intraday::complete_pending(ctl,next);
   require(!ctl.pending_admission.active&&ctl.intraday_activated==1&&!ctl.plans[0].a.empty(),"stocked project not activated from real next state");cases++;
   require(ctl.target.size()==1&&ctl.target[0]==std::make_pair(44,kind),"daily task owner not updated");cases++;
  }
  p.bias.fill(0);p.bias[W]=1;Controller ctl(p);ctl.day=5;ctl.phase=3;ctl.plans.resize(2);
  priv.shed={};priv.seeds={};a.market.clear();
  require(!intraday::propose(ctl,v,a).valid,"inventory-only branch fabricated a seed");cases++;
  ctl.p.intraday_procurement=true;auto choice=intraday::propose(ctl,v,a);
  require(choice.valid&&choice.buy.size()==1&&choice.buy[0].op==Op::BUY_SEED&&choice.buy[0].item==Item::WHEAT,"funded branch missing ordered seed purchase");cases++;
  intraday::consider(ctl,v,a);auto next=v;next.step++;next.hour++;
  intraday::complete_pending(ctl,next);require(ctl.intraday_unfilled==1&&ctl.intraday_activated==0&&ctl.plans[0].a.empty(),"intent was treated as actual fill");cases++;
  a.market.clear();intraday::consider(ctl,v,a);priv.seeds[W]=1;intraday::complete_pending(ctl,next);
  require(ctl.intraday_activated==1&&ctl.plans[0].a.size()==2,"filled seed did not compile PLANT WATER");cases++;
  ctl.plans.assign(2,{});ctl.target.clear();ctl.p.intraday_procurement=false;a.market.clear();
  auto late=v;late.hour=23;require(!intraday::propose(ctl,late,a).valid,"intraday crossed watering deadline");cases++;
  auto oldkind=p;oldkind.bias.fill(0);oldkind.bias[M]=1;Controller latectl(oldkind);latectl.day=28;latectl.phase=3;latectl.plans.resize(2);auto latev=v;latev.day=28;latev.step=28*24;
  require(!intraday::propose(latectl,latev,a).valid,"project admitted without terminal first output");cases++;
  a.units[1]=action(Op::PLANT,W);own.tiles[12]=Tile{};
  require(!intraday::propose(ctl,v,a).valid,"issued seed action double reserved");cases++;
  a.units[1]=action(Op::WATER);own.tiles[12].kind=TileKind::PLANT;own.tiles[12].crop=Item::STRAWBERRY;
  ctl.plans[1]={{action(Op::PLANT,W)},{13},0};require(!intraday::propose(ctl,v,a).valid,"future seed commitment stolen");cases++;
  ctl.plans[1]={};priv.seeds={};ctl.p.intraday_procurement=true;a.market.assign(10,action(Op::SELL,W));
  require(!intraday::propose(ctl,v,a).valid,"intraday exceeded ten market orders");cases++;
  a.market.clear();own.money=0;a.market.push_back(action(Op::SELL,WO,100));priv.shed[WO]=100;
  require(!intraday::propose(ctl,v,a).valid,"unexecuted sale was treated as cash");cases++;
  own.money=5000;priv.shed={};priv.shed[W]=1;priv.shed[CO]=1;ctl.p.bias.fill(0);ctl.p.bias[CO]=1;a.market={action(Op::SELL,W,1)};
  intraday::consider(ctl,v,a);require(ctl.pending_admission.active&&a.market.empty(),"reserved animal feed sold during admission");cases++;
  ctl.p.intraday_admission=false;ctl.pending_admission={};auto unchanged=a;intraday::consider(ctl,v,a);
  require(!ctl.pending_admission.active&&a.market.size()==unchanged.market.size(),"disabled intraday branch changed actions");cases++;
  ctl.p.intraday_admission=true;intraday::consider(ctl,v,a);auto tomorrow=v;tomorrow.day++;tomorrow.step+=24;
  intraday::complete_pending(ctl,tomorrow);require(!ctl.pending_admission.active&&ctl.intraday_cancelled==1,"pending project leaked across days");cases++;
 }
 {
  namespace h=dp7audit::handoff;using namespace dp7;
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);own.farmer={4,4};own.hands={{5,4}};
  PrivateState priv{};priv.inventories.resize(2);priv.inventories[0][W]=1;priv.inventories[1][W]=1;
  for(int pos:{34,35}){own.tiles[pos].kind=TileKind::ANIMAL;own.tiles[pos].animal=Item::SHEEP;}
  Market market{};std::vector<int8_t>shops;View v{120,5,0,own,rival,priv,market,shops};Controller ctl;ctl.day=5;ctl.phase=3;
  ctl.plans={h::rebuild(v,0,{{35,{action(Op::FEED)}}}),h::rebuild(v,1,{{34,{action(Op::FEED)}}})};
  auto c=h::propose(ctl,v);require(c.feasible==1&&c.total_saved==2&&c.peak_saved==1,"resource exchange must remove crossed feed travel");cases++;
  require(h::order(c.base)==h::order(c.best),"exchange changed semantic actions");cases++;
  require(ctl.plans[0].target.back()==35&&ctl.plans[1].target.back()==34,"read-only diagnostic mutated planner");cases++;
  exchange::apply(ctl,v);require(ctl.plans[0].target.back()==35&&ctl.resource_exchange_checks==0,"resource flag default must be off");cases++;
  ctl.p.resource_aware_exchange=true;exchange::apply(ctl,v);require(ctl.plans[0].target.back()==34&&ctl.plans[1].target.back()==35&&ctl.resource_exchange_applied==1,"online exchange did not change actual task ownership");cases++;
  ctl.plans=c.base;ctl.p.resource_aware_exchange=false;
  priv.inventories[1][W]=0;c=h::propose(ctl,v);require(!c.base_resources&&c.feasible==0,"resource cannot teleport between workers");cases++;
  priv.inventories[1][W]=1;v.hour=23;c=h::propose(ctl,v);require(!c.base_deadline&&c.feasible==0,"handoff exceeded day deadline");cases++;
  v.hour=0;ctl.plans[1]=h::rebuild(v,1,{{34,{action(Op::FEED),action(Op::CARE)}}});c=h::propose(ctl,v);
  require(c.compatible==0,"different dependent groups were interchanged");cases++;
  auto picks=std::vector<Plan>{h::rebuild(v,0,{{44,{action(Op::PICKUP,W,2)}},{35,{action(Op::FEED)}}}),h::rebuild(v,1,{{45,{action(Op::PICKUP,W,2)}},{34,{action(Op::FEED)}}})};
  priv.shed[W]=3;require(!h::resources(v,picks),"shared warehouse stock double reserved");cases++;
  priv.shed[W]=4;require(h::resources(v,picks),"available distinct pickup reservations rejected");cases++;
  // Both operate at the same tick: swapping the owners changes official unit order.
  auto first=std::vector<Plan>{Plan{{action(Op::FERTILIZE)},{34},0},Plan{{action(Op::WATER)},{34},0}};
  auto reversed=first;std::swap(reversed[0],reversed[1]);require(h::order(first)!=h::order(reversed),"same-tile fertilizer/water ordering invisible");cases++;
  require(!h::transferable({35,{action(Op::HARVEST),action(Op::FEED)}}),"output-producing group misclassified as equal-input maintenance");cases++;
 }
 for(int c:{0,1,4})for(int h=(c==4?10:2);h<=(c==0?4:c==1?3:12);h++)for(bool fertilize:{false,true}){
  auto daily=single_crop(c,h,false,fertilize),efficient=single_crop(c,h,true,fertilize);
  dp7::Controller ctl;require(daily.quantity==ctl.finite_yield(c,h,fertilize),"finite rule output");
  require(efficient.quantity==daily.quantity,"water schedule changed finite yield");require(efficient.water<=daily.water,"water schedule cost");cases++;
 }
 for(int c:{2,3}){
  auto daily=single_crop(c,29,false,true),efficient=single_crop(c,29,true,true);
  require(daily.quantity==8&&efficient.quantity==8,"ongoing four fertilized batches");require(efficient.water<daily.water,"ongoing water action saving");cases++;
 }
 for(int day=0;day<30;day++)for(int kind:{0,1,2,3,4,9,10,11})for(bool renew:{false,true}){
  dp7::Params p;p.renew_ongoing=renew;dp7::Controller ctl(p);ctl.day=day;auto pr=kind>=9?ctl.animal_project(kind):ctl.crop_project(kind);auto cal=ctl.project_calendar(kind);
  double quantity=0,money=0;for(int d=day;d<30;d++){quantity+=cal.quantity[d][pr.item];money+=cal.fixed[d];}
  require(quantity==pr.out,"cash calendar output mismatch");
  if(pr.out>0)require(money==-(kind>=9?pr.capital:pr.seed),"money and commodity units mixed");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);PrivateState priv{};Market market{};std::vector<int8_t>shops;
  auto&t=own.tiles[44];t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=0;t.yield_units=2;t.max_lifespan_step=17*24;
  dp7::Params p;p.renew_ongoing=true;dp7::Controller ctl(p);ctl.day=16;ctl.target={{44,dp7::S}};
  dp7::View view{16*24,16,0,own,rival,priv,market,shops};auto jobs=ctl.jobs(view);
  require(jobs.size()==1&&jobs[0].actions.size()==4,"renewal job size");
  require(jobs[0].actions[0].op==Op::HARVEST&&jobs[0].actions[1].op==Op::DIG&&jobs[0].actions[2].op==Op::PLANT&&jobs[0].actions[3].op==Op::WATER,"renewal causality");cases++;
  auto degraded=ctl.reserve(view,jobs);require(degraded.size()==1&&degraded[0].actions.size()==2,"missing seed must not water empty soil");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);own.money=0;PrivateState priv{};Market market{};std::vector<int8_t>shops{0,0,0,0,0,0,0,7};
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
  priv.shed[dp7::MI]=1;priv.shed[dp7::WO]=1;
  for(int i=0;i<10;i++){auto&t=rival.tiles[i];t.kind=TileKind::PASTURE;t.animal=Item::SHEEP;t.placed_day=0;t.yield_units=6;}
  dp7::View view{28*24,28,0,own,rival,priv,market,shops};dp7::Controller base;base.day=28;
  require(base.holding_demand(view)==base.demand(view),"zero holding weight changed forecast");cases++;
  dp7::Params p;p.hold_opponent_supply_weight=.75;dp7::Controller aware(p);aware.day=28;
  auto d=aware.demand(view),adjusted=aware.holding_demand(view),supply=aware.opponent_supply(view);
  require(adjusted[dp7::WO]==d[dp7::WO]-.75*supply[dp7::WO],"holding supply weight not applied");
  require(adjusted[dp7::W]>d[dp7::W],"feed demand must not become sellable wheat");cases+=2;
  auto a=base.funding(view,1,0),b=aware.funding(view,1,0);
  require(a.size()==1&&b.size()==1&&a[0].item==Item::MILK&&b[0].item==Item::WOOL,"holding feature must change real financing action");cases++;
  rival.tiles.assign(100,Tile{});
  require(aware.holding_demand(view)==base.holding_demand(view),"empty rival changed holdings");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);PrivateState priv{};Market market{};std::vector<int8_t>shops(8,4);
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
  dp7::View view{10*24,10,0,own,rival,priv,market,shops};dp7::Controller base;base.day=10;
  require(base.feed_future_price(view,20)==dp7::price(dp7::W,int(10000-base.demand(view)[dp7::W])),"default feed forecast regression");cases++;
  dp7::Params p;p.feed_forecast_days=3;p.opponent_supply_weight=.75;dp7::Controller ctl(p);ctl.day=10;
  int no_stock=ctl.feed_future_price(view,20);priv.shed[dp7::W]=60;int with_stock=ctl.feed_future_price(view,20);
  require(with_stock<no_stock,"own feed stock must offset predicted external purchases");cases++;
  for(int i=0;i<20;i++){auto&t=rival.tiles[i];t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.placed_day=0;}
  require(ctl.feed_future_price(view,20)>with_stock,"visible rival feed consumption missing");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);own.unlocked_mask=7;own.money=5000;PrivateState priv{};Market market{};std::vector<int8_t>shops;
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
  dp7::Params p;p.economic_land=true;p.max_land=4;p.operating_reserve=100;dp7::Controller ctl(p);ctl.day=10;
  dp7::View v{240,10,0,own,rival,priv,market,shops};
  require(ctl.cost(v,dp7::action(Op::BUY_LAND))==4000,"fourth land actual quote");cases++;
  require(ctl.investment_budget(v,4,0)==600,"fourth land project budget must reserve 4000");cases++;
  require(ctl.investment_budget(v,3,0)==4600,"KEEP land must not reserve unbought land");cases++;
  own.money=4000;require(ctl.investment_budget(v,4,0)==0,"fourth land cannot overcommit cash");cases++;
  ctl.choose(v);require(ctl.planned_land==3,"reserve threshold must keep land");cases++;
  own.unlocked_mask=15;own.money=100000;ctl.choose(v);require(ctl.planned_land==4,"no fifth land");cases++;
  require(std::isinf(dp7::next_land_cost(4)),"no finite fifth land quote");cases++;
 }
 {
  dp7::Params p;p.capacity_hauling=true;dp7::Controller ctl(p),base;ctl.day=base.day=15;
  std::vector<dp7::Job>js;
  for(int pos:{0,9,90,99}){dp7::Job j;j.pos=pos;j.out[dp7::MI]=30;j.actions={dp7::action(Op::HARVEST)};js.push_back(j);}
  require(ctl.hauling_needed(js)&&!base.hauling_needed(js),"capacity hauling opt-in effect");cases++;
  int h=ctl.estimate(js,0);require(h>=base.estimate(js,0),"hauling cannot remove work from staffing");cases++;
  int setup=std::max(1,(h+9)/10);auto rs=ctl.pack(js,ctl.starts(h),23-setup,true);
  require(rs.second==0,"predicted hauling needs complete jobs");cases++;
  for(auto&r:rs.first)require(r.total(true)<=23-setup,"hauling return deadline");cases++;
  js.resize(1);require(!ctl.hauling_needed(js),"small harvest must keep auto-deposit");cases++;
  ctl.day=29;js.resize(4);require(!ctl.hauling_needed(js),"terminal has its own return policy");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);PrivateState priv{};Market market{};std::vector<int8_t>shops;
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
  dp7::View view{240,10,0,own,rival,priv,market,shops};dp7::Controller base;base.day=10;
  dp7::Params p;p.committed_feed_weight=1;dp7::Controller ctl(p);ctl.day=10;
  auto before=ctl.values(view,{},{}),after=ctl.values(view,{},{},200);
  auto old0=base.values(view,{},{}),old200=base.values(view,{},{},200);
  for(size_t i=0;i<before.size();i++){
   require(old0[i].first==old200[i].first,"zero feed commitment regression");cases++;
   require(before[i].first==old0[i].first,"no commitment must not change values");cases++;
   if(before[i].second.kind==dp7::W)require(after[i].first>before[i].first,"new animals must increase marginal wheat value");
   else require(after[i].first==before[i].first,"feed demand leaked to unrelated product supply");cases++;
  }
  require(ctl.animal_project(dp7::CO).feed==19,"feed obligation from placement to terminal");cases++;
 }
 for(int c:{0,1,4})for(int h=(c==4?10:2);h<=(c==0?4:c==1?3:12);h++){
  require(repeated_finite_output(c,h)>0,"finite continuation must complete");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);PrivateState priv{};Market market{};std::vector<int8_t>shops;
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
  dp7::Params p;p.existing_replant_weight=1;dp7::Controller base,ctl(p);base.day=ctl.day=0;
  dp7::View v{0,0,0,own,rival,priv,market,shops};
  require(ctl.existing(v)==base.existing(v),"empty farm must not invent continuation");cases++;
  auto&t=own.tiles[44];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=0;t.yield_units=1;
  require(ctl.replant_supply(v)[dp7::W]==24,"W4 seven batches include six future replants");cases++;
  require(ctl.existing(v)[dp7::W]==base.existing(v)[dp7::W]+24,"continuation field changes actual pricing input");cases++;
  auto before=base.values(v,base.existing(v),{}),after=ctl.values(v,ctl.existing(v),{});
  for(size_t i=0;i<before.size();i++)if(before[i].second.kind==dp7::W){require(after[i].first<before[i].first,"committed supply must reduce new wheat score");cases++;}
  t.crop=Item::STRAWBERRY;t.yield_units=0;
  require(ctl.replant_supply(v)[dp7::S]==4&&ctl.replant_supply(v)[dp7::F]==-1,"strawberry partial second cycle timing and fertilizer");cases++;
  base.day=ctl.day=29;require(ctl.replant_supply(v)==dp7::Counts{},"terminal must not create future crops");cases++;
 }
 {
  std::mt19937 gen(41073);int improved=0;
  for(int i=0;i<200;i++){
   dp7::Params p;p.insertion_hire_estimate=true;dp7::Controller ctl(p),base;ctl.day=base.day=i%30;
   std::vector<dp7::Job>jobs;std::array<bool,100>used{};
   for(int j=0;j<8+i%24;j++){int pos;do{pos=gen()%100;}while(used[pos]);used[pos]=true;
    dp7::Job job;job.pos=pos;job.priority=j%3;job.out[dp7::MI]=j%4;
    if(j%3==0)job.needs[dp7::W]=1;
    for(int k=0;k<1+j%4;k++)job.actions.push_back(dp7::action(Op::PASS));jobs.push_back(job);
   }
   int old=base.estimate(jobs,5),now=ctl.estimate(jobs,5);require(now<=old,"insertion estimate increased hiring");cases++;
   if(now<old){improved++;int setup=(5+now+9)/10;int budget=(ctl.day>=29?22:23)-std::max(1,setup)+1;
    require(ctl.pack(jobs,ctl.starts(now),budget,ctl.day>=29).second==0||ctl.pack(jobs,ctl.starts(now),budget,ctl.day>=29,true).second==0,"reduced hire without feasible submitted tasks");cases++;}
  }
  require(improved>0,"insertion estimate is a dead flag");cases++;
 }
 for(int c=0;c<5;c++)for(int day:{0,7,14})for(int pos:{0,44,99}){
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);own.farmer={4,4};own.money=1000;own.unlocked_mask=15;
  PrivateState priv{};priv.inventories.resize(1);Market market{};std::vector<int8_t>shops;
  own.tiles[pos].kind=TileKind::WEED;
  dp7::Params p;p.reconcile_seed_drift=true;dp7::Controller ctl(p);ctl.day=day;ctl.phase=2;ctl.target={{pos,c}};
  dp7::View v{day*24+2,day,2,own,rival,priv,market,shops};
  auto disabled=ctl;disabled.p.reconcile_seed_drift=false;require(!disabled.reconcile_seeds(v),"disabled reconciliation changed behavior");cases++;
  auto old_gap=ctl;old_gap.prepared_seed_need[c]=1;require(!old_gap.reconcile_seeds(v),"old unfunded plan disguised as new seed demand");cases++;
  auto enough=ctl;priv.seeds[c]=1;require(!enough.reconcile_seeds(v),"seed reconciliation bought duplicate stock");priv.seeds[c]=0;cases++;
  auto poor=ctl;own.money=dp7::seed_price[c]-1;require(!poor.reconcile_seeds(v),"seed repair overspent cash");own.money=1000;cases++;
  auto late=ctl;dp7::View end{day*24+23,day,23,own,rival,priv,market,shops};require(!late.reconcile_seeds(end),"seed repair crosses day deadline");cases++;
  auto busy=ctl;busy.phase=3;require(!busy.reconcile_seeds(v),"repair must not interrupt compiled work");cases++;
  auto order=ctl.reconcile_seeds(v);require(order&&order->market.size()==1&&order->market[0].op==Op::BUY_SEED&&int(order->market[0].item)==c&&order->market[0].quantity==1,"state-driven replant seed intent missing");cases++;
  require(!ctl.reconcile_seeds(v)&&ctl.seed_reconciliations==1&&ctl.seed_reconciled_units==1,"reconciliation repeated without settlement");cases++;
  priv.seeds[c]=1;dp7::View filled{day*24+3,day,3,own,rival,priv,market,shops};ctl.compile(filled);
  int plants=0,waters=0;for(auto&plan:ctl.plans)for(auto&a:plan.a){plants+=a.op==Op::PLANT;waters+=a.op==Op::WATER;}
  require(plants==1&&waters==1&&ctl.actual_drop==0,"filled repair must restore whole plant-water intent");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);own.farmer={4,4};own.money=1000;own.unlocked_mask=15;
  PrivateState priv{};priv.inventories.resize(1);Market market{};std::vector<int8_t>shops;
  own.tiles[44].kind=TileKind::WEED;auto&t=own.tiles[45];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=0;t.yield_units=1;t.max_lifespan_step=98;
  dp7::Params p;p.reconcile_seed_drift=true;dp7::Controller ctl(p);ctl.day=4;ctl.phase=2;ctl.target={{44,dp7::C},{45,dp7::W}};
  dp7::View v{98,4,2,own,rival,priv,market,shops};require(!ctl.reconcile_seeds(v),"repair waited through observable immediate crop decay");cases++;
 }
 {
  Simulator env(Config{},61519);dp7::Params p;p.reconcile_seed_drift=true;dp7::Controller dirty(p),fresh(p);
  dirty.day=15;dirty.last_step=400;dirty.phase=2;dirty.seed_reconciliations=12;dirty.seed_reconciled_units=37;dirty.seed_reconcile_checked=true;dirty.prepared_seed_need.fill(19);dirty.anticipated_releases=19;
  dp7::View v{0,0,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  auto a=dirty.act(v),b=fresh.act(v);
  require(dirty.anticipated_releases==0&&dirty.seed_reconciliations==0&&dirty.seed_reconciled_units==0&&dirty.prepared_seed_need==fresh.prepared_seed_need&&dirty.seed_reconcile_checked==fresh.seed_reconcile_checked,"reconciliation carry leaked across episodes");cases++;
  require(a.market.size()==b.market.size()&&dirty.target==fresh.target,"reset changed new opening");cases++;
  for(size_t i=0;i<a.market.size();i++){auto&x=a.market[i];auto&y=b.market[i];require(x.op==y.op&&x.item==y.item&&x.quantity==y.quantity,"reset opening market mismatch");cases++;}
 }
 for(int c:{dp7::T,dp7::S}){zero_expiry_rule_fixture(c);cases++;}
 for(int c=0;c<5;c++)for(int pos:{0,44,99})for(int yield:{0,1,4}){
  Farm farm{};farm.tiles.resize(100);auto&t=farm.tiles[pos];t.kind=TileKind::PLANT;t.crop=Item(c);t.yield_units=yield;t.max_lifespan_step=240;
  auto future=farm;int n=dp7::Controller::project_zero_expiry(future,246);require(n==(yield==0)&&farm.tiles[pos].kind==TileKind::PLANT,"zero-only projection boundary");cases++;
  future=farm;require(dp7::Controller::project_zero_expiry(future,247)==0,"odd decay cadence was invented");cases++;
  future=farm;require(dp7::Controller::project_zero_expiry(future,238)==0,"future crop expiry happened early");cases++;
  future=farm;future.tiles[pos].max_lifespan_step=-1;require(dp7::Controller::project_zero_expiry(future,246)==0,"unknown crop expiry invented");cases++;
 }
 {
  Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);own.farmer={4,4};own.money=10000;own.unlocked_mask=1;
  PrivateState priv{};priv.inventories.resize(1);priv.seeds.fill(100);Market market{};std::vector<int8_t>shops(8,4);
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
  for(auto&t:own.tiles)t.kind=TileKind::LOCKED;
  for(int pos=0;pos<100;pos++)if(dp7::quad(pos)==0){auto&t=own.tiles[pos];t=Tile{};t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=17;t.yield_units=1;t.max_lifespan_step=22*24;}
  auto&t=own.tiles[44];t.crop=Item::STRAWBERRY;t.planted_day=0;t.yield_units=0;t.max_lifespan_step=17*24;
  dp7::Params p;p.plan_zero_expiry=true;p.economic_land=true;p.max_land=1;p.max_animals=0;p.max_strawberry=0;p.max_hands=0;
  dp7::Controller ctl(p);dp7::View v{17*24,17,0,own,rival,priv,market,shops};auto action=ctl.act(v);
  int selected=-1;for(auto[pos,k]:ctl.target)if(pos==44)selected=k;
  require(selected>=0&&selected<5&&selected!=dp7::S,"released land retained forced old industry");cases++;
  require(action.market.empty()&&action.units[0].op==Op::PASS&&ctl.phase==2,"empty preparation queue did not await predicted decay");cases++;
  require(own.tiles[44].kind==TileKind::PLANT&&ctl.anticipated_releases==1,"planning projection changed authoritative map");cases++;
 }
 {
  dp7::Params p;p.cashflow_value_mode=2;dp7::Controller ctl(p);ctl.day=0;Market market{};market.inventory.fill(10000);
  dp7::Controller::Calendar own{},rival{},demand{};dp7::Controller::CashSchedule candidate;
  candidate.quantity[1][dp7::S]=50;candidate.fixed[0]=-100;
  auto independent=ctl.marginal_cash(market,own,rival,demand,candidate);
  require(independent.project>0&&independent.existing_delta==0,"empty baseline has fictitious price externality");cases++;
  own[10][dp7::S]=50;auto competing=ctl.marginal_cash(market,own,rival,demand,candidate);
  require(competing.project==independent.project&&competing.existing_delta<0,"new early supply fails to reprice later owned output");cases++;
  own={};own[10][dp7::MI]=50;auto separate=ctl.marginal_cash(market,own,rival,demand,candidate);
  require(separate.existing_delta==0,"unrelated commodity affected directly");cases++;
  own={};own[0][dp7::S]=50;candidate={};candidate.quantity[10][dp7::S]=50;
  require(ctl.marginal_cash(market,own,rival,demand,candidate).existing_delta==0,"future supply changed already sold inventory");cases++;
  own={};own[10][dp7::W]=50;candidate={};candidate.quantity[1][dp7::W]=-50;
  auto feed=ctl.marginal_cash(market,own,rival,demand,candidate);
  require(feed.project<0&&feed.existing_delta>0,"new feed demand did not affect owned wheat price");cases++;
  own={};own[10][dp7::W]=-50;candidate={};candidate.quantity[1][dp7::W]=50;
  require(ctl.marginal_cash(market,own,rival,demand,candidate).existing_delta>0,"new wheat supply did not reduce later feed expense");cases++;
  for(int item=0;item<9;item++)for(int stock:{9900,10000,12000}){
   double inv=stock;double paid=dp7::Controller::projected_trade(item,inv,-3);
   double got=dp7::Controller::projected_trade(item,inv,3);
   require(paid+got==0,"projected unchanged-market round trip creates money");cases++;
   if(dp7::price(item,stock)>1)require(inv==stock,"nonfloor roundtrip lost inventory");cases++;
  }
  ctl.p.cashflow_value_mode=3;own={};rival={};candidate={};
  candidate.quantity[1][dp7::S]=50;rival[10][dp7::S]=50;
  auto rivalry=ctl.marginal_cash(market,own,rival,demand,candidate);
  require(rivalry.opponent_delta<0&&rivalry.existing_delta==0,"early own supply did not reprice visible rival output");cases++;
  rival={};rival[10][dp7::W]=-50;candidate={};candidate.quantity[1][dp7::W]=50;
  require(ctl.marginal_cash(market,own,rival,demand,candidate).opponent_delta>0,"cheap wheat benefit to rival feed buyer ignored");cases++;
  rival={};require(ctl.marginal_cash(market,own,rival,demand,candidate).opponent_delta==0,"PASS acquired fictitious rival payoff");cases++;
  rival[0][dp7::S]=50;candidate={};candidate.quantity[10][dp7::S]=50;
  require(ctl.marginal_cash(market,own,rival,demand,candidate).opponent_delta==0,"future project changed rival's earlier sales");cases++;
  double high=100000;double floor_cash=dp7::Controller::projected_trade(dp7::S,high,7);
  require(floor_cash==7&&high==100000,"one-dollar sales increased market supply");cases++;
 }
 // Current-market projection must match the frozen own-market kernel. This
 // deliberately does not assert equality to future rival/town/decay events.
 {
  std::mt19937 rng(78162);Simulator env(Config{},71323);
  for(int trial=0;trial<200;trial++){
   dp7::Controller ctl;ctl.day=env.day();ctl.phase=1;
   for(int n=0;n<10;n++){
    int k=rng()%6,q=int(rng()%8);Op op=k==0?Op::BUY_SEED:k==1?Op::BUY_PRODUCT:k==2?Op::SELL:k==3?Op::BUY_ANIMAL:k==4?Op::HIRE:Op::BUY_LAND;
    int item=k==0?int(rng()%5):k==1?(rng()%2?dp7::W:dp7::F):k==2?int(rng()%9):k==3?9+int(rng()%3):-1;
    ctl.queue.push_back(dp7::action(op,item,q));
   }
   dp7::View v{env.step_count(),env.day(),env.hour(),env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
   auto p=ctl.project_preparation(v);auto exact=env.project_own_market(0,ctl.queue);
   require(p.farm.money==exact.farms()[0].money&&p.priv.shed==exact.privates()[0].shed&&p.priv.seeds==exact.privates()[0].seeds,"conditional own-order cash/material projection mismatch");cases++;
   require(p.farm.unlocked_mask==exact.farms()[0].unlocked_mask&&p.market.inventory==exact.market().inventory,"conditional land/market projection mismatch");cases++;
   require(p.farm.hands.size()==exact.farms()[0].hands.size(),"conditional hire count mismatch");cases++;
   for(size_t u=0;u<p.farm.hands.size();u++)require(dp7::cell(p.farm.hands[u])==dp7::cell(exact.farms()[0].hands[u]),"conditional spawn mismatch");
   // Change real state through legal transactions rather than invent a fixture
   // via private-state writes; ordinary daily reset keeps worker counts finite.
   std::array<PlayerAction,2>a;a[0].market=ctl.queue;env.step(a);
   if(env.done())env.reset(71323+trial);
  }
 }
 {
  int changes=0;
  for(int money:{30,120,500,1200,3000})for(int hands:{0,2,5}){
   Simulator env(Config{},13102);Farm own=env.farms()[0];own.money=money;PrivateState priv=env.privates()[0];
   dp7::Params p;p.funded_bundle_mode=2;p.max_hands=hands;p.economic_land=true;p.efficient_water=true;p.efficient_care=true;
   dp7::Controller ctl(p);ctl.day=11;ctl.planned_land=2;
   for(int pos=0;pos<100;pos++)if(dp7::quad(pos)<2)ctl.target.emplace_back(pos,pos%3==0?dp7::CO:dp7::S);
   auto&t=own.tiles[44];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=10;t.yield_units=1;t.max_lifespan_step=16*24;
   for(auto&[pos,k]:ctl.target)if(pos==44)k=dp7::W;
   dp7::View v{11*24,11,0,own,env.farms()[1],priv,env.market(),env.shops()};
   ctl.prepare_orders(v,v,0);auto before=ctl.preview_bundle(v);ctl.compare_funded_bundles(v,v,0);auto after=ctl.preview_bundle(v);
   require(after.score>=before.score-1e-6,"joint selection degraded its own declared score");cases++;
   for(int pos=0;pos<100;pos++)require((before.existing_actions[pos]&after.existing_actions[pos])==before.existing_actions[pos],"joint selection removed baseline-schedulable existing work");cases++;
   require(ctl.bundle_evaluations>0&&after.started<=after.proposed,"bundle candidate accounting invalid");cases++;
   require(own.money==money&&own.tiles[44].kind==TileKind::PLANT,"preview mutated authoritative farm");cases++;
   changes+=ctl.bundle_switches;
  }
  require(changes>0,"funded bundle mode has no behavioral effect in budget/capacity fixtures");cases++;
 }
 // Procurement is a market phase, not a reason to idle already present units.
 // Only a real, resource-free first operation may be brought forward.
 for(int kind:{dp7::W,dp7::C,dp7::T,dp7::S,dp7::M,dp7::G,dp7::CO,dp7::SH})for(bool weed:{false,true}){
  Simulator env(Config{},81924);auto own=env.farms()[0];auto priv=env.privates()[0];
  own.money=1000;own.hands={{4,4}};priv.inventories.resize(2);priv.inventory_order.resize(2);
  own.tiles[44]=Tile{};if(weed)own.tiles[44].kind=TileKind::WEED;
  dp7::Params p;p.preparation_work=true;dp7::Controller ctl(p);ctl.day=0;ctl.phase=1;ctl.target={{44,kind}};
  dp7::View v{0,0,0,own,env.farms()[1],priv,env.market(),env.shops()};
  auto empty=ctl.preparation_prefix(v);require(empty[0].op==Op::PASS&&empty[1].op==Op::PASS,"unfunded new project started during preparation");cases++;
  ctl.queue={dp7::action(kind<5?Op::BUY_SEED:Op::BUY_ANIMAL,kind,1),dp7::action(Op::HIRE)};
  auto a=ctl.preparation_prefix(v);Op expected=weed?Op::DIG:kind<5?Op::PASS:kind==dp7::G?Op::BUILD_COOP:Op::BUILD_PASTURE;
  require(a.size()==2&&a[0].op==expected&&a[1].op==Op::PASS,"preparation emitted future worker or duplicate plot operation");cases++;
  require(priv.shed[kind]==0&&priv.seeds==env.privates()[0].seeds&&own.money==1000,"preparation spent projected resources in authoritative state");cases++;
  own.money=0;require(ctl.preparation_prefix(v)[0].op==Op::PASS,"unaffordable queue was treated as funded");cases++;
 }
 {
  Simulator env(Config{},91241);dp7::Params p;p.preparation_work=true;dp7::Controller ctl(p);
  ctl.day=0;ctl.phase=1;ctl.target={{44,dp7::CO}};
  ctl.queue={dp7::action(Op::BUY_ANIMAL,dp7::CO,1),dp7::action(Op::HIRE)};
  dp7::View v{0,0,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  auto disabled=ctl;disabled.p.preparation_work=false;auto old=disabled.act(v);auto now=ctl.act(v);
  require(old.units.size()==1&&old.units[0].op==Op::PASS&&now.units.size()==1&&now.units[0].op==Op::BUILD_PASTURE,"preparation toggle did not affect actual action");cases++;
  require(now.market.size()==old.market.size()&&ctl.preparation_actions==1,"preparation changed market queue or accounting");cases++;
  std::array<PlayerAction,2>aa;aa[0]=now;env.step(aa);
  require(env.farms()[0].tiles[44].kind==TileKind::PASTURE&&env.privates()[0].shed[dp7::CO]==1&&env.farms()[0].hands.size()==1,"simultaneous legal construction and purchase did not settle");cases++;
  // A new episode must not inherit preparation counters or task ownership.
  env.reset(91241);auto fresh=dp7::Controller(p);auto reset=ctl.act(v),clean=fresh.act(v);
  require(ctl.preparation_actions==fresh.preparation_actions&&reset.units.size()==clean.units.size()&&reset.units[0].op==clean.units[0].op,"preparation carry crossed episode reset");cases++;
 }
 {
  Simulator env(Config{},21942);auto own=env.farms()[0];auto priv=env.privates()[0];
  own.hands={{4,4}};priv.inventories.resize(2);priv.inventory_order.resize(2);
  dp7::Params p;p.preparation_work=true;dp7::Controller ctl(p);ctl.day=3;ctl.phase=1;ctl.target={{44,dp7::T}};
  dp7::View v{72,3,0,own,env.farms()[1],priv,env.market(),env.shops()};
  auto&t=own.tiles[44];t=Tile{};t.kind=TileKind::PLANT;t.crop=Item::TOMATO;t.planted_day=2;
  require(ctl.preparation_prefix(v)[0].op==Op::WATER,"available water prefix missed");cases++;
  t.watered_today=true;require(ctl.preparation_prefix(v)[0].op==Op::PASS,"already watered crop repeated");cases++;
  t.watered_today=false;t.yield_units=4;
  require(ctl.preparation_prefix(v)[0].op==Op::PASS,"preparation skipped HARVEST prerequisite");cases++;
  t=Tile{};t.kind=TileKind::ANIMAL;t.animal=Item::SHEEP;t.placed_day=0;t.fed_today=true;ctl.target={{44,dp7::SH}};
  auto a=ctl.preparation_prefix(v);require(a[0].op==Op::CARE&&a[1].op==Op::PASS,"CARE prefix or plot ownership wrong");cases++;
  t.fed_today=false;require(ctl.preparation_prefix(v)[0].op==Op::PASS,"preparation skipped FEED prerequisite");cases++;
  t.fed_today=true;t.fertilizer_available=true;require(ctl.preparation_prefix(v)[0].op==Op::PASS,"preparation skipped fertilizer collection prerequisite");cases++;
  own.farmer={3,4};own.hands.clear();priv.inventories.resize(1);priv.inventory_order.resize(1);
  require(ctl.preparation_prefix(v)[0].op==Op::PASS,"preparation invented movement to distant project");cases++;
 }
 // Independent service branches must partition, not duplicate, their parent.
 for(int kind:{dp7::W,dp7::C,dp7::T,dp7::S,dp7::M,dp7::G,dp7::CO,dp7::SH})for(int day:{3,10,18,29})for(bool fresh:{false,true}){
  Simulator env(Config{},21461);auto own=env.farms()[0];auto priv=env.privates()[0];own.money=10000;
  auto&t=own.tiles[44];t=Tile{};if(!fresh){t.kind=kind<5?TileKind::PLANT:TileKind::ANIMAL;t.planted_day=t.placed_day=0;
   if(kind<5){t.crop=Item(kind);t.yield_units=4;t.max_lifespan_step=600;}
   else{t.animal=Item(kind);t.yield_units=6;t.fertilizer_available=true;}}
  dp7::Controller base;auto split=base;split.p.split_service_jobs=true;base.day=split.day=day;base.target=split.target={{44,kind}};
  dp7::View v{24*day,day,0,own,env.farms()[1],priv,env.market(),env.shops()};auto before=base.jobs(v),after=split.jobs(v);
  auto signature=[](const std::vector<dp7::Job>&jobs){dp7::Counts needs{},seeds{},out{};std::vector<std::tuple<int,int,int>>acts;
   for(auto&j:jobs){dp7::add(needs,j.needs);dp7::add(seeds,j.seeds);dp7::add(out,j.out);for(auto a:j.actions)acts.emplace_back(int(a.op),int(a.item),a.quantity);}
   std::sort(acts.begin(),acts.end());return std::tuple(needs,seeds,out,acts);};
  require(signature(before)==signature(after),"shared service duplicated or lost operations/resources/output");cases++;
  if(fresh||kind==dp7::W||kind==dp7::C||kind==dp7::M)require(after.size()==before.size(),"causal establishment/replant chain was split");cases++;
  if(!fresh&&kind>=9&&day<29)require(after.size()==3,"existing animal service stayed locked to one unit");cases++;
  for(auto&j:after){bool fertilized=false;for(auto a:j.actions){if(a.op==Op::FERTILIZE)fertilized=true;if(a.op==Op::WATER&&j.needs[dp7::F]>0)require(fertilized,"water moved ahead of fertilizer");}}cases++;
 }
 {
  std::mt19937 rng(90182);int different=0;
  for(int trial=0;trial<120;trial++){
   std::vector<dp7::Job>jobs;for(int i=0;i<10+int(rng()%16);i++){dp7::Job j;j.pos=rng()%100;j.priority=i%3;
    int n=1+rng()%4;while(n--)j.actions.push_back(dp7::action(Op::PASS));if(i%2)j.needs[dp7::W]=1;if(i%3)j.out[dp7::MI]=2;jobs.push_back(j);}
   dp7::Route route(0,44);for(int i=0;i<4;i++)route.append(jobs[i]);
   for(bool ret:{false,true})for(int at=0;at<=int(route.jobs.size());at++){
    auto inserted=route.jobs;inserted.insert(inserted.begin()+at,jobs[4]);
    require(dp7::Controller::insertion_cost(route,jobs[4],at,ret)==route.ordered(inserted,ret),"constant-time insertion cost mismatch");cases++;
   }
   dp7::Controller base,smart;smart.p.regret_schedule=true;auto starts=base.starts(2+trial%5);int budget=18+trial%8;
   auto before=base.pack(jobs,starts,budget,true),after=smart.pack(jobs,starts,budget,true);
   require(after.second<=before.second,"joint scheduler lost baseline-scheduled jobs");cases++;
   for(auto&r:after.first)require(r.total(true)<=budget&&r.cost==r.ordered(r.jobs,false),"joint schedule exceeded daily budget or corrupted route cost");cases++;
   std::vector<std::tuple<int,int,int>>seen;for(auto&r:after.first)for(auto&j:r.jobs)seen.emplace_back(j.pos,j.priority,int(j.actions.size()));
   require(int(seen.size())==int(jobs.size())-after.second,"joint scheduler duplicate/missing job count");cases++;
   different+=smart.regret_improvements;
  }
  require(different>0,"regret scheduling has no behavioral effect");cases++;
 }
 {
  // The execution and workforce-sizing contexts are independently switchable.
  // The legacy S3S switch must continue to enable both contexts.
  std::vector<dp7::Job>jobs;for(int i=0;i<12;i++){dp7::Job j;j.pos=(i*17)%100;j.priority=i%2;for(int k=0;k<1+i%3;k++)j.actions.push_back(dp7::action(Op::PASS));jobs.push_back(j);}
  dp7::Controller compile_only;compile_only.p.regret_compile=true;
  auto starts=compile_only.starts(4);compile_only.pack(jobs,starts,22,true);
  require(compile_only.regret_trials==1,"compile-only switch missed execution context");cases++;
  compile_only.pack(jobs,starts,22,true,false,true);
  require(compile_only.regret_trials==1,"compile-only switch leaked into hire estimate");cases++;
  dp7::Controller hire_only;hire_only.p.regret_hire_estimate=true;
  hire_only.pack(jobs,starts,22,true);
  require(hire_only.regret_trials==0,"hire-only switch leaked into execution context");cases++;
  hire_only.pack(jobs,starts,22,true,false,true);
  require(hire_only.regret_trials==1,"hire-only switch missed workforce context");cases++;
  dp7::Controller legacy;legacy.p.regret_schedule=true;
  legacy.pack(jobs,starts,22,true);legacy.pack(jobs,starts,22,true,false,true);
  require(legacy.regret_trials==2,"legacy S3S switch no longer enables both contexts");cases++;
 }
 // The frozen unit rules allow CARE and FEED in either order while both remain
 // before the same day-end. Test natural animals, not patched simulator state.
 for(int kind:{dp7::G,dp7::CO,dp7::SH}){
  Simulator env(Config{},34912);bool checked=false;
  while(env.step_count()<30){std::array<PlayerAction,2>a;a[0].units.resize(1);
   int step=env.step_count();auto&t=env.farms()[0].tiles[44];
   if(step==0)a[0].market={dp7::action(Op::BUY_ANIMAL,kind,1),dp7::action(Op::BUY_PRODUCT,dp7::W,20)};
   else if(step==1)a[0].units[0]=dp7::action(Op::PICKUP,kind,1);
   else if(step==2)a[0].units[0]=dp7::action(kind==dp7::G?Op::BUILD_COOP:Op::BUILD_PASTURE);
   else if(step==3)a[0].units[0]=dp7::action(Op::PLACE,kind);
   else if(step==4||step==24)a[0].units[0]=dp7::action(Op::PICKUP,dp7::W,1);
   else if(dp7::animal(t)&&!t.fed_today&&env.privates()[0].inventories[0][dp7::W]>0){
    auto x=env.project_unit_phase(0,{dp7::action(Op::CARE)}).project_unit_phase(0,{dp7::action(Op::FEED)});
    auto y=env.project_unit_phase(0,{dp7::action(Op::FEED)}).project_unit_phase(0,{dp7::action(Op::CARE)});
    require(x.farms()[0].tiles[44].cared_today==y.farms()[0].tiles[44].cared_today&&x.farms()[0].tiles[44].fed_today==y.farms()[0].tiles[44].fed_today&&x.privates()[0].inventories==y.privates()[0].inventories,"CARE/FEED commutation failed");cases++;
    while(x.day()==env.day()){std::array<PlayerAction,2>pass;x.step(pass);y.step(pass);}
    auto&tx=x.farms()[0].tiles[44];auto&ty=y.farms()[0].tiles[44];require(tx.pending_care_bonus==ty.pending_care_bonus&&tx.yield_units==ty.yield_units&&tx.consecutive_unfed==ty.consecutive_unfed,"CARE/FEED day-end differs");cases++;
    checked=true;a[0].units[0]=dp7::action(Op::FEED);
   }
   env.step(a);
  }
  require(checked,"animal commutation fixture never exercised");cases++;
 }
 {
  Simulator env(Config{},91530);auto own=env.farms()[0];auto priv=env.privates()[0];
  own.hands={{4,4}};priv.inventories.resize(2);priv.inventory_order.resize(2);priv.shed={};
  dp7::View v{718,29,22,own,env.farms()[1],priv,env.market(),env.shops()};
  auto cargo_route=[](int u,int qty,int item,int delay){dp7::Route r(u,44);dp7::Job j;j.pos=44;j.out[item]=qty;for(int i=0;i<delay;i++)j.actions.push_back(dp7::action(Op::PASS));r.append(j);return r;};
  std::vector<dp7::Route>same{cargo_route(0,60,dp7::S,0),cargo_route(1,60,dp7::MI,0)};
  auto last=dp7::Controller::terminal_preview(v,same,1);
  require(last.deposited[dp7::S]==60&&last.deposited[dp7::MI]==0&&last.stranded[dp7::MI]==60,"terminal forecast sold between simultaneous unit drops");cases++;
  auto two=dp7::Controller::terminal_preview(v,same,2);
  require(two.deposited[dp7::S]==60&&two.deposited[dp7::MI]==60,"terminal forecast did not free capacity after market");cases++;
  priv.shed[dp7::W]=90;
  auto stocked=dp7::Controller::terminal_preview(v,same,1);
  require(stocked.sold[dp7::W]==90&&dp7::sum(stocked.deposited)==0,"terminal forecast sold old shed before unit phase");cases++;
  priv.shed={};priv.shed[dp7::CO]=50;
  auto blocked=dp7::Controller::terminal_preview(v,same,2);
  require(dp7::sum(blocked.deposited)==0,"terminal forecast sold animals or erased occupied capacity");cases++;
  priv.shed={};std::vector<dp7::Route>large{cargo_route(0,101,dp7::S,0)};
  auto oversize=dp7::Controller::terminal_preview(v,large,4);
  require(oversize.stranded[dp7::S]==101&&oversize.deposited[dp7::S]==0,"whole-DROP guard incorrectly allowed oversized cargo");cases++;
  auto late=same;late[0]=cargo_route(0,60,dp7::S,1);late[1]=cargo_route(1,60,dp7::MI,1);
  auto expired=dp7::Controller::terminal_preview(v,late,1);
  require(dp7::sum(expired.deposited)==0,"terminal forecast executed after final available tick");cases++;
  std::vector<dp7::Route>empty{dp7::Route(0,44)};
  require(dp7::Controller::terminal_preview(v,empty,2).cash==0,"empty routes generated revenue");cases++;
  priv.shed[dp7::F]=5;dp7::Route supplied(0,44);dp7::Job j;j.pos=44;j.actions={dp7::action(Op::FERTILIZE)};j.needs[dp7::F]=2;j.out[dp7::S]=1;supplied.append(j);
  auto inputs=dp7::Controller::terminal_preview(v,{supplied},3);
  require(inputs.sold[dp7::F]==3&&inputs.sold[dp7::S]==1,"terminal forecast sold resources already picked up");cases++;
  priv.shed[dp7::W]=5;j.needs[dp7::W]=2;dp7::Route late_input(0,44);late_input.append(j);
  require(!std::isfinite(dp7::Controller::terminal_preview(v,{late_input},4).cash),"terminal forecast invented material after market liquidation");cases++;
 }
 {
  Simulator env(Config{},99310);auto own=env.farms()[0];auto priv=env.privates()[0];priv.shed={};
  dp7::View v{698,29,2,own,env.farms()[1],priv,env.market(),env.shops()};std::mt19937 rng(71713);int changed=0;
  for(int trial=0;trial<30;trial++){
   dp7::Controller c;c.day=29;auto starts=c.starts(10);std::vector<dp7::Job>jobs;
   for(int i=0;i<28;i++){dp7::Job j;j.pos=rng()%100;j.priority=-1;j.actions={dp7::action(Op::HARVEST)};j.out[i%2?dp7::S:dp7::MI]=4+rng()%8;jobs.push_back(j);}
   auto base=c.pack(jobs,starts,21,true);double cash=dp7::Controller::terminal_preview(v,base.first,21).cash;
   auto selected=c.terminal_schedule(v,jobs,starts,21,base);double value=dp7::Controller::terminal_preview(v,selected.first,21).cash;
   require(value>=cash,"terminal candidate selection reduced its explicit cash objective");cases++;
   for(auto&r:selected.first)require(r.total(true)<=21&&r.cost==r.ordered(r.jobs,false),"terminal candidate breaks route budget/accounting");cases++;
   int assigned=0;for(auto&r:selected.first)assigned+=r.jobs.size();require(assigned+selected.second==int(jobs.size()),"terminal candidate loses job accounting");cases++;
   changed+=c.terminal_schedule_switches;
  }
  require(changed>0,"terminal schedule never changes a congested candidate");cases++;
 }
 {
  Simulator env(Config{},23017);auto own=env.farms()[0];auto priv=env.privates()[0];
  own.farmer={4,4};own.hands={{4,4}};priv.inventories.resize(2);priv.inventory_order.resize(2);
  priv.inventories[0][dp7::S]=60;priv.inventories[1][dp7::MI]=60;
  dp7::View v{260,10,20,own,env.farms()[1],priv,env.market(),env.shops()};
  auto control=[](){dp7::Controller c;c.day=10;c.phase=3;c.plans.resize(2);return c;};
  auto disabled=control();disabled.dispatch_idle(v);
  require(disabled.plans[0].a.empty()&&disabled.overflow_dispatch_units==0,"disabled overflow dispatch changed route");cases++;
  auto c=control();c.p.overflow_idle_dispatch=true;
  require(c.expected_auto_deposit(v)==120,"idle cargo forecast wrong");cases++;
  c.dispatch_idle(v);
  require(c.overflow_dispatch_units==1&&c.overflow_dispatch_quantity==60&&c.plans[0].a[0].op==Op::DROP,"spare unit was not used for an overflow wave");cases++;
  c.dispatch_idle(v);require(c.overflow_dispatch_units==1,"already scheduled deposit counted twice");cases++;
  auto busy=control();busy.p.overflow_idle_dispatch=true;
  for(auto&plan:busy.plans){plan.a={dp7::action(Op::WATER)};plan.target={44};}
  busy.dispatch_idle(v);require(busy.overflow_dispatch_units==0&&busy.plans[0].a[0].op==Op::WATER,"dispatch overwrote live maintenance task");cases++;
  auto small=control();small.p.overflow_idle_dispatch=true;priv.inventories[0][dp7::S]=30;
  small.dispatch_idle(v);require(small.overflow_dispatch_units==0,"unneeded early sale/haul at low incoming volume");cases++;
  priv.inventories[0][dp7::S]=60;own.farmer={3,4};own.hands={{3,4}};
  auto late=control();late.p.overflow_idle_dispatch=true;
  dp7::View end{262,10,22,own,env.farms()[1],priv,env.market(),env.shops()};late.dispatch_idle(end);
  require(late.overflow_dispatch_units==0,"dispatch promised warehouse turnover after day end");cases++;
  own.farmer={4,4};late.dispatch_idle(end);require(late.overflow_dispatch_units==1,"last useful on-depot wave rejected");cases++;
  auto terminal=control();terminal.day=29;terminal.p.overflow_idle_dispatch=true;terminal.dispatch_idle(v);
  require(terminal.overflow_dispatch_units==0,"normal-day dispatch overwrote terminal scheduler");cases++;
  auto material=control();material.plans[0].a={dp7::action(Op::PICKUP,dp7::W,3),dp7::action(Op::FEED)};material.plans[0].target={44,44};
  require(material.expected_auto_deposit(v)==122,"remaining pickup/consumption forecast mismatch");cases++;
  auto&tile=own.tiles[44];tile.kind=TileKind::PLANT;tile.crop=Item::STRAWBERRY;tile.yield_units=4;
  auto harvest=control();for(auto&plan:harvest.plans){plan.a={dp7::action(Op::HARVEST)};plan.target={44};}
  require(harvest.expected_auto_deposit(v)==124,"same visible yield counted twice");cases++;
  auto beyond=control();beyond.plans[0].a={dp7::action(Op::PASS),dp7::action(Op::PASS),dp7::action(Op::PASS),dp7::action(Op::PASS),dp7::action(Op::HARVEST)};beyond.plans[0].target={-1,-1,-1,-1,44};
  require(beyond.expected_auto_deposit(v)==120,"forecast harvested beyond day end");cases++;
 }
 {
  dp7::Controller ctl;ctl.day=0;Market m;for(int i=0;i<9;i++)m.inventory[i]=10000;
  dp7::Controller::Calendar no{};dp7::Controller::CashSchedule slow,fast;
  slow.fixed[0]=-100;slow.fixed[3]=150;fast.fixed[0]=-100;fast.fixed[1]=150;
  auto s=ctl.marginal_cash(m,no,no,no,slow),f=ctl.marginal_cash(m,no,no,no,fast);
  require(s.project==50&&f.project==50,"turnover changed terminal project value");cases++;
  require(s.capital_days==300&&f.capital_days==100,"capital-days integral incorrect");cases++;
  require(ctl.turnover_rank(50,f.capital_days,100)>ctl.turnover_rank(50,s.capital_days,100),"early payback not preferred under scarce capital");cases++;
  require(ctl.turnover_rank(50,0,0)==50,"zero exposed capital penalized");cases++;
  require(std::abs(ctl.turnover_rank(50,300,1e15)-50)<1e-9,"ample capital failed to recover raw ranking");cases++;
  auto environment=Simulator(Config{},8129);auto&own=environment.farms()[0];auto&priv=environment.privates()[0];
  dp7::View v{0,0,0,own,environment.farms()[1],priv,environment.market(),environment.shops()};
  auto vs=ctl.values(v,{},{});auto scores=ctl.investment_ranks(v,vs,{},100);
  for(auto[value,pr]:vs){require(scores[pr.kind]==value,"disabled investment rank changed value");cases++;}
  dp7audit::InvestmentAudit audit;
  std::array<PlayerAction,2>actions;actions[0].units={dp7::action(Op::PASS)};actions[0].market={dp7::action(Op::BUY_SEED,dp7::W,1)};
  auto before=environment;environment.step(actions);audit.finish(before,environment,0,actions[0]);
  require(audit.events.size()==1&&audit.events[0].quantity==1,"audit missed actual purchase");cases++;
  actions[0].market.clear();actions[0].units={dp7::action(Op::PLANT,dp7::W)};
  auto projected=environment.project_unit_phase(0,actions[0].units);audit.unit(environment,projected,0,0,actions[0].units[0]);
  require(audit.cohorts.size()==1&&audit.cohorts[0].start_step==1,"audit missed actual planting cohort");cases++;
  require(audit.events.size()==2&&audit.events.back().op==int(Op::PLANT),"audit planting event wrong");cases++;
  auto planning_env=Simulator(Config{},99117);dp7::Controller incoming;auto before_target=incoming.target;
  dp7::View opening{0,0,0,planning_env.farms()[0],planning_env.farms()[1],planning_env.privates()[0],planning_env.market(),planning_env.shops()};
  auto candidates=dp7branch::generate(incoming,opening);std::set<int>rebuilt;
  for(auto&c:candidates)if(c.family=="REBUILD_NEW")rebuilt.insert(c.kind);
  require(candidates.size()>20&&candidates[0].family=="KEEP","investment candidate pool missing KEEP/diversity");cases++;
  require(rebuilt==std::set<int>({0,1,2,3,4,9,10,11}),"candidate pool omitted an industry");cases++;
  require(incoming.target==before_target&&incoming.day==-1,"candidate generator mutated live controller");cases++;
  for(auto&c:candidates)for(auto[pos,k]:c.controller.target)require(pos>=0&&pos<100&&k>=-1&&k<12,"candidate target outside schema");cases++;
 }
 // S4A-1: rolling repair must discard effects already present in the observed
 // state, rebuild movement from the actual position, and preserve dependent
 // actions after a still-required prefix.
 {
  Simulator env(Config{},77104);auto own=env.farms()[0];auto priv=env.privates()[0];own.farmer={3,4};priv.inventories.resize(1);priv.inventory_order.resize(1);
  auto&w=own.tiles[44];w.kind=TileKind::PLANT;w.crop=Item::WHEAT;w.watered_today=true;
  auto&h=own.tiles[45];h.kind=TileKind::PLANT;h.crop=Item::STRAWBERRY;h.yield_units=4;
  dp7::View v{120,5,0,own,env.farms()[1],priv,env.market(),env.shops()};dp7::Controller c;c.day=5;c.phase=3;c.plans.resize(1);
  c.plans[0].a={dp7::action(Op::EAST),dp7::action(Op::WATER),dp7::action(Op::EAST),dp7::action(Op::HARVEST),dp7::action(Op::WEST),dp7::action(Op::DROP)};c.plans[0].target={-1,44,-1,45,-1,44};
  int stale=0;auto repaired=c.repair_plan(v,0,c.plans[0],stale);
  require(stale==1&&repaired.a.size()==4&&repaired.a[0].op==Op::EAST&&repaired.a[1].op==Op::EAST&&repaired.a[2].op==Op::HARVEST&&repaired.a[3].op==Op::DROP,"rolling repair followed stale coordinates or lost future cargo deposit");cases++;
  w.kind=TileKind::WEED;w.watered_today=false;c.plans[0].a={dp7::action(Op::DIG),dp7::action(Op::PLANT,dp7::W),dp7::action(Op::WATER)};c.plans[0].target={44,44,44};
  stale=0;auto chain=c.repair_plan(v,0,c.plans[0],stale);require(stale==0&&chain.a.size()==4&&chain.a[0].op==Op::EAST&&chain.a[1].op==Op::DIG&&chain.a[2].op==Op::PLANT&&chain.a[3].op==Op::WATER,"rolling repair broke a causal plant chain");cases++;
 }
 // A still-valid local assignment is retained, while uncovered independent
 // service is assigned to another worker without duplicating the first task.
 {
  Simulator env(Config{},77105);auto own=env.farms()[0];auto priv=env.privates()[0];own.farmer={4,4};own.hands={{4,4}};priv.inventories.resize(2);priv.inventory_order.resize(2);
  for(int pos:{44,45}){auto&t=own.tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item::SHEEP;t.placed_day=0;t.fed_today=true;t.cared_today=false;}
  dp7::Params p;p.shared_task_atoms_v2=true;p.stepwise_recoordination=true;dp7::Controller c(p);c.day=5;c.phase=3;c.target={{44,dp7::SH},{45,dp7::SH}};c.plans.resize(2);c.plans[0].a={dp7::action(Op::CARE)};c.plans[0].target={44};
  dp7::View v{120,5,0,own,env.farms()[1],priv,env.market(),env.shops()};c.recoordinate(v);
  int care44=0,care45=0;for(auto&pl:c.plans)for(size_t k=0;k<pl.a.size();k++)if(pl.a[k].op==Op::CARE){care44+=pl.target[k]==44;care45+=pl.target[k]==45;}
  require(care44==1&&care45==1&&c.step_recoord_checks==1&&c.step_recoord_rebuilds==1&&c.step_recoord_added_groups==2&&c.step_recoord_reassigned_groups==2,"rolling coordination did not preserve/add unique service work");cases++;
  auto frozen=c;frozen.p.stepwise_recoordination=false;auto before=frozen.plans;frozen.recoordinate(v);require(dp7::Controller::same_remaining(frozen.plans[0],before[0])&&frozen.step_recoord_checks==c.step_recoord_checks,"disabled rolling coordination changed plans");cases++;
 }
 {
  Simulator env(Config{},77106);dp7::Params p;p.stepwise_recoordination=true;dp7::Controller dirty(p),fresh(p);dirty.day=12;dirty.phase=3;dirty.last_step=400;dirty.plans.resize(1);dirty.step_recoord_checks=99;dirty.step_recoord_rebuilds=77;dirty.step_recoord_stale_groups=55;dirty.step_recoord_added_groups=33;
  dp7::View v{0,0,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};auto a=dirty.act(v),b=fresh.act(v);
  require(dirty.step_recoord_checks==0&&dirty.step_recoord_rebuilds==0&&dirty.step_recoord_stale_groups==0&&dirty.step_recoord_added_groups==0,"rolling coordination state leaked across games");cases++;
  require(a.market.size()==b.market.size()&&dirty.target==fresh.target,"rolling reset changed fresh opening");cases++;
 }
 // S4A-3: procurement may overlap with real, already available maintenance,
 // including the first movement toward it. It must never consume projected
 // purchases or assign one service atom to two units.
 {
  Simulator env(Config{},77107);auto own=env.farms()[0];auto priv=env.privates()[0];own.money=1000;own.farmer={4,4};own.hands={{3,4}};priv.inventories.resize(2);priv.inventory_order.resize(2);
  auto&crop=own.tiles[46];crop.kind=TileKind::PLANT;crop.crop=Item::TOMATO;crop.planted_day=2;crop.watered_today=false;
  auto&sheep=own.tiles[45];sheep.kind=TileKind::ANIMAL;sheep.animal=Item::SHEEP;sheep.placed_day=0;sheep.fed_today=true;sheep.cared_today=false;
  dp7::Params p;p.preparation_pipeline_v2=true;dp7::Controller ctl(p);ctl.day=3;ctl.phase=1;ctl.target={{45,dp7::SH},{46,dp7::T}};
  dp7::View v{72,3,0,own,env.farms()[1],priv,env.market(),env.shops()};auto prep=ctl.preparation_pipeline(v);
  int active=(prep[0].op!=Op::PASS)+(prep[1].op!=Op::PASS);
  require(prep.size()==2&&active==1&&prep[0].op==Op::EAST,"preparation pipeline did not follow a unique projected schedule prefix");cases++;
  // A new crop project depends on a seed that is not yet owned; projected
  // market fill must not authorize PLANT or movement toward that project.
  ctl.target={{47,dp7::M}};auto no_future=ctl.preparation_pipeline(v);
  require(no_future[0].op==Op::PASS&&no_future[1].op==Op::PASS,"preparation pipeline spent a projected purchase");cases++;
 }
 {
  Simulator env(Config{},77108);auto own=env.farms()[0];auto priv=env.privates()[0];own.money=1000;own.farmer={4,4};priv.inventories.resize(1);priv.inventory_order.resize(1);
  auto&crop=own.tiles[46];crop.kind=TileKind::PLANT;crop.crop=Item::TOMATO;crop.planted_day=2;crop.watered_today=false;
  dp7::Params p;p.preparation_pipeline_v2=true;dp7::Controller ctl(p);ctl.day=3;ctl.phase=1;ctl.target={{46,dp7::T}};ctl.queue={dp7::action(Op::BUY_SEED,dp7::M,1)};
  dp7::View v{72,3,0,own,env.farms()[1],priv,env.market(),env.shops()};auto out=ctl.act(v);
  require(out.market.size()==1&&out.market[0].op==Op::BUY_SEED&&out.units.size()==1&&out.units[0].op==Op::EAST,"market procurement and existing production did not overlap");cases++;
  std::array<PlayerAction,2>aa;aa[0]=out;env.step(aa);
  require(env.farms()[0].farmer.x==5&&env.farms()[0].farmer.y==4&&env.privates()[0].seeds[dp7::M]==1,"simultaneous movement/purchase official settlement mismatch");cases++;
  require(ctl.preparation_pipeline_checks==1&&ctl.preparation_pipeline_actions==1&&ctl.preparation_pipeline_moves==1,"preparation pipeline behavior counters mismatch");cases++;
  auto disabled=dp7::Controller(p);disabled.p.preparation_pipeline_v2=false;disabled.day=3;disabled.phase=1;disabled.target={{46,dp7::T}};disabled.queue={dp7::action(Op::BUY_SEED,dp7::M,1)};auto base=disabled.act(v);
  require(base.units.size()==1&&base.units[0].op==Op::PASS&&base.market.size()==1,"disabled preparation pipeline changed baseline market or unit action");cases++;
  // Rewinding the public step creates a fresh episode and must clear all
  // pipeline-local counters and assignments.
  auto fresh=dp7::Controller(p);auto reset=ctl.act(v),clean=fresh.act(v);
  require(ctl.preparation_pipeline_checks==fresh.preparation_pipeline_checks&&ctl.preparation_pipeline_actions==fresh.preparation_pipeline_actions&&ctl.preparation_pipeline_moves==fresh.preparation_pipeline_moves,"preparation pipeline counters leaked across episodes");cases++;
  require(reset.market.size()==clean.market.size()&&reset.units.size()==clean.units.size(),"preparation pipeline reset changed fresh opening shape");cases++;
 }
 // S4A-4: when capacity cannot cover every job, choose the schedule that
 // preserves more realized business value rather than the shorter/earlier
 // map coordinate alone.
 {
  Simulator env(Config{},77109);auto own=env.farms()[0];auto priv=env.privates()[0];priv.inventories.resize(1);priv.inventory_order.resize(1);
  dp7::View v{120,5,0,own,env.farms()[1],priv,env.market(),env.shops()};dp7::Controller ctl;ctl.day=5;
  dp7::Job low;low.pos=44;low.priority=0;low.actions={dp7::action(Op::HARVEST)};low.out[dp7::W]=1;
  dp7::Job high;high.pos=45;high.priority=0;high.actions={dp7::action(Op::HARVEST)};high.out[dp7::MI]=10;
  std::vector<dp7::Job>jobs{low,high};std::vector<int>starts{44};auto baseline=ctl.pack(jobs,starts,2,false);
  require(baseline.second==1&&baseline.first[0].jobs.size()==1&&baseline.first[0].jobs[0].pos==44,"value-schedule fixture baseline did not expose distance bias");cases++;
  auto chosen=ctl.value_schedule(v,jobs,starts,2,false,std::move(baseline));
  require(chosen.second==1&&chosen.first[0].jobs.size()==1&&chosen.first[0].jobs[0].pos==45,"value schedule failed to retain higher-cash work");cases++;
  require(ctl.schedule_value_evaluations==4&&ctl.schedule_value_switches==1&&ctl.schedule_value_gain>0,"value schedule evidence counters mismatch");cases++;
  auto full=ctl.pack(jobs,starts,4,false);auto altered=full;
  for(auto&r:altered.first)for(auto&j:r.jobs)j.priority=-999999;
  auto original_value=ctl.schedule_metric(v,jobs,full,4,false);
  auto altered_value=ctl.schedule_metric(v,jobs,altered,4,false);
  require(original_value.value==altered_value.value,"sorting metadata inflated economic schedule value");cases++;
  dp7::Controller stable;stable.day=5;auto kept=stable.value_schedule(v,jobs,starts,4,false,full);
  require(stable.schedule_value_switches==0&&kept.first[0].jobs[0].pos==full.first[0].jobs[0].pos,"equal completed business effects replaced the stable schedule");cases++;
  auto bogus=full;bogus.first[0].jobs[0].out[dp7::MI]++;
  bool rejected=false;try{ctl.schedule_metric(v,jobs,bogus,4,false);}catch(const std::logic_error&){rejected=true;}
  require(rejected,"schedule evaluator accepted invented task output");cases++;
 }
 // S4B: future crop supply may reduce surplus stock, never today's feed or
 // an earlier prefix deficit. No unplanted target or rival stock is credited.
 {
  Simulator env(Config{},77110);auto own=env.farms()[0];auto priv=env.privates()[0];
  dp7::View v{120,5,0,own,env.farms()[1],priv,env.market(),env.shops()};dp7::Controller ctl;ctl.day=5;
  require(ctl.feed_buffer_goal(v,5,3)==15,"disabled net buffer changed legacy target");cases++;
  ctl.p.net_feed_buffer=true;require(ctl.feed_buffer_goal(v,5,3)==15,"net buffer invented future wheat");cases++;
  auto&t=own.tiles[44];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=1;t.yield_units=4;
  require(ctl.feed_buffer_goal(v,5,3)==11,"mature wheat was not credited after its deposit day");cases++;
  require(ctl.feed_buffer_goal(v,5,1)==5,"future wheat erased today's required feed");cases++;
  priv.shed[dp7::W]=50;require(ctl.feed_buffer_goal(v,5,3)==11,"held stock was double counted as a future arrival");cases++;
  t.planted_day=5;t.yield_units=0;require(ctl.feed_buffer_goal(v,5,3)==15,"out-of-window harvest reduced current reserve");cases++;
  for(int pos:{0,1,2,3,4,10,11,12,13,14}){auto&x=own.tiles[pos];x.kind=TileKind::PLANT;x.crop=Item::WHEAT;x.planted_day=5;x.yield_units=0;}
  require(ctl.feed_buffer_goal(v,5,7)==25,"late harvest incorrectly paid an earlier prefix deficit");cases++;
  ctl.p.net_feed_buffer=false;require(ctl.feed_buffer_goal(v,5,7)==35,"disabled net buffer consumed future credit");cases++;
  ctl.p.net_feed_buffer=true;ctl.day=28;
  dp7::View end{672,28,0,own,env.farms()[1],priv,env.market(),env.shops()};require(ctl.feed_buffer_goal(end,5,1)==5,"terminal next-day crop paid preterminal feed");cases++;
 }
 {
  Simulator env(Config{},77111);auto own=env.farms()[0];auto priv=env.privates()[0];
  dp7::View v{0,0,0,own,env.farms()[1],priv,env.market(),env.shops()};dp7::Controller ctl;ctl.queue={dp7::action(Op::HIRE)};
  require(ctl.preserves_hire_projection(v,0,dp7::action(Op::EAST)),"disabled spawn guard changed preparation");cases++;
  ctl.p.preparation_spawn_guard=true;
  require(!ctl.preserves_hire_projection(v,0,dp7::action(Op::EAST)),"depot movement retained stale HIRE projection");cases++;
  require(ctl.preserves_hire_projection(v,0,dp7::action(Op::BUILD_PASTURE)),"spawn guard blocked nonmoving production");cases++;
  own.farmer={3,4};require(!ctl.preserves_hire_projection(v,0,dp7::action(Op::EAST)),"entry into depot retained stale HIRE projection");cases++;
  require(ctl.preserves_hire_projection(v,0,dp7::action(Op::WEST)),"spawn guard blocked unrelated off-depot movement");cases++;
  ctl.queue.clear();require(ctl.preserves_hire_projection(v,0,dp7::action(Op::EAST)),"spawn guard blocked movement without any future HIRE");cases++;
  auto stayed=env,moved=env;std::array<PlayerAction,2>a;
  a[0].units={dp7::action(Op::PASS)};a[0].market={dp7::action(Op::HIRE)};stayed.step(a);
  a[0].units={dp7::action(Op::EAST)};moved.step(a);
  require(dp7::cell(stayed.farms()[0].hands[0])==45&&dp7::cell(moved.farms()[0].hands[0])==44,"official HIRE did not expose the movement/spawn coupling");cases++;
 }
 {
  Simulator env(Config{},77112);auto own=env.farms()[0];auto priv=env.privates()[0];
  own.farmer={4,3};own.hands={{4,4}};priv.inventories.resize(2);priv.inventory_order.resize(2);
  priv.inventories[0][dp7::MI]=60;priv.inventories[1][dp7::WO]=50;
  auto&t=own.tiles[33];t.kind=TileKind::PLANT;t.crop=Item::TOMATO;t.planted_day=1;
  dp7::View v{135,5,15,own,env.farms()[1],priv,env.market(),env.shops()};
  dp7::Controller ctl;ctl.day=5;ctl.phase=3;ctl.plans.resize(2);
  ctl.plans[0].a={dp7::action(Op::WEST),dp7::action(Op::WATER)};ctl.plans[0].target={-1,33};
  auto baseline=ctl.plans;ctl.dispatch_midroute(v);
  require(dp7::Controller::same_remaining(ctl.plans[0],baseline[0]),"disabled midroute delivery changed tasks");cases++;
  ctl.p.midroute_delivery=true;ctl.dispatch_midroute(v);
  auto&route=ctl.plans[0];int drop=0,water=0;for(auto a:route.a){drop+=a.op==Op::DROP;water+=a.op==Op::WATER;}
  require(drop==1&&water==1&&route.a.back().op==Op::WATER&&route.target.back()==33,"midroute visit lost original production");cases++;
  require(ctl.midroute_delivery_insertions==1&&ctl.midroute_delivery_quantity==60&&v.hour+int(route.a.size())+1<=24,"midroute visit exceeded time or accounting");cases++;
  ctl.dispatch_midroute(v);require(ctl.midroute_delivery_insertions==1,"midroute delivery inserted duplicate DROP");cases++;
  ctl.plans=baseline;priv.inventories[0][dp7::W]=1;ctl.plans[0].a.push_back(dp7::action(Op::FEED));ctl.plans[0].target.push_back(33);
  ctl.dispatch_midroute(v);require(ctl.midroute_delivery_insertions==1,"midroute delivery discarded a future feed commitment");cases++;
  ctl.plans=baseline;priv.inventories[0][dp7::W]=0;priv.inventories[1][dp7::WO]=1;
  ctl.dispatch_midroute(v);require(ctl.midroute_delivery_insertions==1,"midroute delivery intervened without capacity pressure");cases++;
  priv.inventories[1][dp7::WO]=50;dp7::View late{142,5,22,own,env.farms()[1],priv,env.market(),env.shops()};
  ctl.dispatch_midroute(late);require(ctl.midroute_delivery_insertions==1,"midroute delivery ignored end-of-day timing");cases++;
  priv.inventories[0][dp7::SH]=1;ctl.dispatch_midroute(v);
  require(ctl.midroute_delivery_insertions==1,"midroute delivery unloaded unplaced animal capital");cases++;
  priv.inventories[0][dp7::SH]=0;t.crop=Item::WHEAT;t.planted_day=1;t.yield_units=3;t.max_lifespan_step=144;
  ctl.plans=baseline;ctl.plans[0].a.back()=dp7::action(Op::HARVEST);
  ctl.day=6;dp7::View decaying{147,6,3,own,env.farms()[1],priv,env.market(),env.shops()};
  ctl.dispatch_midroute(decaying);require(ctl.midroute_delivery_insertions==1,"midroute detour crossed a within-day crop decay deadline");cases++;
 }
 // S4F: declared future work is a conditional forecast, not current stock.
 {
  using namespace dp7;
  Simulator env(Config{},77201);auto own=env.farms()[0];auto priv=env.privates()[0];
  own.farmer={4,4};own.hands={{3,4},{4,3}};priv.inventories.resize(3);priv.inventory_order.resize(3);
  View v{72,3,0,own,env.farms()[1],priv,env.market(),env.shops()};Controller c;c.day=3;c.phase=3;c.plans.resize(3);
  c.plans[0].a={action(Op::PLANT,S),action(Op::WATER)};c.plans[0].target={44,44};
  c.plans[1].a={action(Op::PLANT,W),action(Op::WATER)};c.plans[1].target={43,43};c.plans[1].index=1;
  c.plans[2].a={action(Op::PLACE,CO)};c.plans[2].target={34};
  PlayerAction a;a.units={action(Op::PASS),action(Op::PLANT,W),action(Op::PASS)};
  auto d=intraday::declared(c,v,a);
  require(d.planned[W]==1&&d.planned[S]==1&&d.planned[CO]==1&&sum(d.planned)==3,"declarations lost current/unexecuted work");cases++;
  require(d.selected[S]==c.crop_project(S).out&&d.selected[MI]==c.animal_project(CO).out,"declaration production forecast mismatch");cases++;
  require(d.feed==c.animal_project(CO).feed,"declared feed obligation missing");cases++;
  require(sum(priv.shed)==0&&std::accumulate(priv.seeds.begin(),priv.seeds.end(),0)==0&&sum(priv.inventories[0])==0,"conditional production became owned inventory");cases++;
  c.plans[2].a.push_back(action(Op::PLANT,T));c.plans[2].target.push_back(44);auto dupe=intraday::declared(c,v,a);
  require(dupe.duplicates==1&&dupe.planned[T]==0&&dupe.planned[S]==1,"declaration double counted shared tile");cases++;
  View late{95,3,23,own,env.farms()[1],priv,env.market(),env.shops()};auto end=intraday::declared(c,late,a);
  require(sum(end.planned)==1&&end.planned[W]==1&&end.late>0,"declaration credited impossible next-day task");cases++;
  own.tiles[34].kind=TileKind::ANIMAL;own.tiles[34].animal=Item::COW;auto existing=intraday::declared(c,v,a);
  require(existing.planned[CO]==0,"declaration double counted existing animal");cases++;
  c.plans[0].index=c.plans[0].a.size();c.plans[1].index=c.plans[1].a.size();c.plans[2].index=c.plans[2].a.size();a.units.assign(3,action(Op::PASS));
  auto finished=intraday::declared(c,v,a);require(sum(finished.planned)==0,"completed tasks remained future commitments");cases++;
  auto sample=intraday::inspect(c,v,a,{});require(!sample.seen,"empty declarations produced a fake decision sample");cases++;
 }
 {
  using namespace dp7;Simulator env(Config{},77202);auto own=env.farms()[0];auto priv=env.privates()[0];own.money=10000;
  own.farmer={4,4};own.hands={{4,3}};priv.inventories.resize(2);priv.inventory_order.resize(2);priv.seeds.fill(10);priv.shed[F]=10;
  View v{130,5,10,own,env.farms()[1],priv,env.market(),env.shops()};Params p;p.intraday_admission=true;p.intraday_procurement=true;Controller c(p);c.day=5;c.phase=3;c.plans.resize(2);c.target={{33,S}};
  c.plans[1].a={action(Op::PLANT,S),action(Op::WATER)};c.plans[1].target={33,33};PlayerAction a;a.units={action(Op::PASS),action(Op::WEST)};
  auto base=intraday::propose(c,v,a);auto known=intraday::declared(c,v,a);auto expected=intraday::propose(c,v,a,&known);
  c.p.intraday_declared_value=true;auto enabled=intraday::propose(c,v,a);
  require(base.valid&&expected.valid&&enabled.valid,"declared-value switch fixture lacks a feasible project");cases++;
  require(std::tuple(enabled.kind,enabled.pos,enabled.unit,enabled.value)==std::tuple(expected.kind,expected.pos,expected.unit,expected.value),"declared-value switch failed to use declared production");cases++;
  c.p.intraday_declared_value=false;auto disabled=intraday::propose(c,v,a);
  require(std::tuple(disabled.kind,disabled.pos,disabled.unit,disabled.value)==std::tuple(base.kind,base.pos,base.unit,base.value),"disabled declared-value switch changed baseline");cases++;
 }
 // S4G: the startup flag changes the source of targets, never the rules.
 {
  using namespace dp7;Simulator env(Config{},77301);auto own=env.farms()[0];auto priv=env.privates()[0];
  View v{0,0,0,own,env.farms()[1],priv,env.market(),env.shops()};Params p;
  p.opening_animals={SH};p.opening_crops={1,0,0,0,0};Controller fixed(p);fixed.day=0;fixed.choose(v);
  auto kinds=[](const Controller& c){Counts q{};for(auto[pos,k]:c.target)if(k>=0)q[k]++;return q;};
  require(kinds(fixed)[SH]==1&&kinds(fixed)[W]==1&&sum(kinds(fixed))==2,"disabled startup changed supplied recipe");cases++;
  p.autonomous_start=true;Controller automatic(p);automatic.day=0;automatic.choose(v);auto plan=automatic.target;
  require(plan!=fixed.target,"autonomous startup still read the fixed opening");cases++;
  p.opening_animals={G,G,G,G};p.opening_crops={0,1,5,9,15};Controller alternate(p);alternate.day=0;alternate.choose(v);
  require(alternate.target==plan,"autonomous targets depend on ignored opening parameters");cases++;
  double invested=0;for(auto[pos,k]:automatic.target)if(k>=0)invested+=k>=9?animal_price[k-9]:seed_price[k];
  require(invested>0&&invested<=automatic.investment_budget(v,automatic.planned_land,0),"autonomous opening exceeds declared capital budget");cases++;
  auto qs=kinds(automatic);require(qs[G]<=p.max_geese&&qs[CO]<=p.max_cows&&qs[SH]<=p.max_sheep&&qs[G]+qs[CO]+qs[SH]<=p.max_animals,"startup bypassed animal caps");cases++;
  auto saved_money=own.money;own.money=0;alternate.choose(v);
  require(sum(kinds(alternate))==0,"startup spent imaginary future sales");cases++;
  own.money=saved_money;p.max_animals=p.max_cows=p.max_sheep=p.max_geese=0;
  Controller crops(p);crops.day=0;crops.choose(v);qs=kinds(crops);
  require(qs[G]+qs[CO]+qs[SH]==0&&sum(qs)>0,"startup cannot adapt to available industry constraints");cases++;
  automatic.p.autonomous_start=false;automatic.day=1;alternate=automatic;alternate.p.autonomous_start=true;
  View next{24,1,0,own,env.farms()[1],priv,env.market(),env.shops()};automatic.choose(next);alternate.choose(next);
  require(automatic.target==alternate.target,"startup flag unexpectedly changed the daily selector");cases++;
 }
 // S4I: small exhaustive resource problems and conditional accounting.
 {
  using namespace dp7;using namespace dp7::portfolio;
  std::array<std::vector<double>,8>curve;for(auto&v:curve)v={0.};
  std::array<int,8>cost{};cost.fill(50);cost[0]=100;
  curve[0]={0,110};curve[1]={0,80,160};
  auto result=generate(curve,cost,100,2,0,128);
  require(result.front().mix[0]==0&&result.front().mix[1]==2&&result.front().score==160,"portfolio kept expensive greedy singleton");cases++;
  require(generate(curve,cost,100,1,0,128).front().mix[0]==1,"portfolio ignored scarce land");cases++;
  require(generate(curve,cost,0,2,0,128).size()==1,"portfolio spent imaginary capital");cases++;
  require(generate(curve,cost,100,0,0,128).front().plots==0,"portfolio ignored zero capacity");cases++;
  for(int a=0;a<8;a++){
   auto only=curve;for(auto&v:only)v={0.};only[a]={0,10,12};
   require(generate(only,cost,200,2,2,16).front().mix[a]==2,"portfolio generation lost an industry");cases++;
  }
  for(int money=0;money<=150;money+=10)for(int plots=0;plots<=3;plots++){
   auto cs=curve;cs[2]={0,55,100,130};cost[2]=30;
   double best=0;std::set<Mix>expected;
   for(int a=0;a<=1;a++)for(int b=0;b<=2;b++)for(int d=0;d<=3;d++)if(a+b+d<=plots&&100*a+50*b+30*d<=money){
    best=std::max(best,cs[0][a]+cs[1][b]+cs[2][d]);Mix m{};m[0]=a;m[1]=b;m[2]=d;expected.insert(m);
   }
   auto exact=generate(cs,cost,money,plots,0,128);std::set<Mix>actual;for(auto&r:exact)actual.insert(r.mix);
   require(exact.front().score==best&&actual==expected,"wide small portfolio search differs from exhaustive ground truth");cases++;
  }
  Simulator env(Config{},77401);auto own=env.farms()[0];auto priv=env.privates()[0];
  View v{0,0,0,own,env.farms()[1],priv,env.market(),env.shops()};Controller c;c.day=0;c.p.timing_discount=1.;
  Controller::PortfolioContext context;Controller::CashSchedule empty,produced,consumed;
  require(cash_delta(c,v,context,empty)==0,"empty portfolio has value");cases++;
  produced.quantity[5][W]=4;consumed.quantity[5][W]=-4;
  std::array<Controller::CashSchedule,8>cal{};cal[0]=produced;cal[5]=consumed;Mix mix{};mix[0]=mix[5]=1;
  require(cash_delta(c,v,context,sum_calendar(cal,mix))==0,"own feed was bought and sold twice in aggregate valuation");cases++;
  auto one=cash_delta(c,v,context,produced);mix[0]=2;mix[5]=0;
  require(cash_delta(c,v,context,sum_calendar(cal,mix))<=2*one,"aggregate price impact ignored additional own supply");cases++;
  auto wheat=delivery(c,W,0),strawberry=delivery(c,S,0),cow=delivery(c,CO,0);
  require(wheat.cash.quantity[c.h_age(W)][W]==0&&wheat.cash.quantity[c.h_age(W)+1][W]>0,"harvest credited before auto-deposit");cases++;
  require(strawberry.cash.quantity[first[S]][S]==0&&strawberry.cash.quantity[first[S]+interval[S]+1][S]==4,"ongoing yield ignored collection threshold");cases++;
  require(cow.cash.quantity[0][F]==0&&cow.cash.quantity[2][F]==1&&cow.cash.quantity[0][W]==-1,"animal maintenance delivery timing mismatch");cases++;
  require(portfolio::future_wages(c,v,{})==0,"empty future work requires hired workers");cases++;
  own.tiles[44].kind=TileKind::ANIMAL;own.tiles[44].animal=Item::COW;own.tiles[44].placed_day=0;
  c.p.own_feed_demand_weight=0;auto physical=physical_context(c,v);
  require(physical.own[1][W]==-1&&physical.own[1][F]==1,"portfolio inherited missing existing-animal feed");cases++;
  c.p.own_feed_demand_weight=1;auto explicit_feed=physical_context(c,v);
  require(physical.own==explicit_feed.own,"physical portfolio maintenance depends on historical feed weight");cases++;
  own.tiles[44]=Tile{};
  Params p;p.autonomous_start=true;p.joint_investment_portfolio=true;p.operating_reserve=100;
  p.max_animals=p.max_geese=p.max_cows=p.max_sheep=0;Controller active(p);active.new_day(v);
  require(active.portfolio_evaluated>1&&active.portfolio_generated>0,"joint switch did not compare portfolios");cases++;
  require(active.preview_bundle(v).started==active.preview_bundle(v).proposed,"new portfolio chose unstarted resources");cases++;
  Controller disabled(p);disabled.p.joint_investment_portfolio=false;disabled.new_day(v);
  require(disabled.portfolio_generated==0&&disabled.portfolio_evaluated==0,"disabled joint switch performed hidden search");cases++;
  auto saved=active.target;active.new_day(v);require(active.target==saved,"joint planning is stateful across identical observations");cases++;
  own.money=0;Controller poor(p);poor.new_day(v);
  require(poor.portfolio_switches==0,"zero-cash portfolio bought unsupported resources");cases++;
 }
 {
  using namespace dp7;Simulator env(Config{},77402);
  View v{0,0,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  Params p;p.operating_reserve=100;p.opening_animals={SH};p.opening_crops={1,0,0,0,0};
  Controller old(p);old.new_day(v);p.joint_investment_portfolio=true;Controller later(p);later.new_day(v);
  require(old.target==later.target&&old.queue.size()==later.queue.size(),"portfolio replaced opening while autonomous_start was off");cases++;
  for(size_t i=0;i<old.queue.size();i++){require(Controller::same_action(old.queue[i],later.queue[i]),"startup market queue changed through independent switch");cases++;}
  require(later.portfolio_evaluated==0,"disabled startup searched hidden alternatives");cases++;
  View next{24,1,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};later.new_day(next);
  require(later.portfolio_evaluated>1,"startup guard disabled later-day economic planning");cases++;
  p.autonomous_start=true;Controller automatic(p);automatic.new_day(v);
  require(automatic.portfolio_evaluated>1,"autonomous start cannot compare opening portfolios");cases++;
 }
 std::cout<<"{\"status\":\"PASS\",\"native_mechanism_checks\":"<<cases<<"}\n";
}
