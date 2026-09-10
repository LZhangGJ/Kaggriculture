#include "../policy/triad.hpp"
#include <iostream>
#include <stdexcept>
using namespace dp7;
static int checks=0;
void require(bool b,const char*s){checks++;if(!b)throw std::runtime_error(s);}
struct Fixture{
 Farm f,r;PrivateState p;Market m;std::vector<int8_t> shops;int day=5,hour=1;
 Fixture(){f.tiles.resize(100);r.tiles.resize(100);f.farmer={4,4};r.farmer={4,4};f.unlocked_mask=r.unlocked_mask=15;p.inventories.resize(1);p.inventory_order.resize(1);for(int i=0;i<9;i++)m.prices[i]=price(i,0);}
 View view()const{return {day*24+hour,day,hour,f,r,p,m,shops};}
 void wheat(int pos,int age=1,int q=1){auto&t=f.tiles[pos];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=day-age;t.yield_units=q;t.consecutive_unwatered=1;}
 Controller ctl(){Controller c;c.day=day;c.phase=3;c.target={{45,W}};c.plans.resize(p.inventories.size());c.p.efficient_water=true;c.plant_not_before[45]=day+1;return c;}
 void bag(int u,int item,int q){p.inventories[u][item]=q;if(q)p.inventory_order[u].push_back(item);}
};
int main(){try{
 {Fixture x;x.wheat(45);auto c=x.ctl();auto v=x.view();auto before=x.f.tiles[45];t3repair::obligations(c,v);
  require(c.t3.water_insertions==1,"missing water assigned");require(t3repair::covered_water(c.plans,45,x.hour),"water is in route");require(c.plans[0].a.size()==2,"move and water both budgeted");require(x.f.tiles[45].watered_today==before.watered_today,"observation not mutated");
  t3repair::obligations(c,v);require(c.t3.water_insertions==1,"same tick idempotence");x.hour++;t3repair::obligations(c,x.view());require(c.t3.water_insertions==1,"no duplicate assignment");}
 {Fixture x;x.wheat(45);x.hour=23;auto c=x.ctl();t3repair::obligations(c,x.view());require(c.t3.water_insertions==0,"reject route beyond midnight");require(c.t3.water_unresolved==1,"record unsatisfied obligation");}
 {Fixture x;x.wheat(45);x.f.tiles[45].watered_today=true;auto c=x.ctl();t3repair::obligations(c,x.view());require(c.plans[0].a.empty(),"skip confirmed watered state");}
 {Fixture x;x.wheat(45);x.day=29;auto c=x.ctl();t3repair::obligations(c,x.view());require(c.plans[0].a.empty(),"terminal layer not disturbed");}
 {Fixture x;x.wheat(45);x.bag(0,F,1);auto c=x.ctl();int pos=44;Controller::walk(c.plans[0],pos,45);c.plans[0].a.push_back(action(Op::FERTILIZE));c.plans[0].target.push_back(45);t3repair::obligations(c,x.view());
  require(c.t3.water_insertions==1,"insert after fertilizer");auto sem=Controller::plan_semantics(c.plans[0]);require(sem[0].first.op==Op::FERTILIZE&&sem[1].first.op==Op::WATER,"fertilize-water causal order");require(x.p.inventories[0][F]==1,"forecast does not spend actual fertilizer");}
 {Fixture x;x.wheat(45,2,2);auto c=x.ctl();int pos=44;Controller::walk(c.plans[0],pos,45);c.plans[0].a.push_back(action(Op::HARVEST));c.plans[0].target.push_back(45);t3repair::obligations(c,x.view());auto sem=Controller::plan_semantics(c.plans[0]);require(sem.size()==2&&sem[0].first.op==Op::WATER&&sem[1].first.op==Op::HARVEST,"water precedes finite harvest");}
 {Fixture x;x.wheat(45);auto c=x.ctl();c.t3.suspended=true;t3repair::obligations(c,x.view());t3repair::capacity(c,x.view());require(c.t3.obligation_checks==0&&c.t3.capacity_checks==0,"conditional probes do not recursively optimize");}
 {Fixture x;x.day=27;x.hour=23;x.f.farmer={0,0};x.bag(0,E,101);auto c=x.ctl();c.target.clear();c.p.early_deposit=0;auto p=t3repair::project(c,x.view(),c.plans);require(p.overflow==1,"actual night capacity loss modeled");require(x.p.inventories[0][E]==101,"capacity probe does not mutate real inventory");}
 {Fixture x;x.day=27;x.hour=23;x.f.farmer={0,0};x.p.shed[E]=100;x.bag(0,E,3);auto c=x.ctl();c.target.clear();c.p.early_deposit=0;auto p=t3repair::project(c,x.view(),c.plans);require(p.overflow==0,"current shed sale creates space before night deposit");}
 {Fixture x;x.wheat(45);x.f.money=0;auto c=x.ctl();c.phase=2;c.t3.receipt_expected_hands=5;c.t3.receipt_day=x.day;auto a=t3repair::receipt(c,x.view());require(!a,"no speculative unfunded procurement");require(x.f.money==0,"receipt probe cannot manufacture cash");}
 {Fixture x;auto c=x.ctl();c.phase=3;require(!t3repair::receipt(c,x.view()),"receipt repair not repeated during execution");}
 {Fixture x;auto c=x.ctl();c.phase=2;c.t3.receipt_attempt_day=x.day;require(!t3repair::receipt(c,x.view()),"bounded receipt retries");}

 {Fixture x;x.day=27;x.hour=2;x.f.money=5000;x.shops={7,7};x.p.shed[WO]=20;
  for(int i=0;i<9;i++){x.m.inventory[i]=10000;x.m.prices[i]=price(i,10000);}
  triad::Settings settings;settings.intraday=0;settings.delay_sale=0;settings.batch_delivery=1;
  triad::Controller c(settings);c.core.day=x.day;c.model.day=x.day;c.core.phase=3;
  c.core.plans.resize(1);c.core.target.clear();c.previous_step=x.view().step-1;
  triad::LocalSaleTiming original=c.local_sale;auto clock=c.sale_memory;auto v=x.view();
  auto actual=c.act(v);
  require(bool(c.core.t3.project_sales),"actual P16 controller installs sale snapshot");
  Counts want{},seen{};want[WO]=seen[WO]=20;auto expected=original;
  expected.filter(v,x.p,{},want,clock,settings.reserve,settings.competition);
  auto probe=c.core.t3.project_sales;probe(v,x.p,{},seen);
  require(want==seen,"repair projection honors P16 quantity timing");
  require(seen[WO]<20,"fixture genuinely activates P16 hold, not dead-field check");
  auto live_deadlines=c.local_sale.deadline;auto inventory=x.p.shed;
  auto fork1=c.core.t3.project_sales, fork2=c.core.t3.project_sales;
  Counts a{},b{};a[WO]=b[WO]=20;fork1(v,x.p,{},a);fork2(v,x.p,{},b);
  require(a==b,"copied probes start from identical independent sale state");
  require(c.local_sale.deadline==live_deadlines,"probe cannot mutate live sale deadlines");
  require(x.p.shed==inventory,"probe cannot mutate actual inventory");
  Counts protected_goods{};protected_goods[W]=3;protected_goods[F]=4;auto protection=protected_goods;
  fork1(v,x.p,{},protected_goods);require(protected_goods==protection,"sale filter preserves feed/fertilizer quantities");
  Counts forced{};forced[WO]=20;fork1(v,x.p,{action(Op::HIRE)},forced);
  require(forced[WO]==20,"an actual procurement request cancels hold in copied model");
 }
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAILED after "<<checks<<": "<<e.what()<<"\n";return 1;}}
