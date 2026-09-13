#include "policy/search.hpp"
#include <cassert>
#include <iostream>
#include <random>
using namespace triad;using namespace dp7;
struct Fixture {
 Farm own,opp;PrivateState pr;Market market;std::vector<int8_t>shops{7};
 Fixture(){own.tiles.resize(100);opp.tiles.resize(100);own.money=1500;opp.money=1500;own.farmer=opp.farmer={4,4};own.unlocked_mask=opp.unlocked_mask=1;pr.inventories.resize(1);pr.inventory_order.resize(1);for(int i=0;i<9;i++){market.inventory[i]=9900;market.prices[i]=price(i,9900);}for(int pos=0;pos<100;pos++)if(quad(pos)!=0)own.tiles[pos].kind=opp.tiles[pos].kind=TileKind::LOCKED;}
 View view(int hour=8){return {240+hour,10,hour,own,opp,pr,market,shops};}
 void sheep(){auto&t=own.tiles[43];t.kind=TileKind::ANIMAL;t.animal=Item::SHEEP;t.placed_day=0;t.yield_units=6;t.fed_today=t.cared_today=true;t.fertilizer_available=false;}
};
int main(){
 // Exhaustive route/subset/permutation comparison for a bounded synthetic set.
 std::mt19937 rng(7711);long long enumerated=0;int cases=0;
 for(int test=0;test<80;test++){
  int n=1+rng()%5,ticks=3+rng()%21,start=depot[rng()%4];Counts free{};free[W]=rng()%5;free[F]=rng()%5;Market m;for(int i=0;i<9;i++){m.inventory[i]=9900;m.prices[i]=price(i,9900);}
  std::vector<r11::Bundle>jobs;
  for(int j=0;j<n;j++){r11::Bundle b;b.pos=int(rng()%50);b.actions.push_back(action(Op::HARVEST));b.out[WO]=1+rng()%6;if(rng()%2){b.needs[W]=1;b.actions.push_back(action(Op::FEED));}jobs.push_back(b);}
  auto result=r11::solve(jobs,free,m,0,start,ticks,false);double value=-1e100;int len=100000;
  for(int mask=1;mask<(1<<n);mask++){
   Counts need{},q{};int actions=0;std::vector<int>ids;for(int j=0;j<n;j++)if(mask&(1<<j)){ids.push_back(j);add(need,jobs[j].needs);add(q,jobs[j].out);actions+=jobs[j].actions.size();}
   bool valid=true;int picks=0;for(int i=0;i<12;i++){if(need[i]>free[i])valid=false;picks+=need[i]>0;}if(!valid)continue;
   do{for(int origin:std::vector<int>{start,44,45,54,55}){if(picks&&!at_depot(origin))continue;int cost=dist(start,origin)+picks+actions+1,last=origin;for(int j:ids){cost+=dist(last,jobs[j].pos);last=jobs[j].pos;}cost+=near(last);enumerated++;double v=r11::gross(q,m)-r11::gross(need,m);if(cost<=ticks&&(v>value+1e-9||(std::abs(v-value)<1e-9&&cost<len))){value=v;len=cost;}}}while(std::next_permutation(ids.begin(),ids.end()));
  }
  assert(bool(result)==(value>-1e90));if(result){assert(std::abs(result->rank-value)<1e-8);assert(int(result->route.a.size())==len);assert(result->route.a.back().op==Op::DROP);}cases++;
 }
 Fixture f;f.sheep();auto v=f.view();Settings st;st.scenario=0;st.intraday=0;st.a06_r11_recovery=1;st.a06_reinvest=4;st.a06_roll_scope=0;st.a06_competitive_contract=2;st.batch_delivery=1;
 triad::Controller c(st);c.core.day=10;c.core.phase=3;c.core.last_step=v.step-1;c.core.target={{43,SH}};c.core.plans.resize(1);c.model.day=10;
 PlayerAction a;a.units.resize(1);c.r11_consider(v,a);assert(!c.labor11.active); // KEEP already collects with the idle farmer; extra hire loses its wage.
 auto no_cost=r11::solve(r11::uncovered(c.core,v,a),Counts{},v.market,0,44,15,false);assert(no_cost);
 no_cost->issue_step=v.step;no_cost->day=v.day;no_cost->before_workers=1;no_cost->before_hires=0;
 // Submitted intent has not created a worker or altered the original route.
 assert(f.own.hands.empty()&&c.core.plans[0].a.empty());
 auto nohire=c;nohire.labor11.active=true;nohire.labor11.pending=*no_cost;auto next=f.view(9);assert(r11::reconcile(nohire.labor11,nohire.core,next));assert(nohire.labor11.committed==1);
 auto late=c;late.labor11.active=true;late.labor11.pending=*no_cost;assert(!r11::reconcile(late.labor11,late.core,f.view(23)));assert(late.core.plans[0].a.empty());
 // Busy farmer: extra paid worker must beat KEEP in exact conditional cash.
 auto busy=c;busy.labor11={};busy.core.plans[0]={};int pos=44;dp7::Controller::walk(busy.core.plans[0],pos,0);busy.core.plans[0].a.push_back(action(Op::WATER));busy.core.plans[0].target.push_back(0);
 auto&t=f.own.tiles[0];t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=1;t.consecutive_unwatered=1;
 busy.core.target.push_back({0,S});PlayerAction wait;wait.units.resize(1);busy.r11_consider(v,wait);assert(busy.labor11.active&&busy.labor11.pending.hire);assert(wait.market.back().op==Op::HIRE);
 auto before=busy.core.plans[0];auto failed=busy;assert(!r11::reconcile(failed.labor11,failed.core,next));assert(dp7::Controller::same_remaining(failed.core.plans[0],before));assert(f.own.hands.empty());
 auto confirmed=busy;f.own.hands.push_back({int16_t(busy.labor11.pending.start%10),int16_t(busy.labor11.pending.start/10)});f.own.hires_today++;f.pr.inventories.push_back({});f.pr.inventory_order.push_back({});assert(r11::reconcile(confirmed.labor11,confirmed.core,f.view(9)));assert(confirmed.core.plans.size()==2);assert(confirmed.labor11.hire_confirmed==1);
 f.own.hands.clear();f.own.hires_today=0;f.pr.inventories.resize(1);f.pr.inventory_order.resize(1);
 auto poor=busy;poor.labor11={};f.own.money=0;PlayerAction no;no.units.resize(1);poor.r11_consider(f.view(),no);assert(!poor.labor11.active);assert(no.market.empty());f.own.money=1500;
 auto buy=busy;buy.labor11={};PlayerAction spending;spending.units.resize(1);spending.market.push_back(action(Op::BUY_SEED,W));buy.r11_consider(f.view(),spending);assert(!buy.labor11.active&&spending.market.size()==1);
 Counts protected_need{},existing{};protected_need[W]=2;existing[W]=3;PrivateState pp;pp.shed[W]=6;PlayerAction sale;sale.market.push_back(action(Op::SELL,W,6));r11::protect_orders(sale,pp,existing,protected_need);assert(sale.market[0].quantity==1);
 // Identical mechanism and inputs in real and predictive invocation origins.
 auto real=busy,pred=busy;real.labor11={};pred.labor11={};pred.call_origin=triad::Controller::CallOrigin::PublicPrediction;PlayerAction aa,bb;aa.units.resize(1);bb.units.resize(1);real.r11_consider(f.view(),aa);pred.r11_consider(f.view(),bb);assert(real.labor11.last==pred.labor11.last);assert(aa.market.size()==bb.market.size());
 std::cout<<"PASS exhaustive DP cases="<<cases<<" route enumerations="<<enumerated<<"\n";
 std::cout<<"PASS idle vs hire, observed funding, failure/late cancellation, receipt commit, paid input protection, real/prediction parity\n";
 std::cout<<"negative_idle="<<c.labor11.last<<"\npositive_hire="<<busy.labor11.last<<"\n";
}
