#define A08_SALE_SCHEDULE_DP 1
#include "../policy/sale_schedule_dp.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace triad;using namespace sale_dp;
static int assertions=0;void check(bool ok,const char*why){assertions++;if(!ok)throw std::runtime_error(why);}
int main(){try{
 std::cout<<"{\"scope\":\"new resume boundary tests, no games\",\"cases\":[";
 // Small bounded edge set, not a repeated old exhaustive grid.
 int rows=0,opts=0;
 for(int qty:{1,5,100})for(int T:{1,2,5}){
  Problem p;p.item=6;p.stock=(qty==100?9900:9981);p.quantity=qty;p.steps=T;p.minimum_now=qty/3;p.competition=2;
  if(T>1){p.rival[T-1]=3;p.demand[T-2]=1;}
  if(!supported(p)){check(qty==100,"unexpected unsupported finite quote");continue;}
  Quotes q(p);auto choice=select(p);int sum=0;for(int t=0;t<T;t++)sum+=choice.chosen[t];check(sum==qty,"volume not conserved");check(choice.chosen[0]>=p.minimum_now,"delayed original immediate sales");check(choice.worst[choice.index]>=choice.worst[0]-1e-7,"conditional selector worsens base");
  for(int traffic=0;traffic<2;traffic++)for(int order=0;order<3;order++){
   auto schedule=solve(p,q,traffic,Ordering(order));auto predicted=evaluate(p,q,schedule,traffic,Ordering(order));int stock=p.stock;Cash actual;
   for(int t=0;t<T;t++){auto c=reference_trade(p.item,stock,schedule[t],traffic?p.rival[t]:0,Ordering(order));actual.own+=c.own;actual.rival+=c.rival;stock-=p.demand[t];}
   check(actual.own==predicted.own&&actual.rival==predicted.rival,"cached exact quote path mismatch");
   if(qty<=5){Schedule s{};double best=-1e100;
    std::function<void(int,int)> dfs=[&](int t,int rem){if(t==T-1){if(t==0&&rem<p.minimum_now)return;s[t]=rem;auto c=evaluate(p,q,s,traffic,Ordering(order));best=std::max(best,c.own-p.competition*c.rival);return;}for(int a=t==0?p.minimum_now:0;a<=rem;a++){s[t]=a;dfs(t+1,rem-a);}};dfs(0,qty);
    check(std::abs(best-(predicted.own-p.competition*predicted.rival))<1e-7,"Bellman/exhaustive mismatch");opts++;
   }
  }
  if(rows++)std::cout<<",";std::cout<<"{\"quantity\":"<<qty<<",\"ticks\":"<<T<<",\"chosen\":"<<choice.index<<",\"gain\":"<<choice.gain<<"}";
 }
 Problem invalid;invalid.item=6;invalid.stock=9981;invalid.quantity=101;invalid.steps=5;check(!supported(invalid),"over capacity must fallback");invalid.quantity=1;invalid.steps=6;check(!supported(invalid),"horizon must fallback");invalid.steps=5;invalid.stock=20001;check(!supported(invalid),"stock domain must fallback");invalid.stock=0;invalid.demand[0]=1;check(!supported(invalid),"negative projected stock must fallback");invalid.stock=10200;invalid.demand[0]=0;if constexpr(A08_SALE_FLOOR_DP)check(supported(invalid),"bounded floor state must be supported");else check(!supported(invalid),"disabled floor branch must fallback");
 // Step458 shape: 8 plans, 10 actually observed units. Earlier plans include
 // a paid-material order and an already consumed prefix: neither may move.
 fastkag::Farm f,r;f.hands.resize(9);fastkag::PrivateState priv;priv.inventories.resize(10);priv.inventory_order.resize(10);fastkag::Market market;std::vector<int8_t>shops;
 dp7::View view{458,19,2,f,r,priv,market,shops};dp7::Controller core;core.phase=3;core.plans.resize(8);
 core.plans[0].a={dp7::action(fastkag::Op::PICKUP,dp7::W,2),dp7::action(fastkag::Op::FEED)};core.plans[0].target={44,35};core.plans[0].index=1;
 SaleScheduleDP dp;dp.reconcile_slots(core,view);check(core.plans.size()==10,"observed slots not appended");check(dp.observed_slots_appended==2,"reconciliation counter");check(core.plans[0].index==1&&core.plans[0].a.size()==2&&core.plans[0].a[0].quantity==2&&core.plans[0].target[1]==35,"paid prefix changed");check(core.plans[8].a.empty()&&core.plans[9].a.empty(),"imagined action on new hand");
 dp.reconcile_slots(core,view);check(dp.observed_slots_appended==2,"duplicate append");core.phase=2;core.plans.resize(8);dp.reconcile_slots(core,view);check(core.plans.size()==8,"preparation plans mutated");core.phase=3;priv.inventories.resize(7);priv.inventory_order.resize(7);f.hands.resize(6);dp.reconcile_slots(core,view);check(core.plans.size()==8,"existing paid route truncated");
 std::cout<<"],\"problems\":"<<rows<<",\"exhaustive_scenarios\":"<<opts<<",\"assertions\":"<<assertions<<",\"passed\":true}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
