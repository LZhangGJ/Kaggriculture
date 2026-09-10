#include "policy/planner.hpp"
#include <cassert>
#include <cmath>
int main(){
 using namespace competitive;
 for(int n=-50;n<=100;++n){
  double x=n*.85;
  int early=int(std::floor(x*rival_early_share));
  int late=int(std::round(x))-early;
  assert(early+late==int(std::round(x)));
  assert(std::abs(x*rival_early_share+rival_late_flow(x)-x)<1e-10);
 }
 Planner p;p.day=29;p.cfg.competition=2;p.cfg.discount=0;p.cfg.risk=0;
 dp7::Farm own{},enemy{};dp7::PrivateState priv{};fastkag::Market market{};
 std::vector<int8_t>shops;market.inventory.fill(10000);own.money=1000;
 dp7::View v{696,29,0,own,enemy,priv,market,shops};
 Asset a;a.f[29][dp7::S]=20;p.rival[29][dp7::S]=12;p.cfg.supply=1;
 double inventory=market.inventory[dp7::S];
 auto trade=[&](double q){double px=Planner::quote(dp7::S,inventory,q);if(q<0||px>1)inventory+=q;return q*px;};
 double other=trade(12*rival_early_share),ours=trade(20);other+=trade(rival_late_flow(12));
 Flow prices{};double predicted=p.value(v,a,&prices);
 assert(std::abs(predicted-(ours-2*other))<1e-8);
 assert(prices[29][dp7::S]==dp7::price(dp7::S,inventory));
}
