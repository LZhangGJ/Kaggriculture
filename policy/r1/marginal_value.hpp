#pragma once
#include "planner.hpp"

namespace competitive {
// Local linearization of Planner::value, NOT a future market quote or a
// feasibility proof. Signed flow perturbations are probes of that objective;
// complete candidate streams must still pass value() and execution admission.
struct MarginalValue {
 struct Trade {double own=0,enemy=0;};
 static Trade day_trade(const Planner&m,int d,int i,double q,double&stock){
  stock-=m.dem[d][i]*.5;
#if R2_SALE_CLOCK_MODE >= 2
  double early=m.cfg.supply*m.rival[d][i]*m.rival_early_share[i];
  double late=m.cfg.supply*m.rival[d][i]-early;
#else
  double early=m.cfg.supply*m.rival[d][i]*.5,late=early;
#endif
  Trade r;r.enemy=Planner::trade(i,stock,early);
  r.own=Planner::trade(i,stock,q);
  r.enemy+=Planner::trade(i,stock,late);stock-=m.dem[d][i]*.5;
  return r;
 }
 static Flow compute(const Planner&m,const View&o,const Asset&a,
                     double local_discount=0.,double epsilon=2.){
  if(!(epsilon>0)||!std::isfinite(epsilon)||local_discount<0||!std::isfinite(local_discount))
   throw std::invalid_argument("invalid marginal value scale");
  Flow before{},own{},enemy{},out{};
  Curve balances{},weights{};std::array<double,9>stock{};
  for(int i=0;i<9;i++)stock[i]=o.market.inventory[i];
  double balance=o.own.money;
  double discount=1+m.cfg.discount*std::max(0.,1-o.own.money/20000.);
  for(int d=m.day;d<30;d++){
   double cash=a.fixed[d];
   for(int i=0;i<9;i++){
    before[d][i]=stock[i];auto r=day_trade(m,d,i,a.f[d][i],stock[i]);
    own[d][i]=r.own;enemy[d][i]=r.enemy;cash+=r.own;
   }
   cash-=m.wages(a.labor[d]);cash-=m.cfg.action_cost*a.labor[d];
   balance+=cash;balances[d]=balance;weights[d]=1/std::pow(discount,d-m.day);
  }
  for(int i=0;i<9;i++)for(int d=m.day;d<30;d++){
   auto change=[&](double delta){
    double inventory=before[d][i],cash_change=0,value=0;
    for(int t=d;t<30;t++){
     auto r=day_trade(m,t,i,a.f[t][i]+(t==d?delta:0.),inventory);
     double dc=r.own-own[t][i];cash_change+=dc;
     value+=(dc-m.cfg.competition*(r.enemy-enemy[t][i]))*weights[t];
     value-=m.cfg.risk*(std::max(0.,-balances[t]-cash_change)-std::max(0.,-balances[t]));
    }
    return value;
   };
   // Normalize to day d's units because the lifecycle DP discounts its
   // continuation. Without this conversion future flows are discounted twice.
   out[d][i]=(change(epsilon)-change(-epsilon))/(2*epsilon)*std::pow(1+local_discount,d-m.day);
  }
  return out;
 }
};
}
