#pragma once
#include "sale_clock.hpp"
#include "conditional_market.hpp"
#ifndef R2_LOCAL_SALE_TIMING
#define R2_LOCAL_SALE_TIMING 0
#endif
namespace triad {
struct LocalSaleTiming {
 std::array<int,9>deadline{{-1,-1,-1,-1,-1,-1,-1,-1,-1}};
 int checks=0,holds=0,held_quantity=0,forced=0;double projected_gain=0;
 static int due(int step){return ((step+3)/4)*4+1;}
 static double payoff(int item,int inventory,int quantity,int now,int demand,int rival,double competition){
  double stock=inventory;double ours=competitive::ConditionalMarket::execute(item,stock,now);
  double enemy=competitive::ConditionalMarket::execute(item,stock,rival);stock-=demand;
  ours+=competitive::ConditionalMarket::execute(item,stock,quantity-now);
  return ours-competition*enemy;
 }
 static int amount_now(int item,int inventory,int quantity,int demand,int rival,double competition,double*gain=nullptr){
  double base0=payoff(item,inventory,quantity,quantity,demand,0,competition);
  double base1=payoff(item,inventory,quantity,quantity,demand,rival,competition);
  int best=quantity;double improvement=0;
  for(int q=quantity-1;q>=0;q--){
   double d0=payoff(item,inventory,quantity,q,demand,0,competition)-base0;
   double d1=payoff(item,inventory,quantity,q,demand,rival,competition)-base1;
   double value=std::min(d0,d1);if(value>improvement+1e-6){improvement=value;best=q;}
  }
  if(gain)*gain=improvement;return best;
 }
 static int public_rival_quantity(const dp7::View&o,int item,int until,const SaleClock&clock){
  int ready=0;for(const auto&t:o.opponent.tiles){
   if(dp7::animal(t)&&dp7::product[int(t.animal)-9]==item)ready+=t.yield_units;
   else if(dp7::plant(t)&&int(t.crop)==item)ready+=PublicTradeLedger::available_upper(t,o.day);
  }
  const auto&history=clock.rival[item];
  if(history.count()<2)return ready;
  double mean=0;for(int k=0;k<3;k++)if(history.day[k]>=0)mean+=std::accumulate(history.sales[k].begin(),history.sales[k].end(),0.);
  mean/=history.count();auto density=history.density();double mass=0;
  for(int step=o.step;step<until;step++)mass+=density[step%24];
  return std::max(ready,int(std::ceil(mean*mass)));
 }
 void filter(const dp7::View&o,const fastkag::PrivateState&priv,const dp7::Acts&buys,dp7::Counts&sell,const SaleClock&clock,double reserve,double competition,bool rival_aware=false){
  if constexpr(R2_LOCAL_SALE_TIMING==0)return;
  int until=due(o.step),load=dp7::sum(priv.shed);for(const auto&b:priv.inventories)load+=dp7::sum(b);
  bool eligible=buys.empty()&&o.own.money>=reserve&&o.day<29&&until/24==o.day&&load<90;
  auto demand=PublicTradeLedger::consumption(until-1,o.shops);
  for(int i=1;i<=7;i++){
   if(deadline[i]>=0){
    if(o.step>=deadline[i]||!eligible){deadline[i]=-1;forced++;continue;}
    sell[i]=0;continue;
   }
   if(!eligible||sell[i]<=0||demand[i]<=0)continue;
   checks++;int rival=(rival_aware||R2_LOCAL_SALE_TIMING>=2)?public_rival_quantity(o,i,until,clock):0;
   double gain=0;int now=amount_now(i,o.market.inventory[i],sell[i],demand[i],rival,competition,&gain);
   if(now<sell[i]){deadline[i]=until;holds++;held_quantity+=sell[i]-now;projected_gain+=gain;sell[i]=now;}
  }
 }
};
}
