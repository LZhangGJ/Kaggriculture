#pragma once
// Offline diagnostics only. Never passed to Controller or its public View.
// Reconstruct prices from actual fills, preserving simultaneous unit quotes.
#include "policy.hpp"
#include <stdexcept>

namespace dp7audit {
using namespace fastkag;
struct CashDay {
 std::array<double,9>sales{},products{};
 std::array<double,5>seeds{};
 std::array<double,3>animals{};
 std::array<int,9>sold{},bought_products{};
 std::array<int,5>bought_seeds{};
 std::array<int,3>bought_animals{};
 double hired=0,land=0,start=0,end=0;
 int hires=0,lands=0,checks=0;
 double net()const{
  auto total=[](const auto&a){return std::accumulate(a.begin(),a.end(),0.);};
  return total(sales)-total(products)-total(seeds)-total(animals)-hired-land;
 }
};
struct BeforeMarket {
 std::array<int32_t,9>inventory{};
 std::array<double,2>cash{};
 std::array<int,2>hires{},land{};
 explicit BeforeMarket(const Simulator&env):inventory(env.market().inventory){
  for(int p=0;p<2;p++){cash[p]=env.farms()[p].money;hires[p]=env.farms()[p].hires_today;land[p]=std::popcount(unsigned(env.farms()[p].unlocked_mask));}
 }
};
inline double hire_price(int n){double a=1,b=1;for(int i=0;i<n;i++){double c=a+b;a=b;b=c;}return a;}
inline void accumulate(const BeforeMarket&before,const Simulator&after,
                       const std::array<PlayerAction,2>&actions,
                       std::array<CashDay,2>&days){
 auto inventory=before.inventory;auto hires=before.hires;auto land=before.land;
 std::array<double,2>step_net{};
 const auto&fills=after.last_market_fills();
 auto slots=std::min<size_t>(after.config().max_market_orders,std::max(actions[0].market.size(),actions[1].market.size()));
 for(size_t oi=0;oi<slots;oi++){
  std::array<int,2>n{};std::array<Action,2>a{};
  for(int p=0;p<2;p++)if(oi<actions[p].market.size()){
   a[p]=actions[p].market[oi];n[p]=fills[p].at(oi);
   if(a[p].op==Op::HIRE&&n[p]){double v=after.config().farm_hand_cost_mult*hire_price(hires[p]++);days[p].hired+=v;days[p].hires++;step_net[p]-=v;n[p]=0;}
   else if(a[p].op==Op::BUY_LAND&&n[p]){double v=land[p]==1?1000:land[p]==2?2000:4000;land[p]++;days[p].land+=v;days[p].lands++;step_net[p]-=v;n[p]=0;}
  }
  while(n[0]>0||n[1]>0){
   std::array<int,2>quote{};
   for(int p=0;p<2;p++)if(n[p]>0){int i=int(a[p].item);switch(a[p].op){
    case Op::SELL:quote[p]=dp7::price(i,inventory[i]);break;
    case Op::BUY_PRODUCT:quote[p]=dp7::price(i,inventory[i]-1);break;
    case Op::BUY_SEED:quote[p]=dp7::seed_price.at(i);break;
    case Op::BUY_ANIMAL:quote[p]=dp7::animal_price.at(i-9);break;
    default:throw std::runtime_error("unexpected cash fill");
   }}
   for(int p=0;p<2;p++)if(n[p]>0){int i=int(a[p].item),v=quote[p];auto&d=days[p];switch(a[p].op){
    case Op::SELL:d.sales[i]+=v;d.sold[i]++;step_net[p]+=v;if(v>1)inventory[i]++;break;
    case Op::BUY_PRODUCT:d.products[i]+=v;d.bought_products[i]++;step_net[p]-=v;inventory[i]--;break;
    case Op::BUY_SEED:d.seeds[i]+=v;d.bought_seeds[i]++;step_net[p]-=v;break;
    case Op::BUY_ANIMAL:d.animals[i-9]+=v;d.bought_animals[i-9]++;step_net[p]-=v;break;
    default:break;
   }n[p]--;}
  }
 }
 for(int p=0;p<2;p++){
  if(before.cash[p]+step_net[p]!=after.farms()[p].money)throw std::runtime_error("cash ledger does not reconcile at step "+std::to_string(after.step_count())+" seat "+std::to_string(p));
  days[p].end=after.farms()[p].money;days[p].checks++;
 }
}
}
