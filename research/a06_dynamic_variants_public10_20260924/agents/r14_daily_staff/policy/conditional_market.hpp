#pragma once
#include "executor/policy.hpp"
#ifndef R2_MARKET_INTEGRAL
#define R2_MARKET_INTEGRAL 0
#endif
static_assert(R2_MARKET_INTEGRAL>=0 && R2_MARKET_INTEGRAL<=2);
namespace competitive {
inline thread_local int runtime_market_integral=0;
struct MarketModeScope {
 int previous;
 explicit MarketModeScope(int mode):previous(runtime_market_integral){runtime_market_integral=mode;}
 ~MarketModeScope(){runtime_market_integral=previous;}
};
// Conditional, sequential price/stock primitive. This is NOT a prediction of
// rival intent, nor does it remove the forecast's daily half-order assumption.
// At integral inputs, mode 2 matches successive official single-side quotes.
// Fractional forecast flows are a piecewise-linear relaxation of those quotes.
struct ConditionalMarket {
 static double saturation(int item){
  static const auto thresholds=[](){std::array<double,9>a{};
   for(int i=0;i<9;i++){
    double lo=10000,hi=10001;
    while(dp7::price(i,hi)>1)hi=10000+2*(hi-10000);
    while(hi-lo>1){double mid=std::floor((lo+hi)*.5);if(dp7::price(i,mid)<=1)hi=mid;else lo=mid;}
    a[i]=hi;
   }return a;
  }();return thresholds[item];
 }
 static double prefix(int item,double x){
  static const auto sums=[](){std::array<std::array<double,20001>,9>a{};
   for(int i=0;i<9;i++)for(int n=0;n<20000;n++)a[i][n+1]=a[i][n]+dp7::price(i,n);
   return a;
  }();
  int n=int(std::floor(x));if(n==20000)return sums[item][n];
  return sums[item][n]+(x-n)*dp7::price(item,n);
 }
 static double integral(int item,double a,double b){
  if(a==b)return 0;
  if(a>=0 && b<=20000)return prefix(item,b)-prefix(item,a);
  // Unusual observations outside the fast lookup remain defined. Runtime is
  // proportional to the requested quantity, never to absolute market stock.
  double value=0;
  while(a<b){double next=std::min(b,std::floor(a)+1);value+=(next-a)*dp7::price(item,std::floor(a));a=next;}
  return value;
 }
 static double execute(int item,double&stock,double quantity){
  if(quantity==0)return 0;
  if(quantity<0){double cash=-integral(item,stock+quantity,stock);stock+=quantity;return cash;}
  double counted=std::min(quantity,std::max(0.,saturation(item)-stock));
  double cash=integral(item,stock,stock+counted)+(quantity-counted);
  stock+=counted;return cash;
 }
 static void update(int item,double&stock,double quantity){
  stock+=quantity<0?quantity:std::min(quantity,std::max(0.,saturation(item)-stock));
 }
};
}
