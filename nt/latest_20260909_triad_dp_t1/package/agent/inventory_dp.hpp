#pragma once
#include "planner.hpp"
#include <unordered_map>
namespace competitive {
// Finite-horizon Bellman DP. State is (time, owned inventory, market inventory).
// Prices, own fills and price-floor inventory transitions are exact CONDITIONAL
// on supplied future arrivals and town consumption. Forecasts are not guarantees.
struct InventoryDP {
 struct Step {int arrival=0,rival=0,demand=0;};
 struct Answer{double value=0;int sell=0;};
 int item,horizon,capacity=100;double competition=1.;std::vector<Step>future;
 std::unordered_map<uint64_t,Answer>memo;
 explicit InventoryDP(int i,std::vector<Step>f,double c):item(i),horizon(f.size()),competition(c),future(std::move(f)){}
 std::pair<double,int> sell(int inv,int q)const{double cash=0;for(int k=0;k<q;k++){int p=price(item,inv);cash+=p;if(p>1)inv++;}return{cash,inv};}
 Answer solve(int t,int stock,int inv){
  stock=std::max(0,stock);if(t==horizon)return {sell(inv,stock).first,stock};
  uint64_t key=(uint64_t(t)<<56)|(uint64_t(stock)<<40)|uint32_t(inv);auto it=memo.find(key);if(it!=memo.end())return it->second;
  Answer best{-1e100,0};auto&f=future[t];
  int minimum=std::max(0,stock+f.arrival-capacity);
  for(int q=minimum;q<=stock;q++){
   auto[own,after]=sell(inv,q);auto[other,last]=sell(after,std::max(0,f.rival));
   auto rest=solve(t+1,stock-q+f.arrival,last-f.demand);double v=own-competition*other+rest.value;
   if(v>best.value+1e-9){best={v,q};}
  }
  memo.emplace(key,best);return best;
 }
};
}
