#include "policy/search.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace triad;
int main(){
 long long checks=0;
 for(int item=1;item<=7;item++)for(int inv:{9700,9900,10000,10050,10100,10300})for(int qty:{1,2,5,10,20})for(int demand:{1,3,6})for(int rival:{0,3,12,30}){
  double gain=0;int q=LocalSaleTiming::amount_now(item,inv,qty,demand,rival,2.,&gain);assert(q>=0&&q<=qty);
  double g0=LocalSaleTiming::payoff(item,inv,qty,q,demand,0,2.)-LocalSaleTiming::payoff(item,inv,qty,qty,demand,0,2.);
  double g1=LocalSaleTiming::payoff(item,inv,qty,q,demand,rival,2.)-LocalSaleTiming::payoff(item,inv,qty,qty,demand,rival,2.);
  assert(std::min(g0,g1)>=-1e-8);assert(std::abs(gain-std::min(g0,g1))<1e-8);checks++;
 }
 Farm own,other;own.tiles.resize(100);other.tiles.resize(100);own.unlocked_mask=other.unlocked_mask=1;own.farmer=other.farmer={4,4};own.money=other.money=3000;
 for(int p=0;p<100;p++)if(quad(p)!=0){own.tiles[p].kind=other.tiles[p].kind=TileKind::LOCKED;}
 PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);priv.seeds[W]=2;
 Market market;for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=price(i,10000);}std::vector<int8_t>shops;
 View v{24,1,0,own,other,priv,market,shops};Settings settings;settings.scenario=0;settings.a06_reinvest=4;settings.a06_execution_candidates=1;
 triad::Controller own_value(settings);own_value.plan(v);auto margin=own_value;margin.s.a06_competitive_contract=1;PlayerAction out;out.units.resize(1);
 auto a=own_value.a06_remaining_values(own_value.core,v,out);auto b=margin.a06_remaining_values(margin.core,v,out);assert(a==b);
 assert(priv.seeds[W]==2&&own.money==3000); // forecast must not mutate observed resources
 // The risk certificate is conditional on the declared 0/ready-supply pair,
 // NOT a certificate about unseen actual opponent intent or future shops.
 std::cout<<"PASS "<<checks<<" bounded short-sale DP certificates\n";
 std::cout<<"PASS cash/margin valuation identical when public rival flow is zero; observed cash/paid inventory immutable\n";
}
