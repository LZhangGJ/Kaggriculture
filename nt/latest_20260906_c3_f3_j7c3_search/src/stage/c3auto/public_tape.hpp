#pragma once
#include "executor/observed_day_scenario.hpp"
namespace competitive {
struct PublicTape {
 int step=-1,last_day=-1,days=0;std::array<int,9> inventory{},owned{},consumed{};
 std::array<double,9>today{},ema{};std::array<std::array<double,6>,9>hours{};
 int accepted=0,skipped_floor=0;
 static std::array<int,9> total(const fastkag::PrivateState&p){std::array<int,9>a{};for(int i=0;i<9;i++){a[i]=p.shed[i];for(auto&b:p.inventories)a[i]+=b[i];}return a;}
 void observe(const View&o){
  if(last_day<0)last_day=o.day;
  if(o.day!=last_day){for(int i=0;i<9;i++)ema[i]=days?.5*ema[i]+.5*today[i]:today[i];today={};days++;last_day=o.day;}
  if(step<0||o.step!=step+1||step%24==23)return;
  auto current=total(o.priv);for(int i=0;i<9;i++){
   // The one-cash price floor makes gross sales unidentifiable; do not pretend otherwise.
   if(o.market.prices[i]<=2||price(i,inventory[i])<=2){skipped_floor++;continue;}
   double q=o.market.inventory[i]-inventory[i]+consumed[i]+current[i]-owned[i];
   today[i]+=q;if(q>0)hours[i][(step%24)/4]+=q;accepted++;
  }
 }
 void remember(const View&o,const PlayerAction&a){
  fastkag::ObservedDayScenario scene(o);auto projected=scene.project_units(a.units,int(a.units.size()));owned=total(projected.privates()[0]);inventory=o.market.inventory;step=o.step;consumed={};
  if(step%24==0)for(int i=0;i<8;i++)consumed[i]++;
  if(step%4==0){static const std::array<std::vector<int>,8>ps{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C,C},{MI,T,W},{S,MI},{WO,WO}}};for(int s:o.shops)for(int i:ps[s])consumed[i]++;}
 }
 double hour_weight(int item,int bucket)const{double n=0;for(double x:hours[item])n+=x;return (hours[item][bucket]+2.)/(n+12.);}
};
}
