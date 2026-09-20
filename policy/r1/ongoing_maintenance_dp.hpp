#pragma once
#include "planner.hpp"
namespace competitive {
// Conditional maintenance DP: today's existing stock is collected by the C2
// harvest executor; future production is assumed collectible next morning.
// It is NOT a joint route/storage/sales DP. State is (day,dry streak,fertility).
struct OngoingMaintenanceDP {
 struct Choice {double value=-1e100;bool water=false,fertilize=false;};
 int kind=2,birth=0,begin=0,end=29,mode=2;
 double work=3.,discount=1.;std::array<double,30> price{},fert{};
 double memo[30][2][4]{};bool seen[30][2][4]{};
 bool production(int next)const{int ds=next-birth-dp7::first[kind];return ds>=0&&ds%dp7::interval[kind]==0&&ds/dp7::interval[kind]<4;}
 bool legacy_water(int d,int dry)const{return dry>=1||production(d+1);}
 bool legacy_fertilize(int d,int f)const{return production(d+1)&&f==0;}
 double q(int d,int dry,int f,bool w,bool z){
  if(d>=end)return 0.;if(z&&f>0)return -1e100;
  // Future choices must be executable by the same subtractive controller.
  if(z&&!legacy_fertilize(d,f))return -1e100;
  if(w&&!legacy_water(d,dry))return -1e100;
  if(mode==1&&w!=legacy_water(d,dry))return -1e100;
  if(mode<=2&&!w&&dry>=1)return -1e100;
  double v=-work*(int(w)+int(z))-(z?fert[d]:0.);
  if(!w&&dry>=1)return v; // death before next dawn's production
  int active=z?3:f;bool event=production(d+1);
  if(event)v+=discount*(1+(w&&active>0))*price[d+1];
  int last=birth+dp7::first[kind]+3*dp7::interval[kind];
  if(d+1<std::min(end,last))v+=discount*solve(d+1,w?0:dry+1,std::max(0,active-1));
  return v;
 }
 double solve(int d,int dry,int f){
  if(d>=end||d>=birth+dp7::first[kind]+3*dp7::interval[kind])return 0.;
  f=std::clamp(f,0,3);if(seen[d][dry][f])return memo[d][dry][f];
  seen[d][dry][f]=true;double best=-1e100;
  for(bool w:{false,true})for(bool z:{false,true})best=std::max(best,q(d,dry,f,w,z));
  return memo[d][dry][f]=best;
 }
 Choice first(int d,int dry,int f,bool oldw,bool oldf){
  Choice best{q(d,dry,f,oldw,oldf),oldw,oldf};
  for(bool w:{false,true})for(bool z:{false,true}){double v=q(d,dry,f,w,z);if(v>best.value+1e-9)best={v,w,z};}
  return best;
 }
};
}

