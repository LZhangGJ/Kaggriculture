#pragma once
#include "planner.hpp"
#ifndef A08_ANIMAL_COLLECTION_LABOR
#define A08_ANIMAL_COLLECTION_LABOR 1
#endif
static_assert(A08_ANIMAL_COLLECTION_LABOR==0 || A08_ANIMAL_COLLECTION_LABOR==1);
namespace competitive {
// Port of our independently authored Python service Bellman recurrence.
// State: day, consecutive unfed days (0/1), previously banked care bonus.
// This is exact for the conditional service model, not a global execution proof.
struct AnimalServiceDP {
 struct Choice{int feed=0,care=0;double value=0;};
 Choice choices[30][2][6]{};double value[30][2][6]{};
 // Keep the Bellman table writer at a real optimized call boundary. Central
 // GCC13 -O3 testing of r2 observed feed=0 with value=65 for a state whose
 // unique optimal action is feed=1. Local GCC14/Clang do not reproduce that
 // failure; this is a targeted code-generation workaround, not a UB diagnosis.
 // noipa prevents caller-specific clones/inlining, NOT optimization of this
 // function's body. The production build remains -O3.
#if defined(__GNUC__) && !defined(__clang__)
 __attribute__((noipa))
#elif defined(__clang__)
 __attribute__((noinline))
#endif
 void solve(int kind,int placed,int day,const Flow&prices,double work){
  const int j=kind-9,cap=held[j]-1;
  for(int d=28;d>=day;--d){
   const int age=d+1-placed;
   const bool tick=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
   for(int h=0;h<2;++h)for(int p=0;p<=cap;++p){
    // Preserve the original action order and 1e-9 tie rule. Select a single
    // immutable candidate index, then commit its action AND value together;
    // do not carry three independent winner fields through nested loops.
    const auto evaluate=[&](int feed,int care)->Choice{
     const int nh=feed?0:h+1;
     if(nh>=2)return {0,0,0};
     const int quantity=tick?1+(feed?p:0):0;
     const int np=std::min(cap,(tick?0:p)+care);
     const double collection=A08_ANIMAL_COLLECTION_LABOR
       ? (quantity>0 ? work : 0.) + work : 0.;
     const double manure=A08_ANIMAL_COLLECTION_LABOR
       ? prices[d+1][F] : std::max(0.,prices[d+1][F]-work);
     const double v=-feed*(prices[d][W]+work)-care*work
       +quantity*prices[d+1][product[j]]+manure-collection+value[d+1][nh][np];
     return {feed,care,v};
    };
    const Choice candidates[3]={evaluate(0,0),evaluate(1,0),
      (p<cap||tick)?evaluate(1,1):Choice{0,0,-1e100}};
    int selected=0;
    for(int i=1;i<3;++i)
     if(candidates[i].value>candidates[selected].value+1e-9)selected=i;
    const Choice winner=candidates[selected];
    choices[d][h][p]=winner;
    value[d][h][p]=winner.value;
   }
  }
 }
 Choice first(int day,const Tile&t)const{if(day>=29)return {};return choices[day][std::min(1,int(t.consecutive_unfed))][std::min(held[int(t.animal)-9]-1,int(t.pending_care_bonus))];}
};
}
