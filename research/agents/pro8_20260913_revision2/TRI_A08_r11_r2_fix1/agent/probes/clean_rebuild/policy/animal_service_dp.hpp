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
 void solve(int kind,int placed,int day,const Flow&prices,double work){
  int j=kind-9,cap=held[j]-1;for(int d=28;d>=day;d--){int age=d+1-placed;bool tick=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
   for(int h=0;h<2;h++)for(int p=0;p<=cap;p++){
    Choice best{0,0,-1e100};
    for(int feed=0;feed<=1;feed++){int nh=feed?0:h+1;if(nh>=2){if(0>best.value)best={0,0,0};continue;}
     for(int care=0;care<=feed;care++){if(care&&p>=cap&&!tick)continue;
      int quantity=tick?1+(feed?p:0):0,np=std::min(cap,(tick?0:p)+care);
      // Every credited future output must be collected. The live executor
      // harvests positive animal yield and always collects available manure;
      // neither is a free/optional cash credit, including on the last day.
      const double collection = A08_ANIMAL_COLLECTION_LABOR
        ? (quantity>0 ? work : 0.) + work : 0.;
      const double manure = A08_ANIMAL_COLLECTION_LABOR
        ? prices[d+1][F] : std::max(0.,prices[d+1][F]-work);
      double v=-feed*(prices[d][W]+work)-care*work
        +quantity*prices[d+1][product[j]]+manure-collection+value[d+1][nh][np];
      if(v>best.value+1e-9)best={feed,care,v};
     }
    }
    choices[d][h][p]=best;value[d][h][p]=best.value;
   }
  }
 }
 Choice first(int day,const Tile&t)const{if(day>=29)return {};return choices[day][std::min(1,int(t.consecutive_unfed))][std::min(held[int(t.animal)-9]-1,int(t.pending_care_bonus))];}
};
}
