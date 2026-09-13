// Standalone reduction of the ORIGINAL delivered r2 recurrence.
// No game engine, simulator, trace, foreign memory or private opponent data.
// GCC13 failure was reported centrally; local compiler matrix is logged.
#include <array>
#include <algorithm>
#include <cstdio>
#ifndef A08_ANIMAL_COLLECTION_LABOR
#define A08_ANIMAL_COLLECTION_LABOR 1
#endif
constexpr int W=0,E=5,F=8;
constexpr std::array<int,3> afirst{4,8,6}, ainterval{1,2,3}, held{4,6,6}, product{5,6,7};
using Flow=std::array<std::array<double,9>,30>;
struct AnimalServiceDP {
 struct Choice{int feed=0,care=0;double value=0;};
 Choice choices[30][2][6]{};double value[30][2][6]{};

#if defined(__GNUC__) && !defined(__clang__)
 __attribute__((noipa))
#elif defined(__clang__)
 __attribute__((noinline))
#endif
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
};
int main() {
 Flow prices{};for(auto &d:prices){d.fill(30.);d[W]=25.;d[F]=2.;}
 AnimalServiceDP poor;poor.solve(9,21,28,prices,4.);
 for(auto &d:prices)d[E]=100.;
 AnimalServiceDP rich;rich.solve(9,21,28,prices,4.);
 auto q=rich.choices[28][1][0];
 std::printf("poor.feed=%d poor.value=%.17g rich.feed=%d rich.care=%d rich.choice_value=%.17g rich.value=%.17g terminal=%.17g\n",poor.choices[28][1][0].feed,poor.value[28][1][0],q.feed,q.care,q.value,rich.value[28][1][0],rich.value[29][0][0]);
 return !(poor.choices[28][1][0].feed==0 && poor.value[28][1][0]==0 && q.feed==1 && q.care==0 && q.value==65 && rich.value[28][1][0]==65 && rich.value[29][0][0]==0);
}
