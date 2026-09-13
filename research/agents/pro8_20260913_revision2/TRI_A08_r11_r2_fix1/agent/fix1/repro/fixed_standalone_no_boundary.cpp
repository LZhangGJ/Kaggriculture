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
 // Keep the Bellman table writer at a real optimized call boundary. Central
 // GCC13 -O3 testing of r2 observed feed=0 with value=65 for a state whose
 // unique optimal action is feed=1. Local GCC14/Clang do not reproduce that
 // failure; this is a targeted code-generation workaround, not a UB diagnosis.
 // noipa prevents caller-specific clones/inlining, NOT optimization of this
 // function's body. The production build remains -O3.
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
