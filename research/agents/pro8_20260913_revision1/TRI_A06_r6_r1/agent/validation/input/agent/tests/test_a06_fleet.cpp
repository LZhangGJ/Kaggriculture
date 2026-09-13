#include "policy/executor/policy.hpp"
#include <random>
#include <iostream>
#include <map>
#include <cassert>
using namespace dp7;
int main(){
 std::mt19937 rng(6106);int checked=0;
 for(int trial=0;trial<80;trial++){
  int n=2+rng()%5;std::vector<Job>jobs;
  for(int j=0;j<n;j++){Job x;x.pos=rng()%100;int len=1+rng()%3;x.actions.assign(len,action(Op::WATER));if(rng()%3==0)x.needs[W]=1;if(rng()%4==0)x.needs[F]=1;if(rng()%2)x.out[M]=1+rng()%6;jobs.push_back(x);}
  int start=trial%3==0?23:depot[trial%4];bool ret=trial%2;
  a06::TourTable tab(jobs,start,ret);
  for(int mask=1;mask<(1<<n);mask++){
   std::vector<int>perm;int work=0,needs=0;bool output=false;
   for(int i=0;i<n;i++)if(mask&(1<<i)){perm.push_back(i);work+=jobs[i].actions.size();for(int k=0;k<12;k++)if(jobs[i].needs[k])needs|=1<<k;output|=dp7::output(jobs[i].out);}
   int best=100000;
   if(!needs||at_depot(start))do{int c=work+std::popcount(unsigned(needs)),pos=start;for(int i:perm){c+=dist(pos,jobs[i].pos);pos=jobs[i].pos;}if(ret&&output)c+=near(pos)+1;best=std::min(best,c);}while(std::next_permutation(perm.begin(),perm.end()));
   assert(tab.cost[mask]==best);if(best<100000){auto r=tab.route(mask,0,jobs);assert(r.total(ret)==best);}checked++;
  }
 }
 for(int trial=0;trial<160;trial++){
  int n=2+rng()%9,excess=trial<100?rng()%80:rng()%1200,budget=23;std::vector<Route>routes;
  for(int i=0;i<n;i++){Route r(i,44);Job j;j.pos=20+rng()%55;j.out[M]=1+rng()%(trial<100?15:150);j.actions.push_back(action(Op::HARVEST));r.append(j);routes.push_back(r);}
  auto selected=a06::delivery_knapsack(routes,excess,budget);int cost=0,q=0;
  for(int i=0;i<n;i++)if(selected[i]){q+=sum(routes[i].jobs[0].out);cost+=near(routes[i].end)+1;assert(routes[i].total(true)<=budget);}
  int best=100000,bestq=0;
  for(int mask=0;mask<(1<<n);mask++){int c=0,out=0;bool feasible=true;for(int i=0;i<n;i++)if(mask&(1<<i)){feasible&=routes[i].total(true)<=budget;c+=near(routes[i].end)+1;out+=sum(routes[i].jobs[0].out);}if(feasible){bestq=std::max(bestq,out);if(out>=excess)best=std::min(best,c);}}
  if(best<100000){assert(q>=excess);assert(cost==best);}else assert(q==bestq);
 }
 std::cout<<"PASS: "<<checked<<" exact tour/subset comparisons; 160 warehouse subset comparisons, including >400-unit demand\n";
}
