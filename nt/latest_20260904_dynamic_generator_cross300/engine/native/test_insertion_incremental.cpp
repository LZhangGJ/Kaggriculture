#include "policy.hpp"
#include <chrono>
#include <iostream>
#include <random>
using namespace dp7;
int checks=0;void check(bool b){checks++;if(!b)throw std::runtime_error("insertion exactness");}
// Frozen pre-optimization implementation (before regret's optional second pass).
std::pair<std::vector<Route>,int> reference(std::vector<Job>js,const std::vector<int>&starts,int budget,bool ret,bool insertion){
 std::stable_sort(js.begin(),js.end(),[&](auto&a,auto&b){if(insertion)return std::tuple(a.priority,-int(a.actions.size()),snake(a.pos))<std::tuple(b.priority,-int(b.actions.size()),snake(b.pos));return std::tuple(a.priority,snake(a.pos),-int(a.actions.size()))<std::tuple(b.priority,snake(b.pos),-int(b.actions.size()));});
 std::vector<Route>rs;for(int i=0;i<int(starts.size());i++)rs.emplace_back(i,starts[i]);int dropped=0;
 for(auto&j:js){int best=-1,at=-1;std::tuple<int,int,int,int,int>bk;for(auto&r:rs){int old=insertion?r.ordered(r.jobs,ret):r.total(ret);int begin=insertion?0:int(r.jobs.size());for(int k=begin;k<=int(r.jobs.size());k++){int c;if(insertion){auto cp=r.jobs;cp.insert(cp.begin()+k,j);c=r.ordered(cp,ret);}else c=r.append_cost(j,ret);if(c<=budget){auto key=std::tuple(c-old,c,int(r.jobs.size()),r.unit,k);if(best<0||key<bk){best=r.unit;at=k;bk=key;}}}}if(best<0){dropped++;continue;}if(insertion)rs[best].jobs.insert(rs[best].jobs.begin()+at,j);else rs[best].append(j);}
 if(insertion)for(auto&r:rs){Route tmp(r.unit,r.start);for(auto&j:r.jobs)tmp.append(j);r=std::move(tmp);}return {rs,dropped};
}
void equal(const std::pair<std::vector<Route>,int>&a,const std::pair<std::vector<Route>,int>&b){
 check(a.second==b.second&&a.first.size()==b.first.size());
 for(size_t u=0;u<a.first.size();u++){const auto&x=a.first[u];const auto&y=b.first[u];
  check(x.start==y.start&&x.end==y.end&&x.cost==y.cost&&x.unit==y.unit&&x.needs==y.needs&&x.has_output==y.has_output);
  check(packmemo::key(x.jobs,{},0,false,false,false)==packmemo::key(y.jobs,{},0,false,false,false));
 }
}
int main(){try{
 Controller c;std::mt19937 rng(59321);double slow=0,fast=0;int cases=0;
 for(int sample=0;sample<2500;sample++){
  std::vector<Job>jobs;for(int n=int(rng()%80);n--;){Job j;j.pos=rng()%100;j.priority=int(rng()%7)-3;j.crop=rng()%2;
   for(int x=1+rng()%5;x--;)j.actions.push_back(action(Op::WATER));
   for(int k=0;k<12;k++)if(rng()%8==0)j.needs[k]=1+rng()%3;
   if(rng()%2)j.out[rng()%9]=1+rng()%5;jobs.push_back(j);
  }
  std::vector<int>starts;for(int n=1+rng()%15;n--;)starts.push_back(rng()%100);
  int budget=int(rng()%27)-1;
  for(bool ret:{false,true})for(bool insert:{false,true}){
   auto t=std::chrono::steady_clock::now();auto a=reference(jobs,starts,budget,ret,insert);slow+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
   t=std::chrono::steady_clock::now();auto b=c.pack_uncached(jobs,starts,budget,ret,insert);fast+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();equal(a,b);cases++;
  }
 }
 std::cout<<"{\"status\":\"PASS\",\"cases\":"<<cases<<",\"checks\":"<<checks<<",\"reference_seconds\":"<<slow<<",\"incremental_seconds\":"<<fast<<"}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
