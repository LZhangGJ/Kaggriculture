#pragma once
// A06: bounded exact two-worker assignment/routing DP. This uses current jobs,
// not replay/seed information. Every atomic job retains its dependency order.
// The official game has NO per-worker bag cap; the common shed is capped at 100.
#ifndef A06_EXEC_MODE
#define A06_EXEC_MODE 0
#endif
namespace a06 {
struct FleetStats {int solves=0,changes=0,added=0,saved=0;};
inline bool identical(const Job&a,const Job&b){
 if(a.pos!=b.pos||a.needs!=b.needs||a.seeds!=b.seeds||a.out!=b.out||a.actions.size()!=b.actions.size())return false;
 for(size_t i=0;i<a.actions.size();i++)if(a.actions[i].op!=b.actions[i].op||a.actions[i].item!=b.actions[i].item||a.actions[i].quantity!=b.actions[i].quantity)return false;
 return true;
}
struct TourTable {
 int n,m,start;std::vector<int>distance,parent,cost,last;
 TourTable(const std::vector<Job>&jobs,int origin,bool ret):n(jobs.size()),m(1<<n),start(origin),distance(m*n,100000),parent(m*n,-1),cost(m,100000),last(m,-1){
  std::vector<int>work(m,0),needs(m,0),outputs(m,0);
  for(int mask=1;mask<m;mask++){
   int j=std::countr_zero(unsigned(mask)),prev=mask&(mask-1),types=0;
   for(int k=0;k<12;k++)if(jobs[j].needs[k]>0)types|=1<<k;
   work[mask]=work[prev]+jobs[j].actions.size();needs[mask]=needs[prev]|types;outputs[mask]=outputs[prev]||output(jobs[j].out);
  }
  cost[0]=0;
  for(int j=0;j<n;j++)distance[(1<<j)*n+j]=dist(start,jobs[j].pos);
  for(int mask=1;mask<m;mask++)for(int j=0;j<n;j++)if(mask&(1<<j)){
   int d=distance[mask*n+j];if(d>=100000)continue;
   // All resource pickups precede departure. No fictitious input borrowing.
   int c=d+work[mask]+std::popcount(unsigned(needs[mask]))+(ret&&outputs[mask]?near(jobs[j].pos)+1:0);
   if((!needs[mask]||at_depot(start))&&c<cost[mask]){cost[mask]=c;last[mask]=j;}
   for(int k=0;k<n;k++)if(!(mask&(1<<k))){int next=mask|(1<<k),nd=d+dist(jobs[j].pos,jobs[k].pos);if(nd<distance[next*n+k]){distance[next*n+k]=nd;parent[next*n+k]=j;}}
  }
 }
 Route route(int mask,int unit,const std::vector<Job>&jobs)const{
  int end=last[mask];std::vector<int>order;
  while(end>=0){order.push_back(end);int p=parent[mask*n+end];mask^=1<<end;end=p;}
  Route r(unit,start);for(auto i=order.rbegin();i!=order.rend();i++)r.append(jobs[*i]);return r;
 }
};
inline std::vector<Job>coalesce(const Route&a,const Route&b,const Job*extra){
 std::vector<Job>jobs;
 auto append=[&](const Job&j){
  auto it=std::find_if(jobs.begin(),jobs.end(),[&](const Job&x){return x.pos==j.pos;});
  if(it==jobs.end())jobs.push_back(j);else{
   // Split collect/feed/care atoms may be reunited. FERTILIZE->WATER and
   // HARVEST->DIG->PLANT->WATER chains remain intact inside their atomic job.
   it->actions.insert(it->actions.end(),j.actions.begin(),j.actions.end());
   add(it->needs,j.needs);add(it->seeds,j.seeds);add(it->out,j.out);it->priority=std::min(it->priority,j.priority);it->crop|=j.crop;
  }
 };
 for(const auto&j:a.jobs)append(j);for(const auto&j:b.jobs)append(j);if(extra)append(*extra);return jobs;
}
inline bool improve_pair(Route&a,Route&b,int budget,bool ret,const Job*extra,FleetStats&stats){
 auto jobs=coalesce(a,b,extra);int n=jobs.size();if(n<2||n>10)return false;
 stats.solves++;TourTable ta(jobs,a.start,ret),tb(jobs,b.start,ret);int all=(1<<n)-1,best=-1;
 auto metric=[](int a,int b){return std::pair(a+b,std::max(a,b));};
 auto old=metric(a.total(ret),b.total(ret));auto score=extra?std::pair(100000,100000):old;
 for(int mask=0;mask<=all;mask++){
  int ca=ta.cost[mask],cb=tb.cost[all^mask];if(ca>budget||cb>budget)continue;
  auto s=metric(ca,cb);if(s<score){score=s;best=mask;}
 }
 if(best<0)return false;
 a=ta.route(best,a.unit,jobs);b=tb.route(all^best,b.unit,jobs);
 stats.changes++;stats.saved+=std::max(0,old.first-score.first);stats.added+=extra!=nullptr;return true;
}
inline void fleet_dp(const std::vector<Job>&authoritative,std::pair<std::vector<Route>,int>&pack,int budget,bool ret,FleetStats&stats){
 auto&rs=pack.first;if(rs.size()<2||budget<2)return;
 std::vector<Job>missing;
 for(const auto&j:authoritative){bool found=false;for(auto&r:rs)for(auto&x:r.jobs)found|=identical(j,x);if(!found)missing.push_back(j);}
 // Geographic candidate ordering bounds work. The actual selected partition,
 // order, pickups and return path are solved exactly for each <=10-tile union.
 auto proximity=[](const Route&a,const Route&b){int d=20;for(auto&x:a.jobs)for(auto&y:b.jobs)d=std::min(d,dist(x.pos,y.pos));return d;};
 std::vector<std::tuple<int,int,int>>pairs;
 for(int a=0;a<int(rs.size());a++)for(int b=a+1;b<int(rs.size());b++)if(!rs[a].jobs.empty()||!rs[b].jobs.empty())pairs.emplace_back(proximity(rs[a],rs[b]),a,b);
 std::sort(pairs.begin(),pairs.end());int attempts=0;
 for(auto[d,a,b]:pairs){if(attempts++>=12)break;improve_pair(rs[a],rs[b],budget,ret,nullptr,stats);}
 std::stable_sort(missing.begin(),missing.end(),[](auto&a,auto&b){return a.priority<b.priority;});
 int accepted=0,tries=0;
 for(auto&j:missing){bool done=false;for(auto[d,a,b]:pairs){
   if(tries++>=16)break;int close=20;for(auto&x:rs[a].jobs)close=std::min(close,dist(x.pos,j.pos));for(auto&x:rs[b].jobs)close=std::min(close,dist(x.pos,j.pos));if(close>3)continue;
   if(improve_pair(rs[a],rs[b],budget,ret,&j,stats)){done=true;accepted++;break;}
  }if(tries>=16)break;
 }
 pack.second=std::max(0,pack.second-accepted);
}
// Minimum-extra-labour subset of tours to return before daily automatic shed
// deposit. Output is physical units, not marked-to-market profit. The compiler
// and observation reconciliation still validate all actual DROP/SELL fills.
inline std::vector<bool> delivery_knapsack(const std::vector<Route>&rs,int excess,int budget){
 int available=0;int n=rs.size();
 for(int i=0;i<n&&i<31;i++)if(rs[i].total(true)<=budget)for(auto&j:rs[i].jobs)available+=sum(j.out);
 // Saturate at the real requirement, not an arbitrary 400-unit ceiling.
 const int cap=std::min(available,std::max(0,excess)),inf=100000;std::vector<int>dp(cap+1,inf);std::vector<unsigned>which(cap+1,0);dp[0]=0;
 for(int i=0;i<n&&i<31;i++){
  int q=0;for(auto&j:rs[i].jobs)q+=sum(j.out);int extra=near(rs[i].end)+1;
  if(q<=0||rs[i].total(true)>budget)continue;
  for(int x=cap;x>=0;x--)if(dp[x]<inf){int nx=std::min(cap,x+q);if(dp[x]+extra<dp[nx]){dp[nx]=dp[x]+extra;which[nx]=which[x]|(1u<<i);}}
 }
 int at=-1;for(int x=std::min(cap,std::max(0,excess));x<=cap;x++)if(dp[x]<inf&&(at<0||dp[x]<dp[at]||(dp[x]==dp[at]&&x>at)))at=x;
 if(at<0)for(int x=cap;x>=0;x--)if(dp[x]<inf){at=x;break;}
 std::vector<bool>out(n,false);if(at>=0)for(int i=0;i<n&&i<31;i++)out[i]=which[at]&(1u<<i);return out;
}
} // namespace a06
