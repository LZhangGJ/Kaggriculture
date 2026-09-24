#pragma once
// Receipt-confirmed, bounded same-day output rescue. All inputs are View/core
// observations and commitments; no live simulator, seed or rival identity.
#include "executor/policy.hpp"
#include <map>
namespace triad::r11 {
using namespace dp7;
struct Bundle {int pos=-1;Acts actions;Counts needs{},out{};Tile identity;};
struct Choice {int unit=-1,start=-1,day=-1,issue_step=-1,before_workers=0,before_hires=0;bool hire=false;Plan route;Counts needs{};std::vector<Bundle>jobs;double rank=0;};
struct State {
 bool active=false;Choice pending;
 long long checks=0,dp_solves=0,dp_masks=0,rollouts=0,offered=0,committed=0,hire_requests=0,hire_confirmed=0,cancelled=0;
 double last_gain=0,last_margin_gain=0;std::string last="null";
};
inline bool same_asset(const Tile&a,const Tile&b){
 return a.kind==b.kind&&a.crop==b.crop&&a.animal==b.animal&&
   (!plant(a)||a.planted_day==b.planted_day)&&(!animal(a)||a.placed_day==b.placed_day);
}
inline bool relevant(Op op){return op==Op::HARVEST||op==Op::COLLECT_FERTILIZER||op==Op::WATER||op==Op::FEED||op==Op::FERTILIZE||op==Op::CARE;}
inline double gross(const Counts&q,const Market&m){double v=0;for(int i=0;i<9;i++)if(q[i]>0)v+=revenue(i,m.inventory[i],q[i]);return v;}
inline std::vector<Bundle> uncovered(const dp7::Controller&c,const View&o,const PlayerAction&issued){
 auto res=intraday::reserved(c);std::array<bool,100>blocked=res.tiles;
 if(c.pending_admission.active&&c.pending_admission.pos>=0)blocked[c.pending_admission.pos]=true;
 for(size_t u=0;u<issued.units.size();u++){
  auto a=issued.units[u];if(a.op!=Op::PASS&&!dp7::Controller::movement(a.op)&&a.op!=Op::DROP&&a.op!=Op::PICKUP)
   blocked[cell(u?o.own.hands[u-1]:o.own.farmer)]=true;
 }
 // All split atoms of an uncovered plot are kept together. Never extract a
 // harvest across an already committed rotation/establishment/service chain.
 std::map<int,Bundle>by;std::set<int>unsafe;
 for(const auto&j:c.jobs(o)){
  const auto&t=o.own.tiles[j.pos];if(blocked[j.pos]||(!plant(t)&&!animal(t)))continue;
  if(sum(j.seeds)||std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return !relevant(a.op);})){unsafe.insert(j.pos);continue;}
  auto&b=by[j.pos];b.pos=j.pos;b.identity=t;b.actions.insert(b.actions.end(),j.actions.begin(),j.actions.end());add(b.needs,j.needs);add(b.out,j.out);
 }
 std::vector<Bundle>out;
 for(auto&[pos,b]:by)if(!unsafe.count(pos)&&sum(b.out)>0)out.push_back(std::move(b));
 std::stable_sort(out.begin(),out.end(),[&](const auto&a,const auto&b){
  double av=gross(a.out,o.market)/(1+int(a.actions.size())+2*near(a.pos));
  double bv=gross(b.out,o.market)/(1+int(b.actions.size())+2*near(b.pos));
  return av!=bv?av>bv:a.pos<b.pos;
 });
 if(out.size()>8)out.resize(8);return out;
}
// Exact minimum travel for each subset/order within this bounded set. No job
// splitting: service/harvest precedence is the order supplied by jobs(). Inputs
// are picked at a depot before work, and every chosen route explicitly DROPs.
inline std::optional<Choice> solve(const std::vector<Bundle>&jobs,const Counts&free,const Market&m,
 int unit,int start,int ticks,bool hire,State*stats=nullptr){
 int n=jobs.size();if(n==0||n>8||ticks<=0)return {};if(stats)stats->dp_solves++;
 constexpr int INF=100000;int N=1<<n;std::vector<Counts>need(N),produce(N);std::vector<int>work(N),picks(N);
 for(int mask=1;mask<N;mask++){int j=std::countr_zero(unsigned(mask)),prev=mask&(mask-1);need[mask]=need[prev];add(need[mask],jobs[j].needs);produce[mask]=produce[prev];add(produce[mask],jobs[j].out);work[mask]=work[prev]+jobs[j].actions.size();for(int i=0;i<12;i++)picks[mask]+=need[mask][i]>0;}
 std::vector<int>origins{start};for(int d:depot)if(d!=start)origins.push_back(d);
 std::optional<Choice>best;double best_value=-1e100;int best_len=INF;
 for(int origin:origins){
  std::vector<int>dp(N*n,INF),parent(N*n,-1);
  for(int j=0;j<n;j++)dp[(1<<j)*n+j]=dist(origin,jobs[j].pos);
  for(int mask=1;mask<N;mask++)for(int last=0;last<n;last++)if(mask&(1<<last)){
   int prev=mask^(1<<last);if(!prev)continue;
   for(int k=0;k<n;k++)if(prev&(1<<k)){int v=dp[prev*n+k]+dist(jobs[k].pos,jobs[last].pos);if(v<dp[mask*n+last]){dp[mask*n+last]=v;parent[mask*n+last]=k;}}
  }
  for(int mask=1;mask<N;mask++){
   if(stats)stats->dp_masks++;bool feasible=true;for(int i=0;i<12;i++)if(need[mask][i]>free[i])feasible=false;
   if(!feasible||(picks[mask]&&!at_depot(origin)))continue;
   double value=gross(produce[mask],m)-gross(need[mask],m);
   for(int last=0;last<n;last++)if(mask&(1<<last)){
    int len=dist(start,origin)+picks[mask]+work[mask]+dp[mask*n+last]+near(jobs[last].pos)+1;
    if(len>ticks||value<best_value-1e-9||(std::abs(value-best_value)<1e-9&&len>=best_len))continue;
    std::vector<int>order;int at=last,rem=mask;while(at>=0){order.push_back(at);int prev=parent[rem*n+at];rem^=1<<at;at=prev;}std::reverse(order.begin(),order.end());
    Choice q;q.unit=unit;q.start=start;q.hire=hire;q.needs=need[mask];q.rank=value;int pos=start;dp7::Controller::walk(q.route,pos,origin);
    std::vector<int>items;for(int i=0;i<12;i++)if(q.needs[i])items.push_back(i);std::sort(items.begin(),items.end(),[](int a,int b){return name(a)<name(b);});
    for(int i:items){q.route.a.push_back(action(Op::PICKUP,i,q.needs[i]));q.route.target.push_back(origin);}
    for(int j:order){const auto&b=jobs[j];q.jobs.push_back(b);dp7::Controller::walk(q.route,pos,b.pos);for(auto a:b.actions){q.route.a.push_back(a);q.route.target.push_back(b.pos);}}
    int dest=depot[0];for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;dp7::Controller::walk(q.route,pos,dest);q.route.a.push_back(action(Op::DROP));q.route.target.push_back(dest);
    if(int(q.route.a.size())!=len)throw std::logic_error("r11 DP route length");best=std::move(q);best_value=value;best_len=len;
   }
  }
 }
 return best;
}
inline void protect_orders(PlayerAction&out,const PrivateState&post,const Counts&existing,const Counts&extra){
 Counts free{};for(int i=0;i<12;i++)free[i]=std::max(0,post.shed[i]-existing[i]-extra[i]);
 for(auto&a:out.market)if(a.op==Op::SELL&&int(a.item)>=0&&int(a.item)<9&&extra[int(a.item)]>0){int i=int(a.item);a.quantity=std::min(a.quantity,free[i]);free[i]-=a.quantity;}
 out.market.erase(std::remove_if(out.market.begin(),out.market.end(),[](auto a){return a.quantity<=0;}),out.market.end());
}
inline bool reconcile(State&s,dp7::Controller&c,const View&o){
 if(!s.active)return false;const auto q=s.pending;
 if(o.step<=q.issue_step)return false;s.active=false;
 int n=int(o.own.hands.size())+1;
 // Only confirmed actual workers can extend the plan vector. Even if the
 // route is cancelled, the paid worker remains real and may receive later work.
 if(c.phase==3&&o.day==c.day&&c.plans.size()<size_t(n))c.plans.resize(n);
 bool ok=o.day==q.day&&o.step==q.issue_step+1&&c.phase==3&&n==q.before_workers+int(q.hire)&&
  o.own.hires_today==q.before_hires+int(q.hire)&&q.unit<n&&q.unit>=0;
 if(ok){auto&pl=c.plans[q.unit];ok=pl.index>=pl.a.size()&&sum(o.priv.inventories[q.unit])==0&&cell(q.unit?o.own.hands[q.unit-1]:o.own.farmer)==q.start;}
 if(ok){auto r=intraday::reserved(c);for(int i=0;i<12;i++)if(q.needs[i]>std::max(0,o.priv.shed[i]-r.shed[i]))ok=false;
  for(const auto&b:q.jobs)if(r.tiles[b.pos]||!same_asset(b.identity,o.own.tiles[b.pos]))ok=false;
  if(o.hour+int(q.route.a.size())>(o.day==29?23:24))ok=false;
 }
 if(!ok){s.cancelled++;return false;}
 c.plans[q.unit]=q.route;c.intraday_units.insert(q.unit);s.committed++;s.hire_confirmed+=q.hire;return true;
}
inline bool protected_assets(const Farm&initial,const Farm&keep,const Farm&trial){
 for(int pos=0;pos<100;pos++){
  const auto&a=initial.tiles[pos],&b=keep.tiles[pos],&c=trial.tiles[pos];
  if(!plant(a)&&!animal(a))continue;if(!same_asset(a,b))continue;
  if(!same_asset(a,c))return false;
  if(plant(b)&&(c.consecutive_unwatered>b.consecutive_unwatered||c.fertilized_until_day<b.fertilized_until_day))return false;
  if(animal(b)&&(c.consecutive_unfed>b.consecutive_unfed||c.pending_care_bonus<b.pending_care_bonus))return false;
 }
 return true;
}
}
