#pragma once
// A08 r10: bounded cash-funded continuation of already-paid field capital.
// No new seed/crop/animal is created here. Routes are appended to material
// commitments, and a HIRE request does not create a worker until observed.
#include "public_flow_scenario.hpp"
#include <limits>
namespace triad::paid_detail {
using namespace dp7;
struct Routing {
 std::vector<Plan> plans;
 std::vector<PaidTarget> covered;
 int added_steps=0;
};
inline bool target_alive(const View&o,const PaidTarget&t){
 const auto&x=o.own.tiles[t.pos];return plant(x)&&int(x.crop)==t.kind&&x.planted_day==t.birth&&!x.watered_today;
}
// Held--Karp over <=8 water sites followed by a subset-allocation DP over
// workers. Existing remaining semantic actions are retained in their order.
// The routing objective nominates maximal deadline-feasible coverage; the
// economic selector below must independently accept the WHOLE continuation.
inline Routing routes(const Controller&c,const View&o,const std::vector<PaidTarget>&asked,int first_unit=0){
 Routing r;r.plans=c.core.plans;
 if(r.plans.size()!=o.priv.inventories.size())return r;
 std::vector<PaidTarget>t;for(auto x:asked)if(target_alive(o,x)&&t.size()<8)t.push_back(x);
 int n=int(t.size()),N=1<<n,U=int(r.plans.size()),limit=24-o.hour;
 // Preserve a paid fertilizer-before-water obligation even when the new
 // water route belongs to another worker. Offsets refer to current time.
 std::vector<int>ready(n,0);
 for(const auto&pl:c.core.plans)for(size_t k=pl.index;k<pl.a.size();k++)
  if(pl.a[k].op==Op::FERTILIZE)for(int j=0;j<n;j++)
   if(pl.target[k]==t[j].pos)ready[j]=std::max(ready[j],int(k-pl.index)+2);
 if(!n||first_unit>=U)return r;
 const int INF=100000;
 struct Option {std::vector<int>cost,last;std::vector<std::vector<int>>prev;};
 std::vector<Option>opt(U);
 for(int u=0;u<U;u++){
  auto&pl=r.plans[u];int stale=0;pl=c.core.repair_plan(o,u,pl,stale);
  auto&v=opt[u];v.cost.assign(N,INF);v.last.assign(N,-1);v.prev.assign(N,std::vector<int>(n,-1));v.cost[0]=0;
  if(u<first_unit||int(pl.a.size())>=limit)continue;
  int end=dp7::Controller::plan_end(o,u,pl);std::vector<std::vector<int>>d(N,std::vector<int>(n,INF));
  for(int j=0;j<n;j++)d[1<<j][j]=std::max(dist(end,t[j].pos)+1,ready[j]-int(pl.a.size()));
  for(int mask=1;mask<N;mask++)for(int j=0;j<n;j++)if((mask>>j&1)&&d[mask][j]<INF){
   if(int(pl.a.size())+d[mask][j]<=limit&&d[mask][j]<v.cost[mask]){v.cost[mask]=d[mask][j];v.last[mask]=j;}
   for(int k=0;k<n;k++)if(!(mask>>k&1)){
    int q=std::max(d[mask][j]+dist(t[j].pos,t[k].pos)+1,ready[k]-int(pl.a.size())),m=mask|(1<<k);
    if(q<d[m][k]){d[m][k]=q;v.prev[m][k]=j;}
   }
  }
 }
 std::vector<std::vector<int>>dp(U+1,std::vector<int>(N,INF)),chosen(U+1,std::vector<int>(N));dp[0][0]=0;
 for(int u=0;u<U;u++)for(int mask=0;mask<N;mask++)if(dp[u][mask]<INF){
  int remaining=(N-1)^mask;
  for(int sub=remaining;;sub=(sub-1)&remaining){
   int value=dp[u][mask]+opt[u].cost[sub];
   if(value<dp[u+1][mask|sub]){dp[u+1][mask|sub]=value;chosen[u+1][mask|sub]=sub;}
   if(!sub)break;
  }
 }
 int best=0;auto rank=[&](int mask){int paid=0;for(int j=0;j<n;j++)if(mask>>j&1)paid+=seed_price[t[j].kind];return std::tuple(std::popcount(unsigned(mask)),paid,-dp[U][mask],-mask);};
 for(int m=1;m<N;m++)if(dp[U][m]<INF&&rank(m)>rank(best))best=m;
 r.added_steps=dp[U][best];int mask=best;
 for(int u=U-1;u>=0;u--){int sub=chosen[u+1][mask];mask^=sub;if(!sub)continue;
  std::vector<int>order;int a=sub,j=opt[u].last[a];
  while(a){order.push_back(j);int prev=opt[u].prev[a][j];a^=1<<j;j=prev;}
  std::reverse(order.begin(),order.end());auto&pl=r.plans[u];int pos=dp7::Controller::plan_end(o,u,pl);
  for(int k:order){dp7::Controller::walk(pl,pos,t[k].pos);while(int(pl.a.size())+1<ready[k]){pl.a.push_back(action(Op::PASS));pl.target.push_back(t[k].pos);}pl.a.push_back(action(Op::WATER));pl.target.push_back(t[k].pos);r.covered.push_back(t[k]);}
 }
 return r;
}
inline std::vector<PaidTarget> urgent(const Controller&c,const View&o){
 std::array<bool,100>covered{},removing{};
 for(const auto&pl:c.core.plans)for(size_t k=pl.index;k<pl.a.size();k++){
  int pos=pl.target[k];if(pos<0||o.hour+int(k-pl.index)>=24)continue;
  auto op=pl.a[k].op;
  if(op==Op::WATER)covered[pos]=true;
  if(op==Op::DIG||op==Op::PLANT||(op==Op::HARVEST&&plant(o.own.tiles[pos])&&!ongoing(int(o.own.tiles[pos].crop))))removing[pos]=true;
 }
 std::vector<PaidTarget>out;
 for(auto[pos,kind]:c.core.target){if(pos<0||kind<0||kind>=5)continue;const auto&t=o.own.tiles[pos];
  if(plant(t)&&int(t.crop)==kind&&t.consecutive_unwatered>=1&&!t.watered_today&&!covered[pos]&&!removing[pos])out.push_back({pos,kind,t.planted_day});
 }
 std::stable_sort(out.begin(),out.end(),[](auto a,auto b){return std::tuple(-seed_price[a.kind],a.birth,a.pos)<std::tuple(-seed_price[b.kind],b.birth,b.pos);});
 if(out.size()>8)out.resize(8);return out;
}
// A labelled, loose per-field reachability scenario, NOT a belief that the hidden warehouse contains
// this quantity. Only current public field output reachable by a public worker
// and then a depot before the boundary is used. Fields share workers, so the
// sum is an UPPER scenario, not a jointly schedulable or actual stock estimate.
// No private stock is read; pre-existing hidden warehouse supply is unknown.
inline competitive::Flow reachable_field_scenario(const View&o){
 competitive::Flow f{};int budget=24-o.hour;std::vector<int>starts{cell(o.opponent.farmer)};
 for(auto p:o.opponent.hands)starts.push_back(cell(p));
 for(int pos=0;pos<100;pos++){
  const auto&t=o.opponent.tiles[pos];int travel=1000;
  for(int st:starts)travel=std::min(travel,dist(st,pos)+1+near(pos)+1);
  if(travel>budget)continue;
  if(animal(t)){f[o.day][product[int(t.animal)-9]]+=std::max(0,int(t.yield_units));if(t.fertilizer_available)f[o.day][F]+=1;}
  else if(plant(t))f[o.day][int(t.crop)]+=std::max(0,int(t.yield_units));
 }
 return f;
}
struct Score {double value=0,cash=0,enemy_cash=0,tail=0;int ticks=0,dry=0;Counts shed{};};
// Finite current-day execution, identical action semantics to live execution.
// All later prices, opponent arrivals and the tail are explicitly conditional.
inline Score evaluate(Controller after,const View&start,const PlayerAction&first,const competitive::Flow&flow){
 after.continuation.probe=true;
 fastkag::PublicFlowScenario world(start,flow,1.,&after.sale_memory);
 world.advance(first);int ticks=1;
 while(!world.done()&&world.view().day==start.day){auto v=world.view();auto a=after.act(v);world.advance(a);ticks++;}
 auto end=world.view();Score r;r.cash=world.own_cash();r.enemy_cash=world.rival_cash();r.shed=end.priv.shed;r.ticks=ticks;
 for(int p=0;p<100;p++)if(plant(start.own.tiles[p])&&end.own.tiles[p].kind==TileKind::WEED)r.dry++;
 r.value=r.cash-after.s.competition*r.enemy_cash;
 if(!world.done()){
  Controller tail=after;tail.continuation.probe=true;tail.plan(end);r.tail=tail.predicted;r.value+=r.tail;
 }
 return r;
}
inline void write_score(std::ostream&o,const Score&s){o<<"{\"value\":"<<s.value<<",\"cash\":"<<s.cash<<",\"opponent_cash\":"<<s.enemy_cash<<",\"conditional_tail\":"<<s.tail<<",\"ticks\":"<<s.ticks<<",\"plant_to_weed\":"<<s.dry<<",\"warehouse\":[";for(int i=0;i<12;i++){if(i)o<<",";o<<s.shed[i];}o<<"]}";}
}
namespace triad {
inline void paid_record(Controller&c,const View&o,const PlayerAction&a){
 auto&x=c.continuation;int n=0;for(auto b:a.market)n+=b.op==Op::HIRE;
 if(n){x.request_step=o.step;x.request_count=n;x.request_workers=o.priv.inventories.size();}
}
inline void paid_observe(Controller&c,const View&o){
 auto&x=c.continuation;
 if(x.hire_day!=o.day){x.hire_day=o.day;x.hire_deficit=0;x.request_count=0;x.request_step=-1;}
 if(x.request_count&&o.step>x.request_step){
  int arrived=std::max(0,int(o.priv.inventories.size())-x.request_workers);
  if(arrived<x.request_count)x.hire_deficit+=x.request_count-arrived;
  else x.hire_deficit=std::max(0,x.hire_deficit-arrived);
  x.request_count=0;
 }
 if(x.pending_day<0)return;
 if(o.day!=x.pending_day){x.pending.clear();x.pending_day=-1;return;}
 if(o.step<=x.pending_step)return;
 int old=x.pending_workers,actual=int(o.priv.inventories.size());
 // The authoritative observation, NOT the requested number of HIREs, decides
 // which routes may be committed. Partial fills cannot create phantom hands.
 if(c.core.phase==3&&actual>old){
  c.core.plans.resize(actual);
  auto r=paid_detail::routes(c,o,x.pending,old);
  for(int u=old;u<actual;u++){
   if(!r.plans[u].a.empty()){c.core.plans[u]=std::move(r.plans[u]);c.core.intraday_units.insert(u);x.routes_committed++;}
  }
  x.observed_new_workers+=actual-old;x.water_committed+=r.covered.size();
  for(auto t:r.covered){c.core.crop_water[t.pos]=1;c.core.crop_birth[t.pos]=t.birth;c.core.crop_kind[t.pos]=t.kind;}
 }else x.unfilled++;
 x.pending.clear();x.pending_day=-1;
}
inline bool paid_choose(Controller&c,const View&o,PlayerAction&out){
 using namespace paid_detail;
 if(c.continuation.hire_deficit<=0||c.core.phase!=3||o.day>=29||o.hour>=23||c.core.plans.size()!=o.priv.inventories.size())return false;
 auto need=urgent(c,o);if(need.empty())return false;
 auto&state=c.continuation;
 if(state.checked_day!=o.day){state.checked_day=o.day;state.checks_today=0;state.last_cash=-1;state.last_hands=-1;}
 // At most two evaluations per day; a rejected poor-cash state is revisited
 // only after actual money/workforce has changed, never from predicted sales.
 if(state.checks_today>=2||(state.checks_today&&state.last_cash==int(o.own.money)&&state.last_hands==int(o.own.hands.size())))return false;
 int next=int(o.own.hires_today);
 bool affordable=next<15&&o.own.hands.size()<size_t(c.s.max_hands)&&o.own.money>=fib[next];
 if(!affordable){auto route=routes(c,o,need);if(route.covered.empty())return false;}
 state.checks_today++;state.checks++;state.last_cash=int(o.own.money);state.last_hands=o.own.hands.size();
 auto checkpoint=c;
 auto base=c;base.continuation.probe=true;base.previous_step=o.step-1;
 auto original=base.act(o);
 std::array<competitive::Flow,2>flow{{competitive::Flow{},reachable_field_scenario(o)}};
 std::array<Score,2>bs{evaluate(base,o,original,flow[0]),evaluate(base,o,original,flow[1])};
 // Do not intervene on a raw warning when the current rolling execution
 // already maintains every nominated prepaid crop through this boundary.
 if(bs[0].dry==0&&bs[1].dry==0)return false;
 Controller winner=base;PlayerAction action_win=original;double gain=0;int chosen=-1;
 std::ostringstream log;log.precision(17);log<<"{\"step\":"<<o.step<<",\"cash_observed\":"<<o.own.money<<",\"workers_observed\":"<<o.priv.inventories.size()<<",\"sites\":[";
 for(size_t i=0;i<need.size();i++){if(i)log<<",";log<<need[i].pos;}
 log<<"],\"observed_hire_deficit\":"<<state.hire_deficit<<",\"baseline\":[";write_score(log,bs[0]);log<<",";write_score(log,bs[1]);log<<"],\"alternatives\":[";
 bool sep=false;
 for(int hires=0;hires<=2;hires++){
  Controller trial=checkpoint;trial.continuation.probe=true;trial.previous_step=o.step-1;
  PlayerAction first;bool valid=true;int route_sites=0;double wages=0;
  if(!hires){
   auto r=routes(trial,o,need);if(r.covered.empty())continue;
   trial.core.plans=std::move(r.plans);route_sites=r.covered.size();
   for(auto t:r.covered){trial.core.crop_water[t.pos]=1;trial.core.crop_birth[t.pos]=t.birth;trial.core.crop_kind[t.pos]=t.kind;}
   trial.continuation.routes_committed++;trial.continuation.water_committed+=route_sites;
   first=trial.act(o);
  }else{
   if(int(o.own.hands.size())+hires>int(c.s.max_hands)||int(o.own.hires_today)+hires>15||original.market.size()+hires>10)continue;
   for(int h=0;h<hires;h++)wages+=fib[o.own.hires_today+h];
   if(wages>o.own.money)continue; // Never collateralise future/this-step sales.
   if(std::any_of(original.market.begin(),original.market.end(),[](auto a){return a.op==Op::HIRE;}))continue;
   trial=base;first=original;for(int h=0;h<hires;h++)first.market.push_back(action(Op::HIRE));
   fastkag::ObservedDayScenario one(o);auto post=one.project_units(first.units,-1);
   auto without=post.project_own_market(0,original.market);auto with=post.project_own_market(0,first.market);
   auto&fills=with.last_market_fills()[0];
   for(size_t j=0;j<original.market.size();j++)if(original.market[j].op!=Op::SELL&&fills[j]<without.last_market_fills()[0][j])valid=false;
   for(int h=0;h<hires;h++)if(fills[original.market.size()+h]!=1)valid=false;
   if(!valid)continue;
   auto&x=trial.continuation;x.pending=need;x.pending_day=o.day;x.pending_step=o.step;x.pending_workers=o.priv.inventories.size();
   x.hire_requested+=hires;route_sites=need.size();
  }
  auto a=evaluate(trial,o,first,flow[0]),b=evaluate(trial,o,first,flow[1]);
  double safe=std::min(a.value-bs[0].value,b.value-bs[1].value);
  // Do not buy a worker unless the proposed real route actually reduces
  // crop loss in BOTH declared supply scenarios. Forecast profit is separate.
  valid=(a.dry<bs[0].dry&&b.dry<bs[1].dry);
  if(sep)log<<",";sep=true;log<<"{\"hires\":"<<hires<<",\"paid_wage_upper\":"<<wages<<",\"nominated_sites\":"<<route_sites<<",\"reduces_loss_both\":"<<(valid?"true":"false")<<",\"minimum_conditional_gain\":"<<safe<<",\"scenarios\":[";write_score(log,a);log<<",";write_score(log,b);log<<"]}";
  state.alternatives++;
  if(valid&&safe>gain+1e-6){gain=safe;chosen=hires;winner=std::move(trial);action_win=std::move(first);}
 }
 log<<"],\"chosen_hires\":"<<chosen<<",\"minimum_conditional_gain\":"<<gain<<"}";
 auto info=log.str();state.last_json=info;
 if(chosen<0)return false;
 // Commit only the policy state AFTER THE FIRST action, never a rolled future.
 auto counts=state;winner.continuation.checks=counts.checks;winner.continuation.alternatives=counts.alternatives;winner.continuation.selected=counts.selected+1;
 winner.continuation.checked_day=counts.checked_day;winner.continuation.checks_today=counts.checks_today;
 winner.continuation.last_cash=counts.last_cash;winner.continuation.last_hands=counts.last_hands;
 winner.continuation.last_json=info;winner.continuation.probe=false;
 c=std::move(winner);out=std::move(action_win);return true;
}
}
