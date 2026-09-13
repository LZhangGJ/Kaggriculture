#pragma once
#include "planner.hpp"
#include "local_sale_timing.hpp"
#include <limits>
#include <sstream>
#ifndef A08_SALE_SCHEDULE_DP
#define A08_SALE_SCHEDULE_DP 0
#endif
static_assert(A08_SALE_SCHEDULE_DP==0 || A08_SALE_SCHEDULE_DP==1);
namespace triad {
namespace sale_dp {
// Current guaranteed warehouse batch only. Future harvest/carry is NEVER cash.
// The horizon ends at the ALREADY chosen local hold deadline (<=4 ticks away).
// A new schedule must sell at least the original immediate quantity and clear
// the remainder no later: no additional inventory/cash delay is introduced.
constexpr int MAX_Q=100, MAX_T=5;
enum Ordering {RIVAL_FIRST=0, LOCKSTEP=1, OWN_FIRST=2};
struct Cash {double own=0,rival=0;};
struct Problem {
 int item=0,stock=0,quantity=0,steps=0,minimum_now=0;
 double competition=0;
 std::array<int,MAX_T> demand{},rival{};
};
using Schedule=std::array<int,MAX_T>;
// Exact per-unit lockstep for a single pair of same-item orders. Non-buyable
// products have no cash-admission constraint when selling actual shed units.
inline Cash reference_trade(int item,int&stock,int own,int rival,Ordering order){
 Cash c;
 auto sell=[&](int n,double&money){for(int k=0;k<n;k++){int p=dp7::price(item,stock);money+=p;if(p>1)stock++;}};
 if(order==RIVAL_FIRST){sell(rival,c.rival);sell(own,c.own);return c;}
 if(order==OWN_FIRST){sell(own,c.own);sell(rival,c.rival);return c;}
 int both=std::min(own,rival);
 for(int k=0;k<both;k++){int p=dp7::price(item,stock);c.own+=p;c.rival+=p;if(p>1)stock+=2;}
 sell(own-both,c.own);sell(rival-both,c.rival);return c;
}
// Certified non-floor region gives an exact inventory lattice indexed by
// cumulative sold units. Saturating-price paths deliberately retain r10.
inline bool supported(const Problem&p){
 if(p.quantity<1||p.quantity>MAX_Q||p.steps<1||p.steps>MAX_T||p.minimum_now<0||p.minimum_now>p.quantity)return false;
 long long high=static_cast<long long>(p.stock)+p.quantity,low=p.stock;
 for(int t=0;t<p.steps;t++){if(p.rival[t]<0||p.demand[t]<0)return false;high+=p.rival[t];low-=p.demand[t];}
 // The bounded lookup domain is explicit. Exotic observations defer to the
 // unchanged parent rather than overflow an integer or allocate an unbounded table.
 if(low<0||high>20000)return false;
 return p.item>=1&&p.item<=7&&dp7::price(p.item,double(high))>1;
}
struct Quotes {
 int lo=0,hi=0;std::vector<double>prefix,even,odd;
 explicit Quotes(const Problem&p){
  lo=p.stock;hi=p.stock+p.quantity;
  for(int t=0;t<p.steps;t++){lo-=p.demand[t];hi+=p.rival[t];}
  hi+=2;int n=hi-lo+1;prefix.resize(n+1);even.resize(n+1);odd.resize(n+1);
  for(int k=0;k<n;k++){double v=dp7::price(p.item,lo+k);prefix[k+1]=prefix[k]+v;even[k+1]=even[k]+(k%2==0?v:0);odd[k+1]=odd[k]+(k%2?v:0);}
 }
 double sum(int start,int n)const{return prefix.at(start-lo+n)-prefix.at(start-lo);}
 double stride2(int start,int n)const{const auto&a=(start-lo)%2?odd:even;return a.at(start-lo+2*n)-a.at(start-lo);}
 Cash trade(int stock,int own,int rival,Ordering order)const{
  if(order==RIVAL_FIRST)return {sum(stock+rival,own),sum(stock,rival)};
  if(order==OWN_FIRST)return {sum(stock,own),sum(stock+own,rival)};
  int both=std::min(own,rival);double locked=stride2(stock,both);
  return {locked+sum(stock+2*both,own-both),locked+sum(stock+2*both,rival-both)};
 }
};
inline Schedule solve(const Problem&p,const Quotes&q,bool traffic,Ordering ordering,int*states=nullptr){
 const double NEG=-1e100;
 std::array<std::array<double,MAX_Q+1>,MAX_T+1>v{};
 std::array<std::array<int,MAX_Q+1>,MAX_T>take{};
 for(auto&row:v)row.fill(NEG);v[p.steps][0]=0;
 std::array<int,MAX_T+1>rival_before{},demand_before{};
 for(int t=0;t<p.steps;t++){rival_before[t+1]=rival_before[t]+(traffic?p.rival[t]:0);demand_before[t+1]=demand_before[t]+p.demand[t];}
 for(int t=p.steps-1;t>=0;t--)for(int remaining=0;remaining<=p.quantity;remaining++){
  const int stock=p.stock+p.quantity-remaining+rival_before[t]-demand_before[t];
  const int enemy=traffic?p.rival[t]:0;
  int low=t==0?p.minimum_now:0;if(t==p.steps-1)low=remaining;
  for(int amount=low;amount<=remaining;amount++){
   if(v[t+1][remaining-amount]<NEG/2)continue;
   Cash cash=q.trade(stock,amount,enemy,ordering);
   double value=cash.own-p.competition*cash.rival+v[t+1][remaining-amount];
   if(value>v[t][remaining]+1e-9){v[t][remaining]=value;take[t][remaining]=amount;}
  }
  if(states)(*states)++;
 }
 Schedule result{};int remaining=p.quantity;
 for(int t=0;t<p.steps;t++){result[t]=take[t][remaining];remaining-=result[t];}
 return result;
}
inline Cash evaluate(const Problem&p,const Quotes&q,const Schedule&a,bool traffic,Ordering order){
 int stock=p.stock;Cash total;
 for(int t=0;t<p.steps;t++){
  int rival=traffic?p.rival[t]:0;auto c=q.trade(stock,a[t],rival,order);total.own+=c.own;total.rival+=c.rival;
  stock+=a[t]+rival-p.demand[t];
 }return total;
}
struct Choice {
 Schedule original{},chosen{};std::vector<Schedule>alternatives;
 std::vector<std::array<Cash,4>>cash;std::vector<double>worst;
 int index=0,states=0;double gain=0;
};
inline Choice select(const Problem&p){
 Choice r;r.original[0]=p.minimum_now;r.original[p.steps-1]+=p.quantity-p.minimum_now;r.chosen=r.original;
 if(!supported(p))return r;
 Quotes quotes(p);
 auto insert=[&](Schedule s){if(std::find(r.alternatives.begin(),r.alternatives.end(),s)==r.alternatives.end())r.alternatives.push_back(s);};
 insert(r.original);Schedule early{};early[0]=p.quantity;insert(early);
 insert(solve(p,quotes,false,RIVAL_FIRST,&r.states));
 for(int order=0;order<3;order++)insert(solve(p,quotes,true,Ordering(order),&r.states));
 // Scenario-optimal Bellman profiles are cross-evaluated as COMMON complete
 // schedules, NOT clairvoyant different future schedules for each scenario.
 // This is a bounded robust candidate set, not an exact minimax DP theorem.
 for(const auto&s:r.alternatives){
  std::array<Cash,4>c{};c[0]=evaluate(p,quotes,s,false,RIVAL_FIRST);
  for(int o=0;o<3;o++)c[1+o]=evaluate(p,quotes,s,true,Ordering(o));
  double worst=1e100;for(auto x:c)worst=std::min(worst,x.own-p.competition*x.rival);
  r.cash.push_back(c);r.worst.push_back(worst);
 }
 for(size_t i=1;i<r.worst.size();i++)if(r.worst[i]>r.worst[r.index]+1e-6)r.index=i;
 r.chosen=r.alternatives[r.index];r.gain=r.worst[r.index]-r.worst[0];return r;
}
}
struct SaleScheduleDP {
 long checks=0,eligible=0,changed=0,advanced_units=0,states=0,skip_floor=0,skip_history=0;
 std::string last_json="{}";
 long observed_slots_appended=0;
 // Reconcile only ACTUALLY observed arrivals with an executing plan vector.
 // A changed earlier decision plus a later legal observation can expose more
 // workers than a conditional plan contained. Preserve all paid existing
 // routes and indices; extra workers start idle, never with imagined routes.
 // This is also used by conditional rollouts, not a saved-replay special case.
 void reconcile_slots(dp7::Controller&core,const dp7::View&o){
  if constexpr(!A08_SALE_SCHEDULE_DP)return;
  if(core.phase==3&&core.plans.size()<o.priv.inventories.size()){
   observed_slots_appended+=o.priv.inventories.size()-core.plans.size();
   core.plans.resize(o.priv.inventories.size());
  }
 }
 void filter(const dp7::View&o,const fastkag::PrivateState&post,const dp7::Acts&buys,
             const dp7::Counts&available,dp7::Counts&sell,std::array<int,9>&deadline,
             const SaleClock&clock,double reserve,double competition){
  if constexpr(!A08_SALE_SCHEDULE_DP)return;
  if(!buys.empty()||o.own.money<reserve||o.day>=29)return;
  int load=dp7::sum(post.shed);for(auto&b:post.inventories)load+=dp7::sum(b);
  // Same conservative capacity guard as the old hold owner. The new action
  // only advances its sales, never consumes reserved feed/fertilizer/seeds,
  // never changes unit routes, never waits for an assumed future harvest.
  if(load>=90)return;
  std::ostringstream log;log.precision(17);bool any=false;
  log<<"{\"step\":"<<o.step<<",\"observed_cash\":"<<o.own.money<<",\"projected_current_load\":"<<load<<",\"items\":[";
  for(int i=1;i<=7;i++){
   int end=deadline[i],amount=available[i];
   if(amount<=sell[i]||end<=o.step||end/24!=o.day||end-o.step>=sale_dp::MAX_T)continue;
   checks++;sale_dp::Problem p;p.item=i;p.stock=o.market.inventory[i];p.quantity=amount;p.steps=end-o.step+1;p.minimum_now=sell[i];p.competition=competition;
   const auto&h=clock.rival[i];std::array<double,24>mean{};int observed_days=0;
   for(int k=0;k<3;k++)if(h.day[k]>=0&&h.day[k]<o.day&&h.day[k]>=o.day-3){observed_days++;for(int t=0;t<24;t++)mean[t]+=h.sales[k][t];}
   if(observed_days<2){skip_history++;continue;}
   double sum=0;int prior=0;int total_rival=0;
   for(int t=0;t<p.steps;t++){
    p.demand[t]=PublicTradeLedger::consumption(o.step+t,o.shops)[i];
    sum+=mean[(o.step+t)%24]/observed_days;int cumulative=int(std::floor(sum+.5));p.rival[t]=cumulative-prior;prior=cumulative;total_rival+=p.rival[t];
   }
   if(total_rival==0){skip_history++;continue;}
   if(!sale_dp::supported(p)){skip_floor++;continue;}
   eligible++;auto result=sale_dp::select(p);states+=result.states;
   if(any)log<<",";any=true;
   log<<"{\"item\":"<<i<<",\"warehouse_surplus_after_current_units\":"<<amount<<",\"original_now\":"<<p.minimum_now<<",\"deadline\":"<<end<<",\"market_inventory\":"<<p.stock<<",\"completed_observed_sale_days\":"<<observed_days<<",\"traffic_not_known_inventory\":[";
   for(int t=0;t<p.steps;t++){if(t)log<<",";log<<p.rival[t];}log<<"],\"known_consumption\":[";
   for(int t=0;t<p.steps;t++){if(t)log<<",";log<<p.demand[t];}log<<"],\"profiles\":[";
   for(size_t n=0;n<result.alternatives.size();n++){
    if(n)log<<",";log<<"{\"quantity_by_tick\":[";
    for(int t=0;t<p.steps;t++){if(t)log<<",";log<<result.alternatives[n][t];}
    log<<"],\"conditional_cash\":[";
    for(int s=0;s<4;s++){if(s)log<<",";log<<"["<<result.cash[n][s].own<<","<<result.cash[n][s].rival<<"]";}
    log<<"],\"minimum_conditional_rank\":"<<result.worst[n]<<"}";
   }
   log<<"],\"selected_profile\":"<<result.index<<",\"minimum_rank_gain\":"<<result.gain<<"}";
   const int now=result.chosen[0];
   if(now>sell[i]){changed++;advanced_units+=now-sell[i];sell[i]=now;if(now>=amount)deadline[i]=-1;}
  }
  log<<"]}";if(any)last_json=log.str();
 }
};
}
