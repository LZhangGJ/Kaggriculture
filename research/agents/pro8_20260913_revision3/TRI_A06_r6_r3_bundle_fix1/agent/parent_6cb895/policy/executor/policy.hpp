#pragma once
#ifndef R2_FINITE_FERTILIZER
#define R2_FINITE_FERTILIZER 0
#endif
static_assert(R2_FINITE_FERTILIZER==0||R2_FINITE_FERTILIZER==1);
// Native v7 policy. Only public farms, own private state and current market are
// visible here. No Simulator/seed/opponent-private access enters the policy API.
#include "vendor/simulator.hpp"
#include "observation_view.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <functional>
#include <numeric>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace dp7 {
using namespace fastkag;
using Counts=std::array<int,12>;
using Quantities=std::array<double,12>;
using Acts=std::vector<Action>;
constexpr int W=0,C=1,T=2,S=3,M=4,E=5,MI=6,WO=7,F=8,G=9,CO=10,SH=11;
constexpr std::array<int,5> seed_price{10,20,50,100,80}, first{2,2,8,10,10}, interval{0,0,1,2,0}, harvest_age{4,3,8,10,10};
constexpr std::array<int,3> animal_price{300,400,500}, afirst{4,8,6}, ainterval{1,2,3}, held{4,6,6}, product{E,MI,WO};
constexpr std::array<int,16> fib{1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987};
constexpr std::array<int,4> depot{44,45,54,55};
inline int sum(const Counts&a){return std::accumulate(a.begin(),a.end(),0);}
inline int quad(int c){return (c/10>=5?2:0)+(c%10>=5?1:0);}
inline int dist(int a,int b){return std::abs(a%10-b%10)+std::abs(a/10-b/10);}
inline int near(int a){int r=100;for(int d:depot)r=std::min(r,dist(a,d));return r;}
inline bool at_depot(int a){return std::find(depot.begin(),depot.end(),a)!=depot.end();}
inline int cell(Position p){return p.y*10+p.x;}
inline auto snake(int c){int x=c%10,y=c/10,q=quad(c);int row=q<2?4-y:y-5;int col=q%2==0?(row%2==0?4-x:x):(row%2==0?x-5:9-x);return std::tuple(q,row,col);}
inline bool plant(const Tile&t){return t.kind==TileKind::PLANT;}
inline bool animal(const Tile&t){return t.animal!=Item::NONE;}
inline bool ongoing(int c){return c==T||c==S;}
// Official incremental prices, indexed by currently owned quadrant count.
inline double next_land_cost(int owned){return owned==1?1000.:owned==2?2000.:owned==3?4000.:INFINITY;}
inline bool output(const Counts&a){return std::any_of(a.begin(),a.end(),[](int x){return x!=0;});}
inline std::string name(int i){return item_name(i);}
inline Action action(Op op,int item=-1,int n=1){return {op,Item(item),n};}
inline void add(Counts&a,const Counts&b){for(int i=0;i<12;i++)a[i]+=b[i];}
inline double shape(int k,double x,double scale){x=std::max(0.,x);switch(k){case 0:return x;case 1:return x*x;case 2:return std::sqrt(x);case 3:return std::log1p(x);default:{double u=x/scale;return u+8*std::pow(std::max(0.,u-1),2);}}}
inline int price(int item,double inventory){
 static constexpr double base[]{25,35,60,120,250,50,160,200,100},scale[]{400,450,200,100,300,332,122,105,200},below[]{.8,1,.4,.7,.2,.4,.6,.2,.4},above[]{.2,.7,.6,1.6,3.6,.2,1.6,3.2,.4};
 static constexpr int bf[]{2,4,4,2,3,4,2,3,0},af[]{3,2,2,0,1,3,0,1,0};
 const bool low=inventory<10000;int f=low?bf[item]:af[item];double v=base[item]+(low?1:-1)*(low?below[item]:above[item])*base[item]/shape(f,scale[item],scale[item])*shape(f,std::abs(inventory-10000),scale[item]);return std::max(1,int(std::nearbyint(v)));
}
inline double revenue(int item,double inv,int q){int n=int(std::nearbyint(inv));double out=0;for(int i=0;i<q;i++)out+=price(item,n++);return out;}
struct Params {
 bool frequent_harvest=false;int frequent_threshold=1;
 bool triad_tour_dp=false;
 bool triad_batch_delivery=false;
 double triad_delivery_pressure=0.;
 bool compile_consequence=false,compile_replant_choices=false;
 bool compile_bounded_rollout=false;
 bool exact_schedule_cache=false,incremental_regret_cost=false;
 bool continuous_market_execution=false,insert_missing_feed=false;
 bool funded_labor=false; // Daily staffing is an observed commitment, not a consumed queue token.
 bool fix_resources=true,fix_expiry=true,fix_values=true,fix_calendar=true,fix_liquidity=false,fix_logistics=false;
 bool capacity_hauling=false;
 bool insertion_hire_estimate=false;
 bool reconcile_seed_drift=false;
 bool plan_zero_expiry=false;
 int cashflow_value_mode=0; // 0 legacy; 1 local; 2 own portfolio; 3 payoff margin.
 int funded_bundle_mode=0; // 0 legacy; 1 repair unstarted proposals; 2 compare all.
 bool preparation_work=false;
 bool split_service_jobs=false,regret_schedule=false;
 // `regret_schedule` is the frozen S3S compatibility switch: it enables both
 // contexts.  The two switches below isolate whether the same general
 // scheduler is used to size the workforce or to compile the actual routes.
 bool regret_compile=false,regret_hire_estimate=false;
 bool terminal_deposit_schedule=false;
 bool overflow_idle_dispatch=false;
 bool capital_time_rank=false;
 // S4A capabilities are deliberately independent switches.  They must remain
 // off unless their own mechanism and multi-seed acceptance gates pass.
 bool shared_task_atoms_v2=false;
 bool stepwise_recoordination=false;
 bool preparation_pipeline_v2=false;
 bool schedule_value_compare=false;
 bool preparation_spawn_guard=false;
 bool net_feed_buffer=false;
 bool midroute_delivery=false;
 bool resource_aware_exchange=false;
 bool intraday_admission=false,intraday_procurement=false,intraday_declared_value=false;
 bool autonomous_start=false; // Same public-state investment selector on day zero.
 bool joint_investment_portfolio=false;
 bool portfolio_crop_calendar=false;
 bool recover_service_inputs=false,procure_service_inputs=false,finance_service_inputs=false;
 bool incremental_pickup_repair=false;
 bool shared_service_insertions=false;
 bool intraday_future_workforce=false;
 bool day_consequence_compare=false,day_value_public_supply=false;
 bool day_value_replan_next_day=false;
 bool idle_task_handoff=false;
 int max_land=3,max_animals=18,max_cows=14,max_sheep=12,max_geese=8,max_strawberry=40,max_tomato=16,max_melon=15;
 std::vector<int> opening_animals{CO,CO,SH,SH};int opening_melon=8,opening_strawberry=4,latest_animal_day=15;
 std::array<int,5> opening_crops{-1,-1,-1,-1,-1};
 std::array<int,5> opening_crop_order{M,S,T,C,W};
 double timing_discount=.84,action_shadow=2.,feed_price_mult=.95,capital_fraction=.94,future_shop_weight=1.;
 double opponent_supply_weight=0.,own_feed_demand_weight=0.,temporal_value_weight=0.;
 double committed_feed_weight=0.;
 double existing_replant_weight=0.;
 double hold_opponent_supply_weight=0.;
 double competition_objective_weight=0.;
 double competitive_sell_weight=0.;int sell_horizon_days=3;double sell_advantage=0.;
 bool rotate_finite=false;double rotation_margin=0.;
 bool portfolio_rotation=false,rotation_timing=false;
 bool economic_land=false,finite_fertilizer=false,fix_finite_projection=false;
 int early_deposit=0;
 bool sell_deposits=false;
 bool efficient_water=false,efficient_care=false;
 bool renew_ongoing=false;
 double operating_reserve=450.,land_margin=0.;
 std::array<int,5>crop_harvest_age{4,3,8,10,10};
 int hold_capacity=100,shed_safety=1,max_hands=14,feed_cover_days=7,feed_stock_cap=60;
 int feed_forecast_days=0;
 int force_min_cows=0,force_min_sheep=0,force_min_strawberry=0,force_min_melon=0;
 Quantities bias{.78,.78,.9,1,1,1,1,1,1,.9,1,1};
};
struct Project{int kind,item,out=0,fert=0,feed=0,seed=0,capital=0;double actions=0;};
struct Job{int pos;Acts actions;Counts needs{},seeds{},out{};int priority=20;bool crop=false;};
struct Route {
 int unit,start,end,cost=0;std::vector<Job>jobs;Counts needs{};bool has_output=false;
 Route(int u,int s):unit(u),start(s),end(s){}
 int total(bool ret)const{return cost+(ret&&has_output?near(end)+1:0);}
 int append_cost(const Job&j,bool ret)const{int c=cost+dist(end,j.pos)+int(j.actions.size());for(int k=0;k<12;k++)c+=(j.needs[k]>0&&needs[k]<=0);return c+(ret&&(has_output||output(j.out))?near(j.pos)+1:0);}
 void append(const Job&j){cost=append_cost(j,false);add(needs,j.needs);jobs.push_back(j);end=j.pos;has_output|=output(j.out);}
 int ordered(const std::vector<Job>&js,bool ret)const{Counts ns{};int c=0,pos=start;bool out=false;for(auto&j:js){add(ns,j.needs);c+=dist(pos,j.pos)+int(j.actions.size());pos=j.pos;out|=output(j.out);}for(int v:ns)c+=v>0;return c+(ret&&out?near(pos)+1:0);}
};
}
#include "pack_memo.hpp"
namespace dp7 {

// Bounded exact Held-Karp tour for a prescribed job set. This solves only a
// worker subproblem; fleet assignment and economic planning remain heuristic.
inline Route triad_exact_tour(const Route&old,bool ret){
 int n=int(old.jobs.size());if(n<2||n>9)return old;int m=1<<n;
 std::vector<int>dp(m*n,100000),parent(m*n,-1);
 int work=0,pick=0;for(auto&j:old.jobs)work+=j.actions.size();for(int q:old.needs)pick+=q>0;
 for(int j=0;j<n;j++)dp[(1<<j)*n+j]=dist(old.start,old.jobs[j].pos);
 for(int mask=1;mask<m;mask++)for(int j=0;j<n;j++)if(mask&(1<<j)){
  int t=dp[mask*n+j];if(t>=100000)continue;
  for(int k=0;k<n;k++)if(!(mask&(1<<k))){int next=mask|(1<<k),nt=t+dist(old.jobs[j].pos,old.jobs[k].pos);
   if(nt<dp[next*n+k]){dp[next*n+k]=nt;parent[next*n+k]=j;}
  }
 }
 int last=-1,best=old.total(ret)-work-pick;
 for(int j=0;j<n;j++){int v=dp[(m-1)*n+j]+(ret&&old.has_output?near(old.jobs[j].pos)+1:0);if(v<best){best=v;last=j;}}
 if(last<0)return old;
 std::vector<int>order;int mask=m-1;while(last>=0){order.push_back(last);int p=parent[mask*n+last];mask^=1<<last;last=p;}
 std::reverse(order.begin(),order.end());Route r(old.unit,old.start);for(int i:order)r.append(old.jobs[i]);return r;
}

#include "a06_fleet_dp.hpp"

struct Order{Action a;int priority=9;};
struct Plan{Acts a;std::vector<int>target;size_t index=0;};
struct PendingAdmission {bool active=false;int day=-1,step=-1,unit=-1,pos=-1,kind=-1;};
// Non-owning audit output; never a policy input and not exposed as a parameter.
// The C++ audit runner owns the destination for one act() call, then clears it.
struct AdmissionInspection {
 bool seen=false,base_valid=false,alternative_valid=false;
 int step=-1,base_kind=-1,alternative_kind=-1,base_unit=-1,alternative_unit=-1,base_pos=-1,alternative_pos=-1;
 int duplicates=0,late=0,feed=0;Counts planned{},selected{};
 double base_value=0,alternative_value=0,same_kind_before=0,same_kind_after=0;
};
class Controller;
namespace exchange {void apply(Controller&,const View&);}
namespace intraday {void complete_pending(Controller&,const View&);void consider(Controller&,const View&,PlayerAction&);}
namespace service {void recover(Controller&,const View&);void procure(Controller&,const View&,PlayerAction&);void protect(const Controller&,const View&,PlayerAction&);}
namespace dayvalue {bool accept(Controller&,const View&,const std::vector<Plan>&,const std::vector<Plan>&);}
namespace handoff {void apply(Controller&,const View&,PlayerAction&);}
namespace liverepair {void market(Controller&,const View&,PlayerAction&);void feed(Controller&,const View&);}
namespace compilechoice {void compare(Controller&,const View&);}
namespace labor {void observe(Controller&,const View&);void procure(Controller&,const View&,PlayerAction&);}
class Controller {
 public:
 a06::FleetStats a06_stats;
 bool resume_compiled_tick=false; // Scenario continuation after this tick's compile, never a policy choice.
 // Optional shared C3 valuation. It sees only this controller and the public View.
 std::function<std::array<double,12>(const Controller&,const View&,const PlayerAction&)> admission_values;
 double admission_blend=0.,admission_margin=0.;int admission_scope=1;
 int crop_service_day=-1;
 std::array<int,100>crop_birth{},crop_kind{};
 std::array<int8_t,100>crop_water{},crop_fertilize{};
 int animal_service_day=-1;std::array<int8_t,100>service_feed{},service_care{};
 std::array<int,100>triad_crop_age=[](){std::array<int,100>a;a.fill(-1);return a;}();
 Params p;int day=-1,last_step=-1,phase=0,planned_land=1,feed_stock_target=0,resource_degraded=0,actual_drop=0,liquidity_recoveries=0,unresolved_overflow=0;double proposal_value=0.;
 std::shared_ptr<packmemo::Cache> schedule_cache;int schedule_cache_day=-1;
 std::vector<std::pair<int,int>>target;Acts queue;std::vector<Plan>plans;Counts expected{},daily_need{};
 Counts prepared_seed_need{};
 int labor_target=0,labor_reordered=0,labor_checks=0,labor_offers=0,labor_arrivals=0,labor_jobs=0;
 double labor_last_utility=0;
 // Project intent only, never a worker-owned sequence. Zero means no wait.
 std::array<int,100>plant_not_before{};
 std::array<int,100>deferred_kind{};
 std::array<int,100>deferred_since{};
 int rotation_generated=0,rotation_evaluated=0,rotation_applied=0,rotation_waits=0,rotation_unknown=0;
 int compile_choice_calls=0,compile_choice_candidates=0,compile_choice_evaluations=0,compile_choice_changes=0,compile_choice_unknown=0,compile_choice_unsafe=0;
 bool seed_reconcile_checked=false;
 int seed_reconciliations=0,seed_reconciled_units=0;
 int anticipated_releases=0;
 int bundle_evaluations=0,bundle_switches=0,bundle_removed_targets=0;
 int portfolio_generated=0,portfolio_evaluated=0,portfolio_switches=0,portfolio_rejected=0;
 int preparation_actions=0;
 int shared_service_plots=0;
 mutable int regret_trials=0,regret_improvements=0;
 int terminal_schedule_evaluations=0,terminal_schedule_switches=0;
 double terminal_expected_cash_gain=0.;
 int overflow_dispatch_checks=0,overflow_dispatch_units=0,overflow_dispatch_quantity=0;
 int step_recoord_checks=0,step_recoord_rebuilds=0,step_recoord_stale_groups=0,step_recoord_added_groups=0,step_recoord_reassigned_groups=0;
 int shared_insertion_checks=0,shared_insertion_applied=0,shared_insertion_peak_saved=0,shared_insertion_steps_saved=0;
 int preparation_pipeline_checks=0,preparation_pipeline_actions=0,preparation_pipeline_moves=0;
 mutable int preparation_spawn_guard_blocks=0;
 int schedule_value_evaluations=0,schedule_value_switches=0;
 double schedule_value_gain=0.;
 int midroute_delivery_checks=0,midroute_delivery_insertions=0,midroute_delivery_quantity=0;
 int resource_exchange_checks=0,resource_exchange_pairs=0,resource_exchange_applied=0,resource_exchange_local_steps_saved=0;
 PendingAdmission pending_admission;std::set<int>intraday_units;
 AdmissionInspection* admission_inspection=nullptr;
 int intraday_checks=0,intraday_proposals=0,intraday_activated=0,intraday_cancelled=0,intraday_unfilled=0,intraday_purchase_orders=0;
 Counts intraday_activated_by_kind{};
 mutable int intraday_workforce_checks=0,intraday_workforce_candidates=0,intraday_workforce_switches=0,intraday_workforce_unknown=0;
 int day_value_checks=0,day_value_rejected=0,day_value_unknown=0;
 int idle_handoff_checks=0,idle_handoff_applied=0,idle_handoff_prefix=0;
 int service_checks=0,service_recovered_feed=0,service_recovered_fertilize=0,service_buys=0,service_financed=0,service_deposits=0;
 int live_market_checks=0,live_market_added=0,live_market_slots_blocked=0;
 int feed_insert_checks=0,feed_insert_applied=0,feed_insert_unresolved=0;
 explicit Controller(Params pars={}):p(std::move(pars)){deferred_kind.fill(-1);}
 int h_age(int c)const{return p.crop_harvest_age[c];}
 int finite_yield(int c,int age,bool fertile=false)const{int maxday=c==W?4:c==C?3:12;return std::min(c==C?4:6,1+std::max(0,std::min(age,maxday)-(maxday+1)/2+1)*(fertile?2:1));}
 Quantities demand(const View&o)const{
  static const std::array<std::vector<int>,8>products{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C},{MI,T,W},{S,MI},{WO}}};
  // Shop enum uses the same frozen official shop order (verified by parity).
  Quantities d{},per{};int st=0,ct=0;for(int s=o.step;s<719;s++){st+=s%4==0;ct+=s%24==0;}for(int x=0;x<8;x++)d[x]+=ct;
  for(int sh:o.shops){auto&pr=products.at(sh);for(int x:pr)d[x]+=(pr.size()==1?2:1)*st;}
  for(auto&pr:products)for(int x:pr)per[x]+=(pr.size()==1?2.:1.)/8.;
  for(int i=int(o.shops.size());i<8;i++){int ticks=0;for(int s=std::max(o.step,(i+1)*3*24);s<719;s++)ticks+=s%4==0;for(int x=0;x<9;x++)d[x]+=p.future_shop_weight*per[x]*ticks;}return d;
 }
 Project crop_project(int c)const{
  Project r{c,c};r.capital=seed_price[c];
  if(ongoing(c)){int pd=day;while(pd+first[c]<=29){std::vector<int>ev;for(int i=0;i<4;i++)if(pd+first[c]+i*interval[c]<=29)ev.push_back(pd+first[c]+i*interval[c]);if(ev.empty())break;r.seed+=seed_price[c];r.out+=2*ev.size();r.actions+=2+std::max(0,ev.back()-pd-1);for(size_t i=0;i<ev.size();){int cover=ev[i]-1+2;r.fert++;r.actions++;i++;while(i<ev.size()&&ev[i]-1<=cover)i++;}r.actions+=std::ceil(ev.size()/2.);pd+=first[c]+interval[c]*3+(p.renew_ongoing?0:1);}}
  else{int cycles=std::max(0,((p.fix_calendar?29:28)-day)/h_age(c));r.out=cycles*finite_yield(c,h_age(c));r.seed=cycles*seed_price[c];r.actions=cycles*(h_age(c)+2);}return r;
 }
 Project animal_project(int a)const{int k=a-9;Project r{a,product[k]};r.capital=animal_price[k];if(day+afirst[k]>29)return r;int n=std::max(0,((p.fix_calendar?29:28)-(day+afirst[k]))/ainterval[k]);r.out=held[k]+n*(1+ainterval[k]);r.feed=r.fert=std::max(0,29-day);r.actions=4+r.feed*3+n+1;return r;}
 // Additional supply AFTER the visible crop's current lifecycle. Conditional
 // on keeping that industry and fulfilling future seed/labour obligations.
 // No new land, unseen rival investment or true future event is assumed.
 Counts replant_supply(const View&o)const{
  Counts extra{};int last=p.fix_calendar?29:28;
  for(auto&t:o.own.tiles){if(!plant(t))continue;int c=int(t.crop);
   if(ongoing(c)){
    int period=first[c]+3*interval[c]+(p.renew_ongoing?0:1);
    for(int pd=std::max(day,t.planted_day+period);pd+first[c]<=last;pd+=period){
     int fertile=-1;
     for(int n=0;n<4;n++){int d=pd+first[c]+n*interval[c];if(d>last)break;extra[c]+=2;
      if(fertile<d-1){extra[F]--;fertile=d+1;}}
    }
   }else{
    int pd=std::max(day,t.planted_day+h_age(c));
    for(;pd+h_age(c)<=last;pd+=h_age(c))extra[c]+=finite_yield(c,h_age(c));
   }
  }return extra;
 }
 Counts existing(const View&o)const{const auto&harvest_age=p.crop_harvest_age;Counts q=o.priv.shed;for(auto&inv:o.priv.inventories)add(q,inv);int ongoing_n=0;for(auto&t:o.own.tiles){if(animal(t)){int k=int(t.animal)-9;q[product[k]]+=t.yield_units;for(int d=day+1;d<30;d++){int ds=d-t.placed_day-afirst[k];if(ds>=0&&ds%ainterval[k]==0)q[product[k]]+=1+ainterval[k];}q[F]+=std::max(0,29-day);}else if(plant(t)){int c=int(t.crop);q[c]+=t.yield_units;if(ongoing(c)){ongoing_n++;for(int d=day+1;d<30;d++){int ds=d-t.planted_day-first[c];if(ds>=0&&ds%interval[c]==0&&ds/interval[c]<4)q[c]+=2;}}else if(day+std::max(0,harvest_age[c]-(day-t.planted_day))<=28)q[c]+=std::max(0,finite_yield(c,harvest_age[c])-t.yield_units);}}q[F]=std::max(0,q[F]-ongoing_n*2);
  if(p.existing_replant_weight>0){auto extra=replant_supply(o);for(int i=0;i<9;i++)q[i]+=int(std::nearbyint(p.existing_replant_weight*extra[i]));}
  return q;
 }
 Counts counts(const View&o)const{Counts c{};for(auto&t:o.own.tiles)if(animal(t))c[int(t.animal)]++;else if(plant(t))c[int(t.crop)]++;return c;}
 Quantities opponent_supply(const View&o)const{
  const auto&harvest_age=p.crop_harvest_age;
  // Conditional forecast from visible assets, not an observation of hidden
  // stock or future actions. Living animals consume some of the visible wheat.
  Quantities q{};int feed=0,fert_need=0;
  for(auto&t:o.opponent.tiles){if(animal(t)){int k=int(t.animal)-9;q[product[k]]+=t.yield_units;for(int d=day+1;d<30;d++){int ds=d-t.placed_day-afirst[k];if(ds>=0&&ds%ainterval[k]==0)q[product[k]]+=1+ainterval[k];}q[F]+=std::max(0,29-day);feed+=std::max(0,29-day);}
   else if(plant(t)){int c=int(t.crop);q[c]+=t.yield_units;if(ongoing(c)){for(int d=day+1;d<30;d++){int ds=d-t.planted_day-first[c];if(ds>=0&&ds%interval[c]==0&&ds/interval[c]<4)q[c]+=2;}fert_need+=2;}else if(day+std::max(0,harvest_age[c]-(day-t.planted_day))<=29)q[c]+=std::max(0,finite_yield(c,harvest_age[c])-t.yield_units);}}
  q[W]-=feed; // Negative supply is additional feed demand, not a hidden asset.
 q[F]=std::max(0.,q[F]-fert_need);return q;
 }
 Quantities holding_demand(const View&o)const{
  // A conditional price forecast used only to rank inventory liquidation.
  // Separate from investment risk and short-window proactive selling; zero
  // retains the previous policy exactly. No hidden inventories are available.
  auto d=demand(o);
  if(p.hold_opponent_supply_weight!=0){auto supply=opponent_supply(o);for(int i=0;i<9;i++)d[i]-=p.hold_opponent_supply_weight*supply[i];}
  return d;
 }
 Counts competition_sales(const View&o)const{
  const auto&harvest_age=p.crop_harvest_age;
  Counts sales{};if(p.competitive_sell_weight<=0||day>=29)return sales;
  // Roll only public production calendars over a short window. No current
  // opponent inventory, route identity or actual future shop draw is available.
  int end=std::min(719,o.step+24*p.sell_horizon_days);Quantities demand{},supply{};
  static const std::array<std::vector<int>,8>products{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C},{MI,T,W},{S,MI},{WO}}};
  for(int s=o.step;s<end;s++){if(s%24==0)for(int i=0;i<8;i++)demand[i]++;if(s%4==0)for(int sh:o.shops){auto&items=products.at(sh);for(int i:items)demand[i]+=items.size()==1?2:1;}}
  for(int side=0;side<2;side++){auto&farm=side?o.opponent:o.own;for(auto&t:farm.tiles){if(animal(t)){int k=int(t.animal)-9;supply[product[k]]+=t.yield_units;for(int d=day+1;d<=std::min(29,(end-1)/24);d++){int ds=d-t.placed_day-afirst[k];if(ds>=0&&ds%ainterval[k]==0)supply[product[k]]+=1+ainterval[k];}}else if(plant(t)){int c=int(t.crop);supply[c]+=t.yield_units;if(ongoing(c)){for(int d=day+1;d<=std::min(29,(end-1)/24);d++){int ds=d-t.planted_day-first[c];if(ds>=0&&ds%interval[c]==0&&ds/interval[c]<4)supply[c]+=2;}}else if(t.planted_day+harvest_age[c]<=std::min(29,(end-1)/24))supply[c]+=std::max(0,finite_yield(c,harvest_age[c])-t.yield_units);}}}
  for(int i=1;i<8;i++){int n=o.priv.shed[i];if(n<=0)continue;double inv=o.market.inventory[i];double future_inv=inv-demand[i]+p.competitive_sell_weight*supply[i];if(revenue(i,inv,n)-revenue(i,future_inv,n)>p.sell_advantage*n)sales[i]=n;}return sales;
 }
 std::vector<std::pair<double,Project>>values(const View&o,const Counts&base,const Counts&selected,int committed_feed=0)const{
  auto d=demand(o);if(p.opponent_supply_weight!=0){auto q=opponent_supply(o);for(int i=0;i<9;i++)d[i]-=p.opponent_supply_weight*q[i];}std::vector<std::pair<double,Project>>vs;std::vector<Project>prs;for(int a=9;a<12;a++)prs.push_back(animal_project(a));for(int c=0;c<5;c++)prs.push_back(crop_project(c));
  int own_animals=0;for(auto&t:o.own.tiles)own_animals+=animal(t);d[W]+=p.own_feed_demand_weight*own_animals*std::max(0,29-day);
  // New animal commitments are not yet visible in own.tiles. Reprice later
  // candidates against their feed obligation as well as their product supply.
  // Keep it separate from `selected` output totals used by temporal calendars.
  d[W]+=p.committed_feed_weight*committed_feed;
  auto marg=[&](int i,int q){return revenue(i,o.market.inventory[i]-d[i]+base[i]+selected[i],q);};
  for(auto&pr:prs){double v=-1e9;if(pr.out>0){double gross=marg(pr.item,pr.out);if(pr.fert>0&&(!p.fix_values||pr.kind>=9))gross+=marg(F,pr.fert);double cost=pr.seed+pr.feed*std::max(12.,o.market.prices[W]*p.feed_price_mult);if(p.fix_values&&pr.kind>=9)cost+=pr.capital;if(ongoing(pr.kind))cost+=pr.fert*std::max(25.,o.market.prices[F]*.72);v=(p.timing_discount*gross-cost-p.action_shadow*pr.actions)*p.bias[pr.kind];}vs.emplace_back(v,pr);}
  if(p.temporal_value_weight>0){auto timed=temporal_values(o,prs,selected);for(size_t i=0;i<vs.size();i++)if(vs[i].second.out>0)vs[i].first=(1-p.temporal_value_weight)*vs[i].first+p.temporal_value_weight*timed[i];}
  if(p.cashflow_value_mode){auto timed=portfolio_values(o,prs,selected);for(size_t i=0;i<vs.size();i++)if(vs[i].second.out>0)vs[i].first=timed[i];}
  if(p.competition_objective_weight>0){auto rival=opponent_supply(o);auto dem=demand(o);auto net=[](int i,double inv,int q){if(q>=0)return revenue(i,inv,q);double cost=0;int stock=int(std::nearbyint(inv));for(int n=0;n<-q;n++)cost+=price(i,--stock);return -cost;};
   for(auto&[value,pr]:vs){if(pr.out<=0)continue;Counts delta{};delta[pr.item]+=pr.out;delta[F]+=pr.kind>=9?pr.fert:-pr.fert;delta[W]-=pr.feed;double impact=0;for(int i=0;i<9;i++){if(!delta[i]||rival[i]==0)continue;double inv=o.market.inventory[i]-dem[i]+base[i]+selected[i];int q=int(std::nearbyint(rival[i]));impact+=net(i,inv,q)-net(i,inv+delta[i],q);}value+=p.competition_objective_weight*impact;}}
  return vs;
 }
 using Calendar=std::array<Quantities,30>;
 struct CashSchedule{Calendar quantity{};std::array<double,30>fixed{};};
 CashSchedule project_calendar(int kind)const{
  const auto&harvest_age=p.crop_harvest_age;
  CashSchedule result;auto&out=result.quantity;
  if(kind>=9){int k=kind-9;result.fixed[day]-=animal_price[k];for(int d=day;d<29;d++){out[d][W]--;out[d+1][F]++;}for(int d=day+afirst[k];d<=29;d+=ainterval[k])out[d][product[k]]+=d==day+afirst[k]?held[k]:1+ainterval[k];}
  else if(ongoing(kind)){for(int pd=day;pd+first[kind]<=29;pd+=first[kind]+3*interval[kind]+(p.renew_ongoing?0:1)){result.fixed[pd]-=seed_price[kind];int fertile=-1;for(int n=0;n<4;n++){int d=pd+first[kind]+n*interval[kind];if(d>29)break;out[d][kind]+=2;if(fertile<d-1){out[d-1][F]--;fertile=d+1;}}}}
  else{for(int pd=day;pd+harvest_age[kind]<=29;pd+=harvest_age[kind]){result.fixed[pd]-=seed_price[kind];out[pd+harvest_age[kind]][kind]+=finite_yield(kind,harvest_age[kind]);}}return result;
 }
 std::vector<double>temporal_values(const View&o,const std::vector<Project>&prs,const Counts&selected)const{
  const auto&harvest_age=p.crop_harvest_age;
  // A public-belief cash-flow estimate, never a query of true future RNG.
  // Use separate production and fixed money schedules: crop seed costs are not
  // commodity sales. Forecasts are conditional on today's visible assets.
  Calendar flow{},dem{};static const std::array<std::vector<int>,8>products{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C},{MI,T,W},{S,MI},{WO}}};Quantities mean{};
  for(auto&pr:products)for(int i:pr)mean[i]+=(pr.size()==1?2.:1.)/8.;
  for(int step=o.step;step<719;step++){int d=step/24;if(step%24==0)for(int i=0;i<8;i++)dem[d][i]++;if(step%4==0){for(int sh:o.shops)for(int i:products[sh])dem[d][i]+=products[sh].size()==1?2:1;for(int n=int(o.shops.size());n<8;n++)if(step>=(n+1)*72)for(int i=0;i<9;i++)dem[d][i]+=p.future_shop_weight*mean[i];}}
  for(int i=0;i<9;i++){flow[day][i]+=o.priv.shed[i];for(auto&inv:o.priv.inventories)flow[day][i]+=inv[i];}
  for(int side=0;side<2;side++){auto&farm=side?o.opponent:o.own;double weight=side?p.opponent_supply_weight:1.;for(auto&t:farm.tiles){if(animal(t)){int k=int(t.animal)-9;flow[day][product[k]]+=weight*t.yield_units;for(int d=day;d<29;d++){flow[d][W]-=weight*(side?1.:p.own_feed_demand_weight);flow[d+1][F]+=weight;}for(int d=day+1;d<30;d++){int ds=d-t.placed_day-afirst[k];if(ds>=0&&ds%ainterval[k]==0)flow[d][product[k]]+=weight*(ds==0?held[k]:1+ainterval[k]);}}
   else if(plant(t)){int c=int(t.crop);if(ongoing(c)){flow[day][c]+=weight*t.yield_units;int fertile=t.fertilized_until_day;for(int d=day+1;d<30;d++){int ds=d-t.planted_day-first[c];if(ds>=0&&ds%interval[c]==0&&ds/interval[c]<4){flow[d][c]+=2*weight;if(fertile<d-1){flow[d-1][F]-=weight;fertile=d+1;}}}}else{int h=std::max(day,t.planted_day+harvest_age[c]);if(h<=29)flow[h][c]+=weight*std::max(int(t.yield_units),finite_yield(c,harvest_age[c]));}}}}
  // Selected output totals identify already proposed project counts (one
  // output commodity per project kind). Price their joint supply chronologically.
  std::vector<CashSchedule>cal;for(auto&pr:prs){auto v=project_calendar(pr.kind);double n=pr.out>0?double(selected[pr.item])/pr.out:0;for(int d=day;d<30;d++)for(int i=0;i<9;i++)flow[d][i]+=n*v.quantity[d][i];cal.push_back(v);}
  std::vector<double>values;for(size_t n=0;n<prs.size();n++){auto&pr=prs[n];if(pr.out<=0){values.push_back(-1e9);continue;}auto inv=o.market.inventory;Quantities projected{};for(int i=0;i<9;i++)projected[i]=inv[i];double money=0;for(int d=day;d<30;d++){money+=cal[n].fixed[d];for(int i=0;i<9;i++)projected[i]+=flow[d][i]-dem[d][i];for(int i=0;i<9;i++){double q=cal[n].quantity[d][i];if(q==0)continue;if(q>0){money+=p.timing_discount*revenue(i,projected[i],int(q));projected[i]+=q;}else{for(int j=0;j<int(-q);j++){projected[i]--;money-=price(i,projected[i]);}}}}
   values.push_back((money-p.action_shadow*pr.actions)*p.bias[pr.kind]);}return values;
 }
 // S3O conditional cash scenario, not an authoritative future-state rollout.
 // Only our visible/owned assets, public rival assets and expected town draws
 // enter this estimate. A hypothetical harvest is not a promised sale/fill.
 struct MarginalCash {double project=0,existing_delta=0,opponent_delta=0,capital_days=0;};
 static double projected_trade(int item,double&stock,int quantity){
  int inventory=int(std::nearbyint(stock));double cash=0;
  if(quantity>0)for(int n=0;n<quantity;n++){int quoted=price(item,inventory);cash+=quoted;if(quoted>1)inventory++;}
  else for(int n=0;n<-quantity;n++){inventory--;cash-=price(item,inventory);}
  stock=inventory;return cash;
 }
 MarginalCash marginal_cash(const Market&market,const Calendar&own,const Calendar&rival,
                           const Calendar&dem,const CashSchedule&candidate)const{
  Quantities without{},with{};for(int i=0;i<9;i++)without[i]=with[i]=market.inventory[i];
  MarginalCash result;
  for(int d=day;d<30;d++){
   result.project+=candidate.fixed[d];
   for(int i=0;i<9;i++){
    // Identical exogenous scenario in both worlds. Endogenous own supply from
    // the candidate is allowed to change later quotes, never copied from replay.
    if(p.cashflow_value_mode==3){
     // Valuate the SAME conditional public rival flow in both worlds. This
     // neither reads hidden stock nor invokes an opponent agent/actual future.
     // Keep mode-2 inventory evolution unchanged for the paired ablation.
     double r0=without[i]-dem[d][i],r1=with[i]-dem[d][i];
     int q=int(std::nearbyint(rival[d][i]));
     double c0=projected_trade(i,r0,q),c1=projected_trade(i,r1,q);
     result.opponent_delta+=(q>0?p.timing_discount:1.)*(c1-c0);
    }
    const double outside=rival[d][i]-dem[d][i];without[i]+=outside;with[i]+=outside;
    const int old_quantity=int(std::nearbyint(own[d][i]));
    double old_without=projected_trade(i,without[i],old_quantity);
    double old_with=projected_trade(i,with[i],old_quantity);
    double added=projected_trade(i,with[i],int(std::nearbyint(candidate.quantity[d][i])));
    const double old_discount=old_quantity>0?p.timing_discount:1.;
    result.existing_delta+=old_discount*(old_with-old_without);
    result.project+=(candidate.quantity[d][i]>0?p.timing_discount:1.)*added;
   }
   result.capital_days+=std::max(0.,-result.project);
  }
  return result;
 }
 struct PortfolioContext{Calendar own{},rival{},dem{};std::array<double,30>own_fixed{};};
 PortfolioContext portfolio_context(const View&o)const{
  PortfolioContext context;auto&own=context.own;auto&rival=context.rival;auto&dem=context.dem;
  static const std::array<std::vector<int>,8>products{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C},{MI,T,W},{S,MI},{WO}}};
  Quantities mean{};for(auto&pr:products)for(int i:pr)mean[i]+=(pr.size()==1?2.:1.)/8.;
  for(int step=o.step;step<719;step++){
   int d=step/24;if(step%24==0)for(int i=0;i<8;i++)dem[d][i]++;
   if(step%4==0){for(int sh:o.shops)for(int i:products[sh])dem[d][i]+=products[sh].size()==1?2:1;
    for(int n=int(o.shops.size());n<8;n++)if(step>=(n+1)*72)for(int i=0;i<9;i++)dem[d][i]+=p.future_shop_weight*mean[i];}
  }
  for(int i=0;i<9;i++){own[day][i]+=o.priv.shed[i];for(auto&inv:o.priv.inventories)own[day][i]+=inv[i];}
  // Match the prior temporal forecast conventions rather than simultaneously
  // changing feed weights, rollover assumptions, harvest ages and price impact.
  for(int side=0;side<2;side++){
   auto&flow=side?rival:own;auto&farm=side?o.opponent:o.own;double weight=side?p.opponent_supply_weight:1.;
   for(auto&t:farm.tiles){
    if(animal(t)){int k=int(t.animal)-9;flow[day][product[k]]+=weight*t.yield_units;
     for(int d=day;d<29;d++){flow[d][W]-=weight*(side?1.:p.own_feed_demand_weight);flow[d+1][F]+=weight;}
     for(int d=day+1;d<30;d++){int ds=d-t.placed_day-afirst[k];if(ds>=0&&ds%ainterval[k]==0)flow[d][product[k]]+=weight*(ds==0?held[k]:1+ainterval[k]);}
    }else if(plant(t)){int c=int(t.crop);
     if(ongoing(c)){flow[day][c]+=weight*t.yield_units;int fertile=t.fertilized_until_day;
      for(int d=day+1;d<30;d++){int ds=d-t.planted_day-first[c];if(ds>=0&&ds%interval[c]==0&&ds/interval[c]<4){flow[d][c]+=2*weight;if(fertile<d-1){flow[d-1][F]-=weight;fertile=d+1;}}}
     }else{int h=std::max(day,t.planted_day+h_age(c));if(h<=29)flow[h][c]+=weight*std::max(int(t.yield_units),finite_yield(c,h_age(c)));}
    }
   }
  }
  return context;
 }
 std::vector<double>portfolio_values(const View&o,const std::vector<Project>&prs,const Counts&selected,Quantities*capital_days=nullptr)const{
  auto context=portfolio_context(o);auto&own=context.own;auto&rival=context.rival;auto&dem=context.dem;
  std::vector<CashSchedule>cal;
  for(auto&pr:prs){auto c=project_calendar(pr.kind);double n=pr.out>0?double(selected[pr.item])/pr.out:0.;
   for(int d=day;d<30;d++)for(int i=0;i<9;i++)own[d][i]+=n*c.quantity[d][i];cal.push_back(c);}
  std::vector<double>result;
  for(size_t i=0;i<prs.size();i++){
   const auto&pr=prs[i];if(pr.out<=0){result.push_back(-1e9);continue;}
   auto cash=marginal_cash(o.market,own,rival,dem,cal[i]);
   if(capital_days)(*capital_days)[pr.kind]=cash.capital_days;
   double value=cash.project+(p.cashflow_value_mode>=2?cash.existing_delta:0.);
   if(p.cashflow_value_mode==3)value-=cash.opponent_delta;
   result.push_back((value-p.action_shadow*pr.actions)*p.bias[pr.kind]);
  }
  return result;
 }
 static double turnover_rank(double value,double capital_days,double remaining_budget){
  return value/(1.+std::max(0.,capital_days)/std::max(1.,remaining_budget));
 }
 Quantities investment_ranks(const View&o,const std::vector<std::pair<double,Project>>&vs,const Counts&selected,double remaining_budget)const{
  Quantities result{},capital_days{};std::vector<Project>projects;
  for(auto&[value,pr]:vs){result[pr.kind]=value;projects.push_back(pr);}
  if(!p.capital_time_rank)return result;
  portfolio_values(o,projects,selected,&capital_days);
  for(auto&[value,pr]:vs)result[pr.kind]=turnover_rank(value,capital_days[pr.kind],remaining_budget);
  return result;
 }
 void choose(const View&o){
  if(!p.economic_land||day==0){choose_impl(o);return;}
  int land=std::popcount(unsigned(o.own.unlocked_mask));choose_impl(o,land);if(land>=p.max_land)return;
  auto keep=target;double keep_value=proposal_value;double cost=next_land_cost(land);
  double liquid=o.own.money;for(int i=0;i<9;i++)liquid+=o.priv.shed[i]*o.market.prices[i]*.72;
  if(liquid<cost+p.operating_reserve)return;
  choose_impl(o,land+1);if(proposal_value-cost<keep_value+p.land_margin){target=std::move(keep);planned_land=land;proposal_value=keep_value;}
 }
 double investment_budget(const View&o,int planned,int animals)const{
  int land=std::popcount(unsigned(o.own.unlocked_mask));double liquid=o.own.money;
  for(int i=0;i<9;i++)liquid+=o.priv.shed[i]*o.market.prices[i]*.72;
  return std::max(0.,p.capital_fraction*liquid-(planned>land?next_land_cost(land):0.)-std::max(0,animals-o.priv.shed[W])*o.market.prices[W]-p.operating_reserve);
 }
 void choose_impl(const View&o,int land_override=0){
  proposal_value=0.;
  auto count=counts(o);int land=std::popcount(unsigned(o.own.unlocked_mask));planned_land=land;double liquid=o.own.money;for(int x=0;x<9;x++)liquid+=o.priv.shed[x]*o.market.prices[x]*.55;
  int earliest=land==1?9:land==2?12:99;if(land<p.max_land&&day>=earliest&&liquid>=next_land_cost(land)+350)planned_land++;
  if(land_override)planned_land=land_override;
  std::vector<int>cells,avail;for(int c=0;c<100;c++)if(quad(c)<planned_land)cells.push_back(c);
  std::stable_sort(cells.begin(),cells.end(),[](int a,int b){return std::tuple(near(a),quad(a),a/10,a%10)<std::tuple(near(b),quad(b),b/10,b%10);});target.clear();for(int c:cells){auto&t=o.own.tiles[c];if(animal(t))target.emplace_back(c,int(t.animal));else if(plant(t))target.emplace_back(c,int(t.crop));else avail.push_back(c);}
  auto ordered=avail;std::stable_sort(ordered.begin(),ordered.end(),[](int a,int b){return std::tuple(near(a),snake(a))<std::tuple(near(b),snake(b));});std::vector<int>chosen;
  int animals=count[G]+count[CO]+count[SH];
  if(day==0&&animals==0&&!p.autonomous_start){chosen=p.opening_animals;if(p.opening_crops[0]>=0){for(int c:p.opening_crop_order)for(int i=0;i<p.opening_crops[c];i++)chosen.push_back(c);}else{for(int i=0;i<p.opening_melon;i++)chosen.push_back(M);for(int i=0;i<p.opening_strawberry;i++)chosen.push_back(S);}}
  else{Counts base=existing(o),selected{},newcount{};double budget=investment_budget(o,planned_land,animals),spent=0;int committed_feed=0;
   Counts limits{75,75,p.max_tomato,p.max_strawberry,p.max_melon,0,0,0,0,p.max_geese,p.max_cows,p.max_sheep},floor{};floor[CO]=p.force_min_cows;floor[SH]=p.force_min_sheep;floor[S]=p.force_min_strawberry;floor[M]=p.force_min_melon;
   while(chosen.size()<avail.size()){
    auto vs=values(o,base,selected,committed_feed);auto ranks=investment_ranks(o,vs,selected,budget-spent);
    std::optional<std::pair<double,Project>>best;double best_value=0;
    for(auto[v,pr]:vs){int k=pr.kind;if(count[k]+newcount[k]>=limits[k]||spent+pr.capital>budget)continue;
     if(k>=9&&(day>p.latest_animal_day||animals+newcount[G]+newcount[CO]+newcount[SH]>=p.max_animals))continue;
     double rank=ranks[k];if(count[k]+newcount[k]<floor[k]){double bonus=k>=9?3500:1800;v+=bonus;rank+=bonus;}
     if(v<=0)continue;
     if(!best||std::tuple(rank,-pr.capital,name(k))>std::tuple(best->first,-best->second.capital,name(best->second.kind))){best={{rank,pr}};best_value=v;}
    }
    if(!best)break;auto pr=best->second;proposal_value+=best_value;chosen.push_back(pr.kind);newcount[pr.kind]++;
    spent+=pr.capital;selected[pr.item]+=pr.out;selected[F]+=(!p.fix_values||pr.kind>=9?pr.fert:-pr.fert);committed_feed+=pr.feed;
   }
   std::stable_sort(chosen.begin(),chosen.end(),[](int a,int b){auto intensity=[](int c){return c>=9?4:ongoing(c)?2:1;};return std::tuple(-intensity(a),name(a))<std::tuple(-intensity(b),name(b));});
  }
  Counts unused{};std::array<bool,100>used{};for(size_t i=0;i<chosen.size()&&i<ordered.size();i++){target.emplace_back(ordered[i],chosen[i]);used[ordered[i]]=true;}for(int c:avail)if(!used[c])target.emplace_back(c,-1);
 }
 void rotate(const View&o){
  const auto&harvest_age=p.crop_harvest_age;
  if(!p.rotate_finite||p.portfolio_rotation||day>=29)return;
  Counts base=existing(o),selected{},live=counts(o);int limit[5]{75,75,p.max_tomato,p.max_strawberry,p.max_melon};
  for(auto&[pos,want]:target){auto&t=o.own.tiles[pos];if(!plant(t)||ongoing(int(t.crop)))continue;int current=int(t.crop);bool expiry=t.yield_units>0&&t.max_lifespan_step>=0&&t.max_lifespan_step<=(day+1)*24;if(day-t.planted_day<harvest_age[current]&&!expiry)continue;
   auto vs=values(o,base,selected);double old=-1e9,best=-1e9;int choice=current;Project chosen{};for(auto[v,pr]:vs)if(pr.kind==current)old=best=v;
   for(auto[v,pr]:vs){int c=pr.kind;if(c>=5||c==current||live[c]>=limit[c]||pr.out<=0)continue;int extra=std::max(0,seed_price[c]-seed_price[current]);if(o.priv.seeds[c]<=0&&extra>o.own.money)continue;if(v>best+p.rotation_margin){best=v;choice=c;chosen=pr;}}
   if(choice!=current){want=choice;live[current]--;live[choice]++;selected[chosen.item]+=chosen.out;selected[F]-=chosen.fert;}
  }
 }
 bool fertilize_due(const Tile&t)const{int c=int(t.crop);if(!ongoing(c))return false;int ds=day+1-t.planted_day-first[c];return ds>=0&&ds%interval[c]==0&&ds/interval[c]<4&&t.fertilized_until_day<day;}
 bool water_due(const Tile&t)const{
  if(t.watered_today)return false;if(!p.efficient_water||t.consecutive_unwatered>=1)return true;
  int c=int(t.crop),age=day-t.planted_day;if(ongoing(c)){int ds=day+1-t.planted_day-first[c];return ds>=0&&ds%interval[c]==0&&ds/interval[c]<4;}
  int maxday=c==W?4:c==C?3:12;return age>=(maxday+1)/2&&age<=maxday&&t.yield_units<(c==C?4:6);
 }
 bool care_due(const Tile&t)const{
  if(t.cared_today)return false;if(!p.efficient_care)return true;
  int k=int(t.animal)-9;for(int d=day+2;d<=29;d++){int ds=d-t.placed_day-afirst[k];if(ds>=0&&ds%ainterval[k]==0)return ds>0||t.pending_care_bonus<held[k]-1;}return false;
 }
 bool finite_fertilize_due(const Tile&t,const View&o,int pos=-1)const{
  if(!p.finite_fertilizer||!plant(t)||ongoing(int(t.crop))||t.watered_today||t.fertilized_until_day>=day)return false;
#if R2_FINITE_FERTILIZER
  if(pos>=0&&crop_service_day==day&&crop_birth[pos]==t.planted_day&&crop_kind[pos]==int(t.crop))return crop_fertilize[pos]>0;
#endif
  int c=int(t.crop),age=day-t.planted_day,maxday=c==W?4:c==C?3:12,maxyield=c==C?4:6;
  if(age<(maxday+1)/2||age>std::min(maxday,h_age(c)))return false;
  int remaining=std::max(0,std::min({maxday,h_age(c),age+29-day})-age+1);
  int natural=std::min(maxyield,t.yield_units+remaining),boost=std::min(maxyield,t.yield_units+remaining+std::min(3,remaining));
  return (boost-natural)*o.market.prices[c]>o.market.prices[F]+p.action_shadow;
 }
 std::vector<Job>jobs(const View&o,bool anticipate=false)const{
  const auto&harvest_age=p.crop_harvest_age;
  std::vector<Job>js;for(auto [pos,want]:target){if(want<0)continue;Tile t=o.own.tiles[pos];if(t.kind==TileKind::LOCKED){if(!anticipate)continue;t=Tile{};}Job j;j.pos=pos;auto push=[&](Op op,int it=-1){j.actions.push_back(action(op,it));};
   if(want>=9){if(animal(t)){bool controlled=animal_service_day==day,retiring=controlled&&!service_feed[pos]&&t.consecutive_unfed>=1;int k=int(t.animal)-9;int ds=day+1-t.placed_day-afirst[k];int incoming=(ds>=0&&ds%ainterval[k]==0&&!t.fed_today)?1+t.pending_care_bonus:0;if(t.yield_units>0&&(day>=29||retiring||(p.frequent_harvest&&t.yield_units>=p.frequent_threshold)||t.yield_units+incoming>held[k]||t.yield_units>=held[k])){push(Op::HARVEST);j.out[product[k]]=t.yield_units;}if(t.fertilizer_available){push(Op::COLLECT_FERTILIZER);j.out[F]=1;}if(day<29){if(!t.fed_today&&(!controlled||service_feed[pos])){push(Op::FEED);j.needs[W]=1;}if(controlled?(service_care[pos]&&!t.cared_today):care_due(t))push(Op::CARE);}j.priority=day<29?0:-5;}
    else if(t.kind==TileKind::EMPTY||t.kind==TileKind::WEED||t.kind==TileKind::COOP||t.kind==TileKind::PASTURE){if(t.kind==TileKind::WEED){push(Op::DIG);t=Tile{};}auto st=want==G?TileKind::COOP:TileKind::PASTURE;auto build=want==G?Op::BUILD_COOP:Op::BUILD_PASTURE;if(t.kind==TileKind::EMPTY)push(build);else if(t.kind!=st){push(Op::DIG);push(build);}push(Op::PLACE,want);push(Op::FEED);push(Op::CARE);j.needs[want]=1;j.needs[W]=1;j.priority=6;}
   }else{if(animal(t))continue;j.crop=true;if(t.kind==TileKind::WEED||t.kind==TileKind::COOP||t.kind==TileKind::PASTURE){if(day>=29)continue;push(Op::DIG);t=Tile{};}int desired=want;if(plant(t)){int c=int(t.crop),y=t.yield_units;bool fdue=finite_fertilize_due(t,o,pos);if(fdue){push(Op::FERTILIZE);j.needs[F]=1;}bool expiry=p.fix_expiry&&t.max_lifespan_step>=0&&t.max_lifespan_step<=(day+1)*24;if(ongoing(c)){bool due=fertilize_due(t);if(y>0&&(day>=29||expiry||(p.frequent_harvest&&y>=p.frequent_threshold)||y>=4||(due&&y+2>4))){push(Op::HARVEST);j.out[c]=y;}if(day<29){if(due){push(Op::FERTILIZE);j.needs[F]=1;}if(water_due(t))push(Op::WATER);}desired=c;if(p.renew_ongoing&&day-t.planted_day>=first[c]+3*interval[c]&&want<5&&day+first[want]<=29){if(y>0&&!output(j.out)){push(Op::HARVEST);j.out[c]=y;}j.actions.erase(std::remove_if(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::WATER||a.op==Op::FERTILIZE;}),j.actions.end());j.needs[F]=0;push(Op::DIG);t=Tile{};desired=want;}}else if(((day-t.planted_day)>=(triad_crop_age[pos]>=0?triad_crop_age[pos]:harvest_age[c])||(expiry&&y>0))&&day<=28){if(water_due(t))push(Op::WATER);push(Op::HARVEST);if(p.fix_finite_projection){int age=day-t.planted_day,maxday=c==W?4:c==C?3:12;int bonus=!t.watered_today&&age>=(maxday+1)/2&&age<=maxday?(fdue||t.fertilized_until_day>=day?2:1):0;j.out[c]=std::min(c==C?4:6,y+bonus);}else j.out[c]=std::max(y,finite_yield(c,h_age(c)));t=Tile{};}else if(day>=29){if(y>0&&(!p.fix_finite_projection||day-t.planted_day>=first[c])){int bonus=0;if(p.fix_finite_projection&&!t.watered_today){int age=day-t.planted_day,maxday=c==W?4:c==C?3:12;if(age>=(maxday+1)/2&&age<=maxday){push(Op::WATER);bonus=fdue||t.fertilized_until_day>=day?2:1;}}push(Op::HARVEST);j.out[c]=std::min(c==C?4:6,y+bonus);}}else if(water_due(t))push(Op::WATER);}
    if(t.kind==TileKind::EMPTY&&day<29&&day+harvest_age[desired]<=(p.fix_calendar?29:28)){push(Op::PLANT,desired);push(Op::WATER);j.seeds[desired]=1;}j.priority=plant(o.own.tiles[pos])?0:12;if(output(j.out))j.priority=-1;
   }if(!j.actions.empty())js.push_back(j);
  }
  if(crop_service_day==day)for(auto&j:js){
   const auto&t=o.own.tiles[j.pos];
   if(!plant(t)||!ongoing(int(t.crop))||t.planted_day!=crop_birth[j.pos]||int(t.crop)!=crop_kind[j.pos])continue;
   bool lifecycle=false;for(auto x:j.actions)lifecycle|=x.op==Op::PLANT||x.op==Op::DIG||x.op==Op::PLACE;
   if(lifecycle||t.planted_day==day)continue;
   j.actions.erase(std::remove_if(j.actions.begin(),j.actions.end(),[&](auto x){return (x.op==Op::WATER&&crop_water[j.pos]==0)||(x.op==Op::FERTILIZE&&crop_fertilize[j.pos]==0);}),j.actions.end());
   if(crop_fertilize[j.pos]==0)j.needs[F]=0;
  }
  // A deferred investment must not erase maintenance or the old harvest.
  // PLANT and its downstream WATER are deferred, not the whole plot's work.
  for(auto&j:js)if(plant_not_before[j.pos]>day){
   auto at=std::find_if(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::PLANT;});
   if(at!=j.actions.end()){j.actions.erase(at,j.actions.end());j.seeds={};}
  }
  js.erase(std::remove_if(js.begin(),js.end(),[](const auto&j){return j.actions.empty();}),js.end());
  if(!p.split_service_jobs&&!p.shared_task_atoms_v2)return js;
  std::vector<Job>shared;
  for(auto&j:js){auto&t=o.own.tiles[j.pos];
   bool terminal_chain=std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::DIG||a.op==Op::PLANT||a.op==Op::PLACE;});
   if(terminal_chain||(!animal(t)&&!(plant(t)&&ongoing(int(t.crop))))){shared.push_back(j);continue;}
   Job collect=j,maintain=j,care=j;
   for(auto*q:{&collect,&maintain,&care}){q->actions.clear();q->needs={};q->seeds={};q->out={};}
   // Inventory accounting belongs to exactly one branch. Ongoing-crop water
   // does not immediately change its held harvest. Keep FERTILIZE before WATER.
   for(auto a:j.actions){
    if(a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER)collect.actions.push_back(a);
    else if(a.op==Op::CARE)care.actions.push_back(a);
    else maintain.actions.push_back(a);
   }
   collect.out=j.out;maintain.needs=j.needs;maintain.seeds=j.seeds;
   for(auto*q:{&collect,&maintain,&care})if(!q->actions.empty())shared.push_back(std::move(*q));
  }return shared;
 }
 Counts forecast(const std::vector<Job>&js)const{Counts q{};for(auto&j:js)add(q,j.out);return q;}
 int feed_future_price(const View&o,int today_need)const{
  if(p.feed_forecast_days<=0){auto d=demand(o);return price(W,int(o.market.inventory[W]-d[W]));}
  int end=std::min(719,o.step+24*p.feed_forecast_days),endday=std::min(28,end/24);
  double change=0;int wheat_shops=0;for(int sh:o.shops)wheat_shops+=(sh==0||sh==1||sh==2||sh==3||sh==5);
  // Unknown future shops are an expectation, not actual RNG. Five of eight
  // shop types consume one wheat per four turns.
  for(int s=o.step;s<end;s++){
   if(s%24==0)change--;
   if(s%4==0){change-=wheat_shops;for(int j=int(o.shops.size());j<8;j++)if(s>=(j+1)*72)change-=p.future_shop_weight*5./8.;}
  }
  for(int side=0;side<2;side++){
   double weight=side?p.opponent_supply_weight:1.;auto&farm=side?o.opponent:o.own;int animals=0;
   for(auto&t:farm.tiles){animals+=animal(t);if(plant(t)&&int(t.crop)==W){int ready=std::max(day,t.planted_day+h_age(W));if(ready<=endday)change+=weight*std::max(int(t.yield_units),finite_yield(W,h_age(W)));}}
   if(!side){animals=std::max(animals,today_need);int stock=o.priv.shed[W];for(auto&inv:o.priv.inventories)stock+=inv[W];change+=std::min(stock,animals*std::max(0,endday-day));}
   change-=weight*animals*std::max(0,endday-day);
  }
  return price(W,int(o.market.inventory[W]+change));
 }
 int feed_buffer_goal(const View&o,int today_need,int horizon)const{
  const int legacy=std::min(p.feed_stock_cap,today_need*horizon);
  if(!p.net_feed_buffer||horizon<=0||today_need<=0)return legacy;
  std::array<int,30>arrivals{};
  // Credit only wheat already standing on our real farm. Our current route
  // takes materials at day start and deposits harvest at day end, so even a
  // harvest today cannot finance today's pickup/FEED requirement.
  for(const auto&t:o.own.tiles)if(plant(t)&&int(t.crop)==W){
   const int harvest_day=std::max(day,int(t.planted_day)+h_age(W));
   const int available_day=harvest_day+1;
   if(available_day<30&&available_day<day+horizon){
    arrivals[available_day]+=std::max(int(t.yield_units),finite_yield(W,h_age(W)));
   }
  }
  int deficit=0,required=0;
  for(int d=day;d<std::min(29,day+horizon);d++){
   deficit+=today_need-arrivals[d];required=std::max(required,deficit);
  }
  return std::max(today_need,std::min(legacy,required));
 }
 std::vector<Order>orders(const View&o,const std::vector<Job>&js){
  Counts need{},sn{};for(auto&j:js){add(need,j.needs);add(sn,j.seeds);}std::vector<Order>out;auto push=[&](Op op,int i,int n,int pr){if(n>0)out.push_back({action(op,i,n),pr});};int land=std::popcount(unsigned(o.own.unlocked_mask));if(planned_land>land)push(Op::BUY_LAND,-1,1,1);for(int a:{CO,SH,G})push(Op::BUY_ANIMAL,a,std::max(0,need[a]-o.priv.shed[a]),2);
  push(Op::BUY_PRODUCT,W,std::max(0,need[W]-o.priv.shed[W]),0);int horizon=std::max(0,std::min(p.feed_cover_days,29-day));int goal=feed_buffer_goal(o,need[W],horizon);int future=feed_future_price(o,need[W]);if(day<22&&future>=o.market.prices[W]+5)push(Op::BUY_PRODUCT,W,std::max(0,goal-std::max(o.priv.shed[W],need[W])),4);else goal=need[W];feed_stock_target=goal;
  if(day<15)push(Op::BUY_PRODUCT,F,std::max(0,need[F]-o.priv.shed[F]),4);for(int c:{M,S,T,C,W})push(Op::BUY_SEED,c,std::max(0,sn[c]-o.priv.seeds[c]),3);daily_need=need;return out;
 }
 double cost(const View&o,const Action&a)const{int i=int(a.item);switch(a.op){case Op::BUY_LAND:return next_land_cost(std::popcount(unsigned(o.own.unlocked_mask)));case Op::BUY_ANIMAL:return animal_price[i-9]*a.quantity;case Op::BUY_SEED:return seed_price[i]*a.quantity;case Op::BUY_PRODUCT:{double v=0;int inv=o.market.inventory[i];for(int k=0;k<a.quantity;k++)v+=price(i,--inv);return v;}default:return 0;}}
 Acts sales_sorted(const View&o,const Counts&s)const{std::vector<int>ids;for(int i=0;i<9;i++)ids.push_back(i);std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return o.market.prices[a]>o.market.prices[b];});Acts out;for(int i:ids)if(s[i]>0)out.push_back(action(Op::SELL,i,s[i]));return out;}
 Acts funding(const View&o,double goal,int slots)const{Counts avail=o.priv.shed,sell{};avail[W]=std::max(0,avail[W]-std::max(daily_need[W],std::min(o.priv.shed[W],feed_stock_target)));avail[F]=std::max(0,avail[F]-daily_need[F]);double cash=o.own.money;auto inv=o.market.inventory;sell[F]=avail[F];avail[F]=0;for(int k=0;k<sell[F];k++)cash+=price(F,inv[F]++);int left=std::max(0,slots-std::max(0,100-sum(o.priv.shed)+sum(sell)));auto d=holding_demand(o);
  auto tactical=competition_sales(o);for(int x=1;x<8;x++)if(tactical[x]>0){int n=std::min(avail[x],tactical[x]);for(int k=0;k<n;k++)cash+=price(x,inv[x]+sell[x]+k);sell[x]+=n;avail[x]-=n;left=std::max(0,left-n);}
  while(cash<goal||left>0){int best=-1;std::tuple<double,int,int,std::string>bk;for(int x=0;x<9;x++)if(avail[x]>0){int now=price(x,inv[x]+sell[x]),future=price(x,inv[x]+sell[x]-d[x]);auto key=std::tuple(double(future-now),future,now,name(x));if(best<0||key<bk){best=x;bk=key;}}if(best<0)break;sell[best]++;avail[best]--;cash+=std::get<2>(bk);left=std::max(0,left-1);}return sales_sorted(o,sell);
 }
 double sale_cash(const View&o,const Acts&sales)const{auto inv=o.market.inventory;double cash=o.own.money;for(auto&a:sales)for(int k=0;k<a.quantity;k++)cash+=price(int(a.item),inv[int(a.item)]++);return cash;}
 std::vector<Order>admit(const View&o,std::vector<Order>buy,double cash)const{std::stable_sort(buy.begin(),buy.end(),[](auto&a,auto&b){auto key=[](auto&x){int i=int(x.a.item);return std::pair(x.priority,(i==W||i==M||i==S)?0:1);};return key(a)<key(b);});std::vector<Order>out;auto inv=o.market.inventory;for(auto x:buy){auto&a=x.a;int i=int(a.item);if(a.op==Op::BUY_LAND){double c=cost(o,a);if(c<=cash){out.push_back(x);cash-=c;}}else if(a.op==Op::BUY_PRODUCT){int n=0;double c=0;while(n<a.quantity){int pr=price(i,inv[i]-1);if(c+pr>cash)break;c+=pr;inv[i]--;n++;}if(n){a.quantity=n;out.push_back(x);cash-=c;}}else{Action one=a;one.quantity=1;double u=cost(o,one);int n=std::min(a.quantity,int(std::floor(cash/std::max(1.,u))));if(n>0){a.quantity=n;out.push_back(x);cash-=n*u;}}}return out;}
 std::vector<int>starts(int hands)const{std::vector<int>s{44};std::array<int,4>occ{1,0,0,0};for(int i=0;i<hands;i++){int j=std::min_element(occ.begin(),occ.end())-occ.begin();occ[j]++;s.push_back(depot[j]);}return s;}
 static int insertion_cost(const Route&r,const Job&j,int at,bool ret){
  int before=at? r.jobs[at-1].pos:r.start;
  int extra=dist(before,j.pos)+int(j.actions.size());
  if(at<int(r.jobs.size()))extra+=dist(j.pos,r.jobs[at].pos)-dist(before,r.jobs[at].pos);
  for(int i=0;i<12;i++)extra+=(j.needs[i]>0&&r.needs[i]<=0);
  int end=at==int(r.jobs.size())?j.pos:r.end;
  return r.cost+extra+(ret&&(r.has_output||output(j.out))?near(end)+1:0);
 }
 std::pair<std::vector<Route>,int>regret_pack_legacy(const std::vector<Job>&js,const std::vector<int>&starts,int budget,bool ret)const{
  std::vector<Route>rs;for(int i=0;i<int(starts.size());i++)rs.emplace_back(i,starts[i]);
  std::vector<bool>done(js.size(),false);int remaining=js.size(),dropped=0;
  while(remaining){
   int priority=100000;for(size_t i=0;i<js.size();i++)if(!done[i])priority=std::min(priority,js[i].priority);
   int selected=-1,unit=-1,index=-1;std::tuple<int,int,int,int>regret_key;
   for(size_t i=0;i<js.size();i++)if(!done[i]&&js[i].priority==priority){
    int first=100000,second=100000,best_unit=-1,best_index=-1,best_total=100000;
    for(auto&r:rs){int local=100000,local_at=-1,local_total=100000;
     for(int at=0;at<=int(r.jobs.size());at++){
      int total=insertion_cost(r,js[i],at,ret),delta=total-r.total(ret);
      if(total<=budget&&std::tuple(delta,total,at)<std::tuple(local,local_total,local_at)){local=delta;local_total=total;local_at=at;}
     }
     if(local_at<0)continue;
     if(std::tuple(local,local_total,r.unit)<std::tuple(first,best_total,best_unit)){second=first;first=local;best_total=local_total;best_unit=r.unit;best_index=local_at;}
     else second=std::min(second,local);
    }
    if(best_unit<0){done[i]=true;remaining--;dropped++;continue;}
    auto key=std::tuple(second==100000?100000:second-first,first,int(js[i].actions.size()),-int(i));
    if(selected<0||key>regret_key){selected=i;unit=best_unit;index=best_index;regret_key=key;}
   }
   if(selected<0)continue;
   auto&route=rs[unit];auto ordered=route.jobs;ordered.insert(ordered.begin()+index,js[selected]);
   Route updated(route.unit,route.start);for(auto&j:ordered)updated.append(j);route=std::move(updated);
   done[selected]=true;remaining--;
  }return {rs,dropped};
 }
 #include "regret_pack_incremental.inc"
 std::pair<std::vector<Route>,int>regret_pack(const std::vector<Job>&js,const std::vector<int>&starts,int budget,bool ret)const{
  return p.incremental_regret_cost?regret_pack_incremental(js,starts,budget,ret):regret_pack_legacy(js,starts,budget,ret);
 }
 std::pair<std::vector<Route>,int>pack(std::vector<Job>js,const std::vector<int>&starts,int budget,bool ret,bool insertion=false,bool hire_estimate=false)const{
  auto*cache=packmemo::active;
  if(!cache)return pack_uncached(std::move(js),starts,budget,ret,insertion,hire_estimate);
  cache->calls++;
  bool regret=p.regret_schedule||(hire_estimate?p.regret_hire_estimate:p.regret_compile);
  auto key=packmemo::key(js,starts,budget,ret,insertion,regret,p.incremental_regret_cost);
  auto it=cache->entries.find(key);
  if(it!=cache->entries.end()){
   cache->hits++;regret_trials+=it->second.trials;regret_improvements+=it->second.improvements;return it->second.result;
  }
  cache->misses++;int trials=regret_trials,improvements=regret_improvements;
  auto result=pack_uncached(std::move(js),starts,budget,ret,insertion,hire_estimate);
  cache->remember(std::move(key),result,regret_trials-trials,regret_improvements-improvements);return result;
 }
 std::pair<std::vector<Route>,int>pack_uncached(std::vector<Job>js,const std::vector<int>&starts,int budget,bool ret,bool insertion=false,bool hire_estimate=false)const{
  std::stable_sort(js.begin(),js.end(),[&](auto&a,auto&b){if(insertion)return std::tuple(a.priority,-int(a.actions.size()),snake(a.pos))<std::tuple(b.priority,-int(b.actions.size()),snake(b.pos));return std::tuple(a.priority,snake(a.pos),-int(a.actions.size()))<std::tuple(b.priority,snake(b.pos),-int(b.actions.size()));});std::vector<Route>rs;for(int i=0;i<int(starts.size());i++)rs.emplace_back(i,starts[i]);int dropped=0;
  // Keep each route's cached cost/needs/end exact after insertion. Testing an
  // insertion then changes only its two adjacent edges and new pickup types;
  // no copy and full route traversal is needed for every candidate position.
  for(auto&j:js){int best=-1,at=-1;std::tuple<int,int,int,int,int>bk;for(auto&r:rs){int old=r.total(ret);int begin=insertion?0:int(r.jobs.size());for(int k=begin;k<=int(r.jobs.size());k++){int c=insertion?insertion_cost(r,j,k,ret):r.append_cost(j,ret);if(c<=budget){auto key=std::tuple(c-old,c,int(r.jobs.size()),r.unit,k);if(best<0||key<bk){best=r.unit;at=k;bk=key;}}}}if(best<0){dropped++;continue;}if(insertion){auto ordered=rs[best].jobs;ordered.insert(ordered.begin()+at,j);Route updated(rs[best].unit,rs[best].start);for(const auto&task:ordered)updated.append(task);rs[best]=std::move(updated);}else rs[best].append(j);}
  bool use_regret=p.regret_schedule||(hire_estimate?p.regret_hire_estimate:p.regret_compile);
  if(use_regret){regret_trials++;auto alt=regret_pack(js,starts,budget,ret);
   auto score=[&](const std::vector<Route>&routes){int cost=0,peak=0;for(auto&r:routes){cost+=r.total(ret);peak=std::max(peak,r.total(ret));}return std::pair(cost,peak);};
   if(alt.second==0&&(dropped>0||score(alt.first)<score(rs))){regret_improvements++;return alt;}
  }return {rs,dropped};
 }
 double job_consequence(const View&o,const Job&j)const{
  double value=0.;for(int i=0;i<9;i++)if(j.out[i]>0)value+=revenue(i,o.market.inventory[i],j.out[i]);
  const auto&t=o.own.tiles[j.pos];for(auto&a:j.actions){int item=int(a.item);
   if(a.op==Op::FEED&&animal(t)){int k=int(t.animal)-9;value+=animal_price[k]+o.market.prices[product[k]]*(1+t.yield_units+t.pending_care_bonus);}
   else if(a.op==Op::CARE&&animal(t)){int k=int(t.animal)-9;value+=o.market.prices[product[k]];}
   else if(a.op==Op::WATER&&plant(t))value+=o.market.prices[int(t.crop)]*(t.fertilized_until_day>=day?2:1);
   else if(a.op==Op::FERTILIZE&&plant(t))value+=2*o.market.prices[int(t.crop)];
   else if(a.op==Op::PLANT&&item>=0&&item<5){auto pr=crop_project(item);value+=std::max(0.,double(pr.out*o.market.prices[pr.item]-pr.seed));}
   else if(a.op==Op::PLACE&&item>=9&&item<12){auto pr=animal_project(item);value+=pr.capital+std::max(0.,double(pr.out*o.market.prices[pr.item]-pr.feed*o.market.prices[W]));}
  }
  // Priority remains a deadline signal, but it is not allowed to erase the
  // monetary comparison between two jobs in the same class.
  if(j.priority<0)value+=1000.;else if(j.priority==0)value+=500.;return std::max(1.,value);
 }
 struct ScheduleMetric{double value=0.,timed=0.;int dropped=0,total=0,peak=0;};
 ScheduleMetric schedule_metric(const View&o,const std::vector<Job>&all,const std::pair<std::vector<Route>,int>&candidate,int budget,bool ret)const{
  ScheduleMetric m;m.dropped=candidate.second;
  for(auto&r:candidate.first){int tick=0,pos=r.start;for(int q:r.needs)tick+=q>0;
   for(auto&j:r.jobs){
    // Sorting metadata is not an economic effect. Candidate packers may
    // temporarily replace priority to explore another ordering; always score
    // the authoritative input job, never that altered priority.
    auto original=std::find_if(all.begin(),all.end(),[&](const Job&x){
     if(x.pos!=j.pos||x.needs!=j.needs||x.seeds!=j.seeds||x.out!=j.out||x.actions.size()!=j.actions.size())return false;
     for(size_t k=0;k<x.actions.size();k++)if(!same_action(x.actions[k],j.actions[k]))return false;
     return true;
    });
    if(original==all.end())throw std::logic_error("schedule scored a job absent from its authoritative input");
    tick+=dist(pos,j.pos)+int(j.actions.size());pos=j.pos;double v=job_consequence(o,*original);m.value+=v;
   }
   if(ret&&r.has_output)tick+=near(pos)+1;m.total+=tick;m.peak=std::max(m.peak,tick);
  }
  return m;
 }
 std::pair<std::vector<Route>,int>value_schedule(const View&o,const std::vector<Job>&js,const std::vector<int>&ss,int budget,bool ret,std::pair<std::vector<Route>,int>baseline){
  auto plain=*this;plain.p.regret_schedule=plain.p.regret_compile=plain.p.regret_hire_estimate=false;plain.p.schedule_value_compare=false;
  std::vector<std::pair<std::vector<Route>,int>> candidates;candidates.push_back(std::move(baseline));
  candidates.push_back(plain.pack(js,ss,budget,ret,true));
  auto economic=js;for(auto&j:economic){int bucket=j.priority;long long scaled=std::llround(std::min(99999.,10.*job_consequence(o,j)));j.priority=int(std::clamp<long long>(static_cast<long long>(bucket)*100000-scaled,-2000000000LL,2000000000LL));}
  candidates.push_back(plain.pack(economic,ss,budget,ret));candidates.push_back(plain.pack(economic,ss,budget,ret,true));
  auto best_metric=schedule_metric(o,js,candidates[0],budget,ret);size_t best=0;schedule_value_evaluations+=int(candidates.size());
  // Do not perturb a complete schedule merely because a heuristic says the
  // same cash should arrive earlier. No verified financing/price consequence
  // was modelled for that timing bonus. Equal completed semantics retain KEEP.
  for(size_t i=1;i<candidates.size();i++){auto metric=schedule_metric(o,js,candidates[i],budget,ret);if(metric.value>best_metric.value+1e-6){best=i;best_metric=metric;}}
  auto base_metric=schedule_metric(o,js,candidates[0],budget,ret);if(best){schedule_value_switches++;schedule_value_gain+=best_metric.value-base_metric.value;}
  return std::move(candidates[best]);
 }
 bool hauling_needed(const std::vector<Job>&js)const{return p.capacity_hauling&&day<29&&sum(forecast(js))>100-p.shed_safety;}
 int estimate(const std::vector<Job>&js,int base)const{bool haul=hauling_needed(js);
  auto refine=[&](int h){
#if A06_EXEC_MODE >= 3
   // One bounded adjacent workforce alternative, evaluated with the complete
   // dependency jobs and its own market-order/setup latency. No task may be
   // sacrificed merely to make the Fibonacci wage estimate look smaller.
   if(h>=2){int fewer=h-1;int setup=(base+fewer+9)/10;int budget=(day>=29||haul?22:23)-std::max(1,setup)+1;
    auto probe=pack(js,starts(fewer),budget,day>=29||haul,true,true);
    if(probe.second>0&&probe.second<=3){a06::FleetStats stats;a06::fleet_dp(js,probe,budget,day>=29||haul,stats);if(probe.second==0)return fewer;}
   }
#endif
   return h;
  };
  for(int h=0;h<=p.max_hands;h++){int setup=(base+h+9)/10;int budget=(day>=29||haul?22:23)-std::max(1,setup)+1;
   if(pack(js,starts(h),budget,day>=29||haul,false,true).second==0)return refine(h);
   if(p.insertion_hire_estimate&&pack(js,starts(h),budget,day>=29||haul,true,true).second==0)return refine(h);
  }return p.max_hands;}
 static int hirecost(int n){return std::accumulate(fib.begin(),fib.begin()+n,0);}
 static int project_zero_expiry(Farm&farm,int step){
  int count=0;for(auto&t:farm.tiles)if(plant(t)&&t.yield_units==0&&t.max_lifespan_step>=0&&step>=t.max_lifespan_step&&(step-t.max_lifespan_step)%2==0){t=Tile{};t.kind=TileKind::WEED;count++;}return count;
 }
 void new_day(const View&o){if(pending_admission.active)intraday_cancelled++;pending_admission={};intraday_units.clear();day=o.day;phase=1;queue.clear();plans.clear();resource_degraded=actual_drop=0;seed_reconcile_checked=false;prepared_seed_need={};
  // The preparation turn emits no unit work. Zero-yield crops whose public
  // expiry is due will certainly disappear at its settlement. Re-plan their
  // land use before purchasing, without changing the real farm or forecasting
  // positive-yield harvests, random weeds, future shops or opponent actions.
  std::optional<Farm>projected;int released=0;
  if(p.plan_zero_expiry){projected=o.own;released=project_zero_expiry(*projected,o.step);}
  anticipated_releases+=released;
  View planning{o.step,o.day,o.hour,projected?*projected:o.own,o.opponent,o.priv,o.market,o.shops};
  choose(planning);rotate(planning);
  if(p.portfolio_rotation)for(auto&[pos,kind]:target)if(deferred_kind[pos]>=0){
   auto&t=planning.own.tiles[pos];
   if(animal(t)||(plant(t)&&int(t.crop)==deferred_kind[pos]&&t.planted_day>=deferred_since[pos]&&plant_not_before[pos]<=day)){deferred_kind[pos]=-1;plant_not_before[pos]=0;}
   else kind=deferred_kind[pos];
  }
  prepare_orders(planning,o,released);
  if(p.funded_bundle_mode)compare_funded_bundles(planning,o,released);
  if(p.joint_investment_portfolio)compare_portfolios(planning,o,released);
  if(p.portfolio_rotation)sync_deferred();
  if(p.portfolio_rotation)compare_rotation(planning,o,released);
 }
 void sync_deferred(){
  for(int pos=0;pos<100;pos++)if(deferred_kind[pos]>=0){
   auto it=std::find_if(target.begin(),target.end(),[&](auto x){return x.first==pos;});
   if(it==target.end()||it->second!=deferred_kind[pos]){deferred_kind[pos]=-1;plant_not_before[pos]=deferred_since[pos]=0;}
  }
 }
 void compare_rotation(const View&planning,const View&actual,int released);
 void prepare_orders(const View&planning,const View&o,int released){
  phase=1;queue.clear();prepared_seed_need={};
  auto js=jobs(planning,true);for(auto&j:js)add(prepared_seed_need,j.seeds);expected=forecast(js);auto buy0=orders(o,js);Acts sell;std::vector<Order>buy;int h;
  auto total=[&](const std::vector<Order>&xs){double c=0;for(auto&x:xs)c+=cost(o,x.a);return c;};auto slots=[](const std::vector<Order>&xs){int c=0;for(auto&x:xs)if(x.a.op==Op::BUY_PRODUCT||x.a.op==Op::BUY_ANIMAL)c+=x.a.quantity;return c;};
  if(day>=29){h=estimate(js,0);labor_target=h;}
  else{
   h=estimate(js,buy0.size()+2);sell=funding(o,total(buy0)+hirecost(h)+8,slots(buy0));
   double cash=sale_cash(o,sell);buy=admit(o,buy0,std::max(0.,cash-hirecost(h)-8));
   h=estimate(js,sell.size()+buy.size());labor_target=h;
   sell=funding(o,total(buy)+hirecost(h)+8,slots(buy0));cash=sale_cash(o,sell);
   buy=admit(o,buy0,std::max(0.,cash-hirecost(h)-8));
   while(h>0&&hirecost(h)+total(buy)>cash)h--;
  }
  // Today's feed remains first. Reserve the selected crew in actual order
  // before new land, animals, seeds, fertilizer and future feed buffers.
  // Identical order count: the ten-order pagination model remains applicable.
  queue=sell;
  if(p.funded_labor){
   for(const auto&x:buy)if(x.priority==0)queue.push_back(x.a);
   for(int i=0;i<h;i++)queue.push_back(action(Op::HIRE));
   for(const auto&x:buy)if(x.priority!=0)queue.push_back(x.a);
   labor_reordered+=h>0&&std::any_of(buy.begin(),buy.end(),[](const auto&x){return x.priority!=0;});
  }else{for(const auto&x:buy)queue.push_back(x.a);for(int i=0;i<h;i++)queue.push_back(action(Op::HIRE));}
  if(queue.empty()&&!released)phase=2;
 }
 // A conditional own-order projection, NOT the actual next state. It cannot
 // access opponent orders/private inventory or random events. Recompile from
 // the real observation after the preparation queue has finished.
 struct PreparedState {Farm farm;PrivateState priv;Market market;int elapsed=0;};
 PreparedState project_preparation(const View&o)const{
  PreparedState s{o.own,o.priv,o.market,phase==1?std::max(1,int((queue.size()+9)/10)):0};
  for(auto a:queue){int i=int(a.item),q=std::max(0,a.quantity);if(q<=0)continue;
   if(a.op==Op::SELL&&i>=0&&i<9){q=std::min(q,s.priv.shed[i]);for(int k=0;k<q;k++){int pr=price(i,s.market.inventory[i]);s.farm.money+=pr;s.priv.shed[i]--;if(pr>1)s.market.inventory[i]++;}}
   else if(a.op==Op::BUY_SEED&&i>=0&&i<5){int n=std::min(q,int(s.farm.money/seed_price[i]));s.priv.seeds[i]+=n;s.farm.money-=n*seed_price[i];}
   else if(a.op==Op::BUY_ANIMAL&&i>=9&&i<12){int n=std::min({q,int(s.farm.money/animal_price[i-9]),std::max(0,100-sum(s.priv.shed))});s.priv.shed[i]+=n;s.farm.money-=n*animal_price[i-9];}
   else if(a.op==Op::BUY_PRODUCT&&(i==W||i==F)){for(int k=0;k<q&&sum(s.priv.shed)<100;k++){int pr=price(i,s.market.inventory[i]-1);if(pr>s.farm.money)break;s.farm.money-=pr;s.market.inventory[i]--;s.priv.shed[i]++;}}
   else if(a.op==Op::BUY_LAND){int land=std::popcount(unsigned(s.farm.unlocked_mask));double cost=next_land_cost(land);if(cost<=s.farm.money){s.farm.money-=cost;s.farm.unlocked_mask|=1<<land;for(int pos=0;pos<100;pos++)if(quad(pos)==land&&s.farm.tiles[pos].kind==TileKind::LOCKED)s.farm.tiles[pos]=Tile{};}}
   else if(a.op==Op::HIRE&&s.farm.hires_today<int(fib.size())){int price=fib[s.farm.hires_today];if(price<=s.farm.money){s.farm.money-=price;s.farm.hires_today++;std::array<int,4>counts{};auto add_pos=[&](Position p){for(int j=0;j<4;j++)counts[j]+=cell(p)==depot[j];};add_pos(s.farm.farmer);for(auto pos:s.farm.hands)add_pos(pos);int pos=depot[std::min_element(counts.begin(),counts.end())-counts.begin()];s.farm.hands.push_back({int16_t(pos%10),int16_t(pos/10)});s.priv.inventories.push_back({});s.priv.inventory_order.push_back({});}}
  }
  for(int i=0;i<9;i++)s.market.prices[i]=price(i,s.market.inventory[i]);return s;
 }
 struct BundlePreview {double score=0;int proposed=0,started=0;std::array<unsigned,100>existing_actions{};std::vector<std::pair<int,int>>started_targets;};
 BundlePreview preview_bundle(const View&planning,bool legacy_score=true)const{
  auto prepared=project_preparation(planning);
  View v{planning.step+prepared.elapsed,day,planning.hour+prepared.elapsed,prepared.farm,planning.opponent,prepared.priv,prepared.market,planning.shops};
  auto probe=*this;auto ready=probe.reserve(v,probe.jobs(v));bool ret=day>=29||hauling_needed(ready);
  std::vector<int>ss{cell(v.own.farmer)};for(auto pos:v.own.hands)ss.push_back(cell(pos));
  int budget=(ret?22:23)-v.hour+1;auto rs=pack(ready,ss,budget,ret);
  if(rs.second){auto alt=pack(ready,ss,budget,ret,true);if(alt.second<rs.second)rs=std::move(alt);}
  BundlePreview r;std::array<bool,100>scheduled{},starts_project{};std::array<int,100>kinds;kinds.fill(-1);
  for(auto[pos,k]:target){kinds[pos]=k;if(k>=0&&plant_not_before[pos]<=day&&!plant(planning.own.tiles[pos])&&!animal(planning.own.tiles[pos]))r.proposed++;}
  for(auto&route:rs.first)for(auto&j:route.jobs){scheduled[j.pos]=true;for(auto a:j.actions)if(a.op==Op::PLANT||a.op==Op::PLACE)starts_project[j.pos]=true;if(plant(planning.own.tiles[j.pos])||animal(planning.own.tiles[j.pos]))for(auto a:j.actions)r.existing_actions[j.pos]|=1u<<int(a.op);}
  Counts base=existing(planning),selected{};int feed=0;
  for(auto[pos,k]:target){if(k<0||!scheduled[pos]||plant(planning.own.tiles[pos])||animal(planning.own.tiles[pos]))continue;
   if(starts_project[pos])r.started_targets.emplace_back(pos,k);
   if(!legacy_score){r.started+=starts_project[pos];continue;}
   for(auto[value,pr]:values(planning,base,selected,feed))if(pr.kind==k){r.score+=value;r.started++;selected[pr.item]+=pr.out;selected[F]+=k>=9?pr.fert:-pr.fert;feed+=pr.feed;break;}}
  int before=int(planning.own.hands.size()),after=int(v.own.hands.size());r.score-=hirecost(after)-hirecost(before);
  if(v.own.unlocked_mask!=planning.own.unlocked_mask)r.score-=next_land_cost(std::popcount(unsigned(planning.own.unlocked_mask)));
  return r;
 }
 void compare_funded_bundles(const View&planning,const View&actual,int released){
  if(day>=29)return;
  std::vector<int>optional;for(auto[pos,k]:target)if(k>=0&&!plant(planning.own.tiles[pos])&&!animal(planning.own.tiles[pos]))optional.push_back(pos);
  if(optional.empty())return;
  auto reference=preview_bundle(planning);bundle_evaluations++;
  if(p.funded_bundle_mode==1&&reference.started==reference.proposed)return;
  auto original=*this,best=*this;double best_score=reference.score;
  Counts base=existing(planning),none{};std::array<double,12>val{};for(auto[v,pr]:values(planning,base,none))val[pr.kind]=v;
  std::array<int,100>kind;kind.fill(-1);for(auto[pos,k]:target)kind[pos]=k;
  std::stable_sort(optional.begin(),optional.end(),[&](int a,int b){return std::tuple(-val[kind[a]],near(a),a)<std::tuple(-val[kind[b]],near(b),b);});
  int tested=1,removed=0,land=std::popcount(unsigned(actual.own.unlocked_mask));
  // The full original proposal is KEEP. Candidate count follows the number of
  // current optional projects, not a fixed replay-specific scale or date.
  for(size_t n=0;n<optional.size();n++){
   auto trial=original;std::array<bool,100>keep{};for(size_t j=0;j<n;j++)keep[optional[j]]=true;
   for(auto&[pos,k]:trial.target)if(k>=0&&!plant(planning.own.tiles[pos])&&!animal(planning.own.tiles[pos])&&!keep[pos])k=-1;
   bool uses_new_land=false;for(auto[pos,k]:trial.target)if(k>=0&&quad(pos)>=land)uses_new_land=true;
   if(!uses_new_land)trial.planned_land=land;
   trial.prepare_orders(planning,actual,released);auto pv=trial.preview_bundle(planning);tested++;
   bool preserves=true;for(int pos=0;pos<100;pos++)if((pv.existing_actions[pos]&reference.existing_actions[pos])!=reference.existing_actions[pos])preserves=false;
   if(preserves&&pv.score>best_score+1e-6){best=std::move(trial);best_score=pv.score;removed=int(optional.size()-n);}
  }
  int prior_evaluations=bundle_evaluations,prior_switches=bundle_switches,prior_removed=bundle_removed_targets;
  *this=std::move(best);bundle_evaluations=prior_evaluations+tested-1;bundle_switches=prior_switches+(removed>0);bundle_removed_targets=prior_removed+removed;
 }
 void compare_portfolios(const View&planning,const View&actual,int released);
 std::vector<Job>reserve(const View&o,std::vector<Job>js){auto shed=o.priv.shed;Counts seeds{};std::copy(o.priv.seeds.begin(),o.priv.seeds.end(),seeds.begin());std::vector<Job>out;std::stable_sort(js.begin(),js.end(),[](auto&a,auto&b){return std::tuple(a.priority,snake(a.pos))<std::tuple(b.priority,snake(b.pos));});auto fits=[](const Counts&a,const Counts&b){for(int i=0;i<12;i++)if(a[i]<b[i])return false;return true;};
  for(auto j:js){bool ok=fits(shed,j.needs)&&fits(seeds,j.seeds);auto&t=o.own.tiles[j.pos];if(!ok&&p.fix_resources&&(plant(t)||animal(t))){Acts acts;Counts ns{},sn{};bool present=plant(t);for(auto a:j.actions){int i=int(a.item);if(a.op==Op::PLANT){if(seeds[i]-sn[i]<=0)continue;sn[i]++;present=true;}else if(a.op==Op::FEED||a.op==Op::FERTILIZE){i=a.op==Op::FEED?W:F;if(shed[i]-ns[i]<=0)continue;ns[i]++;}else if(a.op==Op::DIG)present=false;else if(a.op==Op::WATER&&!present)continue;else if(a.op==Op::HARVEST&&present&&!ongoing(int(t.crop)))present=false;acts.push_back(a);}if(!acts.empty()){j.actions=acts;j.needs=ns;j.seeds=sn;ok=true;resource_degraded++;}}
   if(!ok&&j.crop&&j.needs[F]>0){Job tmp=j;tmp.needs[F]=0;tmp.actions.erase(std::remove_if(tmp.actions.begin(),tmp.actions.end(),[](auto&a){return a.op==Op::FERTILIZE;}),tmp.actions.end());if(fits(shed,tmp.needs)&&fits(seeds,tmp.seeds)){j=tmp;ok=true;}}if(!ok)continue;for(int i=0;i<12;i++){shed[i]-=j.needs[i];seeds[i]-=j.seeds[i];}out.push_back(j);
  }return out;
 }
 static void walk(Plan&pl,int&pos,int dest){while(pos%10<dest%10){pl.a.push_back(action(Op::EAST));pl.target.push_back(-1);pos++;}while(pos%10>dest%10){pl.a.push_back(action(Op::WEST));pl.target.push_back(-1);pos--;}while(pos/10<dest/10){pl.a.push_back(action(Op::SOUTH));pl.target.push_back(-1);pos+=10;}while(pos/10>dest/10){pl.a.push_back(action(Op::NORTH));pl.target.push_back(-1);pos-=10;}}
 struct TerminalPreview {Counts deposited{},stranded{},sold{};double cash=0.;};
 static TerminalPreview terminal_preview(const View&o,const std::vector<Route>&routes,int ticks){
  // Conditional logistics forecast using own visible resources only. Real
  // unit ordering and DROP-before-market capacity are retained. Unknown rival
  // orders, future shops and random weeds are NOT queried or treated as known.
  TerminalPreview result;Counts shed=o.priv.shed;
  std::vector<Counts>cargo(routes.size());std::vector<int>arrival(routes.size(),ticks);
  std::vector<std::vector<int>>pickups(routes.size());
  std::vector<bool>done(routes.size(),false);
  for(size_t u=0;u<routes.size();u++){
   const auto&r=routes[u];if(r.unit!=int(u))throw std::logic_error("terminal route unit order");
   if(u<o.priv.inventories.size())cargo[u]=o.priv.inventories[u];
   for(auto&j:r.jobs)add(cargo[u],j.out);
   for(int i=0;i<12;i++)if(r.needs[i]>0)pickups[u].push_back(i);
   std::sort(pickups[u].begin(),pickups[u].end(),[](int a,int b){return name(a)<name(b);});
   if(r.has_output)arrival[u]=r.total(true)-1;
  }
  for(int tick=0;tick<ticks;tick++){
   for(size_t u=0;u<routes.size();u++){
    if(tick<int(pickups[u].size())){int i=pickups[u][tick],need=routes[u].needs[i];
     if(!at_depot(routes[u].start)||shed[i]<need){result.cash=-INFINITY;return result;}
     shed[i]-=need;
    }
    if(!done[u]&&arrival[u]<=tick){
     // The live no-discard guard waits for enough room for the entire DROP.
     if(sum(cargo[u])<=100-sum(shed)){add(shed,cargo[u]);add(result.deposited,cargo[u]);done[u]=true;}
    }
   }
   // On the terminal day the existing market policy sells all sellable shed
   // items after every unit has acted. Unsellable animals still occupy room.
   for(int i=0;i<9;i++){result.sold[i]+=shed[i];shed[i]=0;}
  }
  for(size_t u=0;u<routes.size();u++)if(!done[u])add(result.stranded,cargo[u]);
  for(int i=0;i<9;i++)result.cash+=revenue(i,o.market.inventory[i],result.sold[i]);
  return result;
 }
 std::pair<std::vector<Route>,int>terminal_schedule(const View&o,const std::vector<Job>&js,const std::vector<int>&ss,int budget,std::pair<std::vector<Route>,int>baseline){
  auto best=std::move(baseline);double initial=terminal_preview(o,best.first,budget).cash,best_cash=initial;
  terminal_schedule_evaluations++;
  if(!std::isfinite(initial))return best; // Do not rank with an unfulfilled material forecast.
  auto plain=*this;plain.p.regret_schedule=plain.p.regret_compile=plain.p.regret_hire_estimate=false;
  // Enumerate integer completion deadlines available in this actual remaining
  // terminal window, not a source-specific calendar or a fixed worker count.
  // Shorter routes can trade low-value collection for an earlier deposit wave.
  for(int finish=budget;finish>=1;finish--)for(bool insertion:{false,true}){
   auto candidate=plain.pack(js,ss,finish,true,insertion);
   double cash=terminal_preview(o,candidate.first,budget).cash;terminal_schedule_evaluations++;
   if(cash>best_cash+1e-6){best_cash=cash;best=std::move(candidate);}
  }
  if(best_cash>initial+1e-6){terminal_schedule_switches++;terminal_expected_cash_gain+=best_cash-initial;}
  return best;
 }
 void compile_base(const View&o,int ordering=0){auto js=reserve(o,jobs(o));expected=forecast(js);bool haul=hauling_needed(js);std::vector<int>ss{cell(o.own.farmer)};for(auto s:o.own.hands)ss.push_back(cell(s));int budget=(day>=29||haul?22:23)-o.hour+1;auto rs=pack(js,ss,budget,day>=29||haul);if(rs.second){auto other=pack(js,ss,budget,day>=29||haul,true);if(other.second<rs.second)rs=std::move(other);}if(p.schedule_value_compare&&budget>0)rs=value_schedule(o,js,ss,budget,day>=29||haul,std::move(rs));if(p.terminal_deposit_schedule&&day>=29&&budget>0)rs=terminal_schedule(o,js,ss,budget,std::move(rs));
  if(ordering>0&&day<29){
   auto plain=*this;plain.p.regret_schedule=plain.p.regret_compile=plain.p.regret_hire_estimate=false;
   auto ordered=js;if(ordering>=3)for(auto&j:ordered){long long scaled=std::llround(std::min(99999.,10.*job_consequence(o,j)));j.priority=int(std::clamp<long long>(static_cast<long long>(j.priority)*100000-scaled,-2000000000LL,2000000000LL));}
   rs=plain.pack(ordered,ss,budget,haul,ordering==2||ordering==4);
  }
  // Bounded final-compilation optimization only. Never put exponential DP
  // inside the hundreds of admission/hiring previews. Atomic jobs preserve
  // their internal PICKUP/PLANT/WATER and lifecycle precedences.
  if(p.triad_tour_dp)for(auto&r:rs.first)r=triad_exact_tour(r,day>=29||haul);
#if A06_EXEC_MODE >= 1
  a06::fleet_dp(js,rs,budget,day>=29||haul,a06_stats);
#endif
  actual_drop=rs.second;plans.resize(ss.size());
  std::array<unsigned,100>owners{};for(auto&r:rs.first)for(auto&j:r.jobs)owners[j.pos]|=1u<<r.unit;
  for(auto mask:owners)shared_service_plots+=std::popcount(mask)>1;
  std::vector<bool>deposit_units(rs.first.size(),false);
  if(p.fix_logistics&&day<29){int excess=sum(expected)-100+p.shed_safety;for(auto&i:o.priv.inventories)excess+=sum(i);excess=std::max(0,excess);std::vector<std::tuple<double,int,int,int>>options;for(auto&r:rs.first){int qty=0;for(auto&j:r.jobs)qty+=sum(j.out);int extra=near(r.end)+1;if(qty>0&&r.total(true)<=budget)options.emplace_back(double(extra)/qty,extra,r.unit,qty);}std::sort(options.begin(),options.end());for(auto[ratio,extra,u,qty]:options){if(excess<=0)break;deposit_units[u]=true;excess-=qty;}unresolved_overflow=std::max(0,excess);}
#if A06_EXEC_MODE >= 2
  if(p.fix_logistics&&day<29&&!haul){int excess=sum(expected)-100+p.shed_safety;for(auto&i:o.priv.inventories)excess+=sum(i);deposit_units=a06::delivery_knapsack(rs.first,std::max(0,excess),budget);}
#endif
  for(auto&r:rs.first){Plan pl;int pos=r.start;std::vector<int>ids;for(int i=0;i<12;i++)if(r.needs[i]>0)ids.push_back(i);std::sort(ids.begin(),ids.end(),[](int a,int b){return name(a)<name(b);});for(int x:ids){pl.a.push_back(action(Op::PICKUP,x,r.needs[x]));pl.target.push_back(pos);}for(auto&j:r.jobs){walk(pl,pos,j.pos);for(auto a:j.actions){pl.a.push_back(a);pl.target.push_back(j.pos);}}if((day>=29||haul||deposit_units[r.unit])&&r.has_output){int dest=depot[0];for(int s:depot)if(dist(pos,s)<dist(pos,dest))dest=s;walk(pl,pos,dest);pl.a.push_back(action(Op::DROP));pl.target.push_back(dest);}plans[r.unit]=std::move(pl);}phase=3;liverepair::feed(*this,o);
 }
 static bool movement(Op op){return op==Op::NORTH||op==Op::SOUTH||op==Op::EAST||op==Op::WEST;}
 void compile(const View&o){compile_base(o);if(p.compile_consequence||p.compile_replant_choices)compilechoice::compare(*this,o);}
 bool semantic_needed(const View&o,int unit,const Action&a,int target_pos)const{
  if(target_pos<0)return false;
  const auto&t=o.own.tiles[target_pos];int item=int(a.item);
  const auto&inv=o.priv.inventories.at(unit);
  switch(a.op){
   // Official PICKUP(q) adds q; q is not a target backpack balance. An
   // unexecuted later pickup can be needed after an earlier FEED/FERTILIZE.
   case Op::PICKUP:return at_depot(target_pos)&&item>=0&&item<12&&a.quantity>0&&o.priv.shed[item]>0&&(p.incremental_pickup_repair||inv[item]<a.quantity);
   case Op::HARVEST:return t.yield_units>0;
   case Op::COLLECT_FERTILIZER:return animal(t)&&t.fertilizer_available;
   case Op::FEED:return animal(t)&&!t.fed_today;
   case Op::CARE:return animal(t)&&!t.cared_today;
   case Op::DIG:return t.kind!=TileKind::EMPTY&&!animal(t);
   case Op::BUILD_COOP:return t.kind==TileKind::EMPTY;
   case Op::BUILD_PASTURE:return t.kind==TileKind::EMPTY;
   case Op::PLACE:return item>=9&&item<12&&t.kind==(item==G?TileKind::COOP:TileKind::PASTURE)&&!animal(t);
   case Op::PLANT:return item>=0&&item<5&&t.kind==TileKind::EMPTY;
   case Op::WATER:return plant(t)&&!t.watered_today;
   case Op::FERTILIZE:return plant(t)&&t.fertilized_until_day<day;
   case Op::DROP:return at_depot(target_pos)&&sum(inv)>0;
   case Op::PASS:return true;
   default:return false;
  }
 }
 using TaskKey=std::tuple<int,int,int>;
 static TaskKey task_key(int pos,const Action&a){return {pos,int(a.op),int(a.item)};}
 static bool same_action(const Action&a,const Action&b){return a.op==b.op&&a.item==b.item&&a.quantity==b.quantity;}
 static bool same_remaining(const Plan&a,const Plan&b){
  size_t left=b.a.size()-std::min(b.index,b.a.size());if(a.a.size()!=left||a.target.size()!=left)return false;
  for(size_t k=0;k<left;k++)if(!same_action(a.a[k],b.a[b.index+k])||a.target[k]!=b.target[b.index+k])return false;return true;
 }
 // Rebuild movement from the observed position while preserving every still
 // valid semantic group.  A group is advanced only after its first effect is
 // already visible; later dependent actions are never filtered speculatively.
 Plan repair_plan(const View&o,int unit,const Plan&old,int&stale)const{
  struct Group{int pos=-1;Acts actions;};std::vector<Group>groups;
  for(size_t k=old.index;k<old.a.size();k++){
   if(movement(old.a[k].op)||old.target[k]<0)continue;
   if(groups.empty()||groups.back().pos!=old.target[k])groups.push_back({old.target[k],{}});
   groups.back().actions.push_back(old.a[k]);
  }
  Plan out;int pos=cell(unit?o.own.hands.at(unit-1):o.own.farmer);bool projected_cargo=sum(o.priv.inventories.at(unit))>0;
  for(auto&g:groups){size_t first_needed=0;while(first_needed<g.actions.size()){
    const auto&a=g.actions[first_needed];bool needed=a.op==Op::DROP?projected_cargo:semantic_needed(o,unit,a,g.pos);if(needed)break;first_needed++;stale++;
   }
   if(first_needed==g.actions.size())continue;
   int dest=g.pos;if(g.actions[first_needed].op==Op::PICKUP||g.actions[first_needed].op==Op::DROP){dest=depot[0];for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;}
   walk(out,pos,dest);for(size_t k=first_needed;k<g.actions.size();k++){
    const auto&a=g.actions[k];out.a.push_back(a);out.target.push_back(dest);
    if(a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER)projected_cargo=true;
    else if(p.incremental_pickup_repair&&a.op==Op::PICKUP&&int(a.item)>=0&&int(a.item)<12&&o.priv.shed[int(a.item)]>0&&a.quantity>0)projected_cargo=true;
    else if(a.op==Op::DROP)projected_cargo=false;
   }
  }
  return out;
 }
 static bool resource_free_service(const Job&j){
  if(sum(j.needs)||sum(j.seeds))return false;
  return std::all_of(j.actions.begin(),j.actions.end(),[](const Action&a){return a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER||a.op==Op::CARE||a.op==Op::WATER;});
 }
 static int plan_end(const View&o,int unit,const Plan&pl){
  int pos=cell(unit?o.own.hands.at(unit-1):o.own.farmer);for(size_t k=0;k<pl.target.size();k++)if(pl.target[k]>=0)pos=pl.target[k];return pos;
 }
 Plan strip_tasks(const View&o,int unit,const Plan&old,const std::set<TaskKey>&remove)const{
  Plan out;int pos=cell(unit?o.own.hands.at(unit-1):o.own.farmer),last=-2;
  for(size_t k=0;k<old.a.size();k++){
   if(old.target[k]<0||movement(old.a[k].op)||remove.count(task_key(old.target[k],old.a[k])))continue;
   int dest=old.target[k];if(old.a[k].op==Op::PICKUP||old.a[k].op==Op::DROP){dest=depot[0];for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;}
   if(dest!=last){walk(out,pos,dest);last=dest;}out.a.push_back(old.a[k]);out.target.push_back(dest);
  }return out;
 }
 static std::vector<std::pair<Action,int>> plan_semantics(const Plan&pl){
  std::vector<std::pair<Action,int>>v;for(size_t k=pl.index;k<pl.a.size();k++)
   if(pl.target[k]>=0&&!movement(pl.a[k].op))v.emplace_back(pl.a[k],pl.target[k]);return v;
 }
 static Plan insert_service(const View&o,int u,const Plan&pl,const Job&j,int at){
  auto old=plan_semantics(pl);Plan result;int pos=cell(u?o.own.hands[u-1]:o.own.farmer);
  for(int k=0;k<=int(old.size());k++){
   if(k==at){walk(result,pos,j.pos);for(auto a:j.actions){result.a.push_back(a);result.target.push_back(j.pos);}}
   if(k<int(old.size())){walk(result,pos,old[k].second);result.a.push_back(old[k].first);result.target.push_back(old[k].second);}
  }return result;
 }
 // Same service set, flexible insertion between the immutable material tasks.
 // No rollout, worker purchase, input transfer or economic threshold here.
 std::vector<Plan> insert_shared_services(const View&o,const std::vector<Plan>&baseline,
     const std::vector<Job>&movable,const std::set<TaskKey>&keys,int limit,int&assigned)const{
  auto result=baseline;assigned=0;
  for(size_t u=0;u<result.size();u++)result[u]=strip_tasks(o,int(u),result[u],keys);
  for(const auto&j:movable){
   if(!resource_free_service(j))throw std::logic_error("material task passed to shared service insertion");
   bool delivery_required=false;
   for(const auto&pl:baseline){bool output_seen=false;for(size_t k=0;k<pl.a.size();k++){
    if(pl.target[k]==j.pos&&(pl.a[k].op==Op::HARVEST||pl.a[k].op==Op::COLLECT_FERTILIZER)&&
       std::any_of(j.actions.begin(),j.actions.end(),[&](auto a){return same_action(a,pl.a[k]);}))output_seen=true;
    if(output_seen&&pl.a[k].op==Op::DROP)delivery_required=true;
   }}
   int best=-1,slot=-1;std::tuple<int,int,int,int>rank{100000,100000,100000,100000};
   for(size_t u=0;u<result.size();u++){
    auto sem=plan_semantics(result[u]);int start=cell(u?o.own.hands[u-1]:o.own.farmer);
    for(int at=0;at<=int(sem.size());at++){
     bool precedence=true,delivery_after=!delivery_required;
     for(int k=0;k<int(sem.size());k++){
      if(k>=at&&sem[k].first.op==Op::DROP)delivery_after=true;
      // Keep the crop's fertilizer-before-water causal dependency, not just
      // the relative order of the resource-consuming actions themselves.
      if(k>=at&&sem[k].second==j.pos&&sem[k].first.op==Op::FERTILIZE&&
         std::any_of(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::WATER;}))precedence=false;
     }
     if(!precedence||!delivery_after)continue;
     int prev=at?sem[at-1].second:start,delta=dist(prev,j.pos)+int(j.actions.size());
     if(at<int(sem.size()))delta+=dist(j.pos,sem[at].second)-dist(prev,sem[at].second);
     int total=int(result[u].a.size())+delta;auto candidate=std::tuple(total,delta,int(u),at);
     if(total<=limit&&candidate<rank){rank=candidate;best=int(u);slot=at;}
    }
   }
   if(best<0)break;
   result[best]=insert_service(o,best,result[best],j,slot);assigned++;
  }
  // Different workers also share causal crop obligations. Inserting another
  // job may delay fertilizer on one route past water on a different route.
  // Reject that complete alternative; do not rely on unit execution order.
  std::array<int,100>fert,water;fert.fill(-1);water.fill(100000);
  for(const auto&pl:result)for(size_t k=0;k<pl.a.size();k++)if(pl.target[k]>=0){
   int pos=pl.target[k];if(pl.a[k].op==Op::FERTILIZE)fert[pos]=std::max(fert[pos],int(k));
   if(pl.a[k].op==Op::WATER)water[pos]=std::min(water[pos],int(k));
  }
  for(int pos=0;pos<100;pos++)if(fert[pos]>=water[pos]){assigned=-1;break;}
  return result;
 }
 static std::pair<int,int> plan_load(const std::vector<Plan>&routes){int peak=0,total=0;for(auto&pl:routes){peak=std::max(peak,int(pl.a.size()));total+=pl.a.size();}return {peak,total};}
 // S4A-1 thin rolling layer: check every step, preserve still-valid local
 // assignments, repair paths from actual positions, and give uncovered safe
 // service work to the worker that can finish it earliest.  Resource-bearing
 // chains remain with the compiled owner until S4A-2 adds explicit transfer.
 void recoordinate(const View&o){
  if(!p.stepwise_recoordination||phase!=3)return;step_recoord_checks++;
  if(plans.size()!=1+o.own.hands.size())return;
  std::vector<Plan>next(plans.size());int stale=0;
  for(size_t u=0;u<plans.size();u++)next[u]=repair_plan(o,int(u),plans[u],stale);
  const auto retained=p.day_consequence_compare?next:std::vector<Plan>{};
  const int prior_insertions=shared_insertion_applied,prior_peak=shared_insertion_peak_saved,prior_steps=shared_insertion_steps_saved,prior_reassigned=step_recoord_reassigned_groups;
  auto fresh=jobs(o);std::stable_sort(fresh.begin(),fresh.end(),[](const Job&a,const Job&b){return std::tuple(a.priority,snake(a.pos))<std::tuple(b.priority,snake(b.pos));});
  int added=0,limit=(day>=29?22:23)-o.hour+1;
  if(p.shared_task_atoms_v2&&day<29){
   std::vector<Job>movable;std::set<TaskKey>keys;for(auto&j:fresh)if(resource_free_service(j)){movable.push_back(j);for(auto a:j.actions)keys.insert(task_key(j.pos,a));}
   auto baseline=next,candidate=next;std::set<TaskKey>baseline_keys;for(auto&pl:baseline)for(size_t k=0;k<pl.a.size();k++)if(pl.target[k]>=0&&keys.count(task_key(pl.target[k],pl.a[k])))baseline_keys.insert(task_key(pl.target[k],pl.a[k]));
   for(size_t u=0;u<candidate.size();u++)candidate[u]=strip_tasks(o,int(u),candidate[u],keys);
   int assigned=0;for(auto&j:movable){int best=-1,best_total=100000;for(size_t u=0;u<candidate.size();u++){
     int end=plan_end(o,int(u),candidate[u]);int total=int(candidate[u].a.size())+dist(end,j.pos)+int(j.actions.size());
     if(total<=limit&&std::tuple(total,int(u))<std::tuple(best_total,best<0?100000:best)){best=int(u);best_total=total;}
    }
    if(best<0)break;int pos=plan_end(o,best,candidate[best]);walk(candidate[best],pos,j.pos);for(auto a:j.actions){candidate[best].a.push_back(a);candidate[best].target.push_back(j.pos);}assigned++;
   }
   if(assigned==int(movable.size())&&(baseline_keys.size()<keys.size()||plan_load(candidate)<plan_load(baseline))){next=std::move(candidate);added=assigned;step_recoord_reassigned_groups+=assigned;}
   if(p.shared_service_insertions&&!movable.empty()){
    shared_insertion_checks++;int complete=0;
    auto inserted=insert_shared_services(o,baseline,movable,keys,limit,complete);
    std::set<TaskKey>currently_covered;for(const auto&pl:next)for(size_t k=0;k<pl.a.size();k++)
     if(pl.target[k]>=0&&keys.count(task_key(pl.target[k],pl.a[k])))currently_covered.insert(task_key(pl.target[k],pl.a[k]));
    if(complete==int(movable.size())&&(currently_covered.size()<keys.size()||plan_load(inserted)<plan_load(next))){
     auto before=plan_load(next),after=plan_load(inserted);shared_insertion_applied++;
     shared_insertion_peak_saved+=before.first-after.first;shared_insertion_steps_saved+=before.second-after.second;
     next=std::move(inserted);added=complete;step_recoord_reassigned_groups+=complete;
    }
   }
  }else{
   std::multiset<TaskKey>covered;for(auto&pl:next)for(size_t k=0;k<pl.a.size();k++)if(pl.target[k]>=0&&!movement(pl.a[k].op))covered.insert(task_key(pl.target[k],pl.a[k]));
   for(auto&j:fresh){
    Acts missing;auto available=covered;for(auto a:j.actions){auto key=task_key(j.pos,a);auto it=available.find(key);if(it!=available.end())available.erase(it);else missing.push_back(a);}if(missing.empty())continue;
    Job part=j;part.actions=missing;part.needs={};part.seeds={};part.out={};if(!resource_free_service(part))continue;
    int best=-1,best_total=100000;for(size_t u=0;u<next.size();u++){
     int end=plan_end(o,int(u),next[u]);int total=int(next[u].a.size())+dist(end,j.pos)+int(missing.size());
     if(total<=limit&&std::tuple(total,int(u))<std::tuple(best_total,best<0?100000:best)){best=int(u);best_total=total;}
    }
    if(best<0)continue;int pos=plan_end(o,best,next[best]);walk(next[best],pos,j.pos);for(auto a:missing){next[best].a.push_back(a);next[best].target.push_back(j.pos);covered.insert(task_key(j.pos,a));}added++;
   }
  }
  bool changed=stale||added;for(size_t u=0;u<plans.size()&&!changed;u++)changed=!same_remaining(next[u],plans[u]);
  if(changed&&p.day_consequence_compare&&day<29){
   bool alternate=false;for(size_t u=0;u<next.size();u++)if(!same_remaining(next[u],retained[u]))alternate=true;
   if(alternate&&!dayvalue::accept(*this,o,retained,next)){
    next=retained;added=0;shared_insertion_applied=prior_insertions;shared_insertion_peak_saved=prior_peak;
    shared_insertion_steps_saved=prior_steps;step_recoord_reassigned_groups=prior_reassigned;
   }
  }
  if(changed){plans=std::move(next);step_recoord_rebuilds++;step_recoord_stale_groups+=stale;step_recoord_added_groups+=added;}
 }
 // Conditional remaining cargo: only current held products and scheduled
 // visible yields. Not future RNG, an opponent replay, or a new production plan.
 int expected_auto_deposit(const View&o)const{
  int total=0;std::array<bool,100>harvested{},collected{};
  for(size_t u=0;u<plans.size();u++){
   Counts cargo=o.priv.inventories[u];const auto&pl=plans[u];
   for(size_t k=pl.index;k<pl.a.size()&&o.hour+int(k-pl.index)<24;k++){
    auto a=pl.a[k];int item=int(a.item),pos=pl.target[k];
    if(a.op==Op::DROP&&o.hour+int(k-pl.index)<=22){cargo={};continue;}
    if(a.op==Op::PICKUP&&item>=0)cargo[item]+=a.quantity;
    else if(a.op==Op::FEED)cargo[W]=std::max(0,cargo[W]-1);
    else if(a.op==Op::FERTILIZE)cargo[F]=std::max(0,cargo[F]-1);
    else if(a.op==Op::PLACE&&item>=0)cargo[item]=std::max(0,cargo[item]-a.quantity);
    else if(pos>=0&&a.op==Op::HARVEST&&!harvested[pos]){
     const auto&t=o.own.tiles[pos];harvested[pos]=true;
     if(plant(t))cargo[int(t.crop)]+=t.yield_units;
     else if(animal(t))cargo[product[int(t.animal)-9]]+=t.yield_units;
    }else if(pos>=0&&a.op==Op::COLLECT_FERTILIZER&&!collected[pos]){
     const auto&t=o.own.tiles[pos];collected[pos]=true;if(animal(t)&&t.fertilizer_available)cargo[F]++;
    }
   }
   // The current implementation appends at most one DROP at the route end.
   // If it was too late to clear the shed before automatic deposit, do not
   // pretend it creates a second usable warehouse wave.
   total+=sum(cargo);
  }return total;
 }
 void dispatch_idle(const View&o){
  if(!p.overflow_idle_dispatch||phase!=3||day>=29||o.hour>=23)return;
  overflow_dispatch_checks++;
  int excess=expected_auto_deposit(o)-(100-p.shed_safety);
  if(excess<=0)return;
  std::vector<std::tuple<double,int,int>>options;
  for(size_t u=0;u<plans.size();u++){
   const auto&pl=plans[u];int qty=sum(o.priv.inventories[u]);
   if(pl.index<pl.a.size()||qty<=0||qty>100)continue;
   int pos=cell(u?o.own.hands[u-1]:o.own.farmer),time=near(pos)+1;
   if(o.hour+time-1>22)continue;
   options.emplace_back(-double(qty)/time,time,int(u));
  }
  std::sort(options.begin(),options.end());
  for(auto[ratio,time,u]:options){
   if(excess<=0)break;
   int pos=cell(u?o.own.hands[u-1]:o.own.farmer),dest=depot[0];
   for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;
   Plan route;walk(route,pos,dest);route.a.push_back(action(Op::DROP));route.target.push_back(dest);
   plans[u]=std::move(route);int qty=sum(o.priv.inventories[u]);excess-=qty;
   overflow_dispatch_units++;overflow_dispatch_quantity+=qty;
  }
 }
 // Insert an intermediate warehouse visit only when all remaining semantic
 // work can still finish. Unlike idle dispatch, this can exploit a near-shed
 // point before the route ends far away. No production task is deleted, no
 // extra worker is hired, and already-carried input commitments are retained.
 // Public-state cash comparison for selling carried surplus sooner. Only
 // currently harvestable rival goods and known consumption enter this short
 // window; no hidden inventory, future shops or source-agent identity.
 double earlier_delivery_gain(const View&o,const Counts&cargo,int early,int late)const{
  if(early>=late)return 0.;
  Quantities early_inv{},late_inv{};for(int i=0;i<9;i++)early_inv[i]=late_inv[i]=o.market.inventory[i];
  static const std::array<std::vector<int>,8>items{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C,C},{MI,T,W},{S,MI},{WO,WO}}};
  for(int tick=o.step;tick<late;tick++){
   Counts consumed{};
   if(tick%24==0)for(int i=0;i<8;i++)consumed[i]++;
   if(tick%4==0)for(int shop:o.shops)for(int i:items.at(shop))consumed[i]++;
   for(int i=0;i<9;i++){late_inv[i]-=consumed[i];if(tick<early)early_inv[i]-=consumed[i];}
  }
  std::vector<int>units{cell(o.opponent.farmer)};for(auto u:o.opponent.hands)units.push_back(cell(u));
  for(int pos=0;pos<100;pos++){
   const auto&t=o.opponent.tiles[pos];int item=-1,quantity=std::max(0,int(t.yield_units));
   if(animal(t))item=product[int(t.animal)-9];
   else if(plant(t)&&(ongoing(int(t.crop))||o.day-t.planted_day>=first[int(t.crop)]))item=int(t.crop);
   if(item<0||quantity<=0)continue;
   int distance=100;for(int u:units)distance=std::min(distance,dist(u,pos));
   // A lower-bound haul with existing workers, capped by automatic day-end
   // return. It is a conditional scenario, NOT a prediction of actual orders.
   int arrives=std::min((o.day+1)*24,o.step+distance+1+near(pos));
   double q=quantity*p.triad_delivery_pressure;
   if(arrives<=early)early_inv[item]+=q;
   if(arrives<=late)late_inv[item]+=q;
  }
  double gain=0.;for(int i=0;i<9;i++)if(cargo[i]>0)
   gain+=revenue(i,early_inv[i],cargo[i])-revenue(i,late_inv[i],cargo[i]);
  return gain;
 }
 void dispatch_midroute(const View&o){
  if(!p.midroute_delivery||phase!=3||day>=29||o.hour>=22)return;
  if(plans.size()!=o.priv.inventories.size())return;
  midroute_delivery_checks++;
  int excess=expected_auto_deposit(o)-(100-p.shed_safety);
  if(excess<=0&&p.triad_delivery_pressure<=0)return;
  int best=-1,best_qty=0;double best_score=-1.;Plan best_plan;
  for(size_t u=0;u<plans.size();u++){
   const auto&old=plans[u];const auto&cargo=o.priv.inventories[u];int qty=sum(cargo);
   if(old.index>=old.a.size()||qty<=0||qty>100)continue;
   bool blocked=false;Counts need{};
   for(size_t k=old.index;k<old.a.size();k++){
    auto a=old.a[k];
    if(a.op==Op::DROP)blocked=true;
    if(a.op==Op::FEED)need[W]++;
    if(a.op==Op::FERTILIZE)need[F]++;
    if(a.op==Op::PLACE&&int(a.item)>=9)need[int(a.item)]++;
   }
   for(int i=0;i<12;i++)if(cargo[i]>0&&(need[i]>0||i>=9))blocked=true;
   if(blocked)continue;
   int stale=0;auto remaining=repair_plan(o,int(u),old,stale);
   if(remaining.a.empty())continue;
   int start=cell(u?o.own.hands[u-1]:o.own.farmer);
   for(int depot_pos:depot){
    Plan trial;int pos=start;walk(trial,pos,depot_pos);
    // Leave one tick for the ordinary market policy to clear the warehouse
    // if another unit deposits before us. Actual DROP still has a room guard.
    if(o.hour+int(trial.a.size())+1>22)continue;
    trial.a.push_back(action(Op::DROP));trial.target.push_back(depot_pos);
    int early_sale=o.step+int(trial.a.size())-1;
    for(size_t k=0;k<remaining.a.size();k++){
     auto a=remaining.a[k];int dest=remaining.target[k];
     if(movement(a.op)||dest<0)continue;
     walk(trial,pos,dest);trial.a.push_back(a);trial.target.push_back(dest);
    }
    if(o.hour+int(trial.a.size())+1>24)continue;
    bool decay_risk=false;
    for(size_t k=0;k<trial.a.size();k++)if(trial.a[k].op==Op::HARVEST&&trial.target[k]>=0){
     const auto&t=o.own.tiles[trial.target[k]];
     if(plant(t)&&t.max_lifespan_step>=0&&o.step+int(k)+1>=t.max_lifespan_step){decay_risk=true;break;}
    }
    if(decay_risk)continue;
    int extra=std::max(1,int(trial.a.size())-int(remaining.a.size()));
    double value=0;for(int i=0;i<9;i++)value+=revenue(i,o.market.inventory[i],cargo[i]);
    // Conditional salvage ranking, not claimed as exact terminal cash gain.
    double score=value*std::min(qty,std::max(0,excess))/qty/extra;
    if(excess<=0){
     int last=start;for(int target:remaining.target)if(target>=0)last=target;
     int late_sale=std::min((o.day+1)*24,o.step+int(remaining.a.size())+near(last));
     double gain=earlier_delivery_gain(o,cargo,early_sale,late_sale);
     // Slack feasibility above prevents lost obligations; still charge an
     // opportunity cost for spending extra actions on this detour.
     if(gain<=p.action_shadow*extra+1e-6)continue;
     score=(gain-p.action_shadow*extra)/extra;
    }
    if(score>best_score+1e-6){best=int(u);best_qty=qty;best_score=score;best_plan=std::move(trial);}
   }
  }
  if(best>=0){plans[best]=std::move(best_plan);midroute_delivery_insertions++;midroute_delivery_quantity+=best_qty;}
 }
 int incoming(const View&o)const{int total=0;for(auto&i:o.priv.inventories)total+=sum(i);for(size_t u=0;u<plans.size();u++){auto&pl=plans[u];if(pl.index>=pl.a.size())continue;int pos=cell(u?o.own.hands[u-1]:o.own.farmer);if(pl.target[pl.index]>=0&&pl.target[pl.index]!=pos)continue;auto&t=o.own.tiles[pos];if(pl.a[pl.index].op==Op::HARVEST)total+=std::max(0,int(t.yield_units));else if(pl.a[pl.index].op==Op::COLLECT_FERTILIZER&&animal(t)&&t.fertilizer_available)total++;}return total;}
 Acts market(const View&o)const{
  if(day>=29){Counts shed=o.priv.shed;for(size_t u=0;u<plans.size();u++){auto&pl=plans[u];if(pl.index<pl.a.size()&&pl.a[pl.index].op==Op::DROP)add(shed,o.priv.inventories[u]);}return sales_sorted(o,shed);}
  int pending=0;if(o.hour>=23)pending=incoming(o);else{if(p.fix_logistics)for(size_t u=0;u<plans.size();u++){auto&pl=plans[u];if(pl.index<pl.a.size()&&pl.a[pl.index].op==Op::DROP)pending+=sum(o.priv.inventories[u]);}if(pending<=0)return sales_sorted(o,competition_sales(o));}
  Counts shed=o.priv.shed,sell{};int cap=std::max(0,std::min(p.hold_capacity,100-p.shed_safety-pending));int total=0;for(int i=0;i<9;i++)total+=shed[i];int must=std::max(0,total-cap),locked=std::min(shed[W],feed_stock_target);shed[W]-=locked;for(int x:{F,M}){sell[x]=shed[x];must=std::max(0,must-shed[x]);shed[x]=0;}auto d=holding_demand(o);std::vector<int>xs;for(int i=0;i<9;i++)xs.push_back(i);auto key=[&](int i){int f=price(i,int(o.market.inventory[i]-d[i]));int now=o.market.prices[i];return std::tuple(f-now,f,now,name(i));};std::sort(xs.begin(),xs.end(),[&](int a,int b){return key(a)<key(b);});for(int x:xs){if(must<=0)break;int n=std::min(shed[x],must);sell[x]+=n;must-=n;}if(must>0&&locked>0)sell[W]+=std::min(locked,must);return sales_sorted(o,sell);
 }
 std::optional<Action>valid(const View&o,int u,Action a,int target,Counts&pickup,Counts&plant_used)const{
  int pos=cell(u?o.own.hands[u-1]:o.own.farmer),i=int(a.item);auto&t=o.own.tiles[pos];auto&inv=o.priv.inventories[u];Op op=a.op;if(op==Op::NORTH||op==Op::SOUTH||op==Op::EAST||op==Op::WEST||op==Op::PASS)return a;
  if(target>=0&&pos!=target){Plan tmp;walk(tmp,pos,target);return tmp.a.empty()?std::nullopt:std::optional(tmp.a[0]);}
  bool ok=true;switch(op){case Op::PICKUP:{int av=o.priv.shed[i]-pickup[i];if(!at_depot(pos)||av<=0)ok=false;else{a.quantity=std::min(a.quantity,av);pickup[i]+=a.quantity;}break;}case Op::HARVEST:ok=t.yield_units>0;break;case Op::COLLECT_FERTILIZER:ok=animal(t)&&t.fertilizer_available;break;case Op::FEED:ok=animal(t)&&!t.fed_today&&inv[W]>0;break;case Op::CARE:ok=animal(t)&&!t.cared_today;break;case Op::DIG:ok=t.kind!=TileKind::EMPTY&&!animal(t);break;case Op::BUILD_COOP:case Op::BUILD_PASTURE:ok=t.kind==TileKind::EMPTY;break;case Op::PLACE:ok=inv[i]>0&&t.kind==(i==G?TileKind::COOP:TileKind::PASTURE)&&!animal(t);break;case Op::PLANT:ok=t.kind==TileKind::EMPTY&&o.priv.seeds[i]-plant_used[i]>0;if(ok)plant_used[i]++;break;case Op::WATER:ok=plant(t)&&!t.watered_today;break;case Op::FERTILIZE:ok=plant(t)&&inv[F]>0;break;case Op::DROP:ok=at_depot(pos)&&sum(inv)>0;break;default:break;}return ok?std::optional(a):std::nullopt;
 }
 int feed_shortfall(const View&o)const{if(day>=29)return 0;int need=0;for(int pos=0;pos<100;pos++){auto&t=o.own.tiles[pos];need+=animal(t)&&!t.fed_today&&(animal_service_day!=day||service_feed[pos]);}int stock=o.priv.shed[W];for(auto&i:o.priv.inventories)stock+=i[W];return std::max(0,need-stock);}
 std::optional<Action>early_deposit_action(const View&o,int u,const Plan&plan,int room)const{
  if(!p.early_deposit||day>=29||room<=0)return std::nullopt;
  int pos=cell(u?o.own.hands[u-1]:o.own.farmer),remaining=int(plan.a.size()-plan.index),time=24-o.hour;
  if(time<=remaining+1)return std::nullopt;
  auto&inv=o.priv.inventories[u];Counts need{};for(size_t k=plan.index;k<plan.a.size();k++){if(plan.a[k].op==Op::FEED)need[W]++;if(plan.a[k].op==Op::FERTILIZE)need[F]++;}
  int best=-1,n=0;for(int i=0;i<9;i++){int q=std::max(0,inv[i]-need[i]);if(q<=0)continue;if(best<0||q*o.market.prices[i]>n*o.market.prices[best]){best=i;n=q;}}
  if(best<0)return std::nullopt;
  if(at_depot(pos)){
   if(p.triad_batch_delivery){
    // One DROP replaces several per-product PLACE actions only when every
    // carried item is surplus. Keep feed/fertilizer reserved for this route,
    // and never deposit an animal that still has a placement commitment.
    bool surplus_only=true;int quantity=0;
    for(int i=0;i<12;i++){quantity+=inv[i];if(inv[i]>0&&(i>=9||need[i]>0))surplus_only=false;}
    if(surplus_only&&quantity>0&&quantity<=room)return action(Op::DROP);
   }
   return action(Op::PLACE,best,std::min(n,room));
  }
  if(p.early_deposit<2||remaining>0||near(pos)+2>time)return std::nullopt;
  int dest=depot[0];for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;Plan route;walk(route,pos,dest);return route.a.empty()?std::nullopt:std::optional(route.a[0]);
 }
 std::optional<PlayerAction>recover(const View&o){if(!p.fix_liquidity||phase!=2)return std::nullopt;int shortage=feed_shortfall(o);if(shortage<=0)return std::nullopt;PlayerAction out;out.units.resize(1+o.own.hands.size());Counts deposit{};int room=std::max(0,100-sum(o.priv.shed));for(size_t u=0;u<out.units.size();u++){int pos=cell(u?o.own.hands[u-1]:o.own.farmer);auto&inv=o.priv.inventories[u];int qty=sum(inv);if(at_depot(pos)&&inv[F]>0&&qty>0&&qty<=room){out.units[u]=action(Op::DROP);room-=qty;add(deposit,inv);}}int wheat=o.market.inventory[W],finv=o.market.inventory[F];double cash=o.own.money,desired=0;int available=std::max(0,o.priv.shed[F]+deposit[F]-daily_need[F]);for(int i=0;i<shortage;i++)desired+=price(W,wheat-i-1);int sell=0;while(cash<desired&&sell<available){cash+=price(F,finv+sell);sell++;}int buy=0;double spent=0;while(buy<shortage){int pr=price(W,wheat-buy-1);if(spent+pr>cash)break;spent+=pr;buy++;}if(!buy)return std::nullopt;if(sell)out.market.push_back(action(Op::SELL,F,sell));out.market.push_back(action(Op::BUY_PRODUCT,W,buy));liquidity_recoveries++;return out;}
 std::optional<PlayerAction>reconcile_seeds(const View&o){
  if(!p.reconcile_seed_drift||phase!=2||seed_reconcile_checked)return std::nullopt;
  seed_reconcile_checked=true; // One reconciliation per day, never a retry loop.
  auto js=jobs(o);Counts needs{},repair{};for(auto&j:js)add(needs,j.seeds);
  int expense=0;for(int c=0;c<5;c++){
   // Separate new state-driven demand from the old, possibly unfunded plan.
   repair[c]=std::min(std::max(0,needs[c]-o.priv.seeds[c]),std::max(0,needs[c]-prepared_seed_need[c]));
   expense+=repair[c]*seed_price[c];
  }
  if(expense<=0||expense>o.own.money)return std::nullopt;
  // Do not spend a turn shopping while an already visible crop loses yield now.
  for(auto&t:o.own.tiles)if(plant(t)&&t.yield_units>0&&t.max_lifespan_step>=0&&o.step>=t.max_lifespan_step&&(o.step-t.max_lifespan_step)%2==0)return std::nullopt;
  auto stocked=o.priv;for(int c=0;c<5;c++)stocked.seeds[c]+=repair[c];
  View projected{o.step+1,o.day,o.hour+1,o.own,o.opponent,stocked,o.market,o.shops};
  auto check=*this;auto ready=check.reserve(projected,js);bool haul=hauling_needed(ready);
  int budget=(day>=29||haul?22:23)-projected.hour+1;if(budget<=0)return std::nullopt;
  std::vector<int>ss{cell(o.own.farmer)};for(auto s:o.own.hands)ss.push_back(cell(s));
  auto rs=pack(ready,ss,budget,day>=29||haul);if(rs.second)rs=pack(ready,ss,budget,day>=29||haul,true);
  if(rs.second)return std::nullopt;
  for(auto&r:rs.first)if(output(r.needs)&&!at_depot(r.start))return std::nullopt;
  PlayerAction out;out.units.resize(ss.size());for(int c=0;c<5;c++)if(repair[c]>0)out.market.push_back(action(Op::BUY_SEED,c,repair[c]));
  seed_reconciliations++;seed_reconciled_units+=sum(repair);return out;
 }
 Acts preparation_prefix(const View&o)const{
  Acts result(1+o.own.hands.size());auto proposed=jobs(o);auto supplied=project_preparation(o);
  std::array<bool,100>used{};Counts pickup{},plants{};
  for(size_t u=0;u<result.size();u++){
   int pos=cell(u?o.own.hands[u-1]:o.own.farmer);if(used[pos])continue;
   for(auto&j:proposed)if(j.pos==pos&&!j.actions.empty()){
    auto a=j.actions.front();bool allowed=a.op==Op::WATER||a.op==Op::CARE||a.op==Op::DIG||a.op==Op::BUILD_COOP||a.op==Op::BUILD_PASTURE;
    if(!allowed)break;
    if(a.op==Op::DIG||a.op==Op::BUILD_COOP||a.op==Op::BUILD_PASTURE){
     bool funded=true;for(int k=9;k<12;k++)if(j.needs[k]>supplied.priv.shed[k])funded=false;
     for(int k=0;k<5;k++)if(j.seeds[k]>supplied.priv.seeds[k])funded=false;
     if(!funded)break;
    }
    if(auto ok=valid(o,int(u),a,pos,pickup,plants)){result[u]=*ok;used[pos]=true;
     for(int k=9;k<12;k++)supplied.priv.shed[k]-=j.needs[k];
     for(int k=0;k<5;k++)supplied.priv.seeds[k]-=j.seeds[k];
    }break;
   }
  }return result;
 }
 bool preserves_hire_projection(const View&o,int unit,const Action&a)const{
  if(!p.preparation_spawn_guard||!movement(a.op))return true;
  if(std::none_of(queue.begin(),queue.end(),[](const Action&x){return x.op==Op::HIRE;}))return true;
  int pos=cell(unit?o.own.hands.at(unit-1):o.own.farmer),next=pos;
  if(a.op==Op::NORTH)next-=10;else if(a.op==Op::SOUTH)next+=10;else if(a.op==Op::EAST)next++;else if(a.op==Op::WEST)next--;
  // The projected HIRE spawn uses center occupancy before unit actions. Do
  // not execute a prefix that changes that assumption. This is not a claim
  // that moving away is inherently bad; it is a conditional-preview guard.
  return !at_depot(pos)&&!at_depot(next);
 }
 Acts preparation_pipeline(const View&o)const{
  Acts result(1+o.own.hands.size());
  // "Needs no inventory" is not sufficient: CARE pulled ahead of a pending
  // FEED makes the worker travel to the animal, return for wheat, then travel
  // out again. Only a whole, already-existing one-action job is safely
  // commutable with the live market queue. This deliberately excludes
  // compound HARVEST/WATER, FEED/CARE and all establishment chains.
  auto raw_probe=*this;raw_probe.p.shared_task_atoms_v2=false;raw_probe.p.split_service_jobs=false;
  auto raw_jobs=raw_probe.jobs(o);std::set<TaskKey> commutable;
  for(auto&j:raw_jobs){
   if(j.actions.size()==1&&resource_free_service(j))commutable.insert(task_key(j.pos,j.actions.front()));
  }
  if(commutable.empty())return result;
  // Build the schedule that would follow our remaining own orders. It is a
  // conditional plan only: opponent fills/random events are unknown and the
  // next real observation always recompiles it. A current unit may execute
  // only the first atom of its own projected route, and only when that exact
  // atom is already valid in the authoritative current state.
  auto prepared=project_preparation(o);View projected{o.step+prepared.elapsed,day,o.hour+prepared.elapsed,prepared.farm,o.opponent,prepared.priv,prepared.market,o.shops};
  auto planner=*this;auto js=planner.reserve(projected,planner.jobs(projected));bool haul=planner.hauling_needed(js);
  std::vector<int>ss{cell(projected.own.farmer)};for(auto pos:projected.own.hands)ss.push_back(cell(pos));
  int budget=(day>=29||haul?22:23)-projected.hour+1;if(budget<=0)return result;
  auto rs=planner.pack(js,ss,budget,day>=29||haul);if(rs.second){auto alt=planner.pack(js,ss,budget,day>=29||haul,true);if(alt.second<rs.second)rs=std::move(alt);}
  Counts pickup{},plants{};for(size_t u=0;u<result.size()&&u<rs.first.size();u++){
   auto&r=rs.first[u];if(r.jobs.empty())continue;auto&j=r.jobs.front();if(j.actions.empty())continue;
   auto key=task_key(j.pos,j.actions.front());if(!commutable.contains(key))continue;
   if(auto a=valid(o,int(u),j.actions.front(),j.pos,pickup,plants)){
    if(preserves_hire_projection(o,int(u),*a))result[u]=*a;else preparation_spawn_guard_blocks++;
   }
  }return result;
 }
 PlayerAction act(const View&o){if(o.step<=last_step||o.day<day){Controller fresh(p);*this=std::move(fresh);}last_step=o.step;
  if(p.exact_schedule_cache&&!packmemo::active&&(!schedule_cache||schedule_cache_day!=o.day)){schedule_cache=std::make_shared<packmemo::Cache>();schedule_cache_day=o.day;}
  auto cache_lifetime=schedule_cache;
  packmemo::Scope cache_scope(p.exact_schedule_cache?(packmemo::active?packmemo::active:cache_lifetime.get()):nullptr);
  if(o.day!=day)new_day(o);
  if(p.funded_labor)labor::observe(*this,o);
  if(!resume_compiled_tick){if(auto recovery=recover(o))return *recovery;if(auto repair=reconcile_seeds(o))return *repair;intraday::complete_pending(*this,o);if(phase==3)recoordinate(o);service::recover(*this,o);exchange::apply(*this,o);dispatch_idle(o);dispatch_midroute(o);}
  resume_compiled_tick=false;
  bool preparing=phase==1;Acts prep;if(preparing&&(p.preparation_work||p.preparation_pipeline_v2)){if(p.preparation_pipeline_v2){preparation_pipeline_checks++;prep=preparation_pipeline(o);}else prep=preparation_prefix(o);}PlayerAction out;if(phase==1){int n=std::min(10,int(queue.size()));out.market.assign(queue.begin(),queue.begin()+n);queue.erase(queue.begin(),queue.begin()+n);if(queue.empty())phase=2;}else if(phase==2){compile(o);out.market=market(o);}else out.market=market(o);if(out.market.size()>10)out.market.resize(10);int n=1+o.own.hands.size();out.units.resize(n);Counts pickup{},plant_used{},early_placed{};int drop_qty=0;if(phase==3)for(int u=0;u<n;u++){auto&pl=plans.at(u);if(auto early=early_deposit_action(o,u,pl,100-sum(o.priv.shed)+sum(pickup)-drop_qty)){out.units[u]=*early;if(early->op==Op::PLACE){early_placed[int(early->item)]+=early->quantity;drop_qty+=early->quantity;}else if(early->op==Op::DROP){add(early_placed,o.priv.inventories[u]);drop_qty+=sum(o.priv.inventories[u]);}continue;}while(pl.index<pl.a.size()){
   if(p.fix_logistics&&pl.a[pl.index].op==Op::DROP){int pos=cell(u?o.own.hands[u-1]:o.own.farmer);if(at_depot(pos)){int qty=sum(o.priv.inventories[u]);int room=100-sum(o.priv.shed)+sum(pickup)-drop_qty;if(qty>room)break;drop_qty+=qty;}}
   size_t k=pl.index++;auto a=valid(o,u,pl.a[k],pl.target[k],pickup,plant_used);if(a){out.units[u]=*a;break;}}}
  if(p.sell_deposits&&sum(early_placed)>0){for(int i=0;i<9;i++)if(i==W||i==F)early_placed[i]=0;auto sales=sales_sorted(o,early_placed);sales.insert(sales.end(),out.market.begin(),out.market.end());if(sales.size()>10)sales.resize(10);out.market=std::move(sales);}
  if(p.fix_liquidity&&phase!=3&&feed_shortfall(o)>0)for(int u=0;u<n;u++){int pos=cell(u?o.own.hands[u-1]:o.own.farmer);auto&t=o.own.tiles[pos];if(at_depot(pos)&&animal(t)&&t.fertilizer_available)out.units[u]=action(Op::COLLECT_FERTILIZER);}
  if(preparing&&(p.preparation_work||p.preparation_pipeline_v2))for(int u=0;u<n;u++)if(out.units[u].op==Op::PASS&&prep[u].op!=Op::PASS){out.units[u]=prep[u];preparation_actions++;if(p.preparation_pipeline_v2){preparation_pipeline_actions++;preparation_pipeline_moves+=movement(prep[u].op);}}service::protect(*this,o,out);service::procure(*this,o,out);intraday::consider(*this,o,out);service::protect(*this,o,out);handoff::apply(*this,o,out);liverepair::market(*this,o,out);return out;
 }
};
}
#include "resource_exchange.hpp"
#include "intraday_admission.hpp"
#include "service_recovery.hpp"
#include "joint_portfolio.hpp"
#include "day_consequence.hpp"
#include "idle_task_handoff.hpp"
#include "live_execution_repairs.hpp"
#include "rotation_calendar.hpp"
#include "rotation_cash.hpp"
#include "compile_choices.hpp"

#include "labor_funding.hpp"
