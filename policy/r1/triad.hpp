#pragma once
#include <algorithm>
// Triad DP: autonomous economic intent -> conditional calendar -> live executor.
// No J7 macros, opponent identities, tape library, live Simulator or RNG access.
#include "planner.hpp"
#include "marginal_value.hpp"
#include "animal_service_dp.hpp"
#include "ongoing_maintenance_dp.hpp"
#include "finite_fertilizer.hpp"
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include "executor/observed_day_scenario.hpp"
#include "local_sale_timing.hpp"
#include "joint_bundle_state.hpp"
#include <functional>
#ifndef P16_WORKING_CAPITAL_GATE
#define P16_WORKING_CAPITAL_GATE 1
#endif
#ifndef R2_OPTIMIZER_AUDIT
#define R2_OPTIMIZER_AUDIT 0
#endif
#ifndef R2_STUDENT_SLOT_AUDIT
#define R2_STUDENT_SLOT_AUDIT 0
#endif
#ifndef R2_STUDENT_LIFECYCLE_V4
#define R2_STUDENT_LIFECYCLE_V4 0
#endif
#ifndef R2_GREEDY_VALUE_ORDER
#define R2_GREEDY_VALUE_ORDER 0
#endif
#ifndef R2_OUTER_NEIGHBOR_AUDIT
#define R2_OUTER_NEIGHBOR_AUDIT 0
#endif
#ifndef R2_OUTER_BEAM_WIDTH
#define R2_OUTER_BEAM_WIDTH 0
#endif
namespace triad {
using namespace competitive;
struct Settings {
 double competition=.8,supply=.85,future_shop=.7,capital_power=.4;
 double labor_hours=15,work_price=1.2,animal_work=1.0,reserve=120;
 double max_animals=20,max_hands=14,max_land=4,feed_cover=2;
 double rotation=1,preview=1,delivery=2,intraday=1,service=1;
 double replant=.7,land_rent=2,discount=.015,tour_dp=0;
 double layout=0,repeat=1,animal_bias=1,crop_bias=1,portfolio_passes=1;
 double crop_fert=1,harvest_threshold=1,delay_sale=0,opening_budget=1;
 double scenario=0,keep_commitments=1;
 // Runtime switch for the optimal-selling DP in Planner::value(). A compile-time macro was used
 // before, which forced build.sh to compile the same source twice (base and dp) -- doubling every
 // build. As a setting it toggles from R1_CONFIG_OVERRIDES with a single binary.
 double sale_dp=0;
 // Shared overnight-holding budget for the sale DP. Must leave room for the goods that will
 // still come in through the workers' hands: the shed has to absorb those at end_of_day, and
 // anything that does not fit is destroyed. Measured with 90: the DP arm destroyed 30 units
 // against the base's 1.
 double sale_hold_cap=90;
 // Do not let the sale plan bind before this day. Holding defers cash, and early cash hires
 // workers and buys seed and animals; measured, the DP ran -1,073 behind over days 12-20 alone,
 // the stretch where cash is tightest. 0 = bind from takeover.
 double sale_dp_start_day=12;
 // Weight on value()'s ONLY risk term: -risk * sum_d max(0, -cumulative_balance).
 // value() is a deterministic own-c*forecast-rival surrogate, not E[margin]. This penalty only
 // prices forecast insolvency; it does not model the distribution of terminal margin.
 double risk=1;
 // Recovered candidate expansion; unrelated inactive experiments omitted.
 double candidate_extra=0;
 // T3 switches: 0 is exact legacy behavior; >=1 couples service to forecast;
 // >=2 coordinates alternatives using the SAME portfolio value; >=3 adds
 // finite-difference market externalities to maintenance candidate generation.
 double service_reconcile=0,live_ledger=0,delivery_calendar=0,feed_finance=0;
 // R2: state-checked whole-cargo delivery; no economic forecast changes.
 double batch_delivery=0;
 // Opt-in bounded coordinate improvement of today's new, cash-funded projects.
 // Zero preserves the released greedy portfolio exactly.
 double portfolio_swaps=0,portfolio_swap_min_gain=0;
 // Opt-in: all nine product signals are derivatives of the portfolio objective.
 double marginal_value=0;
};
constexpr int SETTINGS_COUNT=sizeof(Settings)/sizeof(double);
struct Commitment {
 int kind=-1,birth=-1,chosen_day=-1,length=0,successor=-1;
 bool funded=false;
};
struct CropPath {Asset a;int kind=-1,length=0;double value=-1e100;};
struct GreedySlotAudit {
 int pos=-1,picked_kind=-1,raw_kind=-1,value_kind=-1,candidates=0;
 double budget=0,picked_gain=0,raw_gain=-1e100,picked_value_gain=0,raw_value_gain=-1e100,picked_rank=0,raw_rank=0;
};
#if R2_STUDENT_SLOT_AUDIT
struct StudentSlotAudit {
 int pos=-1,label=-1,legal_mask=1,terminal=0,animals=0,owned=0,slot_index=0,slot_count=0;
 double budget=0;
 std::array<int,12>stock_left{};
 Asset prefix{};
};
struct StudentReleaseAudit {
 int pos=-1,label=0,legal_mask=0;
 StudentSlotAudit state{};
};
struct StudentV3SlotAudit {StudentSlotAudit slot{};int label_class=0,legal_mask=0;};
#endif
struct Controller {

 Settings s;Planner model;dp7::Controller core;
 JointBundleState joint;
 std::array<Commitment,100>book{};
 int plan_calls=0,candidates=0,deferred=0,kept=0,rotations=0,started=0;
 int portfolio_swap_trials=0,portfolio_swap_accepts=0,portfolio_pair_trials=0,portfolio_pair_accepts=0;
 double portfolio_swap_gain=0,portfolio_pair_gain=0;
 std::vector<GreedySlotAudit>greedy_audit;
#if R2_STUDENT_SLOT_AUDIT
 std::vector<StudentSlotAudit>student_slot_audit;
 // Offline-only direct actor seam.  It is empty in every production build and
 // lets a diagnostic caller choose from the exact mask while this one plan is
 // being constructed, so resources advance in place rather than by replanning
 // the whole prefix once per slot.
 std::function<int(const StudentSlotAudit&)>student_selector;
 std::vector<StudentReleaseAudit>student_release_audit;
 std::vector<StudentV3SlotAudit>student_v3_slot_audit;
 std::function<int(const StudentReleaseAudit&)>student_release_selector;
 std::function<int(const StudentSlotAudit&,int)>student_v3_selector;
#if R2_STUDENT_LIFECYCLE_V4
 std::array<int8_t,100>student_feed_choice=[](){std::array<int8_t,100>a;a.fill(-1);return a;}();
 std::array<int,100>student_retire_until=[](){std::array<int,100>a;a.fill(-1);return a;}();
#endif
#endif
#if R2_OUTER_NEIGHBOR_AUDIT
 std::array<int8_t,100>forced_kind{};
#endif
 double greedy_budget_initial=0,greedy_budget_final=0;
#if R2_OPTIMIZER_AUDIT
 int greedy_stop_pos=-1,greedy_stop_skipped_empty=0,greedy_stop_examined_empty=0,greedy_stop_positive_empty=0;
 double greedy_stop_best_later_gain=0;
#endif
 int greedy_preview_proposed=0,greedy_preview_started=0,greedy_preview_removed=0;
 bool greedy_preview_checked=false;
 int previous_step=-1;std::array<int,12>previous_stock{};
 Flow prices{},generation_values{};int marginal_value_updates=0;Asset portfolio{};std::array<Asset,100>paths{};
 std::array<int,100>release{},successor{},length{};
 double predicted=0;
 std::array<std::array<int8_t,2>,100>forecast_service{},forecast_crop_service{};
#if R2_FINITE_FERTILIZER
 std::array<int8_t,100>finite_first_fertilizer{};
#endif
 int service_trials=0,service_switches=0;double service_objective_gain=0;
 int preparation_finance_checks=0,preparation_finance_repairs=0;
 int capital_collects=0,capital_financings=0,capital_refills=0;bool capital_pending=false;
 LocalSaleTiming local_sale;SaleClock sale_memory;
 Controller(Settings settings={}):s(settings),model(),core(model.core.p){configure(settings);
#if R2_OUTER_NEIGHBOR_AUDIT
  forced_kind.fill(-2);
#endif
 }
 void configure(Settings settings){s=settings;
  auto&c=model.cfg;c.competition=s.competition;c.supply=s.supply;c.future_shop=s.future_shop;
  c.capital_power=s.capital_power;c.labor_hours=s.labor_hours;c.action_cost=s.work_price;
  c.reserve=s.reserve;c.max_animals=s.max_animals;c.replant=s.replant;c.discount=s.discount;
  c.risk=s.risk;c.land_rent=s.land_rent;c.sale_dp=s.sale_dp;c.sale_hold_cap=s.sale_hold_cap;
  auto&p=core.p;p.max_hands=int(s.max_hands);p.max_land=int(s.max_land);p.max_animals=int(s.max_animals);
  p.max_cows=p.max_sheep=p.max_geese=int(s.max_animals);p.max_strawberry=p.max_tomato=p.max_melon=75;
  p.frequent_harvest=true;p.frequent_threshold=std::max(1,int(s.harvest_threshold));
  p.early_deposit=int(s.delivery);p.triad_batch_delivery=s.batch_delivery>0;p.sell_deposits=true;p.continuous_market_execution=true;
  // 0/1 retain the released execution path. >=2 enables the additional
  // public-price-pressure trigger for already feasible mid-route delivery.
  p.triad_delivery_pressure=s.batch_delivery>=2?s.batch_delivery-1:0;
  p.renew_ongoing=!R2_STUDENT_LIFECYCLE_V4;p.insertion_hire_estimate=true;p.reconcile_seed_drift=true;
  p.regret_hire_estimate=p.regret_compile=p.incremental_regret_cost=true;
  p.feed_cover_days=int(s.feed_cover);p.feed_stock_cap=60;
  p.recover_service_inputs=p.procure_service_inputs=p.finance_service_inputs=true;
  p.intraday_admission=s.intraday>0;p.intraday_procurement=s.intraday>0;
  p.intraday_declared_value=false;p.latest_animal_day=20;
  p.exact_schedule_cache=true;p.triad_tour_dp=s.tour_dp>0;
  p.competitive_sell_weight=s.delay_sale;p.opponent_supply_weight=s.supply;
  p.timing_discount=.94;p.action_shadow=s.work_price;
  p.autonomous_start=true;p.opening_animals.clear();p.opening_crops={0,0,0,0,0};
  p.operating_reserve=s.reserve;p.own_feed_demand_weight=1;
  p.finite_fertilizer=R2_FINITE_FERTILIZER&&s.crop_fert>0;
  p.shared_task_atoms_v2=true;
 }
 const Flow& path_prices()const{return s.marginal_value>0?generation_values:prices;}
 double update_value(const View&o){
  double result=model.value(o,portfolio,&prices);
  if(s.marginal_value>0){generation_values=MarginalValue::compute(model,o,portfolio,s.discount);marginal_value_updates++;}
  return result;
 }
 double scalar(const Asset&a,const Flow&px,int start)const{
  double v=0;for(int d=start;d<30;d++){
   double x=a.fixed[d]-s.work_price*a.labor[d];for(int i=0;i<9;i++)x+=a.f[d][i]*px[d][i];
   v+=x/std::pow(1+s.discount,d-start);
  }return v;
 }
 // Exact crop-state recurrence under the declared maintenance/harvest policy.
 // Resource and route feasibility is a separate admission stage, never assumed.
 Asset crop(int k,int birth,int pos,int finish,const Flow&px,const Tile*existing=nullptr,bool*first_fertilizer=nullptr,int committed_water=-1,int committed_fertilizer=-1,std::array<int8_t,2>*first_service=nullptr)const{
  if(first_fertilizer)*first_fertilizer=false;
  int begin=model.day;Asset a;a.kind=k;
  if(birth>=30||finish>29||finish<begin){a.end=30;return a;}
  int start=std::max(begin,birth);a.end=finish;
  if(!existing){a.first_cost=seed_price[k];a.fixed[birth]-=seed_price[k];a.labor[birth]+=1;}
  int dry=existing?existing->consecutive_unwatered:1;
  int fert=existing?existing->fertilized_until_day:-1;
  int yield=existing?existing->yield_units:(ongoing(k)?0:1);
  const double travel=.06*near(pos)+.10;
  OngoingMaintenanceDP md;md.kind=k;md.birth=birth;md.begin=start;md.mode=2;md.work=s.work_price+travel;md.discount=s.marginal_value>0?1/(1+s.discount):1.;
  for(int d=0;d<30;d++){md.price[d]=px[d][k];md.fert[d]=px[d][F];}
  for(int d=start;d<=finish;d++){
   bool visit=false,watered=existing&&d==begin&&existing->watered_today;
   bool w=false,z=false;int age=d-birth;
   if(ongoing(k)){
    if(yield>0){a.f[d][k]+=yield;a.labor[d]+=1;yield=0;visit=true;}
    if(d<finish){
     bool tick=md.production(d+1);bool oldw=dry>=1||tick;bool oldf=tick&&fert<d&&s.crop_fert>0;
     auto choice=md.first(d,dry,std::clamp(fert-d+1,0,3),oldw,oldf);
     w=choice.water&&!watered;z=choice.fertilize&&s.crop_fert>0&&fert<d;
     if(d==birth&&!watered)w=true;
     if(d==begin&&committed_water>=0)w=committed_water&&!watered;
     if(d==begin&&committed_fertilizer>=0)z=committed_fertilizer&&fert<d;
    }
   }else{
    int last=k==W?4:k==C?3:12;bool window=age>=(last+1)/2&&age<=last;
    w=!watered&&(dry>=1||(window&&yield<(k==C?4:6)));
#if R2_FINITE_FERTILIZER
    z=s.crop_fert>0&&finite_fertilizer_choice(k,age,finish-birth,yield,w,watered,fert-d+1,px[finish][k]/(s.marginal_value>0?std::pow(1+s.discount,finish-d):1.),px[d][F],s.work_price+travel);
    if(d==begin&&committed_fertilizer>=0)z=committed_fertilizer&&fert<d;
    if(z)fert=d+2;
#endif
    if(w&&window)yield=std::min(k==C?4:6,yield+(fert>=d?2:1));
    if(d==finish){a.f[d][k]+=yield;a.labor[d]+=1;yield=0;visit=true;}
   }
   if(z){a.f[d][F]-=1;a.labor[d]+=1;fert=d+2;visit=true;}
   if(d==begin&&first_fertilizer)*first_fertilizer=z;
   if(d==begin&&first_service)*first_service={int8_t(w),int8_t(z)};
   if(w){a.labor[d]+=1;visit=true;}
   if(visit)a.labor[d]+=travel;
   bool wet=w||watered;dry=wet?0:dry+1;
   if(d<finish&&ongoing(k)&&md.production(d+1))yield=1+(wet&&fert>=d?1:0);
  }
  return a;
 }
 // Keep the conditional animal forecast at a real call boundary. The audited
 // GCC 13.3 -O3 build's nullptr constprop clone ignored the solved service
 // table (new animals falsely starved on day 3), while its generic function
 // computed the same forecast as the published GCC 14 build. No rule, payoff,
 // parameter, or action is changed. A post-build trajectory test is mandatory.
#if defined(__GNUC__) && !defined(__clang__)
 __attribute__((noipa))
#elif defined(__clang__)
 __attribute__((noinline))
#endif
 Asset animal_path(int k,int birth,int pos,const Flow&px,const Tile*t=nullptr,std::array<int8_t,2>*first_action=nullptr,const std::array<int8_t,2>*committed_first=nullptr)const{
  if(s.service<=0){auto a=model.animal_stream(k,birth,pos,t);for(auto&w:a.labor)w*=s.animal_work;return a;}
  Asset a;a.kind=k;a.end=30;int day=model.day,start=std::max(day,birth),j=k-9;
  if(!t){a.first_cost=animal_price[j];a.fixed[start]-=a.first_cost;a.labor[start]+=3+.15*near(pos);}
  else {a.f[day][product[j]]+=t->yield_units;a.f[day][F]+=t->fertilizer_available;}
  AnimalServiceDP dp;dp.solve(k,birth,start,px,s.work_price,s.marginal_value>0?1/(1+s.discount):1.);
  int hunger=t?std::min(1,int(t->consecutive_unfed)):0,bonus=t?std::min(held[j]-1,int(t->pending_care_bonus)):0;
  for(int d=start;d<29;d++){
   auto c=dp.choices[d][hunger][bonus];
   if(committed_first&&d==start){c.feed=(*committed_first)[0];c.care=(*committed_first)[1];}
   if(!t&&d==start)c.feed=c.care=1;
   bool fed=t&&d==day&&t->fed_today,cared=t&&d==day&&t->cared_today;
   int f=c.feed&&!fed,z=c.care&&!cared;
   if(first_action&&d==start)*first_action={int8_t(f),int8_t(z)};
   a.f[d][W]-=f;a.labor[d]+=(f+z+1+.20*near(pos))*s.animal_work;
   bool feeding=f||fed;hunger=feeding?0:hunger+1;if(hunger>=2){a.end=d+1;break;}
   int age=d+1-birth;bool tick=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
   if(tick){a.f[d+1][product[j]]+=1+(feeding?bonus:0);a.labor[d+1]+=1;bonus=0;}
   if(feeding&&(z||cared))bonus=std::min(held[j]-1,bonus+1);
   a.f[d+1][F]+=1;
  }return a;
 }
 struct RotationDP{
  std::array<double,31>v{};std::array<int,31>kind{},len{};
 };
 RotationDP rotations_dp(int pos,const Flow&px,int start)const{
  RotationDP dp;dp.kind.fill(-1);
  if(s.rotation<=0)return dp;
  for(int d=29;d>=start;d--){dp.v[d]=dp.v[d+1];
   for(int k=0;k<5;k++){
    int minlen=k==W?2:k==C?2:k==M?10:first[k]+3*interval[k];
    int maxlen=k==W?4:k==C?3:k==M?12:minlen;
    for(int len=minlen;len<=maxlen;len++){
     if(d+(ongoing(k)?first[k]:len)>29)continue;
     Asset a=crop(k,d,pos,std::min(29,d+len),px);int next=d+len+(ongoing(k)?1:0);
     double val=scalar(a,px,d)-s.land_rent*len;
     if(s.repeat>0&&next<30)val+=dp.v[next]/std::pow(1+s.discount,next-d);
     if(val>dp.v[d]+1e-8){dp.v[d]=val;dp.kind[d]=k;dp.len[d]=len;}
    }
   }
  }return dp;
 }
 CropPath choose_crop(int forced,int start,int pos,const Flow&px,const RotationDP&dp)const{
  CropPath out;if(start>=29)return out;
  int minlen=forced==W?2:forced==C?2:forced==M?10:first[forced]+3*interval[forced];
  int maxlen=forced==W?4:forced==C?3:forced==M?12:minlen;
  for(int len=minlen;len<=maxlen;len++){
   if(start+(ongoing(forced)?first[forced]:len)>29)continue;auto a=crop(forced,start,pos,std::min(29,start+len),px);
   int next=start+len+(ongoing(forced)?1:0);double score=scalar(a,px,start)-s.land_rent*len;
   if(s.repeat>0&&next<30)score+=dp.v[next]/std::pow(1+s.discount,next-start);
   if(score>out.value){out={a,forced,len,score};}
  }
  if(out.kind<0)return out;
  int next=start+out.length+(ongoing(forced)?1:0);
  for(int n=0;s.repeat>0&&n<15&&next<29;n++){
   int k=dp.kind[next];if(k<0){next++;continue;}
   auto a=crop(k,next,pos,std::min(29,next+dp.len[next]),px);add(out.a,a);
   next+=dp.len[next]+(ongoing(k)?1:0);
  }return out;
 }
 // One-product market replay: exact same conditional trading order as value().
 // Differencing isolates portfolio/competition effects; it is NOT future truth.
 Flow marginal_prices(const View&o,const Asset&a)const{
  Flow result=prices;
  const double discount=1+model.cfg.discount*std::max(0.,1-o.own.money/20000.);
  for(int i:{W,E,MI,WO,F}){
   std::array<double,30>before{};double inv=o.market.inventory[i];
   auto daytrade=[&](int d,double q,double&stock){
    stock-=model.dem[d][i]*.5;
#if R2_SALE_CLOCK_MODE >= 2
    double r=model.cfg.supply*model.rival[d][i]*model.rival_early_share[i],late=model.cfg.supply*model.rival[d][i]-r;
#else
    double r=model.cfg.supply*model.rival[d][i]*.5,late=r;
#endif
    auto trade=[&](double n){return Planner::trade(i,stock,n);};
    double enemy=trade(r);double own=trade(q);enemy+=trade(late);
    stock-=model.dem[d][i]*.5;return own-model.cfg.competition*enemy;
   };
   for(int d=o.day;d<30;d++){before[d]=inv;daytrade(d,a.f[d][i],inv);}
   for(int d=o.day;d<30;d++){
    double plus=before[d],minus=before[d],v=0,weight=1;
    // Symmetric two-unit difference damps integer-price rounding artifacts.
    for(int t=d;t<30;t++){
     v+=(daytrade(t,a.f[t][i]+(t==d?2:0),plus)-daytrade(t,a.f[t][i]-(t==d?2:0),minus))*weight;
     weight/=discount;
    }
    result[d][i]=v*.25;
   }
  }
  return result;
 }
 // Coordinate full lifecycle streams, including feed and output. Accept only
 // a strict gain in the same portfolio objective used for new investments.
 // Preserve the chosen first maintenance action; never solve it again using
 // a different price table after the forecast has been committed.
 void reconcile_service(const View&o){
  if(o.day>=29||s.service<=0)return;
  Flow marginal{};if(s.marginal_value<=0&&s.service_reconcile>=3)marginal=marginal_prices(o,portfolio);
  double score=model.value(o,portfolio);
  for(int pos=0;pos<100;pos++){
   const auto&t=o.own.tiles[pos];if(!animal(t))continue;
   int alternatives=s.marginal_value<=0&&s.service_reconcile>=3?2:1;
   for(int which=0;which<alternatives;which++){
    std::array<int8_t,2>action{};
    auto trial=animal_path(int(t.animal),t.placed_day,pos,which?marginal:path_prices(),&t,&action);
    auto candidate=portfolio;add(candidate,paths[pos],-1);add(candidate,trial);
    double value=model.value(o,candidate);service_trials++;
    if(value>score+1e-6){
     service_switches++;service_objective_gain+=value-score;score=value;
     portfolio=std::move(candidate);paths[pos]=std::move(trial);forecast_service[pos]=action;
     update_value(o);
    }
   }
  }
 }
 void set_service(const View&o){
  core.animal_service_day=o.day;core.crop_service_day=o.day;
  for(int pos=0;pos<100;pos++){
   const auto&t=o.own.tiles[pos];
   if(animal(t)){
    AnimalServiceDP d;d.solve(int(t.animal),t.placed_day,o.day,path_prices(),s.work_price,s.marginal_value>0?1/(1+s.discount):1.);auto c=d.first(o.day,t);
    if(s.service_reconcile>0){c.feed=forecast_service[pos][0];c.care=forecast_service[pos][1];}
    core.service_feed[pos]=s.service<=0?1:c.feed;core.service_care[pos]=s.service<=0?core.care_due(t):c.care;
#if R2_STUDENT_LIFECYCLE_V4
    if(student_feed_choice[pos]>=0){core.service_feed[pos]=student_feed_choice[pos];if(!student_feed_choice[pos])core.service_care[pos]=0;}
#endif
   }
   if(plant(t)&&ongoing(int(t.crop))){
    int k=int(t.crop);OngoingMaintenanceDP md;md.kind=k;md.birth=t.planted_day;md.mode=2;md.work=s.work_price;md.discount=s.marginal_value>0?1/(1+s.discount):1.;
    for(int d=0;d<30;d++){md.price[d]=path_prices()[d][k];md.fert[d]=path_prices()[d][F];}
    auto c=md.first(o.day,std::min(1,int(t.consecutive_unwatered)),std::clamp(int(t.fertilized_until_day)-o.day+1,0,3),core.water_due(t),core.fertilize_due(t));
    core.crop_birth[pos]=t.planted_day;core.crop_kind[pos]=k;
    if(s.marginal_value>0){c.water=forecast_crop_service[pos][0];c.fertilize=forecast_crop_service[pos][1];}
    core.crop_water[pos]=c.water;core.crop_fertilize[pos]=c.fertilize&&s.crop_fert>0;
   }
#if R2_FINITE_FERTILIZER
   if(plant(t)&&!ongoing(int(t.crop))){
    int k=int(t.crop);
    core.crop_birth[pos]=t.planted_day;core.crop_kind[pos]=k;
    // Do not recompute this commitment with a later updated price table.
    core.crop_fertilize[pos]=finite_first_fertilizer[pos];core.crop_water[pos]=core.water_due(t);
   }
#endif
  }
 }
 void plan(const View&o){
  joint.observe(o);
  if constexpr(P16_JOINT_BUNDLES>0){core.p.max_animals=(joint.active&&o.day==joint.created_day)?std::min(int(s.max_animals),joint.animal_limit):int(s.max_animals);core.p.operating_reserve=s.reserve+joint.seed_reserve(o);}
  plan_calls++;model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);core.day=o.day;core.phase=1;core.queue.clear();core.plans.clear();core.pending_admission={};core.intraday_units.clear();core.resource_degraded=core.actual_drop=0;core.seed_reconcile_checked=false;core.prepared_seed_need={};
  Farm farm=o.own;int expiry=dp7::Controller::project_zero_expiry(farm,o.step);
  View v{o.step,o.day,o.hour,farm,o.opponent,o.priv,o.market,o.shops};
  core.target.clear();core.plant_not_before.fill(0);core.triad_crop_age.fill(-1);
#if R2_STUDENT_LIFECYCLE_V4
  core.student_crop_release.fill(0);core.student_crop_dig.fill(0);student_feed_choice.fill(-1);
#endif
  portfolio={};paths={};forecast_service={};forecast_crop_service={};release.fill(o.day);successor.fill(-1);length.fill(0);
  if constexpr(R2_OPTIMIZER_AUDIT){greedy_audit.clear();greedy_budget_initial=greedy_budget_final=0;
#if R2_OPTIMIZER_AUDIT
   greedy_stop_pos=-1;greedy_stop_skipped_empty=greedy_stop_examined_empty=greedy_stop_positive_empty=0;greedy_stop_best_later_gain=0;
#endif
   greedy_preview_proposed=greedy_preview_started=greedy_preview_removed=0;greedy_preview_checked=false;}
#if R2_STUDENT_SLOT_AUDIT
  student_slot_audit.clear();
  student_release_audit.clear();student_v3_slot_audit.clear();
#endif
#if R2_FINITE_FERTILIZER
  finite_first_fertilizer.fill(0);
#endif
  for(int i=0;i<9;i++){portfolio.f[o.day][i]+=o.priv.shed[i];for(auto&b:o.priv.inventories)portfolio.f[o.day][i]+=b[i];}
  update_value(o);std::vector<int>free;int animals=0;
  int owned=std::popcount(unsigned(o.own.unlocked_mask));core.planned_land=owned;
#if R2_STUDENT_SLOT_AUDIT
  int student_release_count=0,student_existing_animals=0;
  std::array<int,12>student_initial_stock{};
  for(const auto&t:farm.tiles){
#if R2_STUDENT_LIFECYCLE_V4
   student_release_count+=plant(t)||animal(t);
#else
   student_release_count+=plant(t)&&!ongoing(int(t.crop));
#endif
   student_existing_animals+=animal(t);
  }
  for(int k=0;k<5;k++)student_initial_stock[k]=o.priv.seeds[k];
  for(int k=9;k<12;k++){student_initial_stock[k]=o.priv.shed[k];for(const auto&iv:o.priv.inventories)student_initial_stock[k]+=iv[k];}
#endif
  // Resolve existing projects from facts. Successor intention is separate from
  // the current tile and never replaces maintenance of the incumbent crop.
  for(int pos=0;pos<100;pos++){
   auto&t=farm.tiles[pos];if(t.kind==TileKind::LOCKED)continue;
   if(animal(t)){
    paths[pos]=animal_path(int(t.animal),t.placed_day,pos,path_prices(),&t,&forecast_service[pos]);
#if R2_STUDENT_LIFECYCLE_V4
    if(student_release_selector){
     StudentReleaseAudit event;event.pos=pos;event.label=forecast_service[pos][0]?1:12;
     event.legal_mask=student_retire_until[pos]>=o.day?(1<<12):((1<<1)|(1<<12));
     if(!(event.legal_mask&(1<<event.label)))event.label=12;
     auto&state=event.state;state.pos=pos;state.animals=student_existing_animals;state.owned=owned;
     state.slot_index=student_release_audit.size();state.slot_count=student_release_count;
     state.stock_left=student_initial_stock;state.prefix=portfolio;
     int selected=student_release_selector(event);
     if((selected!=1&&selected!=12)||!(event.legal_mask&(1<<selected)))throw std::runtime_error("student animal lifecycle selector returned illegal choice");
     student_feed_choice[pos]=selected==1;
     if(selected==12)student_retire_until[pos]=std::max(student_retire_until[pos],o.day+(t.consecutive_unfed==0));
     else student_retire_until[pos]=-1;
     std::array<int8_t,2>first{student_feed_choice[pos],int8_t(student_feed_choice[pos]?forecast_service[pos][1]:0)};
     paths[pos]=animal_path(int(t.animal),t.placed_day,pos,path_prices(),&t,&forecast_service[pos],&first);
     event.label=selected;student_release_audit.push_back(event);
    }
#endif
    add(portfolio,paths[pos]);core.target.emplace_back(pos,int(t.animal));animals++;book[pos]={int(t.animal),t.placed_day,o.day,0,-1,true};continue;}
#if R2_STUDENT_LIFECYCLE_V4
   student_retire_until[pos]=-1;
#endif
   if(plant(t)){
   int k=int(t.crop),finish=std::max(o.day,t.planted_day+(ongoing(k)?first[k]+3*interval[k]:core.h_age(k)));
#if R2_STUDENT_SLOT_AUDIT
    int student_keep_finish=-1;double student_keep_value=-1e100;
#endif
    if(!ongoing(k)){
     auto dp=rotations_dp(pos,path_prices(),o.day);double best=-1e100;
     int earliest=std::max(o.day,int(t.planted_day)+first[k]);int last=std::min(29,int(t.planted_day)+(k==W?4:k==C?3:12));
     for(int d=earliest;d<=last;d++){
      auto a=crop(k,t.planted_day,pos,d,path_prices(),&t);double val=scalar(a,path_prices(),o.day)+(s.rotation>0?dp.v[d]:0);
      if(val>best){best=val;finish=d;}
#if R2_STUDENT_SLOT_AUDIT
      if(d>o.day&&val>student_keep_value){student_keep_value=val;student_keep_finish=d;}
#endif
     }
    }
    if constexpr(P16_JOINT_BUNDLES>0){int j=joint.index(pos);if(j>=0&&joint.slots[j].stage==1&&k==W)finish=std::max(o.day,int(t.planted_day)+joint.slots[j].source_age);}
    finish=std::min(29,finish);
    int retained=(book[pos].birth==t.planted_day&&book[pos].kind==k)?book[pos].successor:-1;
    if(retained>=0&&o.priv.seeds[retained]>0&&o.day>=t.planted_day+first[k]&&joint.index(pos)<0)finish=o.day;
#if R2_STUDENT_SLOT_AUDIT
#if R2_STUDENT_LIFECYCLE_V4
    if(student_release_selector){
     int earliest=std::max(o.day,int(t.planted_day)+first[k]);
     int last=std::min(29,int(t.planted_day)+(k==W?4:k==C?3:12));
     bool ripe=!ongoing(k)&&earliest<=o.day&&o.day<=last;
     bool keep=ongoing(k)||!ripe||student_keep_finish>=0;
     int mask=(keep?1<<1:0)|(ripe?1<<2:0)|(o.day<29&&!ripe?1<<11:0);
     if(!mask)throw std::runtime_error("student crop lifecycle has no legal choice");
     StudentReleaseAudit event;event.pos=pos;event.label=ripe&&finish==o.day?2:1;event.legal_mask=mask;
     if(!(mask&(1<<event.label)))event.label=ripe?2:11;
     auto&state=event.state;state.pos=pos;state.animals=student_existing_animals;state.owned=owned;
     state.slot_index=student_release_audit.size();state.slot_count=student_release_count;
     state.stock_left=student_initial_stock;state.prefix=portfolio;
     int selected=student_release_selector(event);
     if((selected!=1&&selected!=2&&selected!=11)||!(mask&(1<<selected)))throw std::runtime_error("student crop lifecycle selector returned illegal choice");
     if(selected==2||selected==11)finish=o.day;
     else if(!ongoing(k)&&student_keep_finish>=0)finish=student_keep_finish;
     core.student_crop_release[pos]=selected==2||selected==11;
     core.student_crop_dig[pos]=selected==11;
     event.label=selected;student_release_audit.push_back(event);
    }
#else
    if(!ongoing(k)&&student_release_selector){
     int earliest=std::max(o.day,int(t.planted_day)+first[k]);
     int last=std::min(29,int(t.planted_day)+(k==W?4:k==C?3:12));
     bool joint_later=false;if constexpr(P16_JOINT_BUNDLES>0){int j=joint.index(pos);joint_later=j>=0&&joint.slots[j].stage==1&&k==W&&int(t.planted_day)+joint.slots[j].source_age>o.day;}
     bool retained_now=retained>=0&&o.priv.seeds[retained]>0&&o.day>=t.planted_day+first[k]&&joint.index(pos)<0;
     int mask=0;if(student_keep_finish>=0&&!retained_now)mask|=1<<1;if(earliest<=o.day&&o.day<=last&&!joint_later)mask|=1<<2;
     if(!mask)mask=1<<(finish==o.day?2:1);
     StudentReleaseAudit release_audit;release_audit.pos=pos;release_audit.label=finish==o.day?2:1;release_audit.legal_mask=mask;
     auto&state=release_audit.state;state.pos=pos;state.animals=student_existing_animals;state.owned=owned;
     state.slot_index=student_release_audit.size();state.slot_count=student_release_count;
     state.stock_left=student_initial_stock;state.prefix=portfolio;
     int selected=student_release_selector(release_audit);
     if((selected!=1&&selected!=2)||!(mask&(1<<selected)))throw std::runtime_error("student release selector returned illegal choice");
     if(selected==2)finish=o.day;else if(student_keep_finish>=0)finish=student_keep_finish;
     release_audit.label=selected;student_release_audit.push_back(release_audit);
    }
#endif
#endif
#if R2_STUDENT_LIFECYCLE_V4
    if(core.student_crop_dig[pos]){
     Asset scrap;scrap.kind=k;scrap.end=o.day;
     scrap.labor[o.day]=1+.06*near(pos)+.10;
     if(t.yield_units>0&&o.day-t.planted_day>=first[k]){
      scrap.f[o.day][k]+=t.yield_units;scrap.labor[o.day]+=1+.06*near(pos)+.10;
     }
     paths[pos]=scrap;
    }else
#endif
    paths[pos]=crop(k,t.planted_day,pos,finish,path_prices(),&t,nullptr,-1,-1,s.marginal_value>0?&forecast_crop_service[pos]:nullptr);add(portfolio,paths[pos]);
#if R2_FINITE_FERTILIZER
    if(!ongoing(k))finite_first_fertilizer[pos]=paths[pos].f[o.day][F]<0;
#endif
    core.triad_crop_age[pos]=finish-t.planted_day;core.target.emplace_back(pos,k);book[pos]={k,t.planted_day,o.day,finish-t.planted_day,retained,true};
    release[pos]=finish+(ongoing(k)?0:0);
    if(finish==o.day&&o.day<29)free.push_back(pos);
    continue;
   }
   core.target.emplace_back(pos,-1);free.push_back(pos);
  }
  update_value(o);
  // Near-depot high-maintenance slots are selected first; no fixed day/plot tape.
  std::stable_sort(free.begin(),free.end(),[&](int a,int b){return std::tuple(near(a),a)<std::tuple(near(b),b);});
  double stock=0;for(int i=0;i<9;i++)stock+=revenue(i,o.market.inventory[i],o.priv.shed[i]);
  double budget=std::max(0.,o.own.money+.85*stock-s.reserve-animals*o.market.prices[W]*s.feed_cover);
  if(o.day==0)budget*=s.opening_budget;
  double joint_earmark=P16_JOINT_BUNDLES?joint.seed_reserve(o):0.;
  if constexpr(P16_JOINT_BUNDLES>0)budget-=joint_earmark;
  if constexpr(R2_OPTIMIZER_AUDIT)greedy_budget_initial=budget;
  auto settarget=[&](int pos,int k){for(auto&[p,c]:core.target)if(p==pos){c=k;return;}core.target.emplace_back(pos,k);};
  std::array<int,12>stock_left{};for(int k=0;k<5;k++)stock_left[k]=o.priv.seeds[k];
  for(int k=9;k<12;k++){stock_left[k]=o.priv.shed[k];for(auto&iv:o.priv.inventories)stock_left[k]+=iv[k];}
  // Funded unfinished placements survive day boundaries; unit IDs do not,
  // because the engine dismisses hands at dawn. New owners are safely assigned.
  for(int pos:free){
   auto&b=book[pos];const auto&t=farm.tiles[pos];
   // A harvested incumbent may disappear before its paid successor is planted.
   // Preserve that successor across the EMPTY/WEED boundary too.
   if(s.keep_commitments>0&&!plant(t)&&!animal(t)&&b.successor>=0&&stock_left[b.successor]>0){
    b.kind=b.successor;b.birth=-1;b.successor=-1;b.funded=true;
   }
   if(s.keep_commitments>0&&plant(t)&&b.successor>=0&&stock_left[b.successor]>0){
    int k=b.successor;auto dp=rotations_dp(pos,path_prices(),o.day);auto candidate=choose_crop(k,o.day,pos,path_prices(),dp);
    if(candidate.kind>=0){auto a=candidate.a;a.fixed[o.day]+=seed_price[k];a.first_cost=0;
     stock_left[k]--;paths[pos]=a;add(portfolio,a);settarget(pos,k);successor[pos]=k;length[pos]=candidate.length;kept++;continue;}
   }
   if(s.keep_commitments<=0||plant(t)||animal(t)||b.kind<0||stock_left[b.kind]<=0)continue;
   int k=b.kind;auto dp=rotations_dp(pos,path_prices(),o.day);Asset a=k>=9?animal_path(k,o.day,pos,path_prices()):choose_crop(k,o.day,pos,path_prices(),dp).a;
   if(a.first_cost<=0)continue;a.fixed[o.day]+=a.first_cost;a.first_cost=0;stock_left[k]--;b.funded=true;
   paths[pos]=a;add(portfolio,a);settarget(pos,k);successor[pos]=k;animals+=k>=9;kept++;
  }
  std::vector<int>new_positions;
  // Inject only the current feasible stage; future receipts are never cash.
  if constexpr(P16_JOINT_BUNDLES>0)if(joint.active){
   for(int i=0;i<joint.count;i++){
    const auto&b=joint.slots[i];int pos=b.pos;const auto&t=farm.tiles[pos];
    if(b.stage>=3)continue;
    int k=joint.force_kind(pos,o.day);
    if(k<0||plant(t)||animal(t)||t.kind==TileKind::LOCKED){core.plant_not_before[pos]=30;continue;}
    if(successor[pos]>=0){ // Paid incumbent is protected, not overwritten.
     if(successor[pos]!=k){joint.cancel(5);break;}continue;
    }
    auto dp=rotations_dp(pos,path_prices(),o.day);auto selected=choose_crop(k,o.day,pos,path_prices(),dp);
    if(selected.kind<0){core.plant_not_before[pos]=30;continue;}
    int len=b.stage==0?b.source_age:selected.length;
    auto a=crop(k,o.day,pos,std::min(29,o.day+len),path_prices());
    double cost=stock_left[k]>0?0:a.first_cost;
    if(b.stage==2){double released=std::min(joint_earmark,double(seed_price[k]));budget+=released;joint_earmark-=released;}
    if(cost>budget||a.first_cost<=0){core.plant_not_before[pos]=30;continue;}
    if(stock_left[k]>0){a.fixed[o.day]+=a.first_cost;a.first_cost=0;stock_left[k]--;}
    budget-=cost;paths[pos]=a;add(portfolio,a);settarget(pos,k);successor[pos]=k;length[pos]=len;
    core.triad_crop_age[pos]=len;book[pos]={k,-1,o.day,len,-1,cost==0};new_positions.push_back(pos);
   }
  }
  free.erase(std::remove_if(free.begin(),free.end(),[&](int p){return successor[p]>=0||joint.index(p)>=0;}),free.end());
  double current=update_value(o);
  int used=0;
  while(o.day<29){
   if(used>=int(free.size())){
    if(owned>=int(s.max_land)||o.day>20||budget<next_land_cost(owned)+300)break;
    double cost=next_land_cost(owned);owned++;core.planned_land=owned;budget-=cost;portfolio.fixed[o.day]-=cost;
    for(int p=0;p<100;p++)if(quad(p)==owned-1){free.push_back(p);settarget(p,-1);}
    std::stable_sort(free.begin()+used,free.end(),[](int a,int b){return std::tuple(near(a),a)<std::tuple(near(b),b);});current=update_value(o);
   }
   int pos=free[used++];auto&t=farm.tiles[pos];auto dp=rotations_dp(pos,path_prices(),o.day);
   double best=0,bestval=current,bestgain=0;int bestkind=-1,bestlen=0;Asset bestpath;double bestcost=0;
   GreedySlotAudit audit;if constexpr(R2_OPTIMIZER_AUDIT){audit.pos=pos;audit.budget=budget;}
#if R2_STUDENT_SLOT_AUDIT
   StudentSlotAudit student;student.pos=pos;student.budget=budget;student.animals=animals;
   student.owned=owned;student.slot_index=used-1;student.slot_count=free.size();
   student.stock_left=stock_left;student.prefix=portfolio;
#endif
#if R2_STUDENT_SLOT_AUDIT
   struct StudentAlternative{bool legal=false;Asset path{};int len=0;double val=0,gain=0,rank=0,cost=0;};
   std::array<StudentAlternative,12>student_alternatives{};
#endif
   int student_kind_index=0;for(int k:{0,1,2,3,4,9,10,11}){
    ++student_kind_index;
    if(k>=9&&((plant(t)
#if R2_STUDENT_LIFECYCLE_V4
       &&!core.student_crop_release[pos]
#endif
       )||animals>=core.p.max_animals||o.day+afirst[k-9]>29))continue;
    Asset a;int len=0;
    if(k>=9)a=animal_path(k,o.day,pos,path_prices());else{auto c=choose_crop(k,o.day,pos,path_prices(),dp);if(c.kind<0)continue;a=c.a;len=c.length;}
    double cost=a.first_cost;
    if(cost<=0)continue;if(stock_left[k]>0){a.fixed[o.day]+=cost;cost=0;}
    double immediate=cost+(k>=9?o.market.prices[W]*s.feed_cover:0);
    if(immediate>budget)continue;
#if R2_STUDENT_SLOT_AUDIT
    student.legal_mask|=1<<student_kind_index;
#endif
#if R2_OUTER_NEIGHBOR_AUDIT
    // -2 means unconstrained.  -1 deliberately leaves this slot empty and
    // therefore terminates the greedy suffix; non-negative values force one
    // of the eight production kinds.  Keep this after legality accounting so
    // an external actor receives the full mask, not a one-hot forced mask.
    if(forced_kind[pos]!=-2&&k!=forced_kind[pos])continue;
#endif
    auto next=portfolio;add(next,a);double val=model.value(o,next);candidates++;
    double value_gain=val-current,gain=value_gain-s.land_rent*(k>=9?29-o.day:len);
    double rank=value_gain;
    if constexpr(!R2_GREEDY_VALUE_ORDER){
     rank=gain/std::pow(std::max(10.,cost)+(k>=9?100:0),s.capital_power*std::max(0.,1-o.own.money/16000.));
     rank*=k>=9?s.animal_bias:s.crop_bias;
    }
#if R2_STUDENT_SLOT_AUDIT
    student_alternatives[k]={true,a,len,val,gain,rank,immediate};
#endif
    if constexpr(R2_OPTIMIZER_AUDIT){audit.candidates++;if(value_gain>audit.raw_value_gain){audit.raw_value_gain=value_gain;audit.value_kind=k;}if(gain>audit.raw_gain){audit.raw_gain=gain;audit.raw_kind=k;audit.raw_rank=rank;}}
    // The production optimizer only admits positive marginal projects.  In a
    // diagnostic student build, however, every feasibility-mask bit is an
    // executable action: once the external actor forces that kind, preserve
    // it even when R1's own marginal score is non-positive.
    bool take=rank>best;
#if R2_OUTER_NEIGHBOR_AUDIT
    take=take||forced_kind[pos]==k;
#endif
    if(take){best=rank;bestval=val;bestgain=gain;bestkind=k;bestlen=len;bestpath=a;bestcost=immediate;if constexpr(R2_OPTIMIZER_AUDIT)audit.picked_value_gain=value_gain;}
   }
#if R2_STUDENT_SLOT_AUDIT
   if(student_selector){
    // Expose the native proposal as a diagnostic only; the callback's return
    // is the actual action.  SKIP is -1 and is always represented by bit 0.
    student.label=bestkind;
    int selected=student_selector(student);
    int bit=selected==-1?0:selected>=0&&selected<5?selected+1:selected>=9&&selected<12?selected-3:-1;
    if(bit<0||!(student.legal_mask&(1<<bit)))throw std::runtime_error("student selector returned illegal kind");
    if(selected<0){best=bestval=bestgain=bestcost=0;bestkind=-1;bestlen=0;bestpath={};}
    else{const auto&a=student_alternatives[selected];if(!a.legal)throw std::runtime_error("student selector missing alternative");
     best=a.rank;bestval=a.val;bestgain=a.gain;bestkind=selected;bestlen=a.len;bestpath=a.path;bestcost=a.cost;}
   }
   bool student_v3_none=false,student_v3_stop=false;
   if(student_v3_selector){
    // Unified v3 classes: STOP, NONE_OR_KEEP, RELEASE, then the eight
    // production kinds.  RELEASE is masked here; KEEP is the same class as
    // placement NONE so one recurrent actor serves both event stages.
    int mask=3|(student.legal_mask&~1)<<2;
    student.label=bestkind;int selected=student_v3_selector(student,mask);
    if(selected<0||selected>=11||!(mask&(1<<selected)))throw std::runtime_error("student v3 selector returned illegal class");
    student_v3_stop=selected==0;student_v3_none=selected==1;
    if(selected>=3){static constexpr std::array<int,8>kinds{0,1,2,3,4,9,10,11};int chosen=kinds[selected-3];
     const auto&a=student_alternatives[chosen];if(!a.legal)throw std::runtime_error("student v3 selector missing alternative");
     best=a.rank;bestval=a.val;bestgain=a.gain;bestkind=chosen;bestlen=a.len;bestpath=a.path;bestcost=a.cost;}
    else{best=bestval=bestgain=bestcost=0;bestkind=-1;bestlen=0;bestpath={};}
    student_v3_slot_audit.push_back({student,selected,mask});
   }
#endif
   if constexpr(R2_OPTIMIZER_AUDIT){audit.picked_kind=bestkind;audit.picked_gain=bestgain;audit.picked_rank=best;if(audit.candidates)greedy_audit.push_back(audit);}
#if R2_STUDENT_SLOT_AUDIT
   student.label=bestkind;student.terminal=bestkind<0;
   student_slot_audit.push_back(std::move(student));
#endif
#if R2_STUDENT_SLOT_AUDIT
   if(student_v3_none)continue;if(student_v3_stop)break;
#endif
   if(bestkind<0){
#if R2_OPTIMIZER_AUDIT
    {
     greedy_stop_pos=pos;
     for(int j=used;j<int(free.size());j++){
      int later=free[j];if(farm.tiles[later].kind!=TileKind::EMPTY)continue;
      greedy_stop_skipped_empty++;
      if(greedy_stop_examined_empty>=3)continue;
      greedy_stop_examined_empty++;
      auto probe_model=model;auto probe_dp=rotations_dp(later,path_prices(),o.day);
      double best_later=0;
      for(int k:{0,1,2,3,4,9,10,11}){
       if(k>=9&&(animals>=core.p.max_animals||o.day+afirst[k-9]>29))continue;
       Asset a;int len=0;
       if(k>=9)a=animal_path(k,o.day,later,path_prices());
       else{auto c=choose_crop(k,o.day,later,path_prices(),probe_dp);if(c.kind<0)continue;a=c.a;len=c.length;}
       double cost=a.first_cost;if(cost<=0)continue;
       if(stock_left[k]>0){a.fixed[o.day]+=cost;cost=0;}
       double immediate=cost+(k>=9?o.market.prices[W]*s.feed_cover:0);
       if(immediate>budget)continue;
       auto next=portfolio;add(next,a);
       double gain=probe_model.value(o,next)-current-s.land_rent*(k>=9?29-o.day:len);
       best_later=std::max(best_later,gain);
      }
      greedy_stop_best_later_gain=std::max(greedy_stop_best_later_gain,best_later);
      greedy_stop_positive_empty+=best_later>1e-6;
     }
    }
#endif
    break;
   }
   settarget(pos,bestkind);successor[pos]=bestkind;length[pos]=bestlen;paths[pos]=bestpath;add(portfolio,bestpath);
   current=bestval;budget-=bestcost;animals+=bestkind>=9;if(stock_left[bestkind]>0)stock_left[bestkind]--;
   book[pos].successor=plant(t)?bestkind:-1;
   if(!plant(t))book[pos]={bestkind,-1,o.day,bestlen,-1,bestcost==0};
   else rotations+=int(t.crop)!=bestkind;
   new_positions.push_back(pos);update_value(o);
  }
  // Optional portfolio coordinate search.  Only replace projects created on
  // truly empty plots in this plan: never touch live assets, inventory-funded
  // commitments, joint bundles, or successors behind an incumbent crop.
  for(int pass=0;pass<(int(s.portfolio_swaps)==1);pass++){
   double top=s.portfolio_swap_min_gain,topval=current,topcost=0;int toppos=-1,topkind=-1,toplen=0;Asset toppath;std::array<int8_t,2>topservice{};
   for(int pos:new_positions){
    const auto&t=farm.tiles[pos];int old=successor[pos];
    if(plant(t)||animal(t)||joint.index(pos)>=0||old<0||book[pos].funded||paths[pos].fixed[o.day]>=-1e-9)continue;
    double oldcost=paths[pos].first_cost+(old>=9?o.market.prices[W]*s.feed_cover:0);
    double available=budget+oldcost;int after_remove=animals-(old>=9);auto without=portfolio;add(without,paths[pos],-1);
    auto dp=rotations_dp(pos,path_prices(),o.day);
    for(int k:{0,1,2,3,4,9,10,11}){
     if(k==old||stock_left[k]>0)continue;
     if(k>=9&&(after_remove>=core.p.max_animals||o.day+afirst[k-9]>29))continue;
     Asset a;int len=0;std::array<int8_t,2>service{};
     if(k>=9)a=animal_path(k,o.day,pos,path_prices(),nullptr,&service);
     else{auto c=choose_crop(k,o.day,pos,path_prices(),dp);if(c.kind<0)continue;a=c.a;len=c.length;}
     double immediate=a.first_cost+(k>=9?o.market.prices[W]*s.feed_cover:0);
     if(a.first_cost<=0||immediate>available)continue;
     auto next=without;add(next,a);double val=model.value(o,next);portfolio_swap_trials++;
     int oldspan=old>=9?29-o.day:length[pos],newspan=k>=9?29-o.day:len;
     double gain=val-current-s.land_rent*(newspan-oldspan);
     if(gain>top){top=gain;topval=val;topcost=immediate;toppos=pos;topkind=k;toplen=len;toppath=std::move(a);topservice=service;}
    }
   }
   if(toppos<0)break;
   int old=successor[toppos];double oldcost=paths[toppos].first_cost+(old>=9?o.market.prices[W]*s.feed_cover:0);
   add(portfolio,paths[toppos],-1);add(portfolio,toppath);budget+=oldcost-topcost;animals+=(topkind>=9)-(old>=9);
   paths[toppos]=std::move(toppath);successor[toppos]=topkind;length[toppos]=toplen;settarget(toppos,topkind);
   book[toppos]={topkind,-1,o.day,toplen,-1,false};forecast_service[toppos]=topservice;
   core.triad_crop_age[toppos]=topkind<5?toplen:-1;
#if R2_FINITE_FERTILIZER
   finite_first_fertilizer[toppos]=topkind<5&&!ongoing(topkind)&&paths[toppos].f[o.day][F]<0;
#endif
   current=topval;portfolio_swap_accepts++;portfolio_swap_gain+=top;update_value(o);
  }
  // Mode 2 is a bounded two-site neighbourhood.  It can cross a budget or
  // market-price valley that no profitable one-site replacement can cross.
  if(int(s.portfolio_swaps)>=2){
   struct Option {int kind=-1,len=0;Asset path{};std::array<int8_t,2>service{};double cost=0,solo=-1e100;};
   struct Site {int pos=-1,old=-1;double priority=-1e100;std::vector<Option> options;};
   std::vector<Site>sites;
   for(int pos:new_positions){
    const auto&t=farm.tiles[pos];int old=successor[pos];
    if(plant(t)||animal(t)||joint.index(pos)>=0||old<0||book[pos].funded||paths[pos].fixed[o.day]>=-1e-9)continue;
    auto without=portfolio;add(without,paths[pos],-1);auto dp=rotations_dp(pos,path_prices(),o.day);Site site;site.pos=pos;site.old=old;
    for(int k:{0,1,2,3,4,9,10,11}){
     if(k==old||stock_left[k]>0)continue;
     Asset a;int len=0;std::array<int8_t,2>service{};
     if(k>=9)a=animal_path(k,o.day,pos,path_prices(),nullptr,&service);
     else{auto c=choose_crop(k,o.day,pos,path_prices(),dp);if(c.kind<0)continue;a=c.a;len=c.length;}
     double cost=a.first_cost+(k>=9?o.market.prices[W]*s.feed_cover:0);if(a.first_cost<=0)continue;
     auto trial=without;add(trial,a);double val=model.value(o,trial);
     int oldspan=old>=9?29-o.day:length[pos],newspan=k>=9?29-o.day:len;
     double solo=val-current-s.land_rent*(newspan-oldspan);
     site.options.push_back({k,len,std::move(a),service,cost,solo});site.priority=std::max(site.priority,solo);
    }
    std::stable_sort(site.options.begin(),site.options.end(),[](const Option&a,const Option&b){return a.solo>b.solo;});
    if(site.options.size()>3)site.options.resize(3);if(!site.options.empty())sites.push_back(std::move(site));
    if(sites.size()>=8)break;
   }
   // ponytail: inspect 8 representative near-first sites, then jointly search
   // the top 4 x 3 alternatives; widen only if coverage, not model error, limits.
   std::stable_sort(sites.begin(),sites.end(),[](const Site&a,const Site&b){return a.priority>b.priority;});
   if(sites.size()>4)sites.resize(4);
   double top=s.portfolio_swap_min_gain,topval=current;int bi=-1,bj=-1,ba=-1,bb=-1;
   for(int i=0;i<int(sites.size());i++)for(int j=i+1;j<int(sites.size());j++){
    auto&a=sites[i];auto&b=sites[j];double oldcost=paths[a.pos].first_cost+paths[b.pos].first_cost;
    if(a.old>=9)oldcost+=o.market.prices[W]*s.feed_cover;if(b.old>=9)oldcost+=o.market.prices[W]*s.feed_cover;
    double available=budget+oldcost;int after=animals-(a.old>=9)-(b.old>=9);
    auto without=portfolio;add(without,paths[a.pos],-1);add(without,paths[b.pos],-1);
    for(int x=0;x<int(a.options.size());x++)for(int y=0;y<int(b.options.size());y++){
     auto&u=a.options[x];auto&v=b.options[y];if(u.cost+v.cost>available||after+(u.kind>=9)+(v.kind>=9)>core.p.max_animals)continue;
     if(u.kind>=9&&o.day+afirst[u.kind-9]>29)continue;if(v.kind>=9&&o.day+afirst[v.kind-9]>29)continue;
     auto trial=without;add(trial,u.path);add(trial,v.path);double val=model.value(o,trial);portfolio_pair_trials++;
     int oldspan=(a.old>=9?29-o.day:length[a.pos])+(b.old>=9?29-o.day:length[b.pos]);
     int newsp=(u.kind>=9?29-o.day:u.len)+(v.kind>=9?29-o.day:v.len);
     double gain=val-current-s.land_rent*(newsp-oldspan);
     if(gain>top){top=gain;topval=val;bi=i;bj=j;ba=x;bb=y;}
    }
   }
   if(bi>=0){
    auto install=[&](const Site&site,const Option&op){int pos=site.pos,old=successor[pos];double oldcost=paths[pos].first_cost+(old>=9?o.market.prices[W]*s.feed_cover:0);
     add(portfolio,paths[pos],-1);add(portfolio,op.path);budget+=oldcost-op.cost;animals+=(op.kind>=9)-(old>=9);
     paths[pos]=op.path;successor[pos]=op.kind;length[pos]=op.len;settarget(pos,op.kind);book[pos]={op.kind,-1,o.day,op.len,-1,false};forecast_service[pos]=op.service;
     core.triad_crop_age[pos]=op.kind<5?op.len:-1;
#if R2_FINITE_FERTILIZER
     finite_first_fertilizer[pos]=op.kind<5&&!ongoing(op.kind)&&paths[pos].f[o.day][F]<0;
#endif
    };
    install(sites[bi],sites[bi].options[ba]);install(sites[bj],sites[bj].options[bb]);
    current=topval;portfolio_pair_accepts++;portfolio_pair_gain+=top;update_value(o);
   }
  }
  // No automatic same-crop continuation when a slot was offered but declined.
  for(int pos:free)if(plant(farm.tiles[pos])&&successor[pos]<0)core.plant_not_before[pos]=30;
  if constexpr(P16_JOINT_BUNDLES>0)if(joint.active)for(int i=0;i<joint.count;i++)if(joint.slots[i].stage==1)core.plant_not_before[joint.slots[i].pos]=30;
  if(s.service_reconcile>=2)reconcile_service(v);
  set_service(v);core.prepare_orders(v,o,expiry);
  if(s.preview>0&&!new_positions.empty()){
   for(int pass=0;pass<3;pass++){
    auto preview=core.preview_bundle(v,false);std::set<int>admitted;for(auto[p,k]:preview.started_targets)admitted.insert(p);
    if constexpr(R2_OPTIMIZER_AUDIT)if(!greedy_preview_checked){greedy_preview_checked=true;greedy_preview_proposed=preview.proposed;greedy_preview_started=int(preview.started_targets.size());}
    bool changed=false;
    for(int pos:new_positions){
     if(plant(farm.tiles[pos])||book[pos].funded||successor[pos]<0||admitted.count(pos))continue;
     add(portfolio,paths[pos],-1);paths[pos]={};successor[pos]=-1;settarget(pos,-1);book[pos]={};deferred++;if constexpr(R2_OPTIMIZER_AUDIT)greedy_preview_removed++;changed=true;
    }
    if(!changed)break;update_value(o);if(s.service_reconcile>=2)reconcile_service(v);set_service(v);core.prepare_orders(v,o,expiry);
   }
  }
  if constexpr(R2_OPTIMIZER_AUDIT)greedy_budget_final=budget;
  predicted=model.value(o,portfolio);model.shadow=prices;
  // Controller::plan builds the committed portfolio itself -- Planner::plan is not on this path --
  // so the executor's sale schedule has to be produced here, at the point where the portfolio is
  // final. act() then hands model.exec_plan to the filter copy it captures by value.
  // The DP is open-loop: it optimises against the production trajectory it is handed. The
  // committed portfolio's projection can drift from what the farm will actually produce, and a
  // plan that is optimal for a game we are not playing is worse than the reactive local rule it
  // replaces -- which is exactly what was measured (DP 71.9% vs the rule's 75.0% on thomas).
  // remaining_portfolio() rebuilds the flow from the REAL tiles at their actual ages and yields.
  // NOTE: the sale plan is NOT built here. See SearchController::act in search.hpp -- choose()/
  // install() replace `live` with a proposal copy, so this function never runs on the controller
  // whose plan is executed, and building it here only paid that cost on every evaluated proposal.
 }

 std::string optimizer_json()const{
  std::ostringstream o;o.precision(17);o<<"{\"budget_initial\":"<<greedy_budget_initial<<",\"budget_final\":"<<greedy_budget_final
   <<",\"preview_checked\":"<<(greedy_preview_checked?"true":"false")<<",\"preview_proposed\":"<<greedy_preview_proposed
   <<",\"preview_started\":"<<greedy_preview_started<<",\"preview_removed\":"<<greedy_preview_removed
   <<",\"swap_trials\":"<<portfolio_swap_trials<<",\"swap_accepts\":"<<portfolio_swap_accepts<<",\"swap_gain\":"<<portfolio_swap_gain
   <<",\"pair_trials\":"<<portfolio_pair_trials<<",\"pair_accepts\":"<<portfolio_pair_accepts<<",\"pair_gain\":"<<portfolio_pair_gain;
#if R2_OPTIMIZER_AUDIT
  o<<",\"stop_pos\":"<<greedy_stop_pos<<",\"stop_skipped_empty\":"<<greedy_stop_skipped_empty
   <<",\"stop_examined_empty\":"<<greedy_stop_examined_empty<<",\"stop_positive_empty\":"<<greedy_stop_positive_empty
   <<",\"stop_best_later_gain\":"<<greedy_stop_best_later_gain;
#endif
  o<<",\"slots\":[";
  for(size_t i=0;i<greedy_audit.size();i++){if(i)o<<",";const auto&a=greedy_audit[i];
   o<<"{\"pos\":"<<a.pos<<",\"budget\":"<<a.budget<<",\"candidates\":"<<a.candidates<<",\"picked_kind\":"<<a.picked_kind
    <<",\"raw_kind\":"<<a.raw_kind<<",\"value_kind\":"<<a.value_kind<<",\"picked_gain\":"<<a.picked_gain<<",\"raw_gain\":"<<a.raw_gain
    <<",\"local_regret\":"<<(a.picked_kind<0?std::max(0.,a.raw_gain):a.raw_gain-a.picked_gain)
    <<",\"picked_value_gain\":"<<a.picked_value_gain<<",\"raw_value_gain\":"<<a.raw_value_gain
    <<",\"value_regret\":"<<(a.picked_kind<0?std::max(0.,a.raw_value_gain):a.raw_value_gain-a.picked_value_gain)
    <<",\"picked_rank\":"<<a.picked_rank<<",\"raw_rank\":"<<a.raw_rank<<"}";
  }o<<"]}";return o.str();
 }

#ifndef P16_LIVE_REMAINING_VALUE
#define P16_LIVE_REMAINING_VALUE 0
#endif
 Asset remaining_portfolio(const View&o,Planner&fresh,Flow&px)const{
  fresh=model;fresh.day=o.day;fresh.demand(o);fresh.public_rival(o);
  Asset total;
  for(int i=0;i<9;i++){total.f[o.day][i]+=o.priv.shed[i];for(const auto&iv:o.priv.inventories)total.f[o.day][i]+=iv[i];}
  fresh.value(o,total,&px);
  if(s.marginal_value>0)px=MarginalValue::compute(fresh,o,total,s.discount);
  // Only remaining work/production from real assets. Morning delivered goods
  // are not added for a second time, and paid setup costs are never repaid.
  for(int pos=0;pos<100;pos++){
   const auto&t=o.own.tiles[pos];Asset a;
   if(animal(t)){
    std::array<int8_t,2>chosen{core.service_feed[pos],core.service_care[pos]};
    a=animal_path(int(t.animal),t.placed_day,pos,px,&t,nullptr,core.animal_service_day==o.day?&chosen:nullptr);
   }
   else if(plant(t)){
    int k=int(t.crop),age=core.triad_crop_age[pos];
    int finish=std::min(29,std::max(o.day,int(t.planted_day)+(age>=0?age:core.h_age(k))));
    bool controlled=core.crop_service_day==o.day&&core.crop_birth[pos]==t.planted_day&&core.crop_kind[pos]==k;
    a=crop(k,t.planted_day,pos,finish,px,&t,nullptr,controlled?core.crop_water[pos]:-1,controlled?core.crop_fertilize[pos]:-1);
   }else continue;
   add(total,a);
  }
  fresh.value(o,total,&px);
  if(s.marginal_value>0)px=MarginalValue::compute(fresh,o,total,s.discount);
  return total;
 }
 std::array<double,12>intraday_values(const dp7::Controller&c,const View&o,const PlayerAction&out){
  std::array<double,12>v;v.fill(-1e90);int pos=-1;for(auto[p,k]:c.target)if(!plant(o.own.tiles[p])&&!animal(o.own.tiles[p])){pos=p;break;}if(pos<0)return v;
  // Compare only incremental project streams; the admission compiler checks
  // actual cash, inputs, free workers and PLANT+WATER completion time.
  Planner fresh;Flow current_prices=path_prices();Asset current=portfolio;
  if constexpr(P16_LIVE_REMAINING_VALUE>0)current=remaining_portfolio(o,fresh,current_prices);
  const Planner&eval=P16_LIVE_REMAINING_VALUE?fresh:model;
  auto dp=rotations_dp(pos,current_prices,o.day);
  for(int k:{0,1,2,3,4,9,10,11}){
   Asset a=k>=9?animal_path(k,o.day,pos,current_prices):choose_crop(k,o.day,pos,current_prices,dp).a;if(a.first_cost<=0)continue;
   auto b=current;add(b,a);v[k]=eval.value(o,b)-eval.value(o,current);
  }return v;
 }
 // Own-unit projection has no access to the live simulator or rival inventory.
 // It only confirms which holdings would reach the shed in OUR unit phase;
 // sale quantities are recalculated from the next actual observation.
 void settle_market(const View&o,PlayerAction&out,bool preparing){
  // ORIGINAL bail-out, restored. An earlier revision let this run whenever a sale plan was active,
  // which did far more than apply the plan: it also reached the `preparing` FINANCING branch below,
  // a path the original NEVER executes because this line always returns first. That branch adds and
  // rewrites SELL orders on the day's first step, so `sale_dp=1` differed from `sale_dp=0` by two
  // couplings, not one -- and only one of them was the DP. With the bail-out restored, enabling the
  // DP changes exactly the sell QUANTITIES on non-preparing steps and nothing else.
  if(s.delay_sale<0||(preparing&&s.feed_finance<=0))return;
  fastkag::ObservedDayScenario scenario(o);
  auto post=scenario.project_units(out.units,-1);
  auto &priv=post.privates()[0];Counts reserve{},sell{};
  if(o.day<29){
   reserve=dp7::intraday::reserved(core).shed;
   if(core.phase!=3){
    for(const auto&j:core.jobs(o))for(int i=0;i<12;i++)reserve[i]+=j.needs[i];
    for(auto&iv:priv.inventories)for(int i=0;i<12;i++)reserve[i]=std::max(0,reserve[i]-iv[i]);
   }
   // One feed wave is protected; it is a working-capital reservation, not
   // another copy of physical inventory. Never reserve a phantom future crop.
   int animals=0,heldfeed=0;for(const auto&t:post.farms()[0].tiles)animals+=animal(t);
   for(const auto&iv:priv.inventories)heldfeed+=iv[W];
   if(o.hour<23)reserve[W]=std::max(reserve[W],std::max(0,animals-heldfeed));
  }
  if constexpr(P16_JOINT_BUNDLES>0){int bag=0;for(const auto&iv:priv.inventories)bag+=iv[W];reserve[W]=std::max(reserve[W],std::max(0,joint.feed_cover(o)-bag));}
  for(int i=0;i<9;i++)sell[i]=std::max(0,priv.shed[i]-reserve[i]);
  // FINANCING sells must not be capped by the plan. The `preparing` branch below sells to fund
  // purchases the agent has already committed to (seeds, animals, hires); if the plan has already
  // shrunk sell[] to what it wants to release, that branch finds nothing to sell and the purchase
  // fails. Measured on seed 2780000035 (cash 10,652 at takeover): the DP arm beat the base by
  // +4,479 over days 12-20 and then lost 11,726 and 21,587 in the two following stretches -- the
  // compounding signature of an early failure to buy. Keep the unfiltered budget for that path.
  // The financing branch below needs the FULL sellable budget (it sells to fund purchases the
  // agent has already committed to), so keep an unfiltered copy before anything touches sell[].
  Counts sell_unfiltered=sell;
  // The plan is deliberately NOT applied on this step. It looks like the natural place to bind --
  // it is the day's main market decision -- but shrinking the sells here starves the purchases the
  // same step issues (seeds, animals, hires, feed), and the projection cannot see it because
  // project_own_market does not model the rival. Measured, same-segment paired: binding gave thomas
  // -5.2pp / melon +3.2pp / demand -2.1pp win rate (seed 2780000035: animals 21 -> 13); without it,
  // +0.0 / +8.4 / +4.2pp with no arm regressing. settle_market's own path below already applies the
  // plan on every OTHER step.

  if(preparing){
   // R2P1: only intervene in an already-issued preparation transaction with
   // demonstrable own-cash shortfall. This projection does not know rival
   // orders; real fills are still learned from the next observation.
   preparation_finance_checks++;
   auto before=post.project_own_market(0,out.market);
   bool shortfall=false;for(double x:before.last_market_cash_shortfalls()[0])shortfall|=x>0;
   // Do NOT give up when every slot is taken: financing works by GROWING the plan's own SELL order,
   // which needs no new slot. Returning here is what let the plan starve the day's purchases (feed
   // wheat, seeds, hires) -- measured on seed 2780000035 the DP arm's animals fell 21 -> 13 and it
   // gave up 28,178 of growth purely because this line fired.
   if(!shortfall)return;
   // Preserve every original order and its relative order. Sell only surplus
   // not already promised to an existing SELL, and never remove an order to
   // make room. Require the whole proposed purchase/hire sequence to fill in
   // the own-side projection, not merely a higher projected cash balance.
   sell=sell_unfiltered;
   for(auto a:out.market)if(a.op==Op::SELL&&int(a.item)>=0&&int(a.item)<9)
    sell[int(a.item)]=std::max(0,sell[int(a.item)]-std::max(0,a.quantity));
   // Financing sells are an INCREMENT on the plan's own order for that product, never a second
   // order: a duplicate would burn another of the ten market slots and split the quantity across
   // two quotes. `sell[i]` here is already what is left AFTER the plan's order, so the two are
   // never added together -- the existing order is grown by exactly the extra the purchase needs.
   for(int i=0;i<9&&sell[i]>0;i++){
    // Never finance a purchase out of the very product being purchased. WHEAT and FERTILIZER can be
    // BUY_PRODUCT, and growing a SELL for the same item would have us selling wheat to raise the
    // cash to buy wheat -- the engine would net the two against each other and the shed would not
    // gain the feed it was funded for.
    bool also_buying=false;
    for(const auto&a:out.market)
     if((a.op==Op::BUY_PRODUCT||a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL)&&int(a.item)==i){also_buying=true;break;}
    if(also_buying)continue;
    int slot=-1;
    for(size_t k=0;k<out.market.size();k++)
     if(out.market[k].op==Op::SELL&&int(out.market[k].item)==i){slot=int(k);break;}
    if(slot<0){
     if(out.market.size()>=10)continue;    // no order to grow and no free slot to add one
     out.market.push_back(action(Op::SELL,i,0));slot=int(out.market.size())-1;
    }
    int was=out.market[slot].quantity;
    int extra=0;
    for(int q=1;q<=sell[i];q++){
     out.market[slot].quantity=was+q;
     auto funded=post.project_own_market(0,out.market);auto&fills=funded.last_market_fills()[0];
     bool full=true;
     for(size_t k=0;k<out.market.size();k++){
      auto a=out.market[k];if(a.op==Op::SELL||a.op==Op::PASS)continue;
      int wanted=(a.op==Op::HIRE||a.op==Op::BUY_LAND)?1:std::max(0,a.quantity);
      full&=fills[k]>=wanted;
     }
     if(full){extra=q;break;}
    }
    if(extra>0){preparation_finance_repairs++;return;}
    out.market[slot].quantity=was;
   }
   return;
  }
  Acts buys;for(auto a:out.market)if(a.op!=Op::SELL)buys.push_back(a);
  local_sale.filter(o,priv,buys,sell,sale_memory,s.reserve,s.competition);
  auto sales=core.sales_sorted(o,sell);
  int slots=std::max(0,10-int(buys.size()));if(int(sales.size())>slots)sales.resize(slots);
  sales.insert(sales.end(),buys.begin(),buys.end());out.market=std::move(sales);
 }

 // The scheduler cannot spend future manure, but a recoverable liquidity
 // shortage must not silently erase today's funded maintenance commitment.
 // Materialise only current, physically reachable collateral. Re-admit the
 // original jobs from the NEXT real observation; never credit a hoped-for fill.
 std::optional<PlayerAction> working_capital_gate(const View&o){
  if constexpr(P16_WORKING_CAPITAL_GATE==0)return {};
  if(o.day>=29)return {};
  if(capital_pending){core.prepare_orders(o,o,0);capital_pending=false;}
  if(core.phase!=1&&core.phase!=2)return {};
  int missing=core.feed_shortfall(o);if(missing<=0)return {};
  double required=0;for(int k=0;k<missing;k++)required+=price(W,o.market.inventory[W]-k-1);
  if(o.own.money>=required)return {};
  PlayerAction out;out.units.resize(o.own.hands.size()+1);
  int room=100-sum(o.priv.shed);Counts incoming{};bool deposit=false,collect=false;
  auto pending=dp7::intraday::reserved(core).shed;
  Counts required_inputs{};for(const auto&job:core.jobs(o))add(required_inputs,job.needs);
  for(size_t u=0;u<out.units.size();u++){
   int pos=cell(u?o.own.hands[u-1]:o.own.farmer);if(!at_depot(pos))continue;
   const auto&bag=o.priv.inventories[u];const auto&t=o.own.tiles[pos];
   // Quantified surplus deposit avoids accidental animal/feed disposal.
   int chosen=-1;for(int i=1;i<9;i++)if(bag[i]>0&&(chosen<0||o.market.prices[i]>o.market.prices[chosen]))chosen=i;
   if(chosen>=0&&room>0){int q=std::min(room,bag[chosen]);out.units[u]=action(Op::PLACE,chosen,q);incoming[chosen]+=q;room-=q;deposit=true;}
   else if(animal(t)&&t.fertilizer_available){out.units[u]=action(Op::COLLECT_FERTILIZER);collect=true;}
  }
  // Current unit phase is fully deterministic for these selected atoms.
  fastkag::ObservedDayScenario scenario(o);auto projected=scenario.project_units(out.units,-1);
  Counts held{};for(const auto&bag:projected.privates()[0].inventories)add(held,bag);
  for(int i=0;i<12;i++)pending[i]=std::max(pending[i],std::max(0,required_inputs[i]-held[i]));
  Counts available{};for(int i=1;i<9;i++)available[i]=std::max(0,projected.privates()[0].shed[i]-pending[i]);
  auto sales=core.sales_sorted(o,available);
  double cash=o.own.money;Acts chosen_sales;
  for(auto a:sales){if(cash>=required)break;int q=0;while(q<a.quantity&&cash<required)cash+=price(int(a.item),o.market.inventory[int(a.item)]+q++);
   if(q){a.quantity=q;chosen_sales.push_back(a);}}
  if(cash>=required&&!chosen_sales.empty()){
   out.market=std::move(chosen_sales);out.market.push_back(action(Op::BUY_PRODUCT,W,missing));
   capital_financings++;capital_pending=true;return out;
  }
  if(deposit||collect){out.market=std::move(chosen_sales);capital_collects+=collect;capital_pending=true;return out;}
  return {};
 }

 PlayerAction act(const View&o){
  if(o.step<=previous_step)throw std::runtime_error("non-monotone policy observation");previous_step=o.step;
  joint.observe(o);
  if(o.day!=core.day)plan(o);
  // Hand the planner's committed sale schedule to the executor. This MUST happen before the
  // project_sales lambda below captures `local_sale` BY VALUE -- the filter that actually runs is
  // the captured copy, so setting it at any other call site leaves the executor on its local rule.
  local_sale.carry=(s.sale_dp>0&&model.exec_plan_ready&&o.day>=int(s.sale_dp_start_day))?&model.exec_carry:nullptr;
  local_sale.plan_step0=model.exec_plan_step;
  if constexpr(P16_JOINT_BUNDLES>0){
   core.p.max_animals=(joint.active&&o.day==joint.created_day)?std::min(int(s.max_animals),joint.animal_limit):int(s.max_animals);
   core.p.operating_reserve=s.reserve+joint.seed_reserve(o);
  }
  // Match conditional execution repair to the P16 market policy, rather
  // than silently valuing its work using R2's immediate-liquidation policy.
  // Capture snapshots by value: all controller copies remain self-contained.
  core.t3.project_sales=[timing=local_sale,clock=sale_memory,
                        reserve=s.reserve,competition=s.competition,feed_cover=joint.feed_cover(o)]
    (const View&v,const fastkag::PrivateState&priv,const dp7::Acts&buys,dp7::Counts&sell) mutable {
      if constexpr(P16_JOINT_BUNDLES>0)if(feed_cover>0){int bag=0;for(const auto&iv:priv.inventories)bag+=iv[W];sell[W]=std::min(sell[W],std::max(0,priv.shed[W]-std::max(0,feed_cover-bag)));}
      timing.filter(v,priv,buys,sell,clock,reserve,competition);
    };
  if(auto rescue=working_capital_gate(o)){joint.record(o,*rescue);return *rescue;}
  core.admission_values=[this](const dp7::Controller&c,const View&v,const PlayerAction&a){return intraday_values(c,v,a);};core.admission_blend=1;core.admission_scope=2;
  bool preparing=core.phase==1;auto out=core.act(o);settle_market(o,out,preparing);
  joint.record(o,out);return out;
 }
 std::string debug()const{
  std::ostringstream o;o<<"{\"plan_calls\":"<<plan_calls<<",\"capital_collects\":"<<capital_collects<<",\"capital_financings\":"<<capital_financings<<",\"input_rejected_atoms\":"<<core.input_rejected_atoms<<",\"input_rejected_feed\":"<<core.input_rejected_feed<<",\"input_rejected_fertilize\":"<<core.input_rejected_fertilize<<",\"marginal_value_updates\":"<<marginal_value_updates<<",\"candidates\":"<<candidates<<",\"deferred\":"<<deferred<<",\"kept_commitments\":"<<kept<<",\"rotations\":"<<rotations<<",\"portfolio_swap_trials\":"<<portfolio_swap_trials<<",\"portfolio_swap_accepts\":"<<portfolio_swap_accepts<<",\"portfolio_swap_gain\":"<<portfolio_swap_gain<<",\"portfolio_pair_trials\":"<<portfolio_pair_trials<<",\"portfolio_pair_accepts\":"<<portfolio_pair_accepts<<",\"portfolio_pair_gain\":"<<portfolio_pair_gain<<",\"intraday_started\":"<<core.intraday_activated<<",\"reference_calls\":0,\"predicted\":"<<predicted<<",\"service_trials\":"<<service_trials<<",\"service_switches\":"<<service_switches<<",\"service_value_gain\":"<<service_objective_gain<<",\"targets\":[";
  bool sep=false;for(auto[p,k]:core.target){if(sep)o<<",";sep=true;o<<"["<<p<<","<<k<<"]";}o<<"],\"preparation_finance_checks\":"<<preparation_finance_checks<<",\"preparation_finance_repairs\":"<<preparation_finance_repairs<<",\"midroute_delivery_checks\":"<<core.midroute_delivery_checks<<",\"midroute_delivery_insertions\":"<<core.midroute_delivery_insertions<<",\"midroute_delivery_quantity\":"<<core.midroute_delivery_quantity<<",\"local_sale_checks\":"<<local_sale.checks<<",\"local_sale_holds\":"<<local_sale.holds<<",\"local_sale_quantity\":"<<local_sale.held_quantity<<",\"local_sale_forced\":"<<local_sale.forced<<",\"local_sale_expected_gain\":"<<local_sale.projected_gain<<"}";return o.str();
 }
};
}
