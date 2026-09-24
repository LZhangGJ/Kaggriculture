#pragma once
// Triad DP: autonomous economic intent -> conditional calendar -> live executor.
// No J7 macros, opponent identities, tape library, live Simulator or RNG access.
#include "planner.hpp"
#include "animal_service_dp.hpp"
#include "ongoing_maintenance_dp.hpp"
#include "finite_fertilizer.hpp"
#include <sstream>
#include "executor/observed_day_scenario.hpp"
#include "local_sale_timing.hpp"
#include "r11_labor.hpp"
#include "r12_calendar.hpp"
#include "public_flow_scenario.hpp"
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
 // Recovered candidate expansion; unrelated inactive experiments omitted.
 double candidate_extra=0;
 // T3 switches: 0 is exact legacy behavior; >=1 couples service to forecast;
 // >=2 coordinates alternatives using the SAME portfolio value; >=3 adds
 // finite-difference market externalities to maintenance candidate generation.
 double service_reconcile=0,live_ledger=0,delivery_calendar=0,feed_finance=0;
 // R2: state-checked whole-cargo delivery; no economic forecast changes.
 double batch_delivery=0;
 // A06 r3: 0 legacy; 1 reserve pending inputs; 2 live remaining flows; 3 WAIT/NOW DP.
 double a06_reinvest=0; // modes 4/5: current-call-stream same-day net receipts
 // R4 scope: 0=r3 everywhere; 1=legacy r1 mechanism in prediction only;
 // 2=legacy intraday value in prediction, retaining r3 pending protection.
 double a06_roll_scope=0;
 // R4 secondary ablation: preserve the chosen feed-cover target at final sale.
 double a06_feed_contract=0;
 // R5: economic inputs after the known current own-unit phase.
 double a06_state_contract=0;
 // R5 bounded economic alternatives: select a whole day execution policy.
 double a06_execution_candidates=0;
 // R6 finite structural ablations: bit1=margin-aware incremental cashflow;
 // bit2=public-rival robust short-window sale DP. No policy identity input.
 double a06_competitive_contract=0;
 double a06_r11_recovery=1; // 0 exact r6 control, 1 receipt-confirmed output rescue
 double a06_r12_calendar=0; // full-portfolio finite harvest calendar;0 exact parent
 double r13_clock=0, r13_start_prior=1, r13_liquidity=1, r13_sale_order=0;
 double r14_value_alignment=0; // bit1 margin, bit2 discount, bit4 future work, bit8 liquidity
 double r14_competition_phase=0;
 double r14_terminal_mode=0;
};
constexpr int SETTINGS_COUNT=sizeof(Settings)/sizeof(double);
struct Commitment {
 int kind=-1,birth=-1,chosen_day=-1,length=0,successor=-1;
 bool funded=false;
};
struct CropPath {Asset a;int kind=-1,length=0;double value=-1e100;};
struct Controller {
 Settings s;Planner model;dp7::Controller core;
 r11::State labor11;bool r11_evaluating=false;
 mutable r12::Stats calendar12;
 std::array<Commitment,100>book{};
 int plan_calls=0,candidates=0,deferred=0,kept=0,rotations=0,started=0;
 int previous_step=-1;std::array<int,12>previous_stock{};
 Flow prices{};Asset portfolio{};std::array<Asset,100>paths{};
 std::array<int,100>release{},successor{},length{};
 double predicted=0;
 std::array<std::array<int8_t,2>,100>forecast_service{};
#if R2_FINITE_FERTILIZER
 std::array<int8_t,100>finite_first_fertilizer{};
#endif
 int service_trials=0,service_switches=0;double service_objective_gain=0;
 int preparation_finance_checks=0,preparation_finance_repairs=0;
 enum class CallOrigin { RealExecution, PublicPrediction };
 CallOrigin call_origin=CallOrigin::RealExecution;
 long long a06_gate_checks=0,a06_gate_activations=0,a06_receipt_events=0;
 bool prediction()const{return call_origin==CallOrigin::PublicPrediction;}
 int effective_reinvest()const{
  if(prediction()&&s.a06_roll_scope==1)return 0;
  if(prediction()&&s.a06_roll_scope==2)return 1;
  return int(s.a06_reinvest);
 }
 bool reinvest_gate()const{int mode=effective_reinvest();return mode>=2&&(mode<4||a06_received_today>0);}
 int a06_live_checks=0,a06_wait_rejections=0,a06_pending_conflicts=0;
 int a06_buffer_hold_events=0,a06_buffer_hold_units=0;
 double a06_last_wait=0,a06_last_best_now=0;
 long long a06_contract_calls=0;std::string a06_contract_last="null";
 int a06_receipt_day=-1,a06_receipt_step=-1;
 double a06_previous_cash=0,a06_received_today=0;
 std::array<double,12>a06_last_now{};
 LocalSaleTiming local_sale;SaleClock sale_memory;
 Controller(Settings settings={}):s(settings),model(),core(model.core.p){configure(settings);}
 void configure(Settings settings){s=settings;
  auto&c=model.cfg;c.competition=s.competition;c.supply=s.supply;c.future_shop=s.future_shop;
  c.capital_power=s.capital_power;c.labor_hours=s.labor_hours;c.action_cost=s.work_price;
  c.reserve=s.reserve;c.max_animals=s.max_animals;c.replant=s.replant;c.discount=s.discount;
  c.risk=s.r13_liquidity;c.land_rent=s.land_rent;
  auto&p=core.p;p.max_hands=int(s.max_hands);p.max_land=int(s.max_land);p.max_animals=int(s.max_animals);
  p.max_cows=p.max_sheep=p.max_geese=int(s.max_animals);p.max_strawberry=p.max_tomato=p.max_melon=75;
  p.frequent_harvest=true;p.frequent_threshold=std::max(1,int(s.harvest_threshold));
  p.early_deposit=int(s.delivery);p.triad_batch_delivery=s.batch_delivery>0;p.sell_deposits=true;p.continuous_market_execution=true;
  // 0/1 retain the released execution path. >=2 enables the additional
  // public-price-pressure trigger for already feasible mid-route delivery.
  p.triad_delivery_pressure=s.batch_delivery>=2?s.batch_delivery-1:0;
  p.renew_ongoing=true;p.insertion_hire_estimate=true;p.reconcile_seed_drift=true;
  p.feed_cover_days=int(s.feed_cover);p.feed_stock_cap=60;
  p.recover_service_inputs=p.procure_service_inputs=p.finance_service_inputs=true;
  p.intraday_admission=s.intraday>0;p.intraday_procurement=s.intraday>0;
  p.intraday_declared_value=false;p.latest_animal_day=20;
  p.exact_schedule_cache=false;p.triad_tour_dp=s.tour_dp>0;
  p.competitive_sell_weight=s.delay_sale;p.opponent_supply_weight=s.supply;
  p.timing_discount=.94;p.action_shadow=s.work_price;
  p.autonomous_start=true;p.opening_animals.clear();p.opening_crops={0,0,0,0,0};
  p.operating_reserve=s.reserve;p.own_feed_demand_weight=1;
  p.finite_fertilizer=R2_FINITE_FERTILIZER&&s.crop_fert>0;
  p.shared_task_atoms_v2=true;
  p.finite_calendar_admission=s.a06_r12_calendar>0&&s.repeat<=0;
 }
 double scalar(const Asset&a,const Flow&px,int start)const{
  double v=0;for(int d=start;d<30;d++){
   double x=a.fixed[d]-s.work_price*a.labor[d];for(int i=0;i<9;i++)x+=a.f[d][i]*px[d][i];
   v+=x/std::pow(1+s.discount,d-start);
  }return v;
 }
 // Exact crop-state recurrence under the declared maintenance/harvest policy.
 // Resource and route feasibility is a separate admission stage, never assumed.
 Asset crop(int k,int birth,int pos,int finish,const Flow&px,const Tile*existing=nullptr,bool*first_fertilizer=nullptr)const{
  if(first_fertilizer)*first_fertilizer=false;
  int begin=model.day;Asset a;a.kind=k;
  if(birth>=30||finish>29||finish<begin){a.end=30;return a;}
  int start=std::max(begin,birth);a.end=finish;
  if(!existing){a.first_cost=seed_price[k];a.fixed[birth]-=seed_price[k];a.labor[birth]+=1;}
  int dry=existing?existing->consecutive_unwatered:1;
  int fert=existing?existing->fertilized_until_day:-1;
  int yield=existing?existing->yield_units:(ongoing(k)?0:1);
  const double travel=.06*near(pos)+.10;
  OngoingMaintenanceDP md;md.kind=k;md.birth=birth;md.begin=start;md.mode=2;md.work=s.work_price+travel;
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
    }
   }else{
    int last=k==W?4:k==C?3:12;bool window=age>=(last+1)/2&&age<=last;
    w=!watered&&(dry>=1||(window&&yield<(k==C?4:6)));
#if R2_FINITE_FERTILIZER
    z=s.crop_fert>0&&finite_fertilizer_choice(k,age,finish-birth,yield,w,watered,fert-d+1,px[finish][k],px[d][F],s.work_price+travel);
    if(z)fert=d+2;
#endif
    if(w&&window)yield=std::min(k==C?4:6,yield+(fert>=d?2:1));
    if(d==finish){a.f[d][k]+=yield;a.labor[d]+=1;yield=0;visit=true;}
   }
   if(z){a.f[d][F]-=1;a.labor[d]+=1;fert=d+2;visit=true;}
   if(d==begin&&first_fertilizer)*first_fertilizer=z;
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
 Asset animal_path(int k,int birth,int pos,const Flow&px,const Tile*t=nullptr,std::array<int8_t,2>*first_action=nullptr)const{
  if(s.service<=0){auto a=model.animal_stream(k,birth,pos,t);for(auto&w:a.labor)w*=s.animal_work;return a;}
  Asset a;a.kind=k;a.end=30;int day=model.day,start=std::max(day,birth),j=k-9;
  if(!t){a.first_cost=animal_price[j];a.fixed[start]-=a.first_cost;a.labor[start]+=3+.15*near(pos);}
  else {a.f[day][product[j]]+=t->yield_units;a.f[day][F]+=t->fertilizer_available;}
  AnimalServiceDP dp;dp.solve(k,birth,start,px,s.work_price);
  int hunger=t?std::min(1,int(t->consecutive_unfed)):0,bonus=t?std::min(held[j]-1,int(t->pending_care_bonus)):0;
  for(int d=start;d<29;d++){
   auto c=dp.choices[d][hunger][bonus];
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
 CropPath choose_crop(int forced,int start,int pos,const Flow&px,const RotationDP&dp,
  const View*context=nullptr,const Asset*base=nullptr,const Planner*forecast=nullptr,bool paid=false,bool labor_paid=false)const{
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
  }
  if(context&&base&&forecast)out=r12_select(*context,*forecast,*base,forced,start,pos,px,nullptr,out,paid,labor_paid);
  return out;
 }
 #include "r12_calendar.inc"
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
  Flow marginal{};if(s.service_reconcile>=3)marginal=marginal_prices(o,portfolio);
  double score=model.value(o,portfolio);
  for(int pos=0;pos<100;pos++){
   const auto&t=o.own.tiles[pos];if(!animal(t))continue;
   int alternatives=s.service_reconcile>=3?2:1;
   for(int which=0;which<alternatives;which++){
    std::array<int8_t,2>action{};
    auto trial=animal_path(int(t.animal),t.placed_day,pos,which?marginal:prices,&t,&action);
    auto candidate=portfolio;add(candidate,paths[pos],-1);add(candidate,trial);
    double value=model.value(o,candidate);service_trials++;
    if(value>score+1e-6){
     service_switches++;service_objective_gain+=value-score;score=value;
     portfolio=std::move(candidate);paths[pos]=std::move(trial);forecast_service[pos]=action;
     model.value(o,portfolio,&prices);
    }
   }
  }
 }
 void set_service(const View&o){
  core.animal_service_day=o.day;core.crop_service_day=o.day;
  for(int pos=0;pos<100;pos++){
   const auto&t=o.own.tiles[pos];
   if(animal(t)){
    AnimalServiceDP d;d.solve(int(t.animal),t.placed_day,o.day,prices,s.work_price);auto c=d.first(o.day,t);
    if(s.service_reconcile>0){c.feed=forecast_service[pos][0];c.care=forecast_service[pos][1];}
    core.service_feed[pos]=s.service<=0?1:c.feed;core.service_care[pos]=s.service<=0?core.care_due(t):c.care;
   }
   if(plant(t)&&ongoing(int(t.crop))){
    int k=int(t.crop);OngoingMaintenanceDP md;md.kind=k;md.birth=t.planted_day;md.mode=2;md.work=s.work_price;
    for(int d=0;d<30;d++){md.price[d]=prices[d][k];md.fert[d]=prices[d][F];}
    auto c=md.first(o.day,std::min(1,int(t.consecutive_unwatered)),std::clamp(int(t.fertilized_until_day)-o.day+1,0,3),core.water_due(t),core.fertilize_due(t));
    core.crop_birth[pos]=t.planted_day;core.crop_kind[pos]=k;
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
 #include "ongoing_supply_admission.inc"
 void plan(const View&o){
  plan_calls++;model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);core.day=o.day;core.phase=1;core.queue.clear();core.plans.clear();core.pending_admission={};core.intraday_units.clear();core.resource_degraded=core.actual_drop=0;core.seed_reconcile_checked=false;core.prepared_seed_need={};
  Farm farm=o.own;int expiry=dp7::Controller::project_zero_expiry(farm,o.step);
  View v{o.step,o.day,o.hour,farm,o.opponent,o.priv,o.market,o.shops};
  core.target.clear();core.plant_not_before.fill(0);core.triad_crop_age.fill(-1);
  portfolio={};paths={};forecast_service={};release.fill(o.day);successor.fill(-1);length.fill(0);
#if R2_FINITE_FERTILIZER
  finite_first_fertilizer.fill(0);
#endif
  for(int i=0;i<9;i++){portfolio.f[o.day][i]+=o.priv.shed[i];for(auto&b:o.priv.inventories)portfolio.f[o.day][i]+=b[i];}
  model.value(o,portfolio,&prices);std::vector<int>free;int animals=0;
  int owned=std::popcount(unsigned(o.own.unlocked_mask));core.planned_land=owned;
  // Resolve existing projects from facts. Successor intention is separate from
  // the current tile and never replaces maintenance of the incumbent crop.
  for(int pos=0;pos<100;pos++){
   auto&t=farm.tiles[pos];if(t.kind==TileKind::LOCKED)continue;
   if(animal(t)){paths[pos]=animal_path(int(t.animal),t.placed_day,pos,prices,&t,&forecast_service[pos]);add(portfolio,paths[pos]);core.target.emplace_back(pos,int(t.animal));animals++;book[pos]={int(t.animal),t.placed_day,o.day,0,-1,true};continue;}
   if(plant(t)){
    int k=int(t.crop),finish=std::max(o.day,t.planted_day+(ongoing(k)?first[k]+3*interval[k]:core.h_age(k)));
    if(!ongoing(k)){
     auto dp=rotations_dp(pos,prices,o.day);double best=-1e100;
     int earliest=std::max(o.day,int(t.planted_day)+first[k]);int last=std::min(29,int(t.planted_day)+(k==W?4:k==C?3:12));
     for(int d=earliest;d<=last;d++){
      auto a=crop(k,t.planted_day,pos,d,prices,&t);double val=scalar(a,prices,o.day)+(s.rotation>0?dp.v[d]:0);
      if(val>best){best=val;finish=d;}
     }
    }
    finish=std::min(29,finish);
    int retained=(book[pos].birth==t.planted_day&&book[pos].kind==k)?book[pos].successor:-1;
    if(retained>=0&&o.priv.seeds[retained]>0&&o.day>=t.planted_day+first[k])finish=o.day;
    paths[pos]=crop(k,t.planted_day,pos,finish,prices,&t);add(portfolio,paths[pos]);
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
  model.value(o,portfolio,&prices);
  r12_reconcile_calendar(v,farm,free);
  // Near-depot high-maintenance slots are selected first; no fixed day/plot tape.
  std::stable_sort(free.begin(),free.end(),[&](int a,int b){return std::tuple(near(a),a)<std::tuple(near(b),b);});
  double stock=0;for(int i=0;i<9;i++)stock+=revenue(i,o.market.inventory[i],o.priv.shed[i]);
  double budget=std::max(0.,o.own.money+.85*stock-s.reserve-animals*o.market.prices[W]*s.feed_cover);
  if(o.day==0)budget*=s.opening_budget;
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
    int k=b.successor;auto dp=rotations_dp(pos,prices,o.day);auto candidate=choose_crop(k,o.day,pos,prices,dp,&o,&portfolio,&model,true);
    if(candidate.kind>=0){auto a=candidate.a;a.fixed[o.day]+=seed_price[k];a.first_cost=0;
     stock_left[k]--;paths[pos]=a;add(portfolio,a);settarget(pos,k);successor[pos]=k;length[pos]=candidate.length;kept++;continue;}
   }
   if(s.keep_commitments<=0||plant(t)||animal(t)||b.kind<0||stock_left[b.kind]<=0)continue;
   int k=b.kind;auto dp=rotations_dp(pos,prices,o.day);Asset a=k>=9?animal_path(k,o.day,pos,prices):choose_crop(k,o.day,pos,prices,dp,&o,&portfolio,&model,true).a;
   if(a.first_cost<=0)continue;a.fixed[o.day]+=a.first_cost;a.first_cost=0;stock_left[k]--;b.funded=true;
   paths[pos]=a;add(portfolio,a);settarget(pos,k);successor[pos]=k;animals+=k>=9;kept++;
  }
  free.erase(std::remove_if(free.begin(),free.end(),[&](int p){return successor[p]>=0;}),free.end());
  double current=model.value(o,portfolio,&prices);
  int used=0;std::vector<int>new_positions;
  while(o.day<29){
   if(used>=int(free.size())){
    if(owned>=int(s.max_land)||o.day>20||budget<next_land_cost(owned)+300)break;
    double cost=next_land_cost(owned);owned++;core.planned_land=owned;budget-=cost;portfolio.fixed[o.day]-=cost;
    for(int p=0;p<100;p++)if(quad(p)==owned-1){free.push_back(p);settarget(p,-1);}
    std::stable_sort(free.begin()+used,free.end(),[](int a,int b){return std::tuple(near(a),a)<std::tuple(near(b),b);});current=model.value(o,portfolio,&prices);
   }
   int pos=free[used++];auto&t=farm.tiles[pos];auto dp=rotations_dp(pos,prices,o.day);
   double best=0,bestval=current;int bestkind=-1,bestlen=0;Asset bestpath;double bestcost=0;
   for(int k:{0,1,2,3,4,9,10,11}){
    if(k>=9&&(plant(t)||animals>=s.max_animals||o.day+afirst[k-9]>29))continue;
    Asset a;int len=0;
    if(k>=9)a=animal_path(k,o.day,pos,prices);else{auto c=choose_crop(k,o.day,pos,prices,dp,&o,&portfolio,&model,stock_left[k]>0);if(c.kind<0)continue;a=c.a;len=c.length;}
    double cost=a.first_cost;
    if(cost<=0)continue;if(stock_left[k]>0){a.fixed[o.day]+=cost;cost=0;}
    double immediate=cost+(k>=9?o.market.prices[W]*s.feed_cover:0);
    if(immediate>budget)continue;
    auto next=portfolio;add(next,a);double val=model.value(o,next);candidates++;
    double gain=val-current-s.land_rent*(k>=9?29-o.day:len);
    double rank=gain/std::pow(std::max(10.,cost)+(k>=9?100:0),s.capital_power*std::max(0.,1-o.own.money/16000.));
    rank*=k>=9?s.animal_bias:s.crop_bias;
    if(rank>best){best=rank;bestval=val;bestkind=k;bestlen=len;bestpath=a;bestcost=immediate;}
   }
   if(bestkind<0)break;
   settarget(pos,bestkind);successor[pos]=bestkind;length[pos]=bestlen;paths[pos]=bestpath;add(portfolio,bestpath);
   if(s.a06_r12_calendar>0&&!plant(t)&&bestkind<5&&!ongoing(bestkind))core.triad_crop_age[pos]=bestlen;
   current=bestval;budget-=bestcost;animals+=bestkind>=9;if(stock_left[bestkind]>0)stock_left[bestkind]--;
   book[pos].successor=plant(t)?bestkind:-1;
   if(!plant(t))book[pos]={bestkind,-1,o.day,bestlen,-1,bestcost==0};
   else rotations+=int(t.crop)!=bestkind;
   new_positions.push_back(pos);model.value(o,portfolio,&prices);
  }
  // No automatic same-crop continuation when a slot was offered but declined.
  for(int pos:free)if(plant(farm.tiles[pos])&&successor[pos]<0)core.plant_not_before[pos]=30;
  if(s.service_reconcile>=2)reconcile_service(v);
  set_service(v);prepare_with_ongoing_supply(v,o,expiry);
  if(s.preview>0&&!new_positions.empty()){
   for(int pass=0;pass<3;pass++){
    auto preview=core.preview_bundle(v,false);std::set<int>admitted;for(auto[p,k]:preview.started_targets)admitted.insert(p);
    bool changed=false;
    for(int pos:new_positions){
     if(plant(farm.tiles[pos])||book[pos].funded||successor[pos]<0||admitted.count(pos))continue;
     add(portfolio,paths[pos],-1);paths[pos]={};successor[pos]=-1;settarget(pos,-1);book[pos]={};deferred++;changed=true;
    }
    if(!changed)break;model.value(o,portfolio,&prices);if(s.service_reconcile>=2)reconcile_service(v);set_service(v);prepare_with_ongoing_supply(v,o,expiry);
   }
  }
  predicted=model.value(o,portfolio);model.shadow=prices;
 }
 std::array<double,12>intraday_values(const dp7::Controller&c,const View&o,const PlayerAction&out){
  a06_gate_checks++;
  if(reinvest_gate()){a06_gate_activations++;return a06_live_intraday_values(c,o,out);}
  std::array<double,12>v;v.fill(-1e90);int pos=-1;for(auto[p,k]:c.target)if(!plant(o.own.tiles[p])&&!animal(o.own.tiles[p])){pos=p;break;}if(pos<0)return v;
  // Compare only incremental project streams; the admission compiler checks
  // actual cash, inputs, free workers and PLANT+WATER completion time.
  auto dp=rotations_dp(pos,prices,o.day);
  for(int k:{0,1,2,3,4,9,10,11}){
   Asset a=k>=9?animal_path(k,o.day,pos,prices):choose_crop(k,o.day,pos,prices,dp,&o,&portfolio,&model).a;if(a.first_cost<=0)continue;
   auto b=portfolio;add(b,a);v[k]=model.value(o,b)-model.value(o,portfolio);
  }return v;
 }
 #include "a06_live_reinvestment.inc"
 // Own-unit projection has no access to the live simulator or rival inventory.
 // It only confirms which holdings would reach the shed in OUR unit phase;
 // sale quantities are recalculated from the next actual observation.
 void settle_market(const View&o,PlayerAction&out,bool preparing){
  if(s.delay_sale<0||(preparing&&s.feed_finance<=0))return;
  fastkag::ObservedDayScenario scenario(o);
  auto post=scenario.project_units(out.units,-1);
  if(preparing){
   // R2P1: only intervene in an already-issued preparation transaction with
   // demonstrable own-cash shortfall. This projection does not know rival
   // orders; real fills are still learned from the next observation.
   preparation_finance_checks++;
   auto before=post.project_own_market(0,out.market);
   bool shortfall=false;for(double x:before.last_market_cash_shortfalls()[0])shortfall|=x>0;
   if(!shortfall||out.market.size()>=10)return;
  }
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
   // The two wheat BUY orders may be today's feed plus a deliberate future
   // buffer, NOT duplicate payment for one obligation. Protect that declared
   // cover only when it exceeds today's job requirement. Subtract feed already
   // consumed and stock already in bags; MAX avoids reserving today's pickup
   // twice. This changes no purchases and grants no credit for unfilled orders.
   if(s.a06_feed_contract>0&&core.feed_stock_target>core.daily_need[W]){
    int already_fed=0;for(const auto&t:post.farms()[0].tiles)already_fed+=animal(t)&&t.fed_today;
    int remaining=std::max(0,core.feed_stock_target-already_fed-heldfeed);
    int extra=std::max(0,std::min(priv.shed[W],remaining)-std::min(priv.shed[W],reserve[W]));
    a06_buffer_hold_events+=extra>0;a06_buffer_hold_units+=extra;
    reserve[W]=std::max(reserve[W],remaining);
   }
  }
  // settle_market rebuilds SELLs after intraday::protect_admitted_inputs.
  // Preserve the pending input reservation here as well; otherwise an already
  // paid fertilizer/feed unit may be sold before next-observation activation.
  if(core.pending_admission.active){
   const auto&pending=core.pending_admission;
   if(pending.day==o.day&&pending.unit<int(priv.inventories.size())){
    auto j=dp7::intraday::project_job(core,o,pending.pos,pending.kind);
    for(int i=0;i<9;i++){
     int need=std::max(0,j.needs[i]-priv.inventories[pending.unit][i]);
     if(need>0&&priv.shed[i]>reserve[i])a06_pending_conflicts++;
     if(effective_reinvest()>=1)reserve[i]+=need;
    }
   }
  }
  for(int i=0;i<9;i++)sell[i]=std::max(0,priv.shed[i]-reserve[i]);
  if(preparing){
   // Preserve every original order and its relative order. Sell only surplus
   // not already promised to an existing SELL, and never remove an order to
   // make room. Require the whole proposed purchase/hire sequence to fill in
   // the own-side projection, not merely a higher projected cash balance.
   for(auto a:out.market)if(a.op==Op::SELL&&int(a.item)>=0&&int(a.item)<9)
    sell[int(a.item)]=std::max(0,sell[int(a.item)]-std::max(0,a.quantity));
   auto choices=core.sales_sorted(o,sell);Acts prefix;
   for(auto sale:choices){
    if(prefix.size()+out.market.size()>=10)break;
    int limit=sale.quantity; sale.quantity=0; prefix.push_back(sale);
    for(int q=1;q<=limit;q++){
     prefix.back().quantity=q;Acts candidate=prefix;
     candidate.insert(candidate.end(),out.market.begin(),out.market.end());
     auto funded=post.project_own_market(0,candidate);auto&fills=funded.last_market_fills()[0];
     bool full=true;
     for(size_t i=0;i<out.market.size();i++){
      auto a=out.market[i];if(a.op==Op::SELL||a.op==Op::PASS)continue;
      int wanted=(a.op==Op::HIRE||a.op==Op::BUY_LAND)?1:std::max(0,a.quantity);
      full&=fills[prefix.size()+i]>=wanted;
     }
     if(full){out.market=std::move(candidate);preparation_finance_repairs++;return;}
    }
   }
   return;
  }
  Acts buys;for(auto a:out.market)if(a.op!=Op::SELL)buys.push_back(a);
  local_sale.filter(o,priv,buys,sell,sale_memory,s.reserve,s.competition,(int(s.a06_competitive_contract)&2)!=0);
  auto sales=core.sales_sorted(o,sell);
  int slots=std::max(0,10-int(buys.size()));if(int(sales.size())>slots)sales.resize(slots);
  sales.insert(sales.end(),buys.begin(),buys.end());out.market=std::move(sales);
 }
 #include "r11_recovery.inc"
 PlayerAction act(const View&o){
  // Receipt history belongs to this controller's call stream. Prediction
  // rollouts receive explicit PublicPrediction origin from SearchController;
  // real execution is NEVER tagged by inspecting observation contents.
  // Scope controls the mechanism, not whether the prediction is legitimate.
  if(a06_receipt_day!=o.day){a06_receipt_day=o.day;a06_received_today=0;a06_previous_cash=o.own.money;}
  else if(a06_receipt_step<o.step){double delta=std::max(0.,o.own.money-a06_previous_cash);a06_received_today+=delta;a06_receipt_events+=delta>0;}
  a06_previous_cash=o.own.money;a06_receipt_step=o.step;
  if(o.step<=previous_step)throw std::runtime_error("non-monotone policy observation");previous_step=o.step;
  if(o.day!=core.day)plan(o);
  core.admission_values=[this](const dp7::Controller&c,const View&v,const PlayerAction&a){return intraday_values(c,v,a);};core.admission_blend=1;core.admission_scope=2;
  if(s.a06_r11_recovery>0)r11::reconcile(labor11,core,o);
  bool preparing=core.phase==1;auto out=core.act(o);settle_market(o,out,preparing);r11_consider(o,out);return out;
 }
 std::string debug()const{
  std::ostringstream o;o<<"{\"r3_fix1_terminal_checks\":"<<ongoing_terminal_checks<<",\"r3_fix1_terminal_rejected\":"<<ongoing_terminal_rejected<<",\"r3_fix1_terminal_errors\":"<<ongoing_terminal_errors<<",\"r3_fix1_terminal_last\":"<<ongoing_terminal_last<<",\"r3_supply_previews\":"<<ongoing_supply_previews<<",\"r3_supply_skips\":"<<ongoing_supply_skips<<",\"r3_supply_preview_errors\":"<<ongoing_supply_preview_errors<<",\"r12_calendar_enabled\":"<<s.a06_r12_calendar<<",\"r12_calls\":"<<calendar12.calls<<",\"r12_changes\":"<<calendar12.changes<<",\"r12_existing_changes\":"<<calendar12.existing_changes<<",\"r12_last\":"<<calendar12.last<<",\"r12_last_change\":"<<calendar12.last_change<<",\"r11_checks\":"<<labor11.checks<<",\"r11_rollouts\":"<<labor11.rollouts<<",\"r11_offered\":"<<labor11.offered<<",\"r11_committed\":"<<labor11.committed<<",\"r11_hire_requests\":"<<labor11.hire_requests<<",\"r11_hire_confirmed\":"<<labor11.hire_confirmed<<",\"r11_cancelled\":"<<labor11.cancelled<<",\"r11_pending\":"<<labor11.active<<",\"r11_last\":"<<labor11.last<<",\"a06_state_contract\":"<<s.a06_state_contract<<",\"a06_contract_calls\":"<<a06_contract_calls<<",\"a06_contract_last\":"<<a06_contract_last<<",\"feed_stock_target\":"<<core.feed_stock_target<<",\"daily_need_wheat\":"<<core.daily_need[W]<<",\"a06_feed_contract\":"<<s.a06_feed_contract<<",\"a06_buffer_hold_events\":"<<a06_buffer_hold_events<<",\"a06_buffer_hold_units\":"<<a06_buffer_hold_units<<",\"a06_roll_scope\":"<<s.a06_roll_scope<<",\"a06_origin_prediction\":"<<prediction()<<",\"a06_gate_checks\":"<<a06_gate_checks<<",\"a06_gate_activations\":"<<a06_gate_activations<<",\"a06_receipt_events\":"<<a06_receipt_events<<",\"a06_mode\":"<<s.a06_reinvest<<",\"a06_received_today\":"<<a06_received_today<<",\"a06_live_checks\":"<<a06_live_checks<<",\"a06_wait_rejections\":"<<a06_wait_rejections<<",\"a06_pending_conflicts\":"<<a06_pending_conflicts<<",\"a06_last_wait\":"<<a06_last_wait<<",\"a06_last_best_now\":"<<a06_last_best_now<<",\"intraday_checks\":"<<core.intraday_checks<<",\"intraday_proposals\":"<<core.intraday_proposals<<",\"intraday_cancelled\":"<<core.intraday_cancelled<<",\"intraday_unfilled\":"<<core.intraday_unfilled<<",\"intraday_pending\":"<<core.pending_admission.active<<",\"intraday_pending_kind\":"<<core.pending_admission.kind<<",\"intraday_pending_pos\":"<<core.pending_admission.pos<<",\"a06_fleet_solves\":"<<core.a06_stats.solves<<",\"a06_fleet_changes\":"<<core.a06_stats.changes<<",\"a06_added_jobs\":"<<core.a06_stats.added<<",\"a06_route_steps_saved\":"<<core.a06_stats.saved<<",\"plan_calls\":"<<plan_calls<<",\"candidates\":"<<candidates<<",\"deferred\":"<<deferred<<",\"kept_commitments\":"<<kept<<",\"rotations\":"<<rotations<<",\"intraday_started\":"<<core.intraday_activated<<",\"reference_calls\":0,\"predicted\":"<<predicted<<",\"service_trials\":"<<service_trials<<",\"service_switches\":"<<service_switches<<",\"service_value_gain\":"<<service_objective_gain<<",\"targets\":[";
  bool sep=false;for(auto[p,k]:core.target){if(sep)o<<",";sep=true;o<<"["<<p<<","<<k<<"]";}o<<"],\"preparation_finance_checks\":"<<preparation_finance_checks<<",\"preparation_finance_repairs\":"<<preparation_finance_repairs<<",\"midroute_delivery_checks\":"<<core.midroute_delivery_checks<<",\"midroute_delivery_insertions\":"<<core.midroute_delivery_insertions<<",\"midroute_delivery_quantity\":"<<core.midroute_delivery_quantity<<",\"local_sale_checks\":"<<local_sale.checks<<",\"local_sale_holds\":"<<local_sale.holds<<",\"local_sale_quantity\":"<<local_sale.held_quantity<<",\"local_sale_forced\":"<<local_sale.forced<<",\"local_sale_expected_gain\":"<<local_sale.projected_gain<<",\"r4_route_checks\":"<<core.r4_route_checks<<",\"r4_route_previews\":"<<core.r4_route_previews<<",\"r4_route_insertions\":"<<core.r4_route_insertions<<",\"r4_route_rejected\":"<<core.r4_route_rejected<<",\"r4_route_errors\":"<<core.r4_route_errors<<",\"r4_route_conditional_gain\":"<<core.r4_route_conditional_gain<<"}";return o.str();
 }
};
}
