#pragma once
#include "executor/policy.hpp"
#include "conditional_market.hpp"
#include "sale_plan_dp.hpp"
#include <cstdlib>
#include <cstdio>
// New autonomous cash-flow DP and marginal portfolio planner. The execution
// engine is our pre-existing dp7 controller, NOT one of the seven opponents.
namespace competitive {
using namespace dp7;
using Flow=std::array<std::array<double,9>,30>;
using Curve=std::array<double,30>;
struct Config {
 double competition=.75,supply=1.,replant=.8,capital_power=.5,discount=.01;
 // Replace value()'s fixed "sell the whole day's output immediately" with the optimal multi-day
 // sale schedule (sale_plan_dp.hpp). The DP needs this candidate's own output flow a.f[d][i],
 // which only exists here -- the executor-side sale filter never had it. 0 = released behaviour.
 double sale_dp=0.;
 double sale_hold_cap=90.;   // shed capacity the plan must respect (100 shared, minus margin)
 double work_scale=1.,land_rent=3.,action_cost=1.5,labor_hours=16.,future_shop=1.;
 double reserve=70.,max_animals=18.,new_limit=30.,feed_cover=2.,hold=1.;
 double expansion=0.,margin_weight=1.,risk=0.;
 double finite_fertilize=0.,wheat_age=4.,melon_age=10.,mpc=0.;
 double inventory_dp=0.,tape=0.,service_dp=0.,service_cost=3.;
 double execution_variant=0.;
 double admission_preview=0.;
 double rotation_edits=0.;
 double service_rounds=0.,growth_forecast=0.;
 double harvest_threshold=0.;
 double investment_blend=0.,investment_scope=1.,investment_start=1.,investment_margin=0.,investment_timing=1.,crop_dp=0.,crop_start=0.,crop_gain=0.,crop_action_cost=3.,crop_fert_floor=0.,crop_max_changes=100.,crop_timing=0.,crop_price_mode=0.,investment_consistent=0.,investment_paid_labor=0.,investment_crop_consistent=0.,calendar_samples=0.;
};
struct Asset {Flow f{};Curve labor{},fixed{};int end=30,kind=-1;double first_cost=0;};
struct ValueBreakdown {
 double fixed=0,own_trade=0,wages=0,actions=0,rival_penalty=0,liquidity_penalty=0,total=0;
#if R2_SHOP_BRANCH_AUDIT
 std::array<double,9> own_trade_by_item{},rival_penalty_by_item{};
#endif
};
struct ValueBasis {int day;double money;};
inline void add(Asset&a,const Asset&b,double scale=1.){for(int d=0;d<30;d++){for(int i=0;i<9;i++)a.f[d][i]+=scale*b.f[d][i];a.labor[d]+=scale*b.labor[d];a.fixed[d]+=scale*b.fixed[d];}}
struct Planner {
 dp7::Controller core;
 Config cfg;
 bool use_value_basis=false;
 ValueBasis value_basis{};
 Flow dem{},rival{},shadow{},extra_rival{};
 // A rollout may realize an already-computed aggregate rival flow without
 // simulating the rival's private workers.  In that case the public board is
 // not a valid source from which to rebuild the flow at the tail boundary.
 // Default false in live play; diagnostics and scenario code may carry the
 // original conditional forecast forward explicitly.
 bool use_rival_forecast=false;
 Flow rival_forecast{};
#if R2_SALE_CLOCK_MODE >= 2
 std::array<double,9>rival_early_share{{.5,.5,.5,.5,.5,.5,.5,.5,.5}};
#endif
 std::array<int,5> learned_harvest{-1,-1,-1,-1,-1};
 // Optional public-history distribution of finite-crop removal ages.
 std::array<std::array<double,30>,5>rival_harvest_weights{};
 int day=0,portfolios=0;
 double last_score=0;
 explicit Planner(Config c={}):cfg(c){auto&p=core.p;
  p.opponent_supply_weight=1.;p.competitive_sell_weight=.5;p.sell_horizon_days=1;p.opening_animals.clear();p.opening_crops={0,0,0,0,0};
  p.economic_land=true;p.operating_reserve=100;p.fix_finite_projection=true;p.efficient_water=true;p.efficient_care=true;p.fix_logistics=true;
  p.plan_zero_expiry=true;p.terminal_deposit_schedule=true;p.shared_task_atoms_v2=true;p.stepwise_recoordination=true;p.preparation_pipeline_v2=true;
  p.schedule_value_compare=true;p.net_feed_buffer=true;p.preparation_spawn_guard=true;p.midroute_delivery=true;
  p.incremental_pickup_repair=true;p.shared_service_insertions=true;p.fix_liquidity=true;p.max_land=4;p.max_hands=15;
  // New projects are admitted by this planner; no second investment owner.
  p.intraday_admission=false;p.intraday_procurement=false;p.joint_investment_portfolio=false;
  p.exact_schedule_cache=true;p.renew_ongoing=false;p.feed_cover_days=2;
 }
 double wages(double w)const {double h=std::max(0.,w*cfg.work_scale/cfg.labor_hours-1.);return std::max(0.,(std::pow(1.61803398875,h+2)-1)/2.2360679775-1);}
 static double quote(int i,double inv,double q){if(std::abs(q)<=2)return price(i,inv+.5*q);return (5.*price(i,inv+q*.1127016654)+8.*price(i,inv+.5*q)+5.*price(i,inv+q*.8872983346))/18.;}
 static double trade(int i,double&inv,double q){
#if R2_MARKET_INTEGRAL == 2
  return ConditionalMarket::execute(i,inv,q);
#else
  double p=quote(i,inv,q);
#if R2_MARKET_INTEGRAL == 1
  ConditionalMarket::update(i,inv,q);
#else
  if(q<0||p>1.)inv+=q;
#endif
  return q*p;
#endif
 }
 // Per-candidate sale plan. The greedy changes one tile at a time, so exactly one product's
 // output differs from the base; solving the DP for that product alone costs 1 solve per
 // candidate instead of 9, which is what makes a faithful (not shadowed) integration affordable.
 static constexpr int MAXB=180;   // 30 days x 6 four-step buckets
 mutable std::array<std::array<double,MAXB>,9>live_q{ };
 // Sale schedule for the portfolio the planner COMMITTED to. The executor follows this; without
 // it the DP only shifted the valuation and the actual selling never changed, which is why five
 // integrations showed no gain.
 // Stock the plan wants to CARRY OUT of each bucket. The executor sells down to this, which is
 // self-correcting: if reality has more stock than the plan assumed it still sells the excess,
 // where a cap on the sell quantity would simply under-sell and strand the goods.
 mutable std::array<std::array<double,MAXB+1>,9>exec_carry{};
 bool exec_plan_ready=false;
 int exec_plan_step=0;   // step the plan was built at; the executor indexes buckets from here
 int exec_plan_day=-1;   // day it was built for: the caller must compare DAYS, not steps
 void build_exec_plan(const Asset&base,const View&o){
  exec_plan_ready=false;
  if(std::getenv("DPTRACE"))std::fprintf(stderr,"[BUILD] day=%d ready=%d today=[%.0f %.0f %.0f %.0f %.0f %.0f %.0f]\n",day,(int)exec_plan_ready,exec_carry[1][day],exec_carry[2][day],exec_carry[3][day],exec_carry[4][day],exec_carry[6][day],exec_carry[7][day],exec_carry[8][day]);
  if(std::getenv("DPINPUT")&&day>=12){
   for(int i=0;i<9;i++){
    std::fprintf(stderr,"[IN] d%d i%d inv0=%d prod",day,i,int(o.market.inventory[i]));
    for(int d=day;d<30;d++)std::fprintf(stderr," %.0f",base.f[d][i]);
    std::fprintf(stderr," dem");for(int d=day;d<30;d++)std::fprintf(stderr," %.1f",dem[d][i]);
    std::fprintf(stderr," riv");for(int d=day;d<30;d++)std::fprintf(stderr," %.0f",rival[d][i]);
    std::fprintf(stderr," plan");for(int d=day;d<30;d++)std::fprintf(stderr," %.0f",(cfg.sale_dp>0?live_q[i][d]:base.f[d][i]));
    std::fprintf(stderr,"\n");
   }
  }

  if(cfg.sale_dp<=0)return;
  // The held-stock budget is a HARD constraint inside the DP (Input::hold_cap), not a price found
  // by search: the DP skips every candidate that would carry more, so the plan is feasible by
  // construction and no bisection is needed. The budget is the physical room left once the animals'
  // feed buffer is set aside -- feed is an INPUT, and a plan that fills the shed with held goods
  // starves the herd (seed 2780000035: animals 21 -> 13 when this was not reserved).
  int animals=0;for(const auto&t:o.own.tiles)animals+=animal(t);
  const double room=std::max(0.,100.-double(animals)*std::max(1.,cfg.feed_cover));
  std::array<double,9>flow{};double flowtotal=0.;
  for(int i=0;i<9;i++){
   double rate=0.;for(int d=day;d<30&&d<day+3;d++)rate+=std::max(0.,base.f[d][i]);
   flow[i]=rate;flowtotal+=rate;
  }
  for(int i=0;i<9;i++){
   double cap=flowtotal>1e-9?room*flow[i]/flowtotal:0.;
   solve_one(i,o.market.inventory[i],base,o,cap,true);
  }
  exec_plan_ready=true;exec_plan_step=o.step;exec_plan_day=o.day;
 }
 std::vector<double> solve_one(int i,double inv0,const Asset&a,const View&o,double cap,bool polish)const{
  // Conditional on this fixed supply forecast, sale timing uses cash margin (coefficient 1).
  // This is still a deterministic surrogate, not a terminal win-probability objective.
  sale_plan::Input in;in.competition=1.;in.hold_cap=cap;in.hold_penalty=0.;
  in.discount=cfg.discount;
  // BUCKET granularity, not day: the engine consumes town demand once every FOUR steps and the
  // executor releases one bucket at a time, so a daily period can neither see nor place a sale at
  // the moment the market actually refills. 108 periods at takeover instead of 18.
  const int start_step=o.step;
  const int nb=std::max(0,(720-start_step)/4);
  for(int b=0;b<nb;b++){
   int st=start_step+4*b, d=st/24;
   // Mirrors simulator.cpp::town_consume exactly: one unit per shop per product every 4 steps,
   // plus one per product every 24. Inlined because public_trade_ledger.hpp needs Planner.
   static const std::array<std::vector<int>,8> ps{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C,C},{MI,T,W},{S,MI},{WO,WO}}};
   double cdem=(st%24==0&&i<8)?1.:0.;
   if(st%4==0)for(int sh:o.shops)for(int p:ps[sh])cdem+=(p==i)?1.:0.;
   in.demand.push_back(cdem);
   in.rival.push_back(cfg.supply*rival[d][i]/6.);           // daily forecast spread over buckets
   in.production.push_back((d<30?a.f[d][i]:0.)/6.);         // arrivals spread through the day
  }
  auto pl=sale_plan::solve(i,inv0,0.,in,polish);
  // Beyond the plan's horizon there is NO cap. Filling those buckets with a small number made the
  // executor treat it as a limit and leave stock unsold at the end of the game (measured: 28.9 units
  // left, 38 units fewer sold, -3,365 cash) -- unsold stock at day 29 is worth nothing.
  for(int b=0;b<MAXB;b++)live_q[i][b]=(b<nb&&b<int(pl.sell.size()))?pl.sell[b]:1e9;
  for(int b=0;b<=MAXB;b++)exec_carry[i][b]=(b<nb&&b<int(pl.stock.size()))?pl.stock[b]:0.;
  exec_carry[i][MAXB]=0.;   // nothing may be carried past the end of the game
  return pl.stock;
 }
 double value(const View&o,const Asset&a,Flow*prices=nullptr,int changed=-1,ValueBreakdown*out=nullptr,const ValueBasis*basis=nullptr
#if R2_SHOP_BRANCH_AUDIT
 ,bool exact_market=false
#endif
 )const{
  std::array<double,9>inv{};for(int i=0;i<9;i++)inv[i]=o.market.inventory[i];double val=0.,balance=o.own.money,liquidity=0;
  auto exchange=[&](int i,double&stock,double quantity){
#if R2_SHOP_BRANCH_AUDIT
   if(exact_market)return ConditionalMarket::execute(i,stock,quantity);
#endif
   return trade(i,stock,quantity);
  };
  // Faithful integration: the greedy changed exactly one product (or none, for the base), so solve
  // the DP for THAT product with ITS production and sell on the resulting schedule -- through
  // trade(), so pricing and the inventory advance (own supply impact) are the released ones. The
  // earlier shadow/timing-profile versions reused the BASE's plan for every candidate, which is why
  // they measured consistently worse.
  if(cfg.sale_dp>0&&changed>=0&&changed<9){
   solve_one(changed,o.market.inventory[changed],a,o,1e9,true);
   for(int d=day;d<30;d++)for(int ii=0;ii<9;ii++)live_q[ii][d]=(d-day<30)?live_q[ii][d]:a.f[d][ii];
  }
  for(int d=day;d<30;d++){
   double own_trade=0,enemy=0;
#if R2_SHOP_BRANCH_AUDIT
   std::array<double,9> own_by_item{},enemy_by_item{};
#endif
   for(int i=0;i<9;i++){
    inv[i]-=dem[d][i]*.5;
#if R2_SALE_CLOCK_MODE >= 2
    double r=cfg.supply*rival[d][i]*rival_early_share[i],late=cfg.supply*rival[d][i]-r;
#else
    double r=cfg.supply*rival[d][i]*.5,late=r;
#endif
#if R2_SHOP_BRANCH_AUDIT
    double enemy_early=exchange(i,inv[i],r);
    enemy+=enemy_early;
#else
    enemy+=exchange(i,inv[i],r);
#endif
    const bool own_sched=cfg.sale_dp>0&&i==changed&&changed>=0;
#if R2_SHOP_BRANCH_AUDIT
    double own_item=own_sched?exchange(i,inv[i],live_q[i][d]):exchange(i,inv[i],a.f[d][i]);
    own_trade+=own_item;
    double enemy_late=exchange(i,inv[i],late);
    enemy+=enemy_late;
    own_by_item[i]=own_item;enemy_by_item[i]=enemy_early+enemy_late;
#else
    if(own_sched) own_trade+=exchange(i,inv[i],live_q[i][d]);
    else own_trade+=exchange(i,inv[i],a.f[d][i]);
    enemy+=exchange(i,inv[i],late);
#endif
    inv[i]-=dem[d][i]*.5;if(prices)(*prices)[d][i]=price(i,inv[i]);
   }
   double wage=wages(a.labor[d]),action=cfg.action_cost*a.labor[d];
   double cash=a.fixed[d]+own_trade-wage-action;balance+=cash;
   liquidity+=std::max(0.,-balance);
   const ValueBasis*clock=basis?basis:(use_value_basis?&value_basis:nullptr);
   int origin=clock?clock->day:day;double origin_money=clock?clock->money:o.own.money;
   double discount=std::pow(1+cfg.discount*std::max(0.,1-origin_money/20000.),d-origin);
   val+=(cash-cfg.competition*enemy)/discount;
   if(out){out->fixed+=a.fixed[d]/discount;out->own_trade+=own_trade/discount;out->wages-=wage/discount;out->actions-=action/discount;out->rival_penalty-=cfg.competition*enemy/discount;}
#if R2_SHOP_BRANCH_AUDIT
   if(out)for(int i=0;i<9;i++){
    out->own_trade_by_item[i]+=own_by_item[i]/discount;
    out->rival_penalty_by_item[i]-=cfg.competition*enemy_by_item[i]/discount;
   }
#endif
  }
  double result=val-cfg.risk*liquidity;
  if(out){out->liquidity_penalty=-cfg.risk*liquidity;out->total=result;}
  return result;
 }
 // Counterfactual for one non-buyable product: the same forecast rival output may be sold
 // immediately or carried (up to one shed) and sold on the revenue-maximising day.  Both arms use
 // the same exact market primitive and the same within-day ordering, so the returned delta isolates
 // cross-day rival stock timing.  Diagnostic only; it is not a belief over the hidden stock.
 double rival_stock_response_delta(const View&o,const Asset&a,int item,double stock0=0,
                                    const ValueBasis*basis=nullptr)const{
  sale_plan::Input in;in.competition=0;in.hold_cap=100;in.discount=0;
  for(int d=day;d<30;d++){
   in.demand.push_back(dem[d][item]);
   in.rival.push_back(std::max(0.,a.f[d][item]));       // our candidate flow, split around rival
   in.production.push_back(std::max(0.,cfg.supply*rival[d][item]));
  }
  auto best=sale_plan::solve(item,o.market.inventory[item],stock0,in,true);
  std::vector<double> immediate(in.count());double held=stock0;
  for(int k=0;k<in.count();k++){held+=in.production[k];immediate[k]=held;held=0;}
  auto objective=[&](const std::vector<double>&sell){
   int n=in.count();std::vector<double>total(n+1);total[0]=o.market.inventory[item]+stock0;
   for(int k=0;k<n;k++)total[k+1]=total[k]+in.production[k]+in.rival[k]-in.demand[k];
   double s=stock0,result=0;const ValueBasis*clock=basis?basis:(use_value_basis?&value_basis:nullptr);
   int origin=clock?clock->day:day;double origin_money=clock?clock->money:o.own.money;
   for(int k=0;k<n;k++){
    double available=s+in.production[k],q=std::min(std::max(0.,sell[k]),available);
    double rival_cash=0,own_cash=0;int unused=0;
    sale_plan::day_step_parts(item,total[k]-s,q,in.rival[k],in.demand[k],
                              ConditionalMarket::saturation(item)-1.,&rival_cash,&own_cash,&unused);
    int d=day+k;double discount=std::pow(1+cfg.discount*std::max(0.,1-origin_money/20000.),d-origin);
    result+=(own_cash-cfg.competition*rival_cash)/discount;s=available-q;
   }
   return result;
  };
  return objective(best.sell)-objective(immediate);
 }
 void demand(const View&o){
  dem={};static const std::array<std::vector<int>,8> ps{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C,C},{MI,T,W},{S,MI},{WO,WO}}};
  std::array<double,9>known{},avg{};
  for(int sh:o.shops)for(int p:ps[sh])known[p]+=6.;
  // Official shops are IID draws WITH replacement, including duplicates.
  for(int s=0;s<8;s++)for(int p:ps[s])avg[p]+=6./8.;
  for(int d=day;d<30;d++)for(int i=0;i<9;i++)dem[d][i]=(i<8?1.:0.)+known[i]+cfg.future_shop*std::max(0,std::min(8,d/3)-int(o.shops.size()))*avg[i];
 }
 Asset animal_stream(int k,int start,int pos,const Tile*tile=nullptr)const{
  Asset a;a.kind=k;int j=k-9,prod=product[j],pending=tile?tile->pending_care_bonus:0;
  if(tile){a.f[day][prod]+=tile->yield_units;a.f[day][F]+=tile->fertilizer_available;}
  else {a.fixed[start]-=animal_price[j];a.first_cost=animal_price[j];a.labor[start]+=4.;}
  for(int d=std::max(day,start);d<29;d++){
   a.f[d][W]-=1;a.f[d+1][F]+=1;
   int age=d+1-start;if(age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0){a.f[d+1][prod]+=std::min(held[j],1+pending);pending=0;}
   pending=std::min(held[j]-1,pending+1);
   a.labor[d]+=3.3+.28*near(pos)+1./ainterval[j];
  }
  a.labor[29]+=1.8+.2*near(pos);return a;
 }
 Asset crop_cycle(int k,int start,int pos,const Tile*tile=nullptr)const{
  Asset a;a.kind=k;int len=ongoing(k)?first[k]+3*interval[k]:core.h_age(k);a.end=start+len;
  if(!tile){a.fixed[start]-=seed_price[k];a.first_cost=seed_price[k];a.labor[start]+=2.;}
  else if(ongoing(k))a.f[day][k]+=tile->yield_units;
  if(ongoing(k)){
   int fertile=tile?tile->fertilized_until_day:-1;
   for(int tick=0;tick<4;tick++){
    int d=start+first[k]+tick*interval[k];if(d>29||d<=day&&tile)continue;
    double q=2.;if(d-1>=day&&fertile<d-1){a.f[d-1][F]-=1;fertile=d+1;a.labor[d-1]+=1.;}
    a.f[d][k]+=q;a.labor[d]+=.6;
   }
   for(int d=std::max(day,start);d<std::min(29,a.end);d++)a.labor[d]+=.65+.06*near(pos);
  }else{
   int harvest=std::max(day,start+len);if(harvest<=29){int q=tile?tile->yield_units:1;
    int maxday=k==W?4:k==C?3:12,minage=(maxday+1)/2,cap=k==C?4:6;
    if(tile){for(int d=day;d<=harvest;d++)if(d-start>=minage&&d-start<=maxday&&!(d==day&&tile->watered_today))q=std::min(cap,q+(tile->fertilized_until_day>=d?2:1));}
    else {
     q=1;int fertile=-1;
     for(int d=start+minage;d<=std::min(29,start+std::min(len,maxday));d++){
      int natural=std::min(cap,q+start+len-d+1),boost=std::min(cap,q+start+len-d+1+std::min(3,start+len-d+1));
      // DEAD as shipped: cfg.finite_fertilize is never assigned from Settings (defaults to 0), so this
  // never fires. The live fertilising is the executor's (core.p.finite_fertilizer <- crop_fert,
  // finite_fertilize_due). Left in place rather than deleted because it is the planner-side
  // version of the same rule and would need `finite_fertilize` wired up to be usable.
  if(cfg.finite_fertilize>0&&fertile<d&&(boost-natural)*shadow[d][k]>shadow[d][F]+cfg.action_cost){a.f[d][F]-=1;a.labor[d]+=1.;fertile=d+2;}
      q=std::min(cap,q+(fertile>=d?2:1));
     }
    }
    a.f[harvest][k]+=q;a.labor[harvest]+=1.;
   }
   for(int d=std::max(day,start);d<=std::min(29,harvest);d++)a.labor[d]+=.8+.05*near(pos);
  }return a;
 }
 // Bellman DP: keep this crop project for another cycle, or stop and free land.
 // Executed one day at a time. The actual next crop may change on replanning.
 Asset crop_dp(int k,int start,int pos,const Flow&prices)const{
  std::array<double,31>v{};std::array<bool,30>take{};
  int len=ongoing(k)?first[k]+3*interval[k]+1:core.h_age(k);
  for(int d=29;d>=start;d--){if(d+(ongoing(k)?first[k]:core.h_age(k))>29)continue;
   auto a=crop_cycle(k,d,pos);double x=0;for(int t=d;t<30;t++){x+=a.fixed[t]-cfg.action_cost*a.labor[t]-cfg.land_rent*(t<a.end&&t>=d);for(int i=0;i<9;i++)x+=a.f[t][i]*prices[t][i];}
   x+=v[std::min(30,d+len)];if(x>0){v[d]=x;take[d]=true;}
  }
  Asset a;a.kind=k;for(int d=start;d<30&&take[d];d+=len)add(a,crop_cycle(k,d,pos));a.first_cost=take[start]?seed_price[k]:0;return a;
 }
 // A freed rival tile is NOT necessarily replanted with the SAME crop. Measured: thomas harvests
 // wheat and puts CARROT back, so a same-crop continuation reported rival CARROT supply = 0 while
 // it actually ran 20 carrot tiles. Model the opponent as an optimiser -- the freed tile takes
 // whichever crop is worth most at the projected prices -- instead of as a static board.
 int best_rotation(int keep,int start,int pos,const Flow&neutral)const{
  int best=keep;double bestv=0.;
  for(int kk:{W,C,T,S,M}){
   auto cand=crop_dp(kk,start,pos,neutral);double v=0.;
   for(int t=start;t<30;t++)for(int i=0;i<9;i++)v+=cand.f[t][i]*neutral[t][i];
   if(v>bestv){bestv=v;best=kk;}
  }
  return best;
 }
 void public_rival(const View&o){
  if(use_rival_forecast){rival=rival_forecast;return;}
  rival=extra_rival;
  auto own_calendar=core.p.crop_harvest_age;
  if(cfg.calendar_samples>0)for(int k:{W,C,M})if(learned_harvest[k]>0)core.p.crop_harvest_age[k]=learned_harvest[k];
  // Price basis for scoring the rival's future crops: `neutral` subtracts demand only, so it is
  // supply-free and therefore optimistic, which is what the rotation model was measured on.
  Flow neutral{};
  for(int d=0;d<30;d++)for(int i=0;i<9;i++)neutral[d][i]=price(i,o.market.inventory[i]-dem[d][i]*(d-day+1));
  for(int pos=0;pos<100;pos++){auto&t=o.opponent.tiles[pos];Asset a;
   if(animal(t))a=animal_stream(int(t.animal),t.placed_day,pos,&t);
   else if(plant(t)){
    int k=int(t.crop),age=day-t.planted_day;double mass=0;
    if(!ongoing(k))for(int h=std::max(0,age);h<30;h++)mass+=rival_harvest_weights[k][h];
    if(mass>0){
     int saved_age=core.p.crop_harvest_age[k];
     // Condition on this batch still existing: already-passed removal ages
     // cannot be replayed as new output every day.
     for(int h=std::max(0,age);h<30;h++)if(rival_harvest_weights[k][h]>0){
      core.p.crop_harvest_age[k]=h;
      auto branch=crop_cycle(k,t.planted_day,pos,&t);int next=std::max(day+1,branch.end);
      if(next<29&&cfg.replant>0)add(branch,crop_dp(best_rotation(k,next,pos,neutral),next,pos,neutral),cfg.replant);
      add(a,branch,rival_harvest_weights[k][h]/mass);
     }
     core.p.crop_harvest_age[k]=saved_age;
    }else{
     a=crop_cycle(k,t.planted_day,pos,&t);int next=std::max(day+1,a.end+(ongoing(k)?1:0));
     if(next<29&&cfg.replant>0)add(a,crop_dp(best_rotation(k,next,pos,neutral),next,pos,neutral),cfg.replant);
    }
   }
   else continue;
   for(int d=day;d<30;d++)for(int i=0;i<9;i++)rival[d][i]+=a.f[d][i];
  }
  core.p.crop_harvest_age=own_calendar;
  if(const char*tp=std::getenv("RIVAL_DUMP")){
   static FILE*fp=std::fopen(tp,"a");
   if(fp){
    std::fprintf(fp,"DAY %d",day);
    for(int i=0;i<9;i++){std::fprintf(fp," i%d",i);
     for(int d=day;d<30;d++)std::fprintf(fp," %.1f",rival[d][i]);}
    std::fprintf(fp,"\n");std::fflush(fp);}
  }
 }
 void plan(const View&o){
  day=o.day;core.p.finite_fertilizer=cfg.finite_fertilize>0;core.p.crop_harvest_age[W]=int(cfg.wheat_age);core.p.crop_harvest_age[M]=int(cfg.melon_age);core.p.frequent_harvest=cfg.margin_weight>1.5;core.p.frequent_threshold=int(cfg.margin_weight);demand(o);public_rival(o);core.new_day(o);
  auto farm=o.own;int released=Controller::project_zero_expiry(farm,o.step);
  View v{o.step,o.day,o.hour,farm,o.opponent,o.priv,o.market,o.shops};
  Asset base;for(int i=0;i<9;i++){base.f[day][i]+=o.priv.shed[i];for(auto&inv:o.priv.inventories)base.f[day][i]+=inv[i];}
  value(o,base,&shadow);core.target.clear();std::vector<int>free;int animals=0;
  for(int pos=0;pos<100;pos++){auto&t=farm.tiles[pos];
   if(animal(t)){add(base,animal_stream(int(t.animal),t.placed_day,pos,&t));core.target.emplace_back(pos,int(t.animal));animals++;}
   else if(plant(t)){int k=int(t.crop);auto a=crop_cycle(k,t.planted_day,pos,&t);add(base,a);
    if(!ongoing(k)&&day-t.planted_day>=core.h_age(k)&&day<29){free.push_back(pos);core.target.emplace_back(pos,k);}
    else {core.target.emplace_back(pos,k);int next=std::max(day+1,a.end+(ongoing(k)?1:0));if(next<29)add(base,crop_dp(k,next,pos,shadow));}
   }else if(t.kind!=TileKind::LOCKED){free.push_back(pos);core.target.emplace_back(pos,-1);}
  }
  int owned=std::popcount(unsigned(o.own.unlocked_mask));core.planned_land=owned;
  double stockcash=0.;for(int i=0;i<9;i++)stockcash+=revenue(i,o.market.inventory[i],o.priv.shed[i]);
  double budget=std::max(0.,o.own.money+.8*stockcash-cfg.reserve-animals*o.market.prices[W]);
  if(free.size()<3&&owned<3&&day<20&&budget>next_land_cost(owned)+800){core.planned_land++;budget-=next_land_cost(owned);base.fixed[day]-=next_land_cost(owned);for(int pos=0;pos<100;pos++)if(quad(pos)==owned){free.push_back(pos);core.target.emplace_back(pos,-1);}}
  std::stable_sort(free.begin(),free.end(),[](int a,int b){return std::tuple(near(a),a/10,a%10)<std::tuple(near(b),b/10,b%10);});
  for(int kind=9;kind<12;kind++){
   int stock=o.priv.shed[kind];for(auto&inv:o.priv.inventories)stock+=inv[kind];
   while(stock-->0&&!free.empty()){
    int pos=free.front();free.erase(free.begin());auto a=animal_stream(kind,day,pos);a.fixed[day]+=animal_price[kind-9];a.first_cost=0;add(base,a);
    for(auto&[p,k]:core.target)if(p==pos){k=kind;break;}animals++;
   }
  }
  double initial=value(o,base,&shadow);
  for(int turn=0;turn<int(free.size())&&turn<cfg.new_limit;turn++){
   int pos=free[turn],bestkind=-1;Asset best;Flow bestshadow{};double bestval=initial,bestscore=0.;
   for(int k:{0,1,2,3,4,9,10,11}){
    if(k>=9&&(animals>=cfg.max_animals||day+afirst[k-9]>28))continue;
    if(k<5&&(day+(ongoing(k)?first[k]:core.h_age(k))>29))continue;
    Asset a=k>=9?animal_stream(k,day,pos):crop_dp(k,day,pos,shadow);if(a.first_cost<=0)continue;
    double immediate=a.first_cost+(k>=9?o.market.prices[W]*cfg.feed_cover:0);
    if(immediate>budget)continue;
    auto next=base;add(next,a);Flow pr{};double val=value(o,next,&pr,k);portfolios++;
    double extra=val-initial-cfg.land_rent*(k>=9?29-day:std::min(29-day,core.h_age(k)));
    double power=cfg.capital_power*std::max(0.,1-o.own.money/16000.);
    double rank=extra/std::pow(a.first_cost+(k>=9?100:0),power);
    if(rank>bestscore){bestscore=rank;bestkind=k;best=a;bestval=val;bestshadow=pr;}
   }
   if(bestkind<0)break;for(auto&[p,k]:core.target)if(p==pos){k=bestkind;break;}
   add(base,best);initial=bestval;shadow=bestshadow;budget-=best.first_cost+(bestkind>=9?o.market.prices[W]*cfg.feed_cover:0);animals+=bestkind>=9;
  }
  last_score=initial;core.prepare_orders(v,o,released);
 }
 PlayerAction act(const View&o){if(o.day!=core.day)plan(o);return core.act(o);}
};
}
