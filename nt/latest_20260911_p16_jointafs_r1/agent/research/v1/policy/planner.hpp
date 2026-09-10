#pragma once
#include "executor/policy.hpp"
#include "conditional_market.hpp"
// New autonomous cash-flow DP and marginal portfolio planner. The execution
// engine is our pre-existing dp7 controller, NOT one of the seven opponents.
namespace competitive {
using namespace dp7;
using Flow=std::array<std::array<double,9>,30>;
using Curve=std::array<double,30>;
struct Config {
 double competition=.75,supply=1.,replant=.8,capital_power=.5,discount=.01;
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
inline void add(Asset&a,const Asset&b,double scale=1.){for(int d=0;d<30;d++){for(int i=0;i<9;i++)a.f[d][i]+=scale*b.f[d][i];a.labor[d]+=scale*b.labor[d];a.fixed[d]+=scale*b.fixed[d];}}
struct Planner {
 dp7::Controller core;
 Config cfg;
 Flow dem{},rival{},shadow{},extra_rival{};
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
 double value(const View&o,const Asset&a,Flow*prices=nullptr)const{
  std::array<double,9>inv{};for(int i=0;i<9;i++)inv[i]=o.market.inventory[i];double val=0.,balance=o.own.money,liquidity=0;
  for(int d=day;d<30;d++){
   double cash=a.fixed[d],enemy=0;
   for(int i=0;i<9;i++){
    inv[i]-=dem[d][i]*.5;
#if R2_SALE_CLOCK_MODE >= 2
    double r=cfg.supply*rival[d][i]*rival_early_share[i],late=cfg.supply*rival[d][i]-r;
#else
    double r=cfg.supply*rival[d][i]*.5,late=r;
#endif
    enemy+=trade(i,inv[i],r);
    cash+=trade(i,inv[i],a.f[d][i]);
    enemy+=trade(i,inv[i],late);
    inv[i]-=dem[d][i]*.5;if(prices)(*prices)[d][i]=price(i,inv[i]);
   }
   cash-=wages(a.labor[d]);cash-=cfg.action_cost*a.labor[d];balance+=cash;
   liquidity+=std::max(0.,-balance);
   val+=(cash-cfg.competition*enemy)/std::pow(1+cfg.discount*std::max(0.,1-o.own.money/20000.),d-day);
  }
  return val-cfg.risk*liquidity;
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
 void public_rival(const View&o){rival=extra_rival;
  auto own_calendar=core.p.crop_harvest_age;
  if(cfg.calendar_samples>0)for(int k:{W,C,M})if(learned_harvest[k]>0)core.p.crop_harvest_age[k]=learned_harvest[k];
  Flow neutral{};for(int d=0;d<30;d++)for(int i=0;i<9;i++)neutral[d][i]=price(i,o.market.inventory[i]-dem[d][i]*(d-day+1));
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
      if(next<29&&cfg.replant>0)add(branch,crop_dp(k,next,pos,neutral),cfg.replant);
      add(a,branch,rival_harvest_weights[k][h]/mass);
     }
     core.p.crop_harvest_age[k]=saved_age;
    }else{
     a=crop_cycle(k,t.planted_day,pos,&t);int next=std::max(day+1,a.end+(ongoing(k)?1:0));
     if(next<29&&cfg.replant>0)add(a,crop_dp(k,next,pos,neutral),cfg.replant);
    }
   }
   else continue;
   for(int d=day;d<30;d++)for(int i=0;i<9;i++)rival[d][i]+=a.f[d][i];
  }
  core.p.crop_harvest_age=own_calendar;
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
    auto next=base;add(next,a);Flow pr{};double val=value(o,next,&pr);portfolios++;
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
