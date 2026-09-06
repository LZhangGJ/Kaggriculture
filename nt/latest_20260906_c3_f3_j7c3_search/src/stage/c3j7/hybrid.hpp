#pragma once
#include "planner.hpp"
#include "executor/investment_candidates.hpp"
#include "local_reference.hpp"
#include "reference_pick.hpp"
#include "public_flow_scenario.hpp"
#include "public_tape.hpp"
#include "inventory_dp.hpp"
#include "animal_service_dp.hpp"
#include "ongoing_maintenance_dp.hpp"
namespace competitive {
struct Hybrid {
 dp7::Controller core;
 Config cfg;Planner model;
 int edits=0;PublicTape tape;int sale_changes=0;
 int crop_changes=0,crop_checks=0,investment_checks=0;
 Flow valuation_prices{};bool valuation_ready=false;
 std::optional<Farm> previous_rival;int previous_rival_day=-1,previous_rival_step=-1;
 std::array<std::array<int,30>,5> harvest_observations{};
 std::array<std::array<int,3>,30> observed_counts{};int history_day=-1;
 std::array<Asset,100> service_paths{};int service_path_day=-1;
 explicit Hybrid(Config c={}):core(frozen_params()),cfg(c),model(c){
  // Harvest already produced stock while the crew visits the animal, instead
  // of waiting for the holding cap. The ordinary route/resource compiler still
  // includes that action in task/staffing schedules; no rival ID is used.
  if(cfg.harvest_threshold>0){core.p.frequent_harvest=true;core.p.frequent_threshold=int(cfg.harvest_threshold);}
  int v=int(cfg.execution_variant);
  if(v==1){core.p.max_land=4;core.p.max_hands=15;}
  if(v==2){core.p.intraday_admission=false;core.p.intraday_procurement=false;}
  if(v==3){core.p.intraday_declared_value=true;core.p.intraday_future_workforce=true;}
  if(v==4){core.p.continuous_market_execution=true;}
  if(v==5){core.p.early_deposit=1;core.p.sell_deposits=true;}
  if(v==6){core.p.compile_consequence=true;core.p.compile_bounded_rollout=true;}
  if(v==7){core.p.max_land=4;}
  if(v==8){core.p.opponent_supply_weight=cfg.supply;}
 }
 [[gnu::noinline]] Asset forecast(const View&o,const dp7::Controller&c,const std::set<int>*admitted=nullptr){
  Asset a;for(int i=0;i<9;i++){a.f[o.day][i]+=o.priv.shed[i];for(auto&b:o.priv.inventories)a.f[o.day][i]+=b[i];}
  std::array<int,100>desired;desired.fill(-1);for(auto[pos,k]:c.target)desired[pos]=k;
  for(int pos=0;pos<100;pos++){
   auto&t=o.own.tiles[pos];int want=desired[pos];
   if(animal(t)){if(cfg.investment_consistent>.5&&valuation_ready)add(a,consistent_animal(o,c,int(t.animal),t.placed_day,pos,&t));else if(service_path_day==o.day)add(a,service_paths[pos]);else add(a,model.animal_stream(int(t.animal),t.placed_day,pos,&t));}
   else if(plant(t)){
    int k=int(t.crop);auto x=(cfg.investment_crop_consistent>.5&&valuation_ready&&ongoing(k))?consistent_crop(o,c,k,t.planted_day,pos,&t):model.crop_cycle(k,t.planted_day,pos,&t);add(a,x);int next=std::max(o.day,x.end+(ongoing(k)?1:0));if(next<29&&want>=0&&want<5)add(a,crop_investment(o,c,want,next,pos));
   }else if(want>=0&&(!admitted||admitted->count(pos))){if(want>=9)add(a,(cfg.investment_consistent>.5&&valuation_ready)?consistent_animal(o,c,want,o.day,pos):model.animal_stream(want,o.day,pos));else add(a,crop_investment(o,c,want,o.day,pos));}
  }
  if(c.planned_land>std::popcount(unsigned(o.own.unlocked_mask)))a.fixed[o.day]-=next_land_cost(std::popcount(unsigned(o.own.unlocked_mask)));
  return a;
 }
 void plan(const View&o){
  growth_plan(o);
  service_plan(o);
  if(cfg.crop_timing<.5)crop_plan(o);
  auto menu=dp7branch::generate(core,o);int chosen=0;
  if(o.day<29){auto&r=switch_plans[0][o.day];for(int i=0;i<int(menu.size());i++)if(menu[i].family==r.family&&menu[i].kind==r.kind&&menu[i].amount==r.amount){chosen=i;break;}}
  core=std::move(menu[chosen].controller);
  if(cfg.mpc>0.5){lookahead(o,menu);return;}
  if(cfg.new_limit<=0||o.day==0||o.day>=27)return;
  if(cfg.new_limit>=3){cross_portfolio(o);return;}
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);Asset initial;model.value(o,initial,&model.shadow);
  prepare_consistent_prices(o,core);
  auto base=forecast(o,core);double current=investment_score(o,base,&model.shadow);
  auto pview=o;std::optional<Farm>projected;int released=0;
  if(core.p.plan_zero_expiry){projected=o.own;released=dp7::Controller::project_zero_expiry(*projected,o.step);}
  View planning{o.step,o.day,o.hour,projected?*projected:o.own,o.opponent,o.priv,o.market,o.shops};
  int allowed=int(cfg.new_limit);for(int round=0;round<std::max(1,int(cfg.expansion));round++){
   double best=current+cfg.reserve;int index=-1,kind=-1;
   for(int n=0;n<int(core.target.size());n++){
    auto[pos,old]=core.target[n];auto&t=planning.own.tiles[pos];if(old<0||animal(t)||plant(t))continue;
    if(allowed==1&&old<9)continue;
    // Same count and location: compare project semantics, not opponent ID.
    for(int k:{0,1,2,3,4,9,10,11}){
     if(k==old||(old>=9)!=(k>=9))continue;
     if(k>=9&&(o.day+afirst[k-9]>29||o.day>core.p.latest_animal_day))continue;
     if(k<5&&o.day+core.h_age(k)>29)continue;
     core.target[n].second=k;auto a=forecast(o,core);double v=investment_score(o,a);core.target[n].second=old;
     if(v>best){best=v;index=n;kind=k;}
    }
   }
   if(index<0)break;core.target[index].second=kind;current=best;edits++;
  }
  core.prepare_orders(planning,o,released);
 }



 void cross_portfolio(const View&o){
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);Asset none;model.value(o,none,&model.shadow);
  std::optional<Farm> projected;int released=0;if(core.p.plan_zero_expiry){projected=o.own;released=dp7::Controller::project_zero_expiry(*projected,o.step);}
  View planning{o.step,o.day,o.hour,projected?*projected:o.own,o.opponent,o.priv,o.market,o.shops};
  auto score=[&](dp7::Controller&c,const dp7::Controller::BundlePreview*preview){std::set<int>started;if(preview)for(auto[pos,k]:preview->started_targets)started.insert(pos);auto a=forecast(o,c,preview?&started:nullptr);return model.value(o,a);};
  for(int round=0;round<std::max(1,int(cfg.expansion));round++){
   auto original=core;std::optional<dp7::Controller::BundlePreview> basePreview;if(cfg.admission_preview>0)basePreview=core.preview_bundle(planning);
   double best=score(core,basePreview?&*basePreview:nullptr)+cfg.reserve;bool found=false;auto selected=core;
   struct Proposal{double value;int index,kind;};std::vector<Proposal> alternatives;
   Counts counts{};for(auto[pos,k]:core.target)if(k>=0)counts[k]++;
   for(int n=0;n<int(core.target.size());n++){
    auto[pos,old]=core.target[n];auto&t=planning.own.tiles[pos];if(animal(t)||plant(t))continue;
    for(int k:{0,1,2,3,4,9,10,11}){
     if(k==old)continue;
     if(old<9&&k<9)continue; // strategic cross-sector edits, not wholesale crop recoloring
     if(cfg.new_limit<4&&k<9)continue;
     if(k>=9&&(o.day>core.p.latest_animal_day||o.day+afirst[k-9]>29))continue;
     if(k<5&&o.day+core.h_age(k)>29)continue;
     int animals=counts[G]+counts[CO]+counts[SH]+(k>=9)-(old>=9);
     if(animals>core.p.max_animals)continue;
     int limit=k==CO?core.p.max_cows:k==SH?core.p.max_sheep:k==G?core.p.max_geese:k==S?core.p.max_strawberry:k==T?core.p.max_tomato:k==M?core.p.max_melon:75;
     if(counts[k]+1>std::max(limit,counts[k]))continue;
     core.target[n].second=k;double v=score(core,nullptr);core.target[n].second=old;alternatives.push_back({v,n,k});
    }
   }
   std::stable_sort(alternatives.begin(),alternatives.end(),[](auto&a,auto&b){return a.value>b.value;});
   int limit=cfg.admission_preview>0?int(cfg.admission_preview):int(alternatives.size()),tried=0;
   for(auto&x:alternatives){if(tried++>=limit)break;auto c=original;c.target[x.index].second=x.kind;c.prepare_orders(planning,o,released);
    std::optional<dp7::Controller::BundlePreview> preview;if(basePreview){preview=c.preview_bundle(planning);bool preserves=true;
     for(int pos=0;pos<100;pos++)if((preview->existing_actions[pos]&basePreview->existing_actions[pos])!=basePreview->existing_actions[pos]){preserves=false;break;}
     if(!preserves)continue;
    }
    double value=score(c,preview?&*preview:nullptr);if(value>best){best=value;selected=std::move(c);found=true;}
   }
   if(!found)break;core=std::move(selected);edits++;
  }
 }

 void renewal_plan(const View&o){
  if(cfg.rotation_edits<.5||o.day>=29)return;
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);Asset empty;model.value(o,empty,&model.shadow);
  std::optional<Farm> projected;int released=0;if(core.p.plan_zero_expiry){projected=o.own;released=dp7::Controller::project_zero_expiry(*projected,o.step);}
  View planning{o.step,o.day,o.hour,projected?*projected:o.own,o.opponent,o.priv,o.market,o.shops};
  auto valuation=[&](const dp7::Controller&c){auto a=forecast(o,c);Counts need{};for(auto&j:c.jobs(o))dp7::add(need,j.seeds);
   // Seeds already in our warehouse are sunk cost, not a second purchase.
   for(int i=0;i<5;i++)a.fixed[o.day]+=seed_price[i]*std::min(need[i],o.priv.seeds[i]);return model.value(o,a);};
  std::set<int>used;
  for(int round=0;round<int(cfg.rotation_edits);round++){
   auto base=core;auto baseline=core.preview_bundle(planning);double best=valuation(core)+cfg.reserve;auto selected=core;bool found=false;int selectedpos=-1;
   struct Proposal{double value;int index,kind,pos;};std::vector<Proposal> proposals;
   Counts counts{};for(auto[pos,k]:core.target)if(k>=0)counts[k]++;
   for(int n=0;n<int(core.target.size());n++){
    auto[pos,old]=core.target[n];auto&t=o.own.tiles[pos];if(used.count(pos)||!plant(t)||ongoing(int(t.crop))||o.day-t.planted_day<core.h_age(int(t.crop)))continue;
    for(int k=0;k<5;k++){
     if(k==old||o.day+core.h_age(k)>29)continue;
     int limit=k==S?core.p.max_strawberry:k==T?core.p.max_tomato:k==M?core.p.max_melon:75;if(counts[k]+1>std::max(limit,counts[k]))continue;
     core.target[n].second=k;double v=valuation(core);core.target[n].second=old;proposals.push_back({v,n,k,pos});
    }
   }
   std::stable_sort(proposals.begin(),proposals.end(),[](auto&a,auto&b){return a.value>b.value;});int tried=0;
   for(auto&p:proposals){if(tried++>=6||p.value<=best)break;auto c=base;c.target[p.index].second=p.kind;c.prepare_orders(planning,o,released);auto trial=c.preview_bundle(planning);bool safe=true;
    // Preserve current service and harvest. Only the crop AFTER the old harvest changes.
    uint32_t mandatory=(1u<<int(Op::HARVEST))|(1u<<int(Op::WATER))|(1u<<int(Op::FEED));
    for(int pos=0;pos<100;pos++)if((trial.existing_actions[pos]&baseline.existing_actions[pos]&mandatory)!=(baseline.existing_actions[pos]&mandatory)){safe=false;break;}
    if(!(trial.existing_actions[p.pos]&(1u<<int(Op::PLANT))))safe=false;
    if(!safe)continue;double v=valuation(c);if(v>best){best=v;selected=std::move(c);selectedpos=p.pos;found=true;}
   }
   if(!found)break;core=std::move(selected);used.insert(selectedpos);edits++;
  }
 }
 void growth_plan(const View&o){
  std::array<int,3> now{};for(auto&t:o.opponent.tiles)if(animal(t))now[int(t.animal)-9]++;
  observed_counts[o.day]=now;history_day=o.day;model.extra_rival={};
  if(cfg.growth_forecast<=0||o.day==0)return;
  int lag=std::min(3,o.day),n=now[0]+now[1]+now[2];double available=std::max(0,16-n);
  for(int d=o.day+1;d<=std::min(18,o.day+6);d++)for(int k=0;k<3;k++){
   double rate=cfg.growth_forecast*std::max(0,now[k]-observed_counts[o.day-lag][k])/double(lag)*std::pow(.75,d-o.day-1);
   rate=std::min(rate,available);available-=rate;if(rate<=0)continue;
   auto future=model.animal_stream(k+9,d,55);
   for(int s=d;s<30;s++)for(int i=0;i<9;i++)model.extra_rival[s][i]+=rate*future.f[s][i];
  }
 }
 Asset service_path(const View&o,int pos,const AnimalServiceDP*dp)const{
  auto&t=o.own.tiles[pos];int kind=int(t.animal),j=kind-9,p=t.pending_care_bonus,h=t.consecutive_unfed;
  Asset a;a.kind=kind;a.f[o.day][product[j]]=t.yield_units;a.f[o.day][F]=t.fertilizer_available;
  for(int d=o.day;d<29;d++){
   int age=d+1-t.placed_day;bool tick=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
   int feed=1,care=p<held[j]-1||tick;
   if(dp){auto q=dp->choices[d][std::min(1,h)][std::min(held[j]-1,p)];feed=q.feed;care=q.care;}
   h=feed?0:h+1;if(h>=2)break;
   int q=tick?1+(feed?p:0):0;p=std::min(held[j]-1,(tick?0:p)+care);
   a.f[d][W]-=feed;a.f[d+1][F]+=1;a.f[d+1][product[j]]+=q;
   a.labor[d]+=feed+care+1.+.12*near(pos);a.labor[d+1]+=(q>0)+.12*near(pos);
  }
  return a;
 }
 void coordinated_service(const View&o){
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);
  Asset all;std::vector<int>positions;
  for(int i=0;i<9;i++){all.f[o.day][i]+=o.priv.shed[i];for(auto&b:o.priv.inventories)all.f[o.day][i]+=b[i];}
  for(int pos=0;pos<100;pos++){auto&t=o.own.tiles[pos];if(animal(t)){positions.push_back(pos);service_paths[pos]=service_path(o,pos,nullptr);add(all,service_paths[pos]);}else if(plant(t))add(all,model.crop_cycle(int(t.crop),t.planted_day,pos,&t));}
  core.animal_service_day=o.day;core.service_feed.fill(1);core.service_care.fill(1);
  for(int pos:positions)core.service_care[pos]=core.care_due(o.own.tiles[pos]);
  double value=model.value(o,all);
  for(int round=0;round<int(cfg.service_rounds);round++){
   bool changed=false;
   for(int pos:positions){Flow prices{};model.value(o,all,&prices);
    for(int d=o.day;d<30;d++)for(int i:{W,E,MI,WO,F}){all.f[d][i]+=2;double up=model.value(o,all);all.f[d][i]-=4;double down=model.value(o,all);all.f[d][i]+=2;prices[d][i]=(up-down)/4.;}
    auto&t=o.own.tiles[pos];AnimalServiceDP dp;dp.solve(int(t.animal),t.placed_day,o.day,prices,cfg.service_cost);auto path=service_path(o,pos,&dp);
    auto alternative=all;add(alternative,service_paths[pos],-1);add(alternative,path);double v=model.value(o,alternative);
    if(v>value+1e-6){all=alternative;value=v;service_paths[pos]=path;auto c=dp.first(o.day,t);core.service_feed[pos]=c.feed;core.service_care[pos]=c.care;changed=true;}
   }
   if(!changed)break;
  }
  service_path_day=o.day;
 }
 void service_plan(const View&o){
  if(cfg.service_dp<.5||(cfg.service_dp>1.5&&cfg.service_dp<3.5&&o.day<(cfg.service_dp<2.5?15:20)))return;
  if(cfg.service_rounds>.5){coordinated_service(o);return;}
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);Asset all;
  for(int i=0;i<9;i++){all.f[o.day][i]+=o.priv.shed[i];for(auto&b:o.priv.inventories)all.f[o.day][i]+=b[i];}
  for(int pos=0;pos<100;pos++){auto&t=o.own.tiles[pos];if(animal(t))add(all,model.animal_stream(int(t.animal),t.placed_day,pos,&t));else if(plant(t))add(all,model.crop_cycle(int(t.crop),t.planted_day,pos,&t));}
  model.value(o,all,&model.shadow);
  if(cfg.service_dp>3.5){
   // Portfolio marginal cash, including price effects on the rival and on our
   // own later sales. The derivative is conditional on the same public flow model.
   auto shadows=model.shadow;
   for(int d=o.day;d<30;d++)for(int i:{W,E,MI,WO,F}){
    all.f[d][i]+=2.;double up=model.value(o,all);all.f[d][i]-=4.;double down=model.value(o,all);all.f[d][i]+=2.;
    shadows[d][i]=(up-down)/4.;
   }
   model.shadow=shadows;
  }
  core.animal_service_day=o.day;core.service_feed.fill(1);core.service_care.fill(1);
  for(int pos=0;pos<100;pos++){auto&t=o.own.tiles[pos];if(!animal(t))continue;AnimalServiceDP dp;dp.solve(int(t.animal),t.placed_day,o.day,model.shadow,cfg.service_cost);auto c=dp.first(o.day,t);core.service_feed[pos]=c.feed;core.service_care[pos]=c.care;}
 }
 double rollout_value(const View&o,const dp7::Controller&trial){
  fastkag::PublicFlowScenario scene(o,model.rival,cfg.supply);auto policy=trial;int horizon=cfg.mpc<1.5?o.day+1:30;
  while(!scene.done()&&scene.view().day<horizon){auto v=scene.view();if(policy.day!=v.day)policy=reference_pick(policy,v);scene.advance(policy.act(v));}
  double result=scene.own_cash()-cfg.competition*scene.rival_cash();
  if(!scene.done()){
   auto leaf=scene.view();auto oldday=model.day;model.day=leaf.day;model.demand(leaf);model.public_rival(leaf);Asset empty;model.value(leaf,empty,&model.shadow);auto future=forecast(leaf,policy);result+=model.value(leaf,future);model.day=oldday;model.demand(o);model.public_rival(o);
  }
  return result;
 }
 void lookahead(const View&o,std::vector<dp7branch::Candidate>&menu){
  if(o.day>=28)return;
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);Asset empty;model.value(o,empty,&model.shadow);
  struct Choice{double value;dp7::Controller controller;};std::vector<Choice>choices;
  for(auto&c:menu){auto a=forecast(o,c.controller);choices.push_back({model.value(o,a),c.controller});}
  std::stable_sort(choices.begin(),choices.end(),[](auto&a,auto&b){return a.value>b.value;});
  auto best=core;double score=rollout_value(o,best);int keep=std::max(2,int(cfg.risk));
  std::set<std::vector<std::pair<int,int>>> seen;seen.insert(core.target);
  int evaluated=0;for(auto&c:choices){if(!seen.insert(c.controller.target).second)continue;if(evaluated++>=keep)break;double v=rollout_value(o,c.controller);if(v>score+cfg.reserve){score=v;best=std::move(c.controller);edits++;}}
  core=std::move(best);
 }
 PlayerAction market_control(const View&o,PlayerAction a){
  if((cfg.hold<.5&&cfg.inventory_dp<.5)||o.day>=29)return a;
  for(auto&x:a.market)if(x.op!=Op::SELL)return a;
  Counts shed=o.priv.shed;for(size_t u=0;u<a.units.size();u++){
   int pos=cell(u?o.own.hands[u-1]:o.own.farmer);if(!at_depot(pos))continue;
   auto&x=a.units[u];auto&bag=o.priv.inventories[u];int i=int(x.item);
   if(x.op==Op::PICKUP&&i>=0){shed[i]-=std::min(shed[i],x.quantity);}
   if(x.op==Op::DROP){int room=100-sum(shed);for(int j:o.priv.inventory_order[u]){int q=std::min(room,bag[j]);shed[j]+=q;room-=q;}}
   if(x.op==Op::PLACE&&i>=0&&i<9){shed[i]+=std::min({100-sum(shed),bag[i],x.quantity});}
  }
  Counts need{};for(auto&j:core.jobs(o))dp7::add(need,j.needs);
  for(auto&bag:o.priv.inventories)for(int i:{W,F})need[i]-=bag[i];
  need[W]=std::max(need[W],core.feed_stock_target);for(int i:{W,F})shed[i]=std::max(0,shed[i]-std::max(0,need[i]));
  if(cfg.inventory_dp>.5){
   return inventory_control(o,std::move(a),shed);
  }
  if(cfg.hold>1.5){
   model.day=o.day;model.demand(o);model.public_rival(o);
   for(int i=0;i<8;i++)if(shed[i]>0){
    double inv=o.market.inventory[i],future=inv-model.dem[o.day][i]+cfg.supply*model.rival[o.day][i];
    double now=dp7::revenue(i,inv,shed[i]),later=dp7::revenue(i,future,shed[i]);
    double r=std::max(0.,model.rival[std::min(29,o.day+1)][i]);
    double external=Planner::quote(i,future,r)*r-Planner::quote(i,future+shed[i],r)*r;
    if(later-now>cfg.competition*external&&sum(o.priv.shed)<65&&o.own.money>2000)shed[i]=0;
   }
  }
  a.market=core.sales_sorted(o,shed);return a;
 }

 PlayerAction inventory_control(const View&o,PlayerAction a,const Counts&available){
  // Preserve the original feed/fertilizer orders and protect procurement phases.
  if(core.phase!=3)return a;
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);
  Counts outputs{};for(auto&j:core.jobs(o))dp7::add(outputs,j.out);
  for(auto&b:o.priv.inventories)for(int i=1;i<8;i++)outputs[i]+=b[i];
  Counts selected{};for(auto&x:a.market)if(x.op==Op::SELL&&(int(x.item)==W||int(x.item)==F))selected[int(x.item)]+=x.quantity;
  int H=std::min(int(cfg.inventory_dp),std::max(0,(719-o.step-1)/4));if(H<1)return a;
  for(int i=1;i<8;i++)if(available[i]>0){
   std::vector<InventoryDP::Step> steps(H);double demandcarry=0,rivalcarry=0;
   for(int t=0;t<H;t++){
    int start=o.step+4*t,end=std::min(719,start+4),d=std::min(29,start/24),bucket=(start%24)/4;
    for(int s=start;s<end;s++){
     if(s%24==0)demandcarry+=1.;
     if(s%4==0)demandcarry+=(model.dem[d][i]-1)/6.;
    }
    double daily=std::max(0.,model.rival[d][i]*cfg.supply);
    if(cfg.tape>0&&tape.days>=2&&tape.ema[i]>0&&d<o.day+3)daily=(1-cfg.tape)*daily+cfg.tape*tape.ema[i];
    rivalcarry+=daily*(cfg.tape>0?tape.hour_weight(i,bucket):1./6.);
    steps[t].demand=int(std::floor(demandcarry+1e-9));demandcarry-=steps[t].demand;
    steps[t].rival=int(std::floor(rivalcarry+1e-9));rivalcarry-=steps[t].rival;
    if(start/24!=end/24&&d==o.day)steps[t].arrival=std::min(100,outputs[i]);
   }
   InventoryDP dp(i,std::move(steps),cfg.competition);selected[i]=dp.solve(0,std::min(available[i],100),o.market.inventory[i]).sell;
  }
  // Warehouse capacity couples items. Guarantee at least the existing compiler's
  // capacity evacuation when a deposit is imminent; do not undo that constraint.
  bool deposit=o.hour>=23;for(auto&x:a.units)deposit|=x.op==Op::DROP||x.op==Op::PLACE;
  if(deposit)for(auto&x:a.market)if(x.op==Op::SELL)selected[int(x.item)]=std::max(selected[int(x.item)],x.quantity);
  auto orders=core.sales_sorted(o,selected);bool same=orders.size()==a.market.size();if(same)for(size_t i=0;i<orders.size();i++)same&=orders[i].op==a.market[i].op&&orders[i].item==a.market[i].item&&orders[i].quantity==a.market[i].quantity;
  sale_changes+=!same;a.market=std::move(orders);return a;
 }


 // First-day installation and already-completed service are not optional.
 // Future service uses the SAME finite-horizon animal recurrence as execution,
 // conditional on one frozen price curve shared by every candidate in this call.
 // Conditional ongoing-crop investment flow shares the maintenance recurrence.
 // Collection is still an expected next-dawn event, not a route guarantee.
 [[gnu::noinline]] Asset consistent_crop(const View&o,const dp7::Controller&c,int kind,int birth,int pos,const Tile*t=nullptr){
  Asset a;a.kind=kind;a.end=birth+first[kind]+3*interval[kind];
  int begin=std::max(o.day,birth),dry=t?std::clamp(int(t->consecutive_unwatered),0,1):1;
  int f=t?std::clamp(int(t->fertilized_until_day)-begin+1,0,3):0;
  if(t){a.f[o.day][kind]+=t->yield_units;if(t->yield_units>0)a.labor[o.day]+=1.+.12*near(pos);}
  else{a.fixed[birth]-=seed_price[kind];a.first_cost=seed_price[kind];a.labor[birth]+=1.;}
  OngoingMaintenanceDP dp;dp.kind=kind;dp.birth=birth;dp.begin=begin;dp.mode=std::max(1,int(cfg.crop_dp));dp.work=cfg.crop_action_cost;
  for(int d=begin;d<30;d++){dp.price[d]=valuation_prices[d][kind];dp.fert[d]=std::max(valuation_prices[d][F],cfg.crop_fert_floor);}
  for(int d=begin;d<std::min(29,a.end);d++){
   bool w=dp.legacy_water(d,dry),z=dp.legacy_fertilize(d,f),paid_water=false;
   if(cfg.crop_dp>.5&&d>=cfg.crop_start&&d>birth){
    auto choice=dp.first(d,dry,f,w,z);double gain=choice.value-dp.q(d,dry,f,w,z);
    if(gain>cfg.crop_gain+1e-9){w=choice.water;z=choice.fertilize;}
   }
   if(d==begin){
    if(!t)w=true; // unconditional PLANT+WATER on a new lifecycle
    else {
     if(c.crop_service_day==o.day&&c.crop_birth[pos]==birth&&c.crop_kind[pos]==kind){
      if(c.crop_water[pos]>=0)w=c.crop_water[pos];if(c.crop_fertilize[pos]>=0)z=c.crop_fertilize[pos];
     }
     paid_water=t->watered_today;w|=paid_water;
    }
   }
   if(z&&f>0)z=false;
   a.f[d][F]-=z;a.labor[d]+=(w&&!paid_water)+z+.06*near(pos)*(w||z);
   dry=w?0:dry+1;if(dry>=2)break;
   int active=z?3:f;
   if(dp.production(d+1)){a.f[d+1][kind]+=1+(w&&active>0);a.labor[d+1]+=1.+.12*near(pos);}
   f=std::max(0,active-1);
  }
  return a;
 }
 [[gnu::noinline]] Asset crop_investment(const View&o,const dp7::Controller&c,int kind,int start,int pos){
  if(cfg.investment_crop_consistent<.5||!valuation_ready||!ongoing(kind))return model.crop_dp(kind,start,pos,model.shadow);
  Asset result;result.kind=kind;
  for(int d=start;d+first[kind]<=29;){
   auto cycle=consistent_crop(o,c,kind,d,pos);double value=0;
   for(int s=d;s<30;s++){value+=cycle.fixed[s]-cfg.action_cost*cycle.labor[s]-cfg.land_rent*(s<cycle.end);for(int k=0;k<9;k++)value+=cycle.f[s][k]*valuation_prices[s][k];}
   if(d>start&&value<=0)break;add(result,cycle);if(d==start)result.first_cost=cycle.first_cost;d=cycle.end+1;
  }
  return result;
 }
 [[gnu::noinline]] Asset consistent_animal(const View&o,const dp7::Controller&c,int kind,int birth,int pos,const Tile*t=nullptr)const{
  int j=kind-9,h=t?std::clamp(int(t->consecutive_unfed),0,1):0,p=t?std::clamp(int(t->pending_care_bonus),0,held[j]-1):0;
  AnimalServiceDP dp;dp.solve(kind,birth,o.day,valuation_prices,cfg.service_cost);Asset a;a.kind=kind;
  if(t){a.f[o.day][product[j]]+=t->yield_units;a.f[o.day][F]+=t->fertilizer_available;}
  else{a.fixed[o.day]-=animal_price[j];a.first_cost=animal_price[j];a.labor[o.day]+=4.;}
  for(int d=o.day;d<29;d++){
   auto choice=dp.choices[d][h][p];int feed=choice.feed,care=choice.care;
   bool paid_feed=false,paid_care=false;
   if(d==o.day){
    if(!t){feed=1;care=1;} // current executor's installation contract
    else{
     if(c.animal_service_day==o.day){feed=c.service_feed[pos];care=c.service_care[pos];}
     paid_feed=t->fed_today;paid_care=t->cared_today;feed|=paid_feed;care|=paid_care;
    }
   }
   if(feed&&!paid_feed)a.f[d][W]-=1;
   a.labor[d]+=(feed&&!paid_feed)+(care&&!paid_care);
   h=feed?0:h+1;if(h>=2)break;
   int age=d+1-birth;bool tick=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
   int q=tick?1+(feed?p:0):0;p=std::min(held[j]-1,(tick?0:p)+(feed&&care));
   a.f[d+1][F]+=1;a.f[d+1][product[j]]+=q;
   a.labor[d+1]+=1.+(q>0)+.24*near(pos);
  }
  return a;
 }
 [[gnu::noinline]] void prepare_consistent_prices(const View&o,const dp7::Controller&c,const std::set<int>*started=nullptr){
  valuation_ready=false;if(cfg.investment_consistent<.5&&cfg.investment_crop_consistent<.5)return;
  Asset neutral;model.value(o,neutral,&model.shadow);
  auto incumbent=forecast(o,c,started);model.value(o,incumbent,&valuation_prices);valuation_ready=true;
 }
 double investment_score(const View&o,const Asset&a,Flow*prices=nullptr){
  double value=model.value(o,a,prices);
  // Admission requires an ALREADY PAID idle worker and a feasible route.
  // Do not invent additional current-day hires; future-day staffing is charged.
  if(cfg.investment_paid_labor>.5&&o.hour>0)value+=model.wages(a.labor[o.day]);
  return value;
 }
 [[gnu::noinline]] std::array<double,12> shared_investment_values(const dp7::Controller&c,const View&o,const PlayerAction&a){
  ++investment_checks;model.day=o.day;model.core.p=c.p;model.demand(o);model.public_rival(o);
  if(cfg.investment_timing>1.5){
   std::array<double,9>left{};for(int s=o.step;s<std::min(719,(o.day+1)*24);s++){
    if(s%24==0)for(int i=0;i<8;i++)left[i]+=1.;
    if(s%4==0){static const std::array<std::vector<int>,8> products{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C,C},{MI,T,W},{S,MI},{WO,WO}}};for(int sh:o.shops)for(int i:products[sh])left[i]+=1.;}
   }model.dem[o.day]=left;
  }
  auto commitments=dp7::intraday::committed_new_targets(c,o,a);std::set<int> started;
  for(auto[pos,k]:commitments)started.insert(pos);
  // Current assets + only new targets whose actions are already committed.
  // Unexecuted, unadmitted morning wishes do not become free visible assets.
  prepare_consistent_prices(o,c,&started);
  auto base=forecast(o,c,&started);double v=investment_score(o,base,&model.shadow);
  std::array<double,12> values{};values.fill(-1e9);
  for(int k:{0,1,2,3,4,9,10,11}){
   Asset project=k>=9?((cfg.investment_consistent>.5&&valuation_ready)?consistent_animal(o,c,k,o.day,44):model.animal_stream(k,o.day,44)):crop_investment(o,c,k,o.day,44);
   if(cfg.investment_timing<.5){for(int d=o.day;d<29;d++){for(int i=0;i<9;i++){project.f[29][i]+=project.f[d][i];project.f[d][i]=0;}}}
   auto trial=base;add(trial,project);values[k]=investment_score(o,trial)-v;
  }
  return values;
 }
 void install_investment(const View&o){
  core.admission_values={};core.admission_blend=0;
  if(cfg.investment_blend<=0||o.day<cfg.investment_start)return;
  core.admission_blend=cfg.investment_blend;core.admission_margin=cfg.investment_margin;core.admission_scope=int(cfg.investment_scope);
  core.admission_values=[this](const dp7::Controller&c,const View&v,const PlayerAction&a){return shared_investment_values(c,v,a);};
 }
 void crop_plan(const View&o){
  core.crop_service_day=-1;
  if(cfg.crop_dp<.5||o.day<cfg.crop_start||o.day>=29)return;
  model.day=o.day;model.core.p=core.p;model.demand(o);model.public_rival(o);Asset portfolio;
  for(int i=0;i<9;i++){portfolio.f[o.day][i]+=o.priv.shed[i];for(auto&bag:o.priv.inventories)portfolio.f[o.day][i]+=bag[i];}
  for(int pos=0;pos<100;pos++){auto&t=o.own.tiles[pos];if(animal(t))add(portfolio,model.animal_stream(int(t.animal),t.placed_day,pos,&t));else if(plant(t))add(portfolio,model.crop_cycle(int(t.crop),t.planted_day,pos,&t));}
  Flow prices{};model.value(o,portfolio,&prices);
  if(cfg.crop_price_mode>.5){for(int d=o.day;d<30;d++)for(int i:{T,S,F}){
   portfolio.f[d][i]+=1.;double up=model.value(o,portfolio);portfolio.f[d][i]-=2.;double down=model.value(o,portfolio);portfolio.f[d][i]+=1.;prices[d][i]=(up-down)/2.;
  }}
  // crop_plan runs before the next new_day(). Never consult the previous
  // controller clock when constructing today's legacy counterfactual.
  dp7::Controller today_rules(core.p);today_rules.day=o.day;
  core.crop_service_day=o.day;core.crop_water.fill(-1);core.crop_fertilize.fill(-1);core.crop_birth.fill(-1);core.crop_kind.fill(-1);
  struct Edit{int pos;bool w,f;double gain;};std::vector<Edit> candidates;
  for(int pos=0;pos<100;pos++){
   const auto&t=o.own.tiles[pos];int k=int(t.crop);
   if(!plant(t)||!ongoing(k)||t.planted_day>=o.day||t.watered_today)continue;
   int last=t.planted_day+first[k]+3*interval[k];if(o.day>=last)continue;
   bool oldw=today_rules.water_due(t),oldf=today_rules.fertilize_due(t);int dry=std::clamp(int(t.consecutive_unwatered),0,1),f=std::clamp(int(t.fertilized_until_day)-o.day+1,0,3);
   OngoingMaintenanceDP dp;dp.kind=k;dp.birth=t.planted_day;dp.begin=o.day;dp.mode=int(cfg.crop_dp);dp.work=cfg.crop_action_cost;
   for(int d=o.day;d<30;d++){dp.price[d]=prices[d][k];dp.fert[d]=std::max(prices[d][F],cfg.crop_fert_floor);}
   double baseline=dp.q(o.day,dry,f,oldw,oldf);OngoingMaintenanceDP::Choice choice{baseline,oldw,oldf};
   for(int w=0;w<=int(oldw);w++)for(int z=0;z<=int(oldf);z++){double v=dp.q(o.day,dry,f,w,z);if(v>choice.value+1e-9)choice={v,bool(w),bool(z)};}
   double gain=choice.value-baseline;++crop_checks;
   // C3 is a conservative subtractive controller: it may omit an incumbent
   // maintenance task, never add an unbudgeted task or erase PLANT+WATER.
   if((choice.water&&!oldw)||(choice.fertilize&&!oldf))continue;
   if((choice.water!=oldw||choice.fertilize!=oldf)&&gain>cfg.crop_gain+1e-9)candidates.push_back({pos,choice.water,choice.fertilize,gain});
  }
  std::stable_sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.gain>b.gain;});
  for(int i=0;i<std::min(int(candidates.size()),int(cfg.crop_max_changes));i++){
   auto e=candidates[i];auto&t=o.own.tiles[e.pos];core.crop_birth[e.pos]=t.planted_day;core.crop_kind[e.pos]=int(t.crop);
   core.crop_water[e.pos]=e.w;core.crop_fertilize[e.pos]=e.f;crop_changes++;
  }
 }

 // Learn harvest timing from public lifecycle changes, not opponent identity.
 // Empty/replanted after positive stock is evidence; a weed transition is not.
 void observe_calendar(const View&o){
  if(cfg.calendar_samples<=0)return;
  if(previous_rival&&o.step>previous_rival_step){
   for(int pos=0;pos<100;pos++){
    const auto&old=previous_rival->tiles[pos];const auto&now=o.opponent.tiles[pos];
    if(!plant(old)||ongoing(int(old.crop))||old.yield_units<=0)continue;
    int k=int(old.crop),age=previous_rival_day-old.planted_day;
    bool gone=now.kind==TileKind::EMPTY||(plant(now)&&(now.planted_day!=old.planted_day||now.crop!=old.crop));
    if(!gone||age<first[k]||age>=30)continue;
    harvest_observations[k][age]++;
    int n=std::accumulate(harvest_observations[k].begin(),harvest_observations[k].end(),0);
    if(n<std::max(2,int(cfg.calendar_samples)))continue;
    int count=0;for(int a=1;a<30;a++){count+=harvest_observations[k][a];if(2*count>=n){model.learned_harvest[k]=a;break;}}
   }
  }
  previous_rival=o.opponent;previous_rival_day=o.day;previous_rival_step=o.step;
 }
 PlayerAction act(const View&o){observe_calendar(o);if(cfg.tape>0)tape.observe(o);if(o.day!=core.day){plan(o);renewal_plan(o);if(cfg.crop_timing>.5)crop_plan(o);}install_investment(o);auto a=market_control(o,core.act(o));if(cfg.tape>0)tape.remember(o,a);return a;}
};
}
