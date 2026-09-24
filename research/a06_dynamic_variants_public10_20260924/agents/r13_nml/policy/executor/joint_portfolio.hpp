#pragma once
// S4I: current-observation portfolio proposals and conditional valuation.
// No simulator, random seed, opponent controller or hindsight continuation.
namespace dp7::portfolio {
constexpr std::array<int,8> kinds{W,C,T,S,M,G,CO,SH};
using Mix=std::array<int,8>;
struct Node {Mix mix{};int spent=0,plots=0,animals=0;double score=0;};
inline bool better(const Node&a,const Node&b){
 return std::tuple(-a.score,a.spent,a.mix)<std::tuple(-b.score,b.spent,b.mix);
}
// Bounded count-vector beam. Curves are a cheap proposal heuristic ONLY.
// Keeping every state (sufficient width) is exhaustive on small test problems.
inline std::vector<Node> generate(const std::array<std::vector<double>,8>&curves,
 const std::array<int,8>&cost,int budget,int plots,int animal_limit,int width){
 if(width<=0)throw std::invalid_argument("portfolio width must be positive");
 std::vector<Node> frontier(1),result(1);
 for(int depth=0;depth<plots&&!frontier.empty();depth++){
  std::vector<Node> next;std::set<Mix> seen;
  for(const auto&node:frontier)for(int k=0;k<8;k++){
   int n=node.mix[k];if(n+1>=int(curves[k].size())||cost[k]<0||node.spent+cost[k]>budget)continue;
   if(k>=5&&node.animals>=animal_limit)continue;
   auto child=node;child.mix[k]++;child.spent+=cost[k];child.plots++;child.animals+=k>=5;
   child.score+=curves[k][n+1]-curves[k][n];
   if(std::isfinite(child.score)&&seen.insert(child.mix).second)next.push_back(child);
  }
  std::sort(next.begin(),next.end(),better);if(int(next.size())>width)next.resize(width);
  result.insert(result.end(),next.begin(),next.end());frontier=std::move(next);
 }
 std::sort(result.begin(),result.end(),better);return result;
}
inline Controller::CashSchedule sum_calendar(const std::array<Controller::CashSchedule,8>&cal,const Mix&mix){
 Controller::CashSchedule result;
 for(int k=0;k<8;k++)if(mix[k])for(int d=0;d<30;d++){
  result.fixed[d]+=mix[k]*cal[k].fixed[d];
  for(int i=0;i<9;i++)result.quantity[d][i]+=mix[k]*cal[k].quantity[d][i];
 }return result;
}
inline Controller::PortfolioContext crop_context(const Controller&c,const View&o);
inline Controller::PortfolioContext physical_context(const Controller&c,const View&o){
 // The new aggregate balance must include known maintenance obligations.
 // Do not inherit the old optional risk weight that could erase real feed.
 if(c.p.portfolio_crop_calendar)return crop_context(c,o);
 auto forecast=c;forecast.p.own_feed_demand_weight=1.;return forecast.portfolio_context(o);
}
// Net self-produced feed against same-day requirements BEFORE pricing. The
// netting is for the cash forecast, never a credit to actual owned resources.
inline double cash_delta(const Controller&c,const View&o,const Controller::PortfolioContext&context,
 const Controller::CashSchedule&added){
 Quantities before{},after{};for(int i=0;i<9;i++)before[i]=after[i]=o.market.inventory[i];
 double value=0;
 for(int d=c.day;d<30;d++){
  value+=added.fixed[d];
  for(int i=0;i<9;i++){
   const double outside=context.rival[d][i]-context.dem[d][i];before[i]+=outside;after[i]+=outside;
   const int q0=int(std::nearbyint(context.own[d][i]));
   const int q1=int(std::nearbyint(context.own[d][i]+added.quantity[d][i]));
   double c0=Controller::projected_trade(i,before[i],q0),c1=Controller::projected_trade(i,after[i],q1);
   value+=(q1>0?c.p.timing_discount:1.)*c1-(q0>0?c.p.timing_discount:1.)*c0;
  }
 }return value;
}
struct DeliveryCalendar {Controller::CashSchedule cash;std::array<int,30>harvest{},water{},feed{},fert{},care{},planting{},dig{};};
inline DeliveryCalendar crop_delivery(const Controller&c,const View&o,int kind,Tile initial);
// This is a transparent maintained-project scenario, NOT an exact future game:
// same crop replanted; successful care/water; no unknown weeds; auto-deposit
// before the next day's sale, except an explicit final-day haul. It reflects
// collection thresholds instead of pretending every produced unit is cash.
inline DeliveryCalendar delivery(const Controller&c,int kind,int start){
 auto clock=c;clock.day=start;auto raw=clock.project_calendar(kind);DeliveryCalendar r;
 r.cash.fixed=raw.fixed;int yield=0;
 for(int d=start;d<30;d++){
  int output_item=kind>=9?product[kind-9]:kind;
  yield+=int(raw.quantity[d][output_item]);
  int next=d<29?int(raw.quantity[d+1][output_item]):0;
  int cap=kind>=9?held[kind-9]:ongoing(kind)?4:100;
  bool collect=yield>0&&((!ongoing(kind)&&kind<5)||yield>=cap||yield+next>cap||d==29);
  if(collect){r.harvest[d]=1;r.cash.quantity[std::min(29,d+1)][output_item]+=yield;yield=0;}
  if(kind>=9){
   if(d<29){r.feed[d]=r.care[d]=1;r.cash.quantity[d][W]--;}
   if(d>start){r.fert[d]=1;r.cash.quantity[std::min(29,d+1)][F]++;}
  }else{
   if(raw.fixed[d]<0){r.planting[d]=1;if(d>start&&ongoing(kind))r.dig[d]=1;}
   int current_start=start;
   for(int prior=start;prior<=d;prior++)if(raw.fixed[prior]<0)current_start=prior;
   int age=d-current_start;
   const bool active=ongoing(kind)?age<=first[kind]+3*interval[kind]+1:age<=c.h_age(kind);
   if(d<29&&active){
    if(!c.p.efficient_water||r.planting[d]||age%2==0)r.water[d]=1;
    if(ongoing(kind)&&next>0)r.water[d]=1;
    if(!ongoing(kind)&&age>=((kind==C?3:kind==W?4:12)+1)/2)r.water[d]=1;
   }
   if(raw.quantity[d][F]<0){r.fert[d]=int(-raw.quantity[d][F]);r.cash.quantity[d][F]+=raw.quantity[d][F];}
  }
 }return r;
}
// Estimate the shared future workforce by packing date-specific job sets on
// their real planned plots, not by charging every project one full worker.
// Existing assets use the same maintained-project assumption; current-day
// executable starts and funding are checked separately by preview_bundle.
inline double future_wages(const Controller&c,const View&o,const std::vector<std::pair<int,int>>&new_targets){
 struct Work{int pos,kind;DeliveryCalendar cal;};std::vector<Work>works;
 for(int pos=0;pos<100;pos++){
  const auto&t=o.own.tiles[pos];int k=animal(t)?int(t.animal):plant(t)?int(t.crop):-1;
  if(k<0)continue;int start=animal(t)?t.placed_day:t.planted_day;
  works.push_back({pos,k,c.p.portfolio_crop_calendar&&k<5?crop_delivery(c,o,k,t):delivery(c,k,std::clamp(start,0,c.day))});
 }
 for(auto[pos,k]:new_targets)works.push_back({pos,k,c.p.portfolio_crop_calendar&&k<5?crop_delivery(c,o,k,Tile{}):delivery(c,k,c.day)});
 double result=0;
 for(int d=c.day+1;d<30;d++){
  std::vector<Job>jobs;
  for(auto&w:works){Job j;j.pos=w.pos;j.priority=0;auto&cal=w.cal;
   if(cal.harvest[d]){j.actions.push_back(action(Op::HARVEST));j.out[w.kind>=9?product[w.kind-9]:w.kind]=1;}
   if(cal.fert[d]){
    if(w.kind>=9){j.actions.push_back(action(Op::COLLECT_FERTILIZER));j.out[F]=1;}
    else{j.actions.push_back(action(Op::FERTILIZE));j.needs[F]=cal.fert[d];}
   }
   if(cal.dig[d])j.actions.push_back(action(Op::DIG));
   if(cal.planting[d])j.actions.push_back(action(Op::PLANT,w.kind));
   if(cal.feed[d]){j.actions.push_back(action(Op::FEED));j.needs[W]=1;}
   if(cal.care[d])j.actions.push_back(action(Op::CARE));
   if(cal.water[d])j.actions.push_back(action(Op::WATER));
   if(!j.actions.empty())jobs.push_back(j);
  }
  int hands=0;for(;hands<=c.p.max_hands;hands++){
   // One preparation step, plus explicit terminal returns. Future positions
   // reset to depot by the rules; no opponent route is available here.
   auto packed=c.pack(jobs,c.starts(hands),d==29?22:23,d==29,false,true);
   if(packed.second==0)break;
  }
  if(hands>c.p.max_hands)return INFINITY;
  result+=Controller::hirecost(hands);
 }return result;
}
}

namespace dp7 {
inline void Controller::compare_portfolios(const View&planning,const View&actual,int released){
 if(!p.joint_investment_portfolio||day>=29)return;
 // Startup selection is an independent capability. Turning on later-day
 // portfolio comparison must not silently replace the supplied opening.
 if(day==0&&!p.autonomous_start)return;
 auto original=*this;auto reference=preview_bundle(planning,false);
 int land=std::popcount(unsigned(actual.own.unlocked_mask));
 std::vector<int>free;for(auto[pos,k]:target)if(plant_not_before[pos]<=day&&!plant(planning.own.tiles[pos])&&!animal(planning.own.tiles[pos]))free.push_back(pos);
 if(free.empty())return;
 std::stable_sort(free.begin(),free.end(),[](int a,int b){return std::tuple(near(a),snake(a))<std::tuple(near(b),snake(b));});
 auto live=counts(planning);int animal_count=live[G]+live[CO]+live[SH];
 int budget=int(investment_budget(planning,planned_land,animal_count));
 Counts limits{75,75,p.max_tomato,p.max_strawberry,p.max_melon,0,0,0,0,p.max_geese,p.max_cows,p.max_sheep};
 std::array<std::vector<double>,8>curves;std::array<int,8>cost{};
 std::array<CashSchedule,8>cal;auto demand_remaining=demand(planning);auto base=existing(planning);
 if(p.opponent_supply_weight){auto rival=opponent_supply(planning);for(int i=0;i<9;i++)demand_remaining[i]-=p.opponent_supply_weight*rival[i];}
 auto context=portfolio::physical_context(*this,planning);
 if(p.portfolio_crop_calendar){for(int i=0;i<9;i++){double supply=0;for(int d=day;d<30;d++)supply+=context.own[d][i];base[i]=int(std::nearbyint(supply));}}
 // Whole-season supply curves cheaply cover different quantities/industries.
 // They do NOT choose the final portfolio; the second stage nets chronological
 // feed/production and prices the complete executable mixture jointly.
 for(int i=0;i<8;i++){
  int k=portfolio::kinds[i];Project pr=k>=9?animal_project(k):crop_project(k);cost[i]=pr.capital;
  cal[i]=p.portfolio_crop_calendar&&k<5?portfolio::crop_delivery(*this,planning,k,Tile{}).cash:portfolio::delivery(*this,k,day).cash;curves[i]={0.};
  if(p.portfolio_crop_calendar&&k<5){pr.out=pr.fert=pr.seed=0;for(int d=day;d<30;d++){pr.out+=int(cal[i].quantity[d][k]);pr.fert-=int(cal[i].quantity[d][F]);pr.seed-=int(cal[i].fixed[d]);}}
  int limit=std::min({int(free.size()),std::max(0,limits[k]-live[k]),budget/std::max(1,pr.capital)});
  if(pr.out<=0||(k>=9&&day>p.latest_animal_day))limit=0;
  double inv=planning.market.inventory[pr.item]-demand_remaining[pr.item]+base[pr.item];
  double fert_inv=planning.market.inventory[F]-demand_remaining[F]+base[F];
  for(int n=1;n<=limit;n++){
   double gross=revenue(pr.item,inv+(n-1)*pr.out,pr.out);
   if(k>=9)gross+=revenue(F,fert_inv+(n-1)*pr.fert,pr.fert);
   double expense=pr.seed+(k>=9?pr.capital:0)+pr.feed*planning.market.prices[W]+(ongoing(k)?pr.fert*planning.market.prices[F]:0);
   curves[i].push_back(curves[i].back()+p.timing_discount*gross-expense-p.action_shadow*pr.actions);
  }
 }
 auto candidates=portfolio::generate(curves,cost,budget,int(free.size()),std::max(0,p.max_animals-animal_count),8);
 int generated=int(candidates.size());std::vector<portfolio::Mix>shortlist(1);std::set<portfolio::Mix>seen;seen.insert({});
 for(const auto&node:candidates)if(seen.insert(node.mix).second){shortlist.push_back(node.mix);if(shortlist.size()>=9)break;}
 // Keep one best pure-industry proposal too: a cheap score must not erase an
 // entire branch before chronological and labour consequences are compared.
 for(int i=0;i<8;i++){
  int n=int(std::max_element(curves[i].begin(),curves[i].end())-curves[i].begin());
  if(i>=5)n=std::min(n,std::max(0,p.max_animals-animal_count));
  portfolio::Mix mix{};mix[i]=n;if(seen.insert(mix).second)shortlist.push_back(mix);
 }
 double old_future=portfolio::future_wages(*this,planning,{});
 auto evaluate=[&](Controller&trial,const BundlePreview&preview){
  portfolio::Mix mix{};for(auto[pos,k]:preview.started_targets){auto it=std::find(portfolio::kinds.begin(),portfolio::kinds.end(),k);if(it!=portfolio::kinds.end())mix[it-portfolio::kinds.begin()]++;}
  auto added=portfolio::sum_calendar(cal,mix);
  double future=portfolio::future_wages(trial,planning,preview.started_targets);
  if(!std::isfinite(future)||!std::isfinite(old_future))return -1.e300;
  auto prepared=trial.project_preparation(planning);
  double wage=hirecost(int(prepared.farm.hands.size()))-hirecost(int(planning.own.hands.size()));
  double land_cost=trial.planned_land>land?next_land_cost(land):0.;
  return portfolio::cash_delta(trial,planning,context,added)-wage-(future-old_future)-land_cost;
 };
 auto best=original;double best_value=evaluate(best,reference);int evaluated=1,rejected=0;
 for(const auto&mix:shortlist){
  auto trial=original;std::vector<int>chosen;
  for(int i=0;i<8;i++)for(int n=0;n<mix[i];n++)chosen.push_back(portfolio::kinds[i]);
  std::stable_sort(chosen.begin(),chosen.end(),[](int a,int b){auto intensity=[](int k){return k>=9?4:ongoing(k)?2:1;};return std::tuple(-intensity(a),name(a))<std::tuple(-intensity(b),name(b));});
  // Clear only this proposal's editable set, not deferred plots excluded from
  // `free`. Otherwise another investment search silently cancels their wait.
  for(auto&[pos,k]:trial.target)if(std::find(free.begin(),free.end(),pos)!=free.end())k=-1;
  for(size_t n=0;n<chosen.size();n++)for(auto&[pos,k]:trial.target)if(pos==free[n])k=chosen[n];
  bool uses_new=false;for(auto[pos,k]:trial.target)if(k>=0&&quad(pos)>=land)uses_new=true;
  if(!uses_new)trial.planned_land=land;
  trial.prepare_orders(planning,actual,released);auto preview=trial.preview_bundle(planning,false);evaluated++;
  bool preserves=true;for(int pos=0;pos<100;pos++)if((preview.existing_actions[pos]&reference.existing_actions[pos])!=reference.existing_actions[pos])preserves=false;
  // New bundles must fund and actually schedule every new start. KEEP remains
  // available even if its older projection is imperfect; no fake output credit.
  if(!preserves||preview.started!=preview.proposed){rejected++;continue;}
  double value=evaluate(trial,preview);
  if(value>best_value+1e-6){best=std::move(trial);best_value=value;}
 }
 bool changed=best.target!=original.target||best.planned_land!=original.planned_land;
 *this=std::move(best);portfolio_generated=original.portfolio_generated+generated;
 portfolio_evaluated=original.portfolio_evaluated+evaluated;portfolio_rejected=original.portfolio_rejected+rejected;
 portfolio_switches=original.portfolio_switches+int(changed);
}
}
#include "crop_delivery.hpp"
