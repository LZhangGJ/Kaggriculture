#pragma once
#include "proposals.hpp"
#include "public_flow_scenario.hpp"
#include "learned_value.hpp"
#include "observed_crop_clock.hpp"
#include "startup_supply.hpp"
#ifndef A08_PREFIX_DAYS
#define A08_PREFIX_DAYS 1
#endif
namespace triad {
// Bounded public-information MPC or offline-calibrated value selector.
// Candidate interventions last only today, then a common base policy resumes.

// Observation-only readout. This code is gated off unless the offline harness
// explicitly requests a day. It never writes controller/market state.
inline std::string cash_probe(const Controller&c,const View&o){
 const auto&m=c.model;const auto&a=c.portfolio;
 Flow flows=a.f,opening{};
#if A08_REALIZATION_MODE==2
 auto cal=m.calendar(a);flows=cal.trade;opening=cal.opening;
#endif
 std::array<double,9> inv{};for(int i=0;i<9;i++)inv[i]=o.market.inventory[i];
 double virtual_balance=o.own.money,financial_balance=o.own.money,legacy_risk=0,financial_risk=0;
 std::ostringstream out;out.precision(17);
 out<<"{\"observed_cash\":"<<o.own.money<<",\"predicted_rank\":"<<c.predicted<<",\"days\":[";
 for(int d=o.day;d<30;d++){
  double wage=m.wages(a.labor[d]),shadow=m.cfg.action_cost*a.labor[d],cash=a.fixed[d],open=a.fixed[d]-wage;
  double sell=0,buy=0,open_sell=0,open_buy=0;
  for(int i=0;i<9;i++){
   inv[i]-=m.dem[d][i]*.5;double r=m.cfg.supply*m.rival[d][i]*.5;
   m.trade(i,inv[i],r);
#if A08_REALIZATION_MODE==2
   double q=opening[d][i]+a.f[d][i]-a.physical_output[d][i];double vi=inv[i];double cv=m.trade(i,vi,q);
   open+=cv;open_sell+=std::max(0.,cv);open_buy+=std::max(0.,-cv);
#endif
   double delta=m.trade(i,inv[i],flows[d][i]);cash+=delta;sell+=std::max(0.,delta);buy+=std::max(0.,-delta);
   m.trade(i,inv[i],r);inv[i]-=m.dem[d][i]*.5;
  }
  double vo=virtual_balance+open-shadow,fo=financial_balance+open;
  virtual_balance+=cash-wage-shadow;financial_balance+=cash-wage;
#if A08_REALIZATION_MODE==2
  legacy_risk+=std::max(0.,-vo);financial_risk+=std::max({0.,-fo,-financial_balance});
#else
  financial_risk+=std::max(0.,-financial_balance);
#endif
  legacy_risk+=std::max(0.,-virtual_balance);
  if(d>o.day)out<<",";
  out<<"{\"day\":"<<d<<",\"fixed_net\":"<<a.fixed[d]<<",\"conditional_sales\":"<<sell<<",\"conditional_resource_buys\":"<<buy<<",\"opening_sales\":"<<open_sell<<",\"opening_buys\":"<<open_buy<<",\"wages\":"<<wage<<",\"labor_shadow_non_cash\":"<<shadow<<",\"old_opening_balance\":"<<vo<<",\"financial_opening_balance\":"<<fo<<",\"old_end_balance\":"<<virtual_balance<<",\"financial_end_balance\":"<<financial_balance<<"}";
 }
 out<<"],\"legacy_liquidity_penalty\":"<<legacy_risk<<",\"financial_liquidity_penalty\":"<<financial_risk<<",\"seed_inventory\":[";
 for(int k=0;k<5;k++){if(k)out<<",";out<<o.priv.seeds[k];}
 out<<"],\"shed\":[";for(int k=0;k<12;k++){if(k)out<<",";out<<o.priv.shed[k];}
 out<<"],\"queue\":[";bool first=true;for(auto a:c.core.queue){if(!first)out<<",";first=false;out<<"["<<int(a.op)<<","<<int(a.item)<<","<<a.quantity<<"]";}out<<"]}";return out.str();
}

struct SearchController {
 int probe_day=-1;std::string probe_json="{}";
 Settings base;Controller live;
 PublicTradeLedger ledger;SaleClock sale_clock;
 ObservedCropClock crop_clock;competitive::Flow removal_forecast{};
 int searches=0,scenarios=0,changes=0,restore_day=-1;double predicted_gain=0;
 std::array<int,32>chosen{};
 explicit SearchController(Settings s={}):base(s),live(s){if(s.scenario<0&&!learned::available)throw std::invalid_argument("No trained value model installed");}
 std::vector<Proposal> prepare(const View&o,bool names=false)const{return generate_proposals(live,base,o,names);}
 void install(const Proposal&p,int day){live=p.policy;restore_day=day+1;}
 void choose(const View&o){
  searches++;int horizon=std::clamp(std::max(int(base.scenario),A08_PREFIX_DAYS),1,5);auto proposals=prepare(o);
  int winner=0;double bestscore=-1e100,baseline=0;
  std::ostringstream probe;probe.precision(17);bool probing=probe_day==o.day;if(probing)probe<<"{\"step\":"<<o.step<<",\"candidates\":[";
  for(int i=0;i<int(proposals.size());i++){
   auto&proposal=proposals[i];double score=0;
   if(probing){if(i)probe<<",";probe<<"{\"id\":"<<proposal.id<<",\"ledger\":"<<cash_probe(proposal.policy,o);}
   if(base.scenario<0){
    score=!learned::available?0:learned::score(proposal.features.x,proposal.id==0);
   }else{
    auto roll=proposal.policy;roll.previous_step=o.step-1;
    fastkag::PublicFlowScenario world(o,proposal.policy.model.rival,base.supply,
#if R2_SALE_CLOCK_MODE >= 1
      &sale_clock
#else
      nullptr
#endif
    );
    int stop=std::min(719,o.step+24*horizon);Settings common=base;common.scenario=0;
    while(!world.done()&&world.view().step<stop){auto v=world.view();if(v.day!=o.day)roll.configure(common);auto act=roll.act(v);world.advance(act);}
    score=world.own_cash()-base.competition*world.rival_cash();
    if(probing)probe<<",\"prefix_end_step\":"<<world.view().step<<",\"prefix_cash\":"<<world.own_cash();
    if(!world.done()){auto v=world.view();Controller tail=live;tail.configure(common);tail.book=roll.book;tail.plan(v);score+=tail.predicted;}
   }
   if(probing)probe<<",\"full_score\":"<<score<<"}";
   scenarios++;if(i==0)baseline=score;
   if(score>bestscore+1e-6){bestscore=score;winner=i;}
  }
  if(probing){probe<<"],\"winner_id\":"<<proposals[winner].id<<"}";probe_json=probe.str();}
  int id=proposals[winner].id;changes+=id!=0;chosen[id]++;predicted_gain+=bestscore-baseline;install(proposals[winner],o.day);
 }
 PlayerAction act(const View&o){
#if R2_STARTUP_SUPPLY_MODE >= 1
  if(o.step==0){
   live.model.extra_rival=startup_supply_prior(o,base);
   constexpr double weight=R2_STARTUP_SUPPLY_MODE==1?.5:1.;
   for(auto&d:live.model.extra_rival)for(auto&x:d)x*=weight;
  }else if(o.day>0)live.model.extra_rival={};
#endif
#if R2_CROP_CLOCK_MODE >= 1
  crop_clock.observe(o);
#if R2_CROP_CLOCK_MODE == 1
  live.model.rival_harvest_weights=crop_clock.weights();
#endif
#endif
  if constexpr(PublicTradeLedger::enabled){ledger.observe(o);sale_clock.observe(ledger);}
  live.sale_memory=sale_clock;
#if R2_SALE_CLOCK_MODE >= 2
  if(o.day!=live.core.day)live.model.rival_early_share=sale_clock.early_shares();
#endif
  if(restore_day>=0&&o.day>=restore_day){live.configure(base);restore_day=-1;}
  if(base.scenario!=0&&o.day!=live.core.day)choose(o);
  auto out=live.act(o);
#if R2_CROP_CLOCK_MODE >= 1
  if(o.hour==0){auto model=live.model;model.rival_harvest_weights=crop_clock.weights();model.public_rival(o);removal_forecast=model.rival;}
#endif
  if constexpr(PublicTradeLedger::enabled)ledger.record(o,out);
  return out;
 }
 std::string debug()const{
  auto text=live.debug();text.pop_back();std::ostringstream s;s<<text<<",\"search_calls\":"<<searches<<",\"search_scenarios\":"<<scenarios<<",\"search_changes\":"<<changes<<",\"search_predicted_gain\":"<<predicted_gain<<",\"learned_value_active\":"<<(base.scenario<0?"true":"false")<<",\"learned_value_available\":"<<(learned::available?"true":"false")<<",\"search_choices\":[";
  for(int i=0;i<32;i++){if(i)s<<",";s<<chosen[i];}s<<"]}";return s.str();
 }
};
}
