#pragma once
#include "proposals.hpp"
#include "public_flow_scenario.hpp"
#include "learned_value.hpp"
#include "observed_crop_clock.hpp"
#include "startup_supply.hpp"
namespace triad {
// Bounded public-information MPC or offline-calibrated value selector.
// Candidate interventions last only today, then a common base policy resumes.
struct SearchController {
 Settings base;Controller live;
 PublicTradeLedger ledger;SaleClock sale_clock;
 ObservedCropClock crop_clock;competitive::Flow removal_forecast{};
 int searches=0,scenarios=0,changes=0,restore_day=-1;double predicted_gain=0;
 std::array<int,32>chosen{};
 explicit SearchController(Settings s={}):base(s),live(s){if(s.scenario<0&&!learned::available)throw std::invalid_argument("No trained value model installed");}
 std::vector<Proposal> prepare(const View&o,bool names=false)const{return generate_proposals(live,base,o,names);}
 void install(const Proposal&p,int day){live=p.policy;restore_day=day+1;}
 void choose(const View&o){
  searches++;int horizon=std::clamp(int(base.scenario),1,5);auto proposals=prepare(o);
  int winner=0;double bestscore=-1e100,baseline=0;
  for(int i=0;i<int(proposals.size());i++){
   auto&proposal=proposals[i];double score=0;
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
    if(!world.done()){auto v=world.view();Controller tail=live;tail.configure(common);tail.book=roll.book;tail.plan(v);score+=tail.predicted;}
   }
   scenarios++;if(i==0)baseline=score;
   if(score>bestscore+1e-6){bestscore=score;winner=i;}
  }
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
  for(int i=0;i<32;i++){if(i)s<<",";s<<chosen[i];}s<<"],\"route_mode\":"<<R2_ROUTE_ECONOMICS<<",\"route_checks\":"<<live.logistics.checks<<",\"route_evaluations\":"<<live.logistics.evaluations<<",\"route_changes\":"<<live.logistics.changes<<",\"route_unsafe\":"<<live.logistics.unsafe<<",\"route_hire_checks\":"<<live.logistics.hire_checks<<",\"route_hire_changes\":"<<live.logistics.hire_changes<<",\"capacity_repairs\":"<<live.logistics.capacity_repairs<<",\"finance_repairs\":"<<live.logistics.finance_repairs<<",\"fill_mismatches\":"<<live.logistics.fill_mismatches<<",\"route_replans\":"<<live.logistics.replans<<",\"route_restored\":"<<live.logistics.restored<<",\"route_unresolved_checks\":"<<live.logistics.unresolved<<",\"route_witness_used\":"<<live.logistics.witness_used<<",\"route_chain_repairs\":"<<live.logistics.chain_repairs<<",\"route_proof_rejected\":"<<live.logistics.proof_rejected<<"}";return s.str();
 }
};
}
