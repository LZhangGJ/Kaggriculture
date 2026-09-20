#pragma once
#include "proposals.hpp"
#include "public_flow_scenario.hpp"
#include "learned_value.hpp"
#include "observed_crop_clock.hpp"
#include "startup_supply.hpp"
#include "joint_candidates.hpp"
namespace triad {
// Bounded public-information MPC or offline-calibrated value selector.
// Candidate interventions last only today, then a common base policy resumes.
struct SearchController {
 Settings base;Controller live;
 PublicTradeLedger ledger;SaleClock sale_clock;
 ObservedCropClock crop_clock;competitive::Flow removal_forecast{};
 int searches=0,scenarios=0,changes=0,restore_day=-1;double predicted_gain=0;
 std::array<int,32>chosen{};
 int joint_searches=0,joint_rollouts=0,joint_switches=0,joint_rejected=0;
 double joint_predicted_gain=0;
 struct JointResult {int id,removed,feeders,age,next,successors,cancelled;double score;bool feasible;};
 std::vector<JointResult>last_joint;
 double event_value(const View&o,Controller roll,const Flow&rival,int horizon,int required,bool&feasible,int&successors){
  roll.previous_step=o.step-1;int count0=roll.joint.successor_started,cancel0=roll.joint.cancelled;
  fastkag::PublicFlowScenario world(o,rival,base.supply);
  int stop=std::min(719,o.step+24*horizon);Settings common=base;common.scenario=0;
  while(!world.done()&&world.view().step<stop){
   auto v=world.view();if(v.day!=o.day){roll.configure(common);roll.model.extra_rival={};}
   auto action=roll.act(v);world.advance(action);
  }
  roll.joint.observe(world.view());successors=roll.joint.successor_started-count0;
  feasible=required==0||(successors>=required&&roll.joint.cancelled==cancel0);
  double score=world.own_cash()-base.competition*world.rival_cash();
  if(!world.done()){
   auto v=world.view();Controller tail=roll;tail.configure(common);tail.plan(v);score+=tail.predicted;
  }
  joint_rollouts++;return score;
 }
 explicit SearchController(Settings s={}):base(s),live(s){if(s.scenario<0&&!learned::available)throw std::invalid_argument("No trained value model installed");}
 std::vector<Proposal> prepare(const View&o,bool names=false)const{return generate_proposals(live,base,o,names);}
 void install(const Proposal&p,int day){live=p.policy;restore_day=day+1;}
 double score(const View&o,const Proposal&proposal,int requested_horizon=0)const{
  if(base.scenario<0)return !learned::available?0:learned::score(proposal.features.x,proposal.id==0);
  auto roll=proposal.policy;roll.previous_step=o.step-1;
  fastkag::PublicFlowScenario world(o,proposal.policy.model.rival,base.supply,
#if R2_SALE_CLOCK_MODE >= 1
    &sale_clock
#else
    nullptr
#endif
  );
  int horizon=requested_horizon?std::clamp(requested_horizon,1,5):std::clamp(int(base.scenario),1,5),stop=std::min(719,o.step+24*horizon);Settings common=base;common.scenario=0;
  while(!world.done()&&world.view().step<stop){auto v=world.view();if(v.day!=o.day)roll.configure(common);auto act=roll.act(v);world.advance(act);}
  double result=world.own_cash()-base.competition*world.rival_cash();
  if(!world.done()){auto v=world.view();Controller tail=live;tail.configure(common);tail.book=roll.book;tail.joint=roll.joint;tail.plan(v);result+=tail.predicted;}
  return result;
 }
 void choose(const View&o){
  searches++;auto proposals=prepare(o);
  int winner=0;double bestscore=-1e100,baseline=0;
  for(int i=0;i<int(proposals.size());i++){
   double value=score(o,proposals[i]);scenarios++;if(i==0)baseline=value;
   if(value>bestscore+1e-6){bestscore=value;winner=i;}
  }
  last_joint.clear();
  if constexpr(P16_JOINT_BUNDLES>0)if(base.scenario>0&&!live.joint.active){
   auto bundles=joint_candidates(live,proposals[winner].policy,o);
   if(!bundles.empty()){
    joint_searches++;int horizon=1;for(const auto&b:bundles)horizon=std::max(horizon,b.age+2);
    bool okay=true;int done=0;
    // KEEP and every exchange have identical event horizon and public rival.
    double keepvalue=event_value(o,proposals[winner].policy,proposals[winner].policy.model.rival,horizon,0,okay,done);
    double top=keepvalue;int selected=-1;
    for(int j=0;j<int(bundles.size());j++){
     auto&b=bundles[j];bool feasible=false;int successors=0;
     double value=event_value(o,b.policy,proposals[winner].policy.model.rival,horizon,b.feeders,feasible,successors);
     last_joint.push_back({b.id,b.removed,b.feeders,b.age,b.next,successors,feasible?0:1,value-keepvalue,feasible});
     if(!feasible){joint_rejected++;continue;}
     if(value>top+1e-6){top=value;selected=j;}
    }
    if(selected>=0){
     auto&b=bundles[selected];joint_switches++;joint_predicted_gain+=top-keepvalue;
     int id=b.id;chosen[id]++;changes++;live=std::move(b.policy);restore_day=o.day+1;return;
    }
   }
  }
  int id=proposals[winner].id;changes+=id!=0;chosen[id]++;predicted_gain+=bestscore-baseline;install(proposals[winner],o.day);
 }
 PlayerAction act(const View&o){
  dp7::packmemo::Cache call_cache;dp7::packmemo::Scope cache_scope(&call_cache);
  live.joint.observe(o);
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
  if(o.day>0)ledger.apply(live.model,o.day,base.supply);
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
  for(int i=0;i<32;i++){if(i)s<<",";s<<chosen[i];}s<<"],\"joint_searches\":"<<joint_searches<<",\"joint_rollouts\":"<<joint_rollouts<<",\"joint_switches\":"<<joint_switches<<",\"joint_rejected\":"<<joint_rejected<<",\"joint_gain\":"<<joint_predicted_gain<<",\"joint_active\":"<<live.joint.active<<",\"joint_sources\":"<<live.joint.source_started<<",\"joint_harvests\":"<<live.joint.source_harvested<<",\"joint_successors\":"<<live.joint.successor_started<<",\"joint_completed\":"<<live.joint.completed<<",\"joint_cancelled\":"<<live.joint.cancelled<<",\"joint_last_cancel\":"<<live.joint.last_cancel_reason<<",\"joint_last\":[";
  bool sep=false;for(const auto&r:last_joint){if(sep)s<<",";sep=true;s<<"{\"id\":"<<r.id<<",\"removed\":"<<r.removed<<",\"feeders\":"<<r.feeders<<",\"age\":"<<r.age<<",\"next\":"<<r.next<<",\"delta\":"<<r.score<<",\"feasible\":"<<r.feasible<<",\"successors\":"<<r.successors<<"}";}s<<"]}";return s.str();
 }
};
}
