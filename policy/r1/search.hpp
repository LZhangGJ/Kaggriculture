#pragma once
#include "proposals.hpp"
#include "public_flow_scenario.hpp"
#include "learned_value.hpp"
#include "observed_crop_clock.hpp"
#include "startup_supply.hpp"
#include "joint_candidates.hpp"
#ifndef R2_SCENARIO_CARRY_RIVAL
#define R2_SCENARIO_CARRY_RIVAL 0
#endif
#ifndef R2_SCENARIO_FIXED_VALUE_CLOCK
#define R2_SCENARIO_FIXED_VALUE_CLOCK 0
#endif
#ifndef R2_RIVAL_STOCK_AUDIT_ITEM
#define R2_RIVAL_STOCK_AUDIT_ITEM -1
#endif
static_assert(R2_RIVAL_STOCK_AUDIT_ITEM>=-1&&R2_RIVAL_STOCK_AUDIT_ITEM<9);
namespace triad {
// Bounded public-information MPC or offline-calibrated value selector.
// Candidate interventions last only today, then a common base policy resumes.
struct SearchController {
 Settings base;Controller live;
 PublicTradeLedger ledger;SaleClock sale_clock;
 ObservedCropClock crop_clock;competitive::Flow removal_forecast{};
 int begun_step=-1;
 int searches=0,scenarios=0,changes=0,restore_day=-1;double predicted_gain=0;
 std::array<int,32>chosen{};
 int joint_searches=0,joint_rollouts=0,joint_switches=0,joint_rejected=0;
 double joint_predicted_gain=0;
 struct JointResult {int id,removed,feeders,age,next,successors,cancelled;double score;bool feasible;};
	 struct ScoreAudit {
	  double tail=0,frozen_clock_tail=0,clock_consistent_tail=0,clock_consistent_score=0,carried_tail=0,carried_replan_tail=0;
	  double rollout_own_cash=0,rollout_rival_cash=0,rollout_objective=0;
	  double rival_stock_delta=0,rival_stock_score=0;
	  bool replan_changed=false,clock_replan_changed=false;
	  Flow initial{},recomputed{};
	  fastkag::PublicFlowScenario::FlowAudit rival_flow{};
	  std::array<int,9>rival_shed{};
	  competitive::ValueBreakdown tail_parts{},frozen_clock_parts{},carried_parts{},carried_replan_parts{};
	 };
 std::vector<JointResult>last_joint;
	 double event_value(const View&o,Controller roll,const Flow&rival,int horizon,int required,bool&feasible,int&successors){
#if R2_SCENARIO_CARRY_RIVAL
  roll.model.use_rival_forecast=true;roll.model.rival_forecast=rival;
#endif
	  roll.previous_step=o.step-1;int count0=roll.joint.successor_started,cancel0=roll.joint.cancelled;
	#if R2_SCENARIO_FIXED_VALUE_CLOCK
	  roll.model.use_value_basis=true;roll.model.value_basis={o.day,o.own.money};
	#endif
	  fastkag::PublicFlowScenario world(o,rival,base.supply);
	  int stop=std::min(719,o.step+24*horizon);Settings common=base;common.scenario=0;
	  double consistent=o.own.money-base.competition*o.opponent.money;
	  double beta=1+base.discount*std::max(0.,1-o.own.money/20000.);
	  while(!world.done()&&world.view().step<stop){
	   auto v=world.view();if(v.day!=o.day){roll.configure(common);roll.model.extra_rival={};}
	   double own0=world.own_cash(),rival0=world.rival_cash();auto action=roll.act(v);world.advance(action);
	   consistent+=((world.own_cash()-own0)-base.competition*(world.rival_cash()-rival0))/std::pow(beta,v.day-o.day);
	  }
  roll.joint.observe(world.view());successors=roll.joint.successor_started-count0;
  feasible=required==0||(successors>=required&&roll.joint.cancelled==cancel0);
  double score=world.own_cash()-base.competition*world.rival_cash();
  if(!world.done()){
	   auto v=world.view();Controller tail=roll;tail.configure(common);
	#if R2_SCENARIO_FIXED_VALUE_CLOCK
	   tail.model.use_value_basis=true;tail.model.value_basis={o.day,o.own.money};
	#endif
	   tail.plan(v);score+=tail.predicted;consistent+=tail.predicted;
	  }
	  joint_rollouts++;
	#if R2_SCENARIO_FIXED_VALUE_CLOCK
	  return consistent;
	#else
	  return score;
	#endif
 }
 explicit SearchController(Settings s={}):base(s),live(s){if(s.scenario<0&&!learned::available)throw std::invalid_argument("No trained value model installed");}
 std::vector<Proposal> prepare(const View&o,bool names=false)const{
  auto current=live;
#if R2_SALE_CLOCK_MODE >= 2
  current.model.rival_early_share=sale_clock.early_shares();
#endif
  return generate_proposals(current,base,o,names);
 }
 void install(const Proposal&p,int day){live=p.policy;restore_day=day+1;}
 double score(const View&o,const Proposal&proposal,int requested_horizon=0,ScoreAudit*audit=nullptr)const{
  if(base.scenario<0)return !learned::available?0:learned::score(proposal.features.x,proposal.id==0);
  auto roll=proposal.policy;
#if R2_SCENARIO_CARRY_RIVAL
  roll.model.use_rival_forecast=true;roll.model.rival_forecast=proposal.policy.model.rival;
#endif
	  roll.previous_step=o.step-1;
	  fastkag::PublicFlowScenario world(o,proposal.policy.model.rival,base.supply,
#if R2_SALE_CLOCK_MODE >= 1
    &sale_clock
#else
    nullptr
#endif
	  );
	  int horizon=requested_horizon?std::clamp(requested_horizon,1,std::max(1,30-o.day)):std::clamp(int(base.scenario),1,5),stop=std::min(719,o.step+24*horizon);Settings common=base;common.scenario=0;
	  double consistent=o.own.money-base.competition*o.opponent.money;
	  double beta=1+base.discount*std::max(0.,1-o.own.money/20000.);
	#if R2_SCENARIO_FIXED_VALUE_CLOCK
	  roll.model.use_value_basis=true;roll.model.value_basis={o.day,o.own.money};
	#endif
	  while(!world.done()&&world.view().step<stop){auto v=world.view();if(v.day!=o.day)roll.configure(common);double own0=world.own_cash(),rival0=world.rival_cash();auto act=roll.act(v);world.advance(act);consistent+=((world.own_cash()-own0)-base.competition*(world.rival_cash()-rival0))/std::pow(beta,v.day-o.day);}
	  double result=world.own_cash()-base.competition*world.rival_cash();
	  if(audit){audit->rollout_own_cash=world.own_cash();audit->rollout_rival_cash=world.rival_cash();audit->rollout_objective=result;audit->rival_flow=world.audit();audit->rival_shed=world.rival_shed();}
  if(!world.done()){
	   auto v=world.view();Controller tail=live;tail.configure(common);tail.book=roll.book;tail.joint=roll.joint;
#if R2_SALE_CLOCK_MODE >= 2
   tail.model.rival_early_share=proposal.policy.model.rival_early_share;
#endif
#if R2_SCENARIO_CARRY_RIVAL
   tail.model.use_rival_forecast=true;tail.model.rival_forecast=proposal.policy.model.rival;
#endif
   tail.plan(v);
   if(audit){
	    audit->tail=tail.model.value(v,tail.portfolio,nullptr,-1,&audit->tail_parts);audit->initial=proposal.policy.model.rival;audit->recomputed=tail.model.rival;
	    competitive::ValueBasis basis{o.day,o.own.money};
	    audit->frozen_clock_tail=tail.model.value(v,tail.portfolio,nullptr,-1,&audit->frozen_clock_parts,&basis);
	   }
	   Controller clocked;
	   if(audit||R2_SCENARIO_FIXED_VALUE_CLOCK){
	    clocked=live;clocked.configure(common);clocked.book=roll.book;clocked.joint=roll.joint;
#if R2_SALE_CLOCK_MODE >= 2
    clocked.model.rival_early_share=proposal.policy.model.rival_early_share;
#endif
	    clocked.model.use_value_basis=true;clocked.model.value_basis={o.day,o.own.money};
#if R2_SCENARIO_CARRY_RIVAL
	    clocked.model.use_rival_forecast=true;clocked.model.rival_forecast=proposal.policy.model.rival;
#endif
	    clocked.plan(v);consistent+=clocked.predicted;
#if R2_RIVAL_STOCK_AUDIT_ITEM >= 0
    if(audit){audit->rival_stock_delta=clocked.model.rival_stock_response_delta(
      v,clocked.portfolio,R2_RIVAL_STOCK_AUDIT_ITEM,ledger.upper[R2_RIVAL_STOCK_AUDIT_ITEM]);
     audit->rival_stock_score=consistent+audit->rival_stock_delta;}
#endif
	   }
	   if(audit){
	    audit->clock_consistent_tail=clocked.predicted;audit->clock_consistent_score=consistent;
	    audit->clock_replan_changed=proposal_key(clocked)!=proposal_key(tail);
	    auto carried=tail.model;carried.rival=proposal.policy.model.rival;
	    audit->carried_tail=carried.value(v,tail.portfolio,nullptr,-1,&audit->carried_parts);
	    Controller replanned=live;replanned.configure(common);replanned.book=roll.book;replanned.joint=roll.joint;
#if R2_SALE_CLOCK_MODE >= 2
    replanned.model.rival_early_share=proposal.policy.model.rival_early_share;
#endif
    replanned.model.use_rival_forecast=true;replanned.model.rival_forecast=proposal.policy.model.rival;
	    replanned.plan(v);audit->carried_replan_tail=replanned.model.value(v,replanned.portfolio,nullptr,-1,&audit->carried_replan_parts);
    audit->replan_changed=proposal_key(replanned)!=proposal_key(tail);
   }
	   result+=tail.predicted;
	  }
	#if R2_SCENARIO_FIXED_VALUE_CLOCK
	  return consistent;
	#else
	  return result;
	#endif
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
 void begin_observation(const View&o){
  if(begun_step==o.step)return;
  if(begun_step>o.step)throw std::runtime_error("observation moved backwards");
  if constexpr(PublicTradeLedger::enabled)if(ledger.previous.step>=0&&o.step!=ledger.previous.step+1)
   throw std::runtime_error("nonconsecutive public trade ledger");
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
  begun_step=o.step;
 }
 PlayerAction act(const View&o){
  dp7::packmemo::Cache call_cache;dp7::packmemo::Scope cache_scope(&call_cache);
  begin_observation(o);
  if(base.scenario!=0&&o.day!=live.core.day)choose(o);
  // Build the sale plan HERE, on `live`, after choose() has committed a portfolio.
  //
  // It cannot live in Controller::plan: choose()/install() replace `live` wholesale with a proposal
  // copy whose core.day is already today's, so live.act() then skips its own plan() entirely -- the
  // live controller has never built a sale plan, and the executor was silently following the WINNING
  // PROPOSAL's plan instead of the committed portfolio's. Gating the call on a flag instead (to stop
  // the ~305 redundant builds a game, one per evaluated proposal) disabled the DP altogether.
  // Building it once here is both correct and cheap.
  // Compare DAYS, not steps. live.model is replaced by choose() every day, so the step marker is
  // stale afterwards; keying on the step rebuilt the plan on every one of the day's 24 steps
  // (measured 431 builds a game).
  if(live.model.cfg.sale_dp>0&&live.model.exec_plan_day!=o.day){
   Planner fresh;Flow px;live.model.build_exec_plan(live.remaining_portfolio(o,fresh,px),o);
  }
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
