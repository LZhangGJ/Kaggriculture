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
 std::string terminal_staff_audit="null";
 int searches=0,scenarios=0,changes=0,restore_day=-1;double predicted_gain=0;
 std::array<int,32>chosen{};
 long long prediction_gate_checks=0,prediction_gate_activations=0,prediction_receipt_events=0;
 std::string last_search="null";
#ifdef A06_PREFIX_AUDIT
 std::vector<Proposal> audit_prepared;
#endif
 explicit SearchController(Settings s={}):base(s),live(s){if(s.scenario<0&&!learned::available)throw std::invalid_argument("No trained value model installed");}
 std::vector<Proposal> prepare(const View&o,bool names=false)const{
  auto original=generate_proposals(live,base,o,names);
  if(base.a06_execution_candidates<=0)return original;
  // Two finite actions in the economic decision, NOT prediction-only gates.
  // Same prepared targets, orders, paid inventory, work/service commitments.
  // Each rollout and its installed real executor use the SAME day mechanism.
  // Horizon, original plan generation and score are unchanged. No WAIT filter.
  std::vector<Proposal> out;out.reserve(original.size()*2);
  for(const auto&p:original){
   auto q=p;q.policy.s.a06_roll_scope=0;q.policy.s.a06_reinvest=4;out.push_back(std::move(q));
   q=p;q.id+=16;q.policy.s.a06_roll_scope=0;q.policy.s.a06_reinvest=1;out.push_back(std::move(q));
  }
  return out;
 }
 void install(const Proposal&p,int day){live=p.policy;restore_day=day+1;}
 void choose(const View&o){
  searches++;int horizon=std::clamp(int(base.scenario),1,5);auto proposals=prepare(o);
  #ifdef A06_PREFIX_AUDIT
  audit_prepared=proposals;
#endif
  int winner=0;double bestscore=-1e100,baseline=0;
  std::ostringstream audit;audit.precision(17);
  audit<<"{\"step\":"<<o.step<<",\"day\":"<<o.day<<",\"real_receipts_before_choose\":"<<live.a06_received_today<<",\"real_receipt_day\":"<<live.a06_receipt_day<<",\"candidates\":[";
  for(int i=0;i<int(proposals.size());i++){
   auto&proposal=proposals[i];double score=0,own_cash=0,rival_cash=0,tail_value=0;
   long long checks=0,activations=0,receipts=0;int first_invest_step=-1;Acts first_invest_orders;
   if(base.scenario<0){
    score=!learned::available?0:learned::score(proposal.features.x,proposal.id==0);
   }else{
    auto roll=proposal.policy;roll.previous_step=o.step-1;
    roll.call_origin=Controller::CallOrigin::PublicPrediction;
    auto before_checks=roll.a06_gate_checks,before_activations=roll.a06_gate_activations,before_receipts=roll.a06_receipt_events;
    fastkag::PublicFlowScenario world(o,proposal.policy.model.rival,base.supply,
#if R2_SALE_CLOCK_MODE >= 1
      &sale_clock
#else
      nullptr
#endif
    );
    int stop=std::min(719,o.step+24*horizon);Settings common=base;common.scenario=0;
    while(!world.done()&&world.view().step<stop){auto v=world.view();if(v.day!=o.day)roll.configure(common);auto act=roll.act(v);
     if(first_invest_step<0&&std::any_of(act.market.begin(),act.market.end(),[](const auto&a){return a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL||a.op==Op::BUY_LAND;})){first_invest_step=v.step;first_invest_orders=act.market;}
     world.advance(act);}
    own_cash=world.own_cash();rival_cash=world.rival_cash();score=own_cash-base.competition*rival_cash;
    checks=roll.a06_gate_checks-before_checks;activations=roll.a06_gate_activations-before_activations;receipts=roll.a06_receipt_events-before_receipts;
    prediction_gate_checks+=checks;prediction_gate_activations+=activations;prediction_receipt_events+=receipts;
    if(!world.done()){auto v=world.view();Controller tail=live;tail.call_origin=Controller::CallOrigin::PublicPrediction;tail.configure(common);tail.book=roll.book;tail.plan(v);tail_value=tail.predicted;score+=tail_value;}
   }
   if(i)audit<<",";
   audit<<"{\"id\":"<<proposal.id<<",\"execution_mode\":"<<proposal.policy.s.a06_reinvest<<",\"execution_scope\":"<<proposal.policy.s.a06_roll_scope<<",\"score\":"<<score<<",\"roll_own_cash\":"<<own_cash<<",\"roll_rival_cash\":"<<rival_cash<<",\"tail_value\":"<<tail_value<<",\"prediction_gate_checks\":"<<checks<<",\"prediction_gate_activations\":"<<activations<<",\"prediction_receipt_events\":"<<receipts<<",\"plan_key\":[";
   auto key=proposal_key(proposal.policy);for(size_t j=0;j<key.size();j++){if(j)audit<<",";audit<<key[j];}audit<<"],\"first_prediction_investment_step\":"<<first_invest_step<<",\"first_prediction_market\":[";
   for(size_t k=0;k<first_invest_orders.size();k++){if(k)audit<<",";auto&a=first_invest_orders[k];audit<<"["<<int(a.op)<<","<<int(a.item)<<","<<a.quantity<<"]";}audit<<"]}";
   scenarios++;if(i==0)baseline=score;
   if(score>bestscore+1e-6){bestscore=score;winner=i;}
  }
  int id=proposals[winner].id;audit<<"],\"selected_id\":"<<id<<",\"score_gain_vs_id0\":"<<bestscore-baseline<<"}";last_search=audit.str();changes+=id!=0;chosen[id]++;predicted_gain+=bestscore-baseline;install(proposals[winner],o.day);
 }
#include "r14_staff.inc"
 void terminal_staff(const View&o){
  if(base.r13_terminal_staff<=0||o.day!=29||o.hour!=0)return;
  if(live.core.day!=o.day)live.plan(o);
  int original=0;for(auto a:live.core.queue)original+=a.op==Op::HIRE;
  if(original<2)return;
  int winner=original;double best=-1e100,original_score=-1e100;
  std::ostringstream audit;audit.precision(17);audit<<"{\"step\":"<<o.step<<",\"original\":"<<original<<",\"candidates\":[";
  auto install_count=[&](Controller&c,int h){
   c.core.p.max_hands=h;
   Acts q;for(auto a:c.core.queue)if(a.op!=Op::HIRE)q.push_back(a);
   for(int u=0;u<h;u++)q.push_back(dp7::action(Op::HIRE));
   c.core.queue=std::move(q);
  };
  // Both worlds begin at this public observation. No future random draw occurs
  // before the true terminal and rival private inventory is never supplied.
  for(int h=original;h>=std::max(0,original-8);--h){
   double score=1e100;std::array<double,2>cash{};
   for(int z=0;z<2;z++){
    auto c=live;install_count(c,h);c.previous_step=o.step-1;c.r11_evaluating=true;
    c.call_origin=Controller::CallOrigin::PublicPrediction;
    fastkag::PublicFlowScenario world(o,c.model.rival,z?base.supply:0.);
    while(!world.done()){auto v=world.view();world.advance(c.act(v));}
    cash[z]=world.own_cash();score=std::min(score,cash[z]-base.competition*world.rival_cash());
   }
   if(h==original)original_score=score;
   if(h<original)audit<<",";
   audit<<"["<<h<<","<<score<<","<<cash[0]<<","<<cash[1]<<"]";
   if(score>best+1e-6||(std::abs(score-best)<=1e-6&&h<winner)){best=score;winner=h;}
  }
  install_count(live,winner);
  audit<<"],\"selected\":"<<winner<<",\"conditional_gain\":"<<best-original_score<<"}";
  terminal_staff_audit=audit.str();
 }
 PlayerAction act(const View&o){
  competitive::MarketModeScope market_mode(int(base.r13_market_integral));
  if(o.hour==0&&o.day>=base.r13_late_day&&base.r13_late_competition>=0){
   base.competition=base.r13_late_competition;
   live.configure(base);
  }
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
  if(base.scenario!=0&&o.day!=live.core.day&&o.day<base.r13_search_until)choose(o);
  if(base.r14_staff>0)daily_staff(o);
  terminal_staff(o);
  auto out=live.act(o);
#if R2_CROP_CLOCK_MODE >= 1
  if(o.hour==0){auto model=live.model;model.rival_harvest_weights=crop_clock.weights();model.public_rival(o);removal_forecast=model.rival;}
#endif
  if constexpr(PublicTradeLedger::enabled)ledger.record(o,out);
  return out;
 }
 std::string debug()const{
  auto text=live.debug();text.pop_back();std::ostringstream s;s<<text<<",\"r14_staff_calls\":"<<staff_calls14<<",\"r14_staff_changes\":"<<staff_changes14<<",\"r14_staff_removed\":"<<staff_removed14<<",\"r14_staff_last\":"<<staff_last14<<",\"terminal_staff_audit\":"<<terminal_staff_audit<<",\"search_calls\":"<<searches<<",\"search_scenarios\":"<<scenarios<<",\"search_changes\":"<<changes<<",\"search_predicted_gain\":"<<predicted_gain<<",\"learned_value_active\":"<<(base.scenario<0?"true":"false")<<",\"learned_value_available\":"<<(learned::available?"true":"false")<<",\"search_choices\":[";
  for(int i=0;i<32;i++){if(i)s<<",";s<<chosen[i];}s<<"],\"prediction_gate_checks\":"<<prediction_gate_checks<<",\"prediction_gate_activations\":"<<prediction_gate_activations<<",\"prediction_receipt_events\":"<<prediction_receipt_events<<",\"execution_candidate_families\":"<<base.a06_execution_candidates<<",\"last_search\":"<<last_search<<"}";return s.str();
 }
};
}
