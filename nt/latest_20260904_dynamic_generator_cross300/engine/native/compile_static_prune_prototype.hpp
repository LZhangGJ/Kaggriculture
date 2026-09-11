#pragma once
// Isolated prototype, NOT included in production policy.hpp.
// Preserve candidate order and the exact original 1e-6 replacement rule.
namespace dp7::compileprune {
inline void compare(Controller&source,const View&o){
 if(source.p.compile_consequence){compilechoice::compare(source,o);return;}
 if(!source.p.compile_replant_choices||o.day>=29)return;
 source.compile_choice_calls++;
 auto cache=source.p.exact_schedule_cache?std::make_unique<packmemo::Cache>():nullptr;
 packmemo::Scope scope(cache?cache.get():packmemo::active);
 auto candidates=compilechoice::generate(source,o);source.compile_choice_candidates+=int(candidates.size());
 if(candidates.size()<2)return;
 double best=compilechoice::static_value(candidates[0].ctl,o);size_t selected=0;
 std::optional<compilechoice::Value>keep;
 for(size_t i=1;i<candidates.size();i++){
  double score=compilechoice::static_value(candidates[i].ctl,o);
  // A non-improving static candidate cannot replace the current winner,
  // irrespective of its conditional safety result. No future reward is used.
  if(!(score>best+1e-6))continue;
  if(!keep){
   keep=compilechoice::evaluate(candidates[0].ctl,o,false);source.compile_choice_evaluations++;
   if(!keep->known){source.compile_choice_unknown++;return;}
  }
  auto v=compilechoice::evaluate(candidates[i].ctl,o,false);source.compile_choice_evaluations++;
  if(!v.known){source.compile_choice_unknown++;continue;}
  bool safe=true;for(int pos=0;pos<100;pos++)if(v.lost[pos]&&!keep->lost[pos])safe=false;
  if(!safe){source.compile_choice_unsafe++;continue;}
  best=score;selected=i;
 }
 if(selected){
  auto&c=candidates[selected].ctl;source.plans=std::move(c.plans);source.expected=c.expected;
  source.actual_drop=c.actual_drop;source.resource_degraded=c.resource_degraded;source.unresolved_overflow=c.unresolved_overflow;
  source.plant_not_before=c.plant_not_before;source.compile_choice_changes++;
 }
}
}
