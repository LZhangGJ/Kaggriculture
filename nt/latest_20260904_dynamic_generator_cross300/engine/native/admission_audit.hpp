#pragma once
// Read-only, offline diagnostics. Never enters a candidate's inputs or scoring.
#include "policy.hpp"
namespace dp7audit {
struct Admission {
 bool captured=false,all_ready_jobs_fit=false;
 int day=0,hour=0,planned_jobs=0,ready_jobs=0,resource_dropped_jobs=0;
 int seed_shortage=0,feed_shortage=0,fertilizer_shortage=0,animal_shortage=0;
 int hands=0,min_hands=-1,unplaced_animals_in_shed=0;
 double cash=0,conditional_wage_saving=0;
};
inline Admission admission(const dp7::Controller&original,const dp7::View&v){
 Admission r;r.captured=true;r.day=v.day;r.hour=v.hour;r.cash=v.own.money;r.hands=int(v.own.hands.size());
 auto probe=original;auto all=probe.jobs(v);r.planned_jobs=int(all.size());
 dp7::Counts need{},seed{};for(auto&j:all){dp7::add(need,j.needs);dp7::add(seed,j.seeds);}
 for(int i=0;i<5;i++)r.seed_shortage+=std::max(0,seed[i]-v.priv.seeds[i]);
 r.feed_shortage=std::max(0,need[dp7::W]-v.priv.shed[dp7::W]);
 r.fertilizer_shortage=std::max(0,need[dp7::F]-v.priv.shed[dp7::F]);
 for(int i=9;i<12;i++){r.animal_shortage+=std::max(0,need[i]-v.priv.shed[i]);r.unplaced_animals_in_shed+=v.priv.shed[i];}
 auto ready=probe.reserve(v,all);r.ready_jobs=int(ready.size());r.resource_dropped_jobs=r.planned_jobs-r.ready_jobs;
 bool ret=v.day>=29||probe.hauling_needed(ready);int budget=(ret?22:23)-v.hour+1;
 // Current controller compiles only at the depot after day-start purchases.
 // Removing a suffix of hires preserves the official spawn order of survivors.
 for(int h=0;h<=r.hands;h++){
  auto starts=probe.starts(h);if(dp7::cell(v.own.farmer)!=starts[0])break;
  bool same=true;for(int u=1;u<=h;u++)same&=dp7::cell(v.own.hands[u-1])==starts[u];if(!same)break;
  auto packed=probe.pack(ready,starts,budget,ret);
  if(packed.second)packed=probe.pack(ready,starts,budget,ret,true);
  if(packed.second==0){r.all_ready_jobs_fit=true;r.min_hands=h;break;}
 }
 if(r.min_hands>=0)r.conditional_wage_saving=dp7::Controller::hirecost(r.hands)-dp7::Controller::hirecost(r.min_hands);
 // A feasible reschedule is NOT an achieved saving: sale timing, resource
 // delivery and new investments can change. Counterfactual testing is needed.
 return r;
}
}
