// Diagnostic only: inspect the failed S4W economic fixture without editing it.
#define main original_mechanism_main
#include "test_service_replacement.cpp"
#undef main
int main(){
 Fixture f;f.day=28;f.c.day=28;f.sheep(44);f.own.tiles[44].placed_day=28;f.own.tiles[44].pending_care_bonus=0;f.care(0,44);f.cargo(0,W,1);
 auto r=rp::score(f.c,f.view());
 std::cout<<"selected="<<r.selected<<" evaluated="<<r.evaluated<<" eligible="<<r.eligible<<" feasible="<<r.feasible<<"\n";
 auto show=[](const char*label,const dayvalue::Value&v){std::cout<<label<<" known="<<v.known<<" score="<<v.score<<" trade="<<v.trade<<" wages="<<v.wages<<" funding_gap="<<v.funding_gap<<"\n";};
 show("KEEP",r.keep);for(auto&v:r.values)show("candidate",v);
 for(bool feed:{false,true}){
  ObservedDayScenario sc(f.view());sc.advance(PlayerAction{{action(feed?Op::FEED:Op::CARE)},{}});
  auto&t=sc.own().tiles[44];std::cout<<(feed?"FEED":"CARE")<<" alive="<<animal(t)<<" fert_available="<<t.fertilizer_available<<"\n";
  if(animal(t)){
   View e{sc.step_count(),29,0,sc.own(),f.opponent,sc.inventory(),sc.market(),sc.shops()};
   auto cal=dayvalue::animal_calendar(f.c,e,t);
   std::cout<<"last_day_wool="<<cal.cash.quantity[29][WO]<<" fertilizer="<<cal.cash.quantity[29][F]<<" feed="<<cal.feed[29]<<"\n";
  }
 }
}
