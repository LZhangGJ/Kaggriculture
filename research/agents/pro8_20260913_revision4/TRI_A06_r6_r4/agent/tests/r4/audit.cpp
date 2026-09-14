// Offline inspection of own-visible production state. Not compiled into agent.
#include "../r3/r3_checks.cpp"
extern "C" const char*td_r4_snapshot(void*ptr,const double*in,size_t n){auto&h=*static_cast<Handle*>(ptr);h.text=with_view(ptr,in,n,[](Handle&h,const dp7::View&o){
 auto c=h.policy.live.core;
 std::ostringstream j;j<<"{\"step\":"<<o.step<<",\"phase\":"<<c.phase<<",\"day\":"<<c.day<<",\"auto\":"<<(c.plans.size()==o.priv.inventories.size()?c.expected_auto_deposit(o):-1)<<",\"mid_checks\":"<<c.midroute_delivery_checks<<",\"mid_insertions\":"<<c.midroute_delivery_insertions<<",\"mid_qty\":"<<c.midroute_delivery_quantity<<",\"shadow\":"<<c.p.action_shadow<<",\"pressure\":"<<c.p.triad_delivery_pressure<<",\"plans\":[";
 for(size_t u=0;u<c.plans.size();u++){if(u)j<<",";const auto&p=c.plans[u];int pos=dp7::cell(u?o.own.hands.at(u-1):o.own.farmer);j<<"{\"unit\":"<<u<<",\"pos\":"<<pos<<",\"bag\":[";for(int i=0;i<12;i++){if(i)j<<",";j<<o.priv.inventories.at(u)[i];}j<<"],\"remaining\":[";for(size_t k=p.index;k<p.a.size();k++){if(k>p.index)j<<",";auto a=p.a[k];j<<"["<<int(a.op)<<","<<int(a.item)<<","<<a.quantity<<","<<p.target[k]<<"]";}j<<"]}";}
 j<<"]}";return j.str();});return h.text.c_str();}
#ifdef R4_TESTING
extern "C" void td_r4_enable(int on){dp7::delivery::testing_enabled=on!=0;}
#endif
extern "C" const char*td_r4_branch(void*ptr,const double*in,size_t n,int mode){auto&h=*static_cast<Handle*>(ptr);h.text=with_view(ptr,in,n,[&](Handle&h,const dp7::View&v){
#ifdef R4_TESTING
 bool previous=dp7::delivery::testing_enabled;struct Restore{bool old;~Restore(){dp7::delivery::testing_enabled=old;}} restore{previous};dp7::delivery::testing_enabled=mode!=0;
#endif
 auto c=h.policy;fastkag::ObservedDayScenario w(v);std::ostringstream j;j.precision(17);int ov=0;
 j<<"{\"mode\":"<<mode<<",\"start\":"<<v.step<<",\"initial_cash\":"<<v.own.money<<",\"ticks\":[";bool comma=false;
 while(!w.finished()){
  auto o=w.view();auto a=c.act(o);auto post=w.project_units(a.units,-1).project_own_market(0,a.market);auto sig=own_signature(post);
  if(comma)j<<",";comma=true;j<<"{\"step\":"<<o.step<<",\"action\":";action_json(j,a);j<<",\"prefix_signature\":[";for(size_t i=0;i<sig.size();i++){if(i)j<<",";j<<sig[i];}j<<"]";
  w.advance(a);ov+=w.overflow();j<<",\"cash\":"<<w.own().money<<",\"overflow\":"<<w.overflow()<<",\"fills\":[";for(size_t i=0;i<w.fills().size();i++){if(i)j<<",";j<<w.fills()[i];}j<<"]}";
 }
 auto&k=c.live.core;j<<"],\"final_cash\":"<<w.own().money<<",\"quote_not_cash\":"<<w.liquidation_quote()<<",\"overflow\":"<<ov<<",\"terminal_checks\":"<<k.delivery_terminal_checks<<",\"terminal_added\":"<<k.delivery_terminal_added<<",\"wave_checks\":"<<k.delivery_wave_checks<<",\"wave_added\":"<<k.delivery_wave_added<<",\"handoffs\":"<<k.delivery_handoffs<<",\"added_steps\":"<<k.delivery_added_steps<<",\"evaluations\":"<<k.delivery_evaluations<<",\"conditional_gain_not_cash\":"<<k.delivery_conditional_gain<<",\"final_stock\":[";for(int i=0;i<12;i++){if(i)j<<",";j<<w.inventory().shed[i];}j<<"]}";return j.str();});return h.text.c_str();}
// Explicit artificial terminal fixture: original paid route ends in DROP,
// while a visible ripe crop next door was not included. Tests actual full
// Controller::act and settle_market, not a hand-written substitute sale queue.
extern "C" const char*td_r4_terminal_branch(void*ptr,const double*in,size_t n,int mode){auto&h=*static_cast<Handle*>(ptr);h.text=with_view(ptr,in,n,[&](Handle&h,const dp7::View&v){
#ifdef R4_TESTING
 bool previous=dp7::delivery::testing_enabled;struct Restore{bool old;~Restore(){dp7::delivery::testing_enabled=old;}} restore{previous};dp7::delivery::testing_enabled=mode!=0;
#endif
 need(v.day==29&&v.priv.inventories.size()==1,"terminal fixture shape");
 auto c=h.policy.live;c.core=dp7::Controller{};c.core.phase=3;c.core.day=v.day;c.core.last_step=v.step-1;c.previous_step=v.step-1;
 c.core.p.stepwise_recoordination=true;c.core.p.fix_logistics=true;c.core.p.sell_deposits=true;c.core.p.terminal_deposit_schedule=true;c.core.p.max_hands=0;
 c.core.target={{43,dp7::T}};c.core.plans.resize(1);c.core.plans[0]=dp7::delivery::rebuild(v,0,{{dp7::action(fastkag::Op::DROP),44}});
 fastkag::ObservedDayScenario w(v);std::ostringstream j;j.precision(17);int ov=0;j<<"{\"mode\":"<<mode<<",\"start\":"<<v.step<<",\"initial_cash\":"<<v.own.money<<",\"ticks\":[";bool comma=false;
 while(!w.finished()){auto o=w.view();auto a=c.act(o);auto post=w.project_units(a.units,-1).project_own_market(0,a.market);auto sig=own_signature(post);if(comma)j<<",";comma=true;j<<"{\"step\":"<<o.step<<",\"action\":";action_json(j,a);j<<",\"prefix_signature\":[";for(size_t i=0;i<sig.size();i++){if(i)j<<",";j<<sig[i];}j<<"]";w.advance(a);ov+=w.overflow();j<<",\"cash\":"<<w.own().money<<",\"overflow\":"<<w.overflow()<<"}";}
 j<<"],\"final_cash\":"<<w.own().money<<",\"overflow\":"<<ov<<",\"terminal_checks\":"<<c.core.delivery_terminal_checks<<",\"terminal_added\":"<<c.core.delivery_terminal_added<<",\"remaining_bag_tomato\":"<<w.inventory().inventories[0][dp7::T]<<"}";return j.str();});return h.text.c_str();}
