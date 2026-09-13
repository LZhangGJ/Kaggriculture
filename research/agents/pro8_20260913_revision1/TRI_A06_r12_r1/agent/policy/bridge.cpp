#include "search.hpp"
#include "observation_codec.hpp"
#include <cstring>
struct Handle {triad::SearchController policy;std::vector<triad::Proposal>prepared;int prepared_day=-1;std::string text;explicit Handle(triad::Settings s):policy(s){}};
extern "C" void*td_new(const double*x,size_t n){try{triad::Settings s;if(n){if(n!=triad::SETTINGS_COUNT)return nullptr;for(size_t i=0;i<n;i++)if(!std::isfinite(x[i]))return nullptr;std::memcpy(&s,x,sizeof(s));}if(s.max_hands<0||s.max_hands>15||s.max_land<1||s.max_land>4||s.max_animals<0||s.max_animals>75||s.labor_hours<1||s.labor_hours>24)return nullptr;return new Handle(s);}catch(...){return nullptr;}}
extern "C" void td_delete(void*p){delete static_cast<Handle*>(p);}
extern "C" const char*td_debug(void*p){auto&h=*static_cast<Handle*>(p);if(h.text.rfind("ERROR",0)!=0)h.text=h.policy.debug();return h.text.c_str();}
extern "C" int td_act(void*p,const dp7::View*v,fastkag::PlayerAction*out){auto&h=*static_cast<Handle*>(p);try{*out=h.policy.act(*v);return 0;}catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return -1;}}
extern "C" int td_observe(void*p,const double*input,size_t count,int32_t*out,size_t cap){try{
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 if(step<0||step>=719||day!=step/24||hour!=step%24||seat<0||seat>1)throw std::runtime_error("clock/seat");
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();if(n>8)throw std::runtime_error("shops");for(int i=0;i<n;i++)shops.push_back(r.integer());if(r.i!=count)throw std::runtime_error("trailing input");
 if(priv.inventories.size()!=(seat==0?a:b).hands.size()+1)throw std::runtime_error("inventory size");
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};fastkag::PlayerAction action;if(td_act(p,&v,&action))return -1;
 size_t size=2+3*(action.units.size()+action.market.size());if(size>cap)throw std::runtime_error("output capacity");
 out[0]=action.units.size();out[1]=action.market.size();int at=2;auto put=[&](const fastkag::Action&a){out[at++]=int(a.op);out[at++]=int(a.item);out[at++]=a.quantity;};for(auto&a:action.units)put(a);for(auto&a:action.market)put(a);return size;
 }catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return -1;}}

// Offline experiment control. No Simulator, seed, opponent ID or hidden private
// state is accepted even by the feature/candidate API.
extern "C" void*td_clone(void*p){try{return new Handle(*static_cast<Handle*>(p));}catch(...){return nullptr;}}
extern "C" int td_prepare(void*p,const dp7::View*v){try{auto&h=*static_cast<Handle*>(p);h.prepared=h.policy.prepare(*v);h.prepared_day=v->day;return h.prepared.size();}catch(const std::exception&e){static_cast<Handle*>(p)->text=std::string("ERROR: ")+e.what();return -1;}}
extern "C" int td_candidate_features(void*p,int index,double*out,size_t cap){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return -1;const auto&f=h.prepared[index].features.x;if(cap<f.size())return -2;std::copy(f.begin(),f.end(),out);return f.size();}
extern "C" int td_candidate_id(void*p,int index){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return -1;return h.prepared[index].id;}
extern "C" int td_install(void*p,int index){auto&h=*static_cast<Handle*>(p);if(index<0||index>=int(h.prepared.size()))return -1;h.policy.install(h.prepared[index],h.prepared_day);h.prepared.clear();return 0;}

extern "C" size_t td_settings_count(){return triad::SETTINGS_COUNT;}
extern "C" const char*td_crop_clock_json(void*p){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy;std::ostringstream out;out.precision(17);
 out<<"{\"step\":"<<c.crop_clock.previous_step<<",\"counts\":[";
 for(int i=0;i<5;i++){if(i)out<<",";out<<c.crop_clock.count[i];}out<<"]";
 auto flow=[&](const char*name,const auto&f){out<<",\""<<name<<"\":[";for(size_t d=0;d<f.size();d++){if(d)out<<",";out<<"[";for(size_t i=0;i<f[d].size();i++){if(i)out<<",";out<<f[d][i];}out<<"]";}out<<"]";};
 flow("weights",c.crop_clock.weights());flow("observed_forecast",c.removal_forecast);flow("live_forecast",c.live.model.rival);out<<"}";h.text=out.str();return h.text.c_str();
}
extern "C" const char*td_clock_json(void*p){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy.sale_clock;auto&l=h.policy.ledger;
 std::ostringstream out;out<<"{\"step\":"<<l.observation_step<<",\"source_step\":"<<l.source_step<<",\"invalid_market\":"<<l.invalid_market;
 auto ints=[&](const char*name,const auto&a){out<<",\""<<name<<"\":[";for(int i=0;i<9;i++){if(i)out<<",";out<<a[i];}out<<"]";};
 ints("own_sales",l.own_sales);ints("own_valid",l.own_valid);ints("rival_sales",l.rival_net);ints("rival_valid",l.valid);
 auto profiles=[&](const char*name,const auto&a){out<<",\""<<name<<"\":[";for(int i=0;i<9;i++){if(i)out<<",";out<<"{\"days\":"<<a[i].count()<<",\"density\":[";auto d=a[i].density();for(int t=0;t<24;t++){if(t)out<<",";out<<d[t];}out<<"]}";}out<<"]";};
 profiles("own",c.own);profiles("rival",c.rival);out<<"}";h.text=out.str();return h.text.c_str();
}

#ifdef A06_PREFIX_AUDIT
// Offline diagnostic only. Does not mutate policy or receive a realized suffix.
extern "C" const char*td_prefix_audit(void*p,const double*input,size_t count,int id,int scope,int ticks,int origin){
 auto&h=*static_cast<Handle*>(p);try{
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
 for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;
 int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 if(r.i!=count||ticks<1||ticks>23-hour||hour!=0)throw std::runtime_error("audit limited to known day, excludes midnight");
 dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
 auto selector=h.policy;selector.act(v);
 const triad::Proposal*chosen=nullptr;for(const auto&q:selector.audit_prepared)if(q.id==id)chosen=&q;
 if(!chosen)throw std::runtime_error("candidate absent");
 auto c=chosen->policy;c.s.a06_roll_scope=scope;c.call_origin=origin?triad::Controller::CallOrigin::PublicPrediction:triad::Controller::CallOrigin::RealExecution;
 fastkag::ObservedDayScenario world(v);std::ostringstream j;j.precision(17);
 j<<"{\"step\":"<<step<<",\"id\":"<<id<<",\"scope\":"<<scope<<",\"origin_prediction\":"<<origin<<",\"assumption\":\"current observation and known rules; rival PASS scenario; excludes midnight and future shops\",\"selection_audit\":"<<selector.last_search<<",\"prefix\":[";
 auto arr=[&](const auto&x){j<<"[";for(size_t i=0;i<x.size();i++){if(i)j<<",";j<<x[i];}j<<"]";};
 auto acts=[&](const auto&x){j<<"[";for(size_t i=0;i<x.size();i++){if(i)j<<",";j<<"["<<int(x[i].op)<<","<<int(x[i].item)<<","<<x[i].quantity<<"]";}j<<"]";};
 for(int t=0;t<ticks&&!world.finished();t++){
  auto o=world.view();if(t)j<<",";j<<"{\"step\":"<<o.step<<",\"cash_before\":"<<o.own.money<<",\"shed\":";arr(o.priv.shed);j<<",\"seeds\":";arr(o.priv.seeds);
  auto action=c.act(o);j<<",\"units\":";acts(action.units);j<<",\"market\":";acts(action.market);j<<",\"debug\":"<<c.debug()<<",\"remaining_plans\":[";
  for(size_t u=0;u<c.core.plans.size();u++){if(u)j<<",";auto&pl=c.core.plans[u];j<<"[";for(size_t k=pl.index;k<pl.a.size();k++){if(k>pl.index)j<<",";auto x=pl.a[k];j<<"["<<int(x.op)<<","<<int(x.item)<<","<<x.quantity<<","<<pl.target[k]<<"]";}j<<"]";}j<<"]";
  world.advance(action);j<<",\"cash_after\":"<<world.own().money<<",\"market_fills\":";arr(world.fills());j<<",\"shed_after\":";arr(world.inventory().shed);j<<",\"seeds_after\":";arr(world.inventory().seeds);j<<"}";
 }
 j<<"],\"endpoint_cash\":"<<world.own().money<<"}";h.text=j.str();return h.text.c_str();
 }catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return h.text.c_str();}
}
#endif

extern "C" const char*td_contract_json(void*p){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy.live;std::ostringstream j;j.precision(17);
 j<<"{\"core_day\":"<<c.core.day<<",\"core_phase\":"<<c.core.phase<<",\"mode\":"<<c.s.a06_reinvest<<",\"book\":[";
 for(int pos=0;pos<100;pos++){if(pos)j<<",";auto&b=c.book[pos];j<<"["<<pos<<","<<b.kind<<","<<b.birth<<","<<b.chosen_day<<","<<b.length<<","<<b.successor<<","<<b.funded<<"]";}
 j<<"],\"queue\":[";bool comma=false;for(const auto&a:c.core.queue){if(comma)j<<",";comma=true;j<<"["<<int(a.op)<<","<<int(a.item)<<","<<a.quantity<<"]";}
 j<<"],\"remaining_plans\":[";for(size_t u=0;u<c.core.plans.size();u++){if(u)j<<",";j<<"[";auto&pl=c.core.plans[u];for(size_t k=pl.index;k<pl.a.size();k++){if(k>pl.index)j<<",";auto&a=pl.a[k];j<<"["<<int(a.op)<<","<<int(a.item)<<","<<a.quantity<<","<<pl.target[k]<<"]";}j<<"]";}
 auto rr=dp7::intraday::reserved(c.core);j<<"],\"reserved_shed\":[";for(size_t i=0;i<rr.shed.size();i++){if(i)j<<",";j<<rr.shed[i];}j<<"],\"reserved_seeds\":[";for(size_t i=0;i<rr.seeds.size();i++){if(i)j<<",";j<<rr.seeds[i];}
 auto&q=c.core.pending_admission;j<<"],\"pending\":["<<q.active<<","<<q.day<<","<<q.unit<<","<<q.pos<<","<<q.kind<<"]}";h.text=j.str();return h.text.c_str();
}

// Offline matched-prefix controls. Neither export is called by the root agent.
// Caller clones the original context before altering a mechanism for a one-step probe.
extern "C" int td_r11_enable(void*p,int enabled){if(!p||(enabled!=0&&enabled!=1))return -1;auto&h=*static_cast<Handle*>(p);h.policy.base.a06_r11_recovery=enabled;h.policy.live.s.a06_r11_recovery=enabled;return 0;}

// Offline short conditional continuation from a matched historical prefix.
// Only a packed legal View is accepted; no historical suffix, opponent private
// state, environment seed, or opponent identifier can be supplied.
extern "C" const char*td_r11_conditional_trace(void*p,const double*input,size_t count,int enabled,int ticks){
 auto&h=*static_cast<Handle*>(p);try{
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
  if(r.i!=count||enabled<0||enabled>1||hour!=step%24||day!=step/24||ticks<1||ticks>24-hour)throw std::runtime_error("r11 audit: current known day only");
  dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  auto choose=h.policy;choose.base.a06_r11_recovery=enabled;choose.live.s.a06_r11_recovery=enabled;
  auto first=choose.act(v);auto controller=choose.live;controller.r11_evaluating=true;
  fastkag::PublicFlowScenario world(v,controller.model.rival,controller.s.supply,nullptr);
  std::ostringstream o;o.precision(17);o<<"{\"kind\":\"same-current-observation conditional trace; not a live game or realized counterfactual\",\"recursive_recovery_search_after_first_decision\":false,\"enabled\":"<<enabled<<",\"decision\":"<<controller.labor11.last<<",\"frames\":[";
  auto arr=[&](const auto&x){o<<"[";for(size_t i=0;i<x.size();i++){if(i)o<<",";o<<x[i];}o<<"]";};
  auto acts=[&](const auto&x){o<<"[";for(size_t i=0;i<x.size();i++){if(i)o<<",";o<<"["<<int(x[i].op)<<","<<int(x[i].item)<<","<<x[i].quantity<<"]";}o<<"]";};
  for(int t=0;t<ticks&&!world.done();t++){
   auto now=world.view();auto action=t?controller.act(now):first;
   if(t)o<<",";o<<"{\"step\":"<<now.step<<",\"cash_before\":"<<now.own.money<<",\"actual_workers_in_scenario\":"<<now.own.hands.size()+1<<",\"actual_hires_in_scenario\":"<<now.own.hires_today<<",\"shed_before\":";arr(now.priv.shed);
   o<<",\"inventories_before\":[";for(size_t u=0;u<now.priv.inventories.size();u++){if(u)o<<",";arr(now.priv.inventories[u]);}o<<"],\"units\":";acts(action.units);o<<",\"market\":";acts(action.market);
   o<<",\"pending_intent\":"<<controller.labor11.active<<",\"confirmed_recovery_workers\":"<<controller.labor11.hire_confirmed<<",\"committed_routes\":"<<controller.labor11.committed<<",\"cancelled_intents\":"<<controller.labor11.cancelled;
   world.advance(action);auto after=world.view();o<<",\"cash_after\":"<<after.own.money<<",\"rival_cash_after\":"<<after.opponent.money<<",\"workers_after\":"<<after.own.hands.size()+1<<",\"shed_after\":";arr(after.priv.shed);o<<"}";
  }
  o<<"],\"endpoint_cash\":"<<world.own_cash()<<",\"endpoint_rival_cash\":"<<world.rival_cash()<<"}";h.text=o.str();return h.text.c_str();
 }catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return h.text.c_str();}
}

extern "C" int td_r12_enable(void*p,int enabled){if(!p||(enabled!=0&&enabled!=1))return -1;auto&h=*static_cast<Handle*>(p);h.policy.base.a06_r12_calendar=enabled;h.policy.live.s.a06_r12_calendar=enabled;h.policy.live.core.p.finite_calendar_admission=enabled&&h.policy.live.s.repeat<=0;return 0;}
