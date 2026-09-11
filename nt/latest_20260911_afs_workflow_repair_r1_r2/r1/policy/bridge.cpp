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

extern "C" const char*td_repair_debug(void*p){
 auto&h=*static_cast<Handle*>(p);const auto&s=h.policy.live.core.t3;
 std::ostringstream o;o<<"{\"obligation_checks\":"<<s.obligation_checks
 <<",\"water_insertions\":"<<s.water_insertions<<",\"water_unresolved_checks\":"<<s.water_unresolved
 <<",\"capacity_checks\":"<<s.capacity_checks<<",\"capacity_rollouts\":"<<s.capacity_rollouts
 <<",\"delivery_insertions\":"<<s.delivery_insertions<<",\"projected_overflow_saved\":"<<s.projected_overflow_saved
 <<",\"receipt_checks\":"<<s.receipt_checks<<",\"receipt_hires\":"<<s.receipt_hires<<",\"receipt_seeds\":"<<s.receipt_seeds
 <<",\"transport_admissions\":"<<s.transport_admissions<<"}";h.text=o.str();return h.text.c_str();}

// Read-only diagnosis for the CURRENT legal observation. No hidden state input.
extern "C" const char* td_obligations_json(void*p,const double*input,size_t count){
 auto&h=*static_cast<Handle*>(p);
 try {
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
  for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
  if(r.i!=count||seat<0||seat>1)throw std::runtime_error("inspect observation");
  dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  const auto&live=h.policy.live;const auto&c=live.core;
  std::ostringstream s;s<<"{\"step\":"<<step<<",\"phase\":"<<c.phase<<",\"feed_shortfall\":"<<c.feed_shortfall(v)<<",\"plan_dropped\":"<<c.actual_drop<<",\"resource_degraded\":"<<c.resource_degraded<<",\"input_rejected_atoms\":"<<c.input_rejected_atoms<<",\"input_rejected_feed\":"<<c.input_rejected_feed<<",\"service\":[";
  bool comma=false;for(int pos=0;pos<100;pos++)if(dp7::animal(v.own.tiles[pos])){if(comma)s<<",";comma=true;s<<"["<<pos<<","<<int(c.service_feed[pos])<<","<<int(c.service_care[pos])<<","<<int(v.own.tiles[pos].consecutive_unfed)<<"]";}s<<"],\"queued\":[";
  comma=false;for(auto x:c.queue){if(comma)s<<",";comma=true;s<<"["<<int(x.op)<<","<<int(x.item)<<","<<x.quantity<<"]";}s<<"],\"raw_jobs\":[";
  comma=false;for(const auto&j:c.jobs(v)){if(comma)s<<",";comma=true;s<<"{\"pos\":"<<j.pos<<",\"actions\":[";bool sep=false;for(auto x:j.actions){if(sep)s<<",";sep=true;s<<int(x.op);}s<<"],\"wheat\":"<<j.needs[0]<<"}";}s<<"],\"remaining\":[";
  comma=false;for(const auto&pl:c.plans){if(comma)s<<",";comma=true;s<<"[";bool sep=false;for(size_t k=pl.index;k<pl.a.size();k++){if(sep)s<<",";sep=true;s<<"["<<int(pl.a[k].op)<<","<<int(pl.a[k].item)<<","<<pl.a[k].quantity<<","<<pl.target[k]<<"]";}s<<"]";}s<<"]}";
  h.text=s.str();return h.text.c_str();
 }catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return h.text.c_str();}
}
