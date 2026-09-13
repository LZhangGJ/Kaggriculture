#include "search.hpp"
#include "observation_codec.hpp"
#include <cstring>
struct Handle {triad::SearchController policy;std::vector<triad::Proposal>prepared;int prepared_day=-1;std::string text;explicit Handle(triad::Settings s):policy(s){}};
extern "C" void*td_new(const double*x,size_t n){try{triad::Settings s;if(n){if(n!=triad::SETTINGS_COUNT)return nullptr;for(size_t i=0;i<n;i++)if(!std::isfinite(x[i]))return nullptr;std::memcpy(&s,x,sizeof(s));}if(s.max_hands<0||s.max_hands>15||s.max_land<1||s.max_land>4||s.max_animals<0||s.max_animals>75||s.labor_hours<1||s.labor_hours>24)return nullptr;return new Handle(s);}catch(...){return nullptr;}}
extern "C" void td_delete(void*p){delete static_cast<Handle*>(p);}
extern "C" const char*td_debug(void*p){auto&h=*static_cast<Handle*>(p);if(h.text.rfind("ERROR",0)!=0)h.text=h.policy.debug();return h.text.c_str();}
extern "C" int td_act(void*p,const dp7::View*v,fastkag::PlayerAction*out){auto&h=*static_cast<Handle*>(p);try{*out=h.policy.act(*v);return 0;}catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what()+" | step="+std::to_string(v->step)+" phase="+std::to_string(h.policy.live.core.phase)+" plans="+std::to_string(h.policy.live.core.plans.size())+" observed_units="+std::to_string(v->priv.inventories.size());return -1;}}
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

// Read-only telemetry; never accepted as input to action selection.
extern "C" const char*td_economy_json(void*p){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy.live;std::ostringstream out;out.precision(17);
 out<<"{\"day\":"<<c.model.day<<",\"predicted\":"<<c.predicted;
 auto flow=[&](const char*name,const auto&f){out<<",\""<<name<<"\":[";for(size_t d=0;d<f.size();d++){if(d)out<<",";out<<"[";for(size_t i=0;i<f[d].size();i++){if(i)out<<",";out<<f[d][i];}out<<"]";}out<<"]";};
 auto arr=[&](const char*name,const auto&a){out<<",\""<<name<<"\":[";for(size_t i=0;i<a.size();i++){if(i)out<<",";out<<a[i];}out<<"]";};
 auto calendar=c.model.calendar(c.portfolio);
 flow("realized_flow",c.model.cash_flow(c.portfolio));flow("warehouse_opening",calendar.opening);flow("same_day_delivery",calendar.same_day);flow("eod_delivery",calendar.overnight);flow("unrealized",calendar.unrealized);
 arr("calendar_workers",calendar.workers);arr("calendar_service_work",calendar.service_work);
 flow("physical_output",c.portfolio.physical_output);flow("net_flow",c.portfolio.f);flow("end_day_shadow",c.prices);flow("demand",c.model.dem);flow("rival",c.model.rival);
 arr("labor",c.portfolio.labor);arr("fixed",c.portfolio.fixed);
 competitive::Curve wages{};for(int d=0;d<30;d++)wages[d]=c.model.wages(c.portfolio.labor[d]);arr("wages",wages);
 out<<",\"config\":{\"competition\":"<<c.model.cfg.competition<<",\"supply\":"<<c.model.cfg.supply<<",\"discount\":"<<c.model.cfg.discount<<",\"action_cost\":"<<c.model.cfg.action_cost<<"},\"paths\":[";
 for(int pos=0;pos<100;pos++){if(pos)out<<",";out<<"{\"pos\":"<<pos<<",\"kind\":"<<c.paths[pos].kind<<",\"end\":"<<c.paths[pos].end;flow("net_flow",c.paths[pos].f);flow("physical_output",c.paths[pos].physical_output);out<<"}";}
 out<<"]}";h.text=out.str();return h.text.c_str();
}

extern "C" void td_probe_enable(void*p,int day){static_cast<Handle*>(p)->policy.probe_day=day;}
extern "C" const char*td_probe_json(void*p){return static_cast<Handle*>(p)->policy.probe_json.c_str();}

// Read-only execution state; no extra policy inputs or hidden state.
extern "C" const char*td_execution_json(void*p){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy.live;std::ostringstream out;
 out<<"{\"day\":"<<c.core.day<<",\"phase\":"<<c.core.phase<<",\"plans\":"<<c.core.plans.size()<<",\"queue\":"<<c.core.queue.size()<<",\"sale_deadlines\":[";
 for(int i=0;i<9;i++){if(i)out<<",";out<<c.local_sale.deadline[i];}out<<"]}";h.text=out.str();return h.text.c_str();
}

// Read-only, deterministic validation seam. It owns a fresh DP object, accepts
// only the supplied synthetic prices, and cannot access or mutate an agent.
// The builder uses this to test the EXACT linked -O3 production solver rather
// than assuming that a separately compiled unit binary proves native behavior.
extern "C" int td_service_dp_snapshot(int kind,int placed,int day,
 const double*prices,std::size_t count,double work,double*out,std::size_t out_count){
 if(!prices||!out||count!=270||out_count!=1440||kind<9||kind>11||
    day<0||day>29||placed< -64||placed>29||!std::isfinite(work))return -1;
 competitive::Flow flow{};
 for(int d=0;d<30;++d)for(int i=0;i<9;++i){
  const double x=prices[d*9+i];if(!std::isfinite(x))return -2;flow[d][i]=x;
 }
 competitive::AnimalServiceDP dp;dp.solve(kind,placed,day,flow,work);
 for(int d=0;d<30;++d)for(int h=0;h<2;++h)for(int b=0;b<6;++b){
  const int n=((d*2+h)*6+b)*4;const auto q=dp.choices[d][h][b];
  out[n]=dp.value[d][h][b];out[n+1]=q.value;out[n+2]=q.feed;out[n+3]=q.care;
 }
 return 0;
}
