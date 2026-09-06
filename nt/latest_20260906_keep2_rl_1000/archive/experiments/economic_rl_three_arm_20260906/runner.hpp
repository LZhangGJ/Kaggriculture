#pragma once
#include "rl_common.hpp"
#include <dlfcn.h>
#include <pybind11/numpy.h>
#include <memory>
namespace econrun {
using namespace fastkag;
struct Library {
 void*lib=nullptr;econrl::Api api;
 explicit Library(const std::string&p){lib=dlopen(p.c_str(),RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
  auto entry=reinterpret_cast<econrl::Api(*)()>(dlsym(lib,"economic_policy_api"));if(!entry)throw std::runtime_error("missing policy API");api=entry();}
 ~Library(){if(lib)dlclose(lib);}
};
struct Game {
 uint64_t seed;int seat,opponent,steps=0;double cash=0,other=0,seconds=0,policy_seconds=0,enemy_seconds=0,env_seconds=0,model_seconds=0,max_action_ms=0;
 int64_t plans=0,references=0,executes=0;std::string error;
 std::vector<econrl::Decision>decisions;std::vector<std::array<PlayerAction,2>>trace;
};
struct Pool {
 std::unique_ptr<bridge::G001>g001,g003;std::unique_ptr<boatlee29::Agent>boatlee;
 std::unique_ptr<kaito58::Agent>kaito;std::unique_ptr<lynn5::Agent>lynn;
 explicit Pool(py::dict d){
  g001=std::make_unique<bridge::G001>(bridge::load_g001(d["g001"]));g003=std::make_unique<bridge::G001>(bridge::load_g001(d["g003"]));
  boatlee=std::make_unique<boatlee29::Agent>(bridge::load_boatlee(d["boatlee_v29"]));
  kaito=std::make_unique<kaito58::Agent>(bridge::load_kaito(d["kaito_v58"]));
  lynn=std::make_unique<lynn5::Agent>(bridge::load_lynn(d["lynn_v5"]));
 }
 Game play(const Library&lib,const std::string&weights,int mode,uint64_t seed,int seat,int opponent,uint64_t sample_seed,bool trace)const{
  Game r;r.seed=seed;r.seat=seat;r.opponent=opponent;auto tic=std::chrono::steady_clock::now();
  auto seconds=[](auto t){return std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();};
  void*policy=nullptr;
  try{
   policy=lib.api.create(weights.c_str(),sample_seed,mode);Simulator env(Config{},seed);
   bridge::G001State gs;boatlee29::State bs;kaito58::State ks;lynn5::State ls;fieldbook::Controller fs;threeday::Controller ts;
   while(!env.done()){
    econrl::Input input{env.step_count(),env.day(),env.hour(),seat,&env.farms()[seat],&env.farms()[1-seat],&env.privates()[seat],&env.market(),&env.shops()};
    std::array<PlayerAction,2>a;auto t=std::chrono::steady_clock::now();a[seat]=lib.api.act(policy,input);double dur=seconds(t);r.policy_seconds+=dur;r.max_action_ms=std::max(r.max_action_ms,dur*1000);
    t=std::chrono::steady_clock::now();auto v=bridge::view(env,1-seat);
    switch(opponent){
     case 0:a[1-seat]=g001->act(env,1-seat,gs);break;case 1:a[1-seat]=g003->act(env,1-seat,gs);break;
     case 2:a[1-seat]=boatlee->act(v,1-seat,bs);break;case 3:a[1-seat]=kaito->act(v,1-seat,ks);break;
     case 4:a[1-seat]=lynn->act(v,1-seat,ls);break;case 5:a[1-seat]=fs.act(v,1-seat);break;case 6:a[1-seat]=ts.act(v,1-seat);break;
     default:throw std::runtime_error("opponent index");}
    r.enemy_seconds+=seconds(t);if(trace)r.trace.push_back(a);t=std::chrono::steady_clock::now();env.step(a);r.env_seconds+=seconds(t);r.steps++;
   }
   if(r.steps!=719)throw std::runtime_error("full game required");r.cash=env.farms()[seat].money;r.other=env.farms()[1-seat].money;
   auto&s=*lib.api.session(policy);r.decisions=s.records;r.model_seconds=s.model_seconds;r.plans=s.plan_calls;r.references=s.reference_calls;r.executes=s.execute_calls;
   if(r.decisions.size()!=30||r.plans!=30||r.executes!=719)throw std::runtime_error("decision lifecycle");
  }catch(const std::exception&e){r.error=e.what();}
  if(policy)lib.api.destroy(policy);r.seconds=seconds(tic);return r;
 }
 py::dict batch(const std::string&library,const std::string&weights,int mode,std::vector<uint64_t>seeds,std::vector<int>seats,std::vector<int>opponents,uint64_t sample_seed,int threads,bool trace)const{
  if(seeds.size()!=seats.size()||seeds.size()!=opponents.size()||threads<1||threads>16)throw std::runtime_error("batch shape");
  for(int p:seats)if(p<0||p>1)throw std::runtime_error("seat");Library lib(library);std::vector<Game>games(seeds.size());
  auto start=std::chrono::steady_clock::now();
  {py::gil_scoped_release release;
   #pragma omp parallel for num_threads(threads) schedule(dynamic)
   for(size_t n=0;n<seeds.size();n++)games[n]=play(lib,weights,mode,seeds[n],seats[n],opponents[n],sample_seed+0x9e3779b97f4a7c15ULL*(n+1),trace);
  }
  py::dict out;out["wall_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();py::list rows,traces;
  size_t count=0;for(auto&r:games)count+=r.decisions.size();
  py::array_t<float>x({py::ssize_t(count),py::ssize_t(econrl::G)}),c({py::ssize_t(count),py::ssize_t(econrl::A),py::ssize_t(econrl::C)}),mask({py::ssize_t(count),py::ssize_t(econrl::A)}),prob({py::ssize_t(count),py::ssize_t(econrl::A)}),logp(count),value(count),reward(count);
  py::array_t<int>choice(count),game(count),day(count),changed(count),pos(count),kind(count);
  size_t at=0;
  for(size_t g=0;g<games.size();g++){
   auto&r=games[g];py::dict row;row["seed"]=r.seed;row["seat"]=r.seat;row["opponent"]=r.opponent;row["steps"]=r.steps;row["cash"]=r.cash;row["opponent_cash"]=r.other;
   row["margin"]=r.cash-r.other;row["win"]=r.cash>r.other;row["tie"]=r.cash==r.other;row["error"]=r.error;row["seconds"]=r.seconds;
   row["policy_seconds"]=r.policy_seconds;row["enemy_seconds"]=r.enemy_seconds;row["env_seconds"]=r.env_seconds;row["model_seconds"]=r.model_seconds;row["max_action_ms"]=r.max_action_ms;
   row["plan_calls"]=r.plans;row["reference_calls"]=r.references;row["execute_calls"]=r.executes;rows.append(row);
   py::list steps;if(trace)for(auto&a:r.trace){py::list pair;pair.append(bridge::pack_player(a[0]));pair.append(bridge::pack_player(a[1]));steps.append(pair);}traces.append(steps);
   for(size_t j=0;j<r.decisions.size();j++,at++){
    auto&d=r.decisions[j];std::copy(d.global.begin(),d.global.end(),x.mutable_data()+at*econrl::G);
    for(int a=0;a<econrl::A;a++)std::copy(d.candidates[a].begin(),d.candidates[a].end(),c.mutable_data()+(at*econrl::A+a)*econrl::C);
    std::copy(d.mask.begin(),d.mask.end(),mask.mutable_data()+at*econrl::A);std::copy(d.probability.begin(),d.probability.end(),prob.mutable_data()+at*econrl::A);
    logp.mutable_data()[at]=d.logp;value.mutable_data()[at]=d.value;choice.mutable_data()[at]=d.choice;game.mutable_data()[at]=g;day.mutable_data()[at]=d.day;changed.mutable_data()[at]=d.changed;pos.mutable_data()[at]=d.pos;kind.mutable_data()[at]=d.newkind;
    reward.mutable_data()[at]=j+1==r.decisions.size()?float((r.cash>r.other?1:r.cash<r.other?-1:0)+.1*std::tanh((r.cash-r.other)/50000.)):0.f;
   }
  }
  out["rows"]=rows;out["traces"]=traces;out["global"]=x;out["candidates"]=c;out["mask"]=mask;out["probability"]=prob;out["logp"]=logp;out["value"]=value;out["reward"]=reward;out["choice"]=choice;out["game"]=game;out["day"]=day;out["changed"]=changed;out["pos"]=pos;out["kind"]=kind;return out;
 }
};
}
inline void econrl_bind(py::module_&m){
 py::class_<econrun::Pool>(m,"RLPool",py::module_local()).def(py::init<py::dict>()).def("batch",&econrun::Pool::batch);
 m.attr("RL_G")=econrl::G;m.attr("RL_C")=econrl::C;m.attr("RL_A")=econrl::A;m.attr("RL_NW")=econrl::NW;
}
