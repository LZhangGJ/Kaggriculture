#include <boost/json.hpp>
#include <boost/json/src.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <unordered_map>
namespace fusion {
#define DP_LIBRARY
#include "agent.cpp"
#undef DP_LIBRARY
}

namespace fusionhost {
fusion::Tile tile(const fastkag::Tile&t){
 fusion::Tile z;z.kind=t.kind==fastkag::TileKind::LOCKED?-1:t.kind==fastkag::TileKind::EMPTY?0:t.kind==fastkag::TileKind::PLANT?2:t.kind==fastkag::TileKind::COOP?3:t.kind==fastkag::TileKind::PASTURE?4:t.kind==fastkag::TileKind::ANIMAL?(t.animal==fastkag::Item::GOOSE?3:4):1;
 z.item=int(t.kind==fastkag::TileKind::PLANT?t.crop:t.animal);z.placed=t.kind==fastkag::TileKind::PLANT?t.planted_day:t.placed_day;
 z.yield=t.yield_units;z.unwatered=t.consecutive_unwatered;z.unfed=t.consecutive_unfed;z.pending=t.pending_care_bonus;
 z.fertilized=t.fertilized_until_day;z.lifespan=t.max_lifespan_step;z.water=t.watered_today;z.feed=t.fed_today;z.care=t.cared_today;z.manure=t.fertilizer_available;return z;
}
fusion::Obs observation(const dp7::View&v,int seat){
 fusion::Obs o;o.day=v.day;o.hour=v.hour;o.step=v.step;o.seat=seat;o.money=v.own.money;o.unlocked=__builtin_popcount(v.own.unlocked_mask);
 for(int p=0;p<100;p++){o.tiles[p]=tile(v.own.tiles[p]);o.opponent_tiles[p]=tile(v.opponent.tiles[p]);}
 o.positions.push_back(fusion::xy(v.own.farmer.x,v.own.farmer.y));for(auto p:v.own.hands)o.positions.push_back(fusion::xy(p.x,p.y));
 for(int i=0;i<12;i++)o.shed[i]=v.priv.shed[i];for(int i=0;i<5;i++)o.seeds[i]=v.priv.seeds[i];
 for(const auto&inv:v.priv.inventories){fusion::Stock s{};for(int i=0;i<12;i++)s[i]=inv[i];o.invs.push_back(s);}
 for(int i=0;i<9;i++){o.market[i]=v.market.inventory[i];o.prices[i]=v.market.prices[i];o.params[i]={fusion::BASE[i],fusion::THR[i],10000,fusion::LT[i],fusion::HT[i],fusion::LO[i],fusion::HI[i]};}
 for(auto sh:v.shops)o.shops.push_back(fastkag::shop_name(sh));return o;
}
fastkag::Action atom(const fusion::J&j){
 const auto&a=j.as_array();fastkag::Action out;std::string name(a[0].as_string());
 for(int k=0;k<24;k++)if(name==bridge::ops[k])out.op=fastkag::Op(k);
 if(a.size()>1&&a[1].is_string())out.item=fastkag::Item(fusion::itemid(std::string(a[1].as_string())));
 if(a.size()>2)out.quantity=int(fusion::num(a[2]));return out;
}
fastkag::PlayerAction act(fusion::Agent&a,const dp7::View&v,int seat){
 auto o=observation(v,seat);fusion::Config c;if(o.step==0||o.day<a.m.day)a.m=fusion::Memory{};
 if(o.day!=a.m.day)a.plan(o,c);auto j=a.execute(o,c);
 fastkag::PlayerAction out;out.units.push_back(atom(j.at("farmer")));
 for(const auto&x:j.at("hands").as_array())out.units.push_back(atom(x));for(const auto&x:j.at("market").as_array())out.market.push_back(atom(x));return out;
}
fusion::Agent create(const std::string&settings){fusion::Context ctx;ctx.agent.s.update(boost::json::parse(settings));return ctx.agent;}
}

namespace safeopening {
#include "opening_params.hpp"
#include "opening_calendar.hpp"
struct Agent {
 int days=0,last=-1,missing=0,handoff=-1;
 dp7::Controller opening{frozen_params()};
 fusion::Agent dynamic;
 explicit Agent(const std::string&settings):dynamic(fusionhost::create(settings)){
  auto j=boost::json::parse(settings);days=int(fusion::num(fusion::get(j,"opening_days")));
  if(days<0||days>30)throw std::invalid_argument("opening_days outside 0..30");
 }
 fastkag::PlayerAction act(const dp7::View&v,int seat){
  if(v.step==0||v.step<last){opening=dp7::Controller(frozen_params());dynamic.m=fusion::Memory{};missing=0;handoff=-1;last=-1;}
  if(last>=0&&v.step!=last+1)throw std::runtime_error("nonsequential safe opening");
  last=v.step;
  if(v.day<days){
   if(v.hour==0&&v.day<=28){
    auto menu=dp7branch::generate(opening,v);
    if(menu.empty()||menu[0].family!="KEEP")throw std::runtime_error("opening KEEP absent");
    const auto&r=switch_plans.at(0).at(v.day);int choice=0;bool found=false;
    for(int i=0;i<int(menu.size());i++)if(menu[i].family==r.family&&menu[i].kind==r.kind&&menu[i].amount==r.amount){choice=i;found=true;break;}
    missing+=!found;opening=menu[choice].controller;
   }
   return opening.act(v);
  }
  if(handoff<0)handoff=v.day;
  // No opener targets, inventory assumptions or later calendar entries transfer.
  // F1 reconstructs its live tasks and finances from the actual legal observation.
  return fusionhost::act(dynamic,v,seat);
 }
};
}

void register_fusion(py::module_&m){
 py::class_<fusion::Agent>(m,"FusionAgent").def(py::init(&fusionhost::create))
 .def("act",[](fusion::Agent&a,const fastkag::Simulator&s,int seat){return bridge::pack_player(fusionhost::act(a,bridge::view(s,seat),seat));})
 .def("debug",[](const fusion::Agent&a){return boost::json::serialize(a.debug());})
 .def("act_json",[](fusion::Agent&a,const std::string&obs){return boost::json::serialize(a.act(boost::json::parse(obs),fusion::JO{}));});
 py::class_<safeopening::Agent>(m,"SafeAgent").def(py::init<const std::string&>())
 .def("act",[](safeopening::Agent&a,const fastkag::Simulator&s,int seat){return bridge::pack_player(a.act(bridge::view(s,seat),seat));})
 .def("debug",[](const safeopening::Agent&a){py::dict d;d["opening_days"]=a.days;d["handoff_day"]=a.handoff;d["missing_opening_recipe"]=a.missing;d["planner"]=boost::json::serialize(a.dynamic.debug());return d;});
 m.def("fusion_batch",[](const std::vector<uint64_t>&seeds,const std::vector<int>&seats,const std::string&settings,int kind,py::object other,int threads){
  if(seeds.size()!=seats.size()||threads<1||threads>16)throw std::invalid_argument("batch dimensions/threads");
  bridge::AuditOpponentSpec spec;spec.kind=kind;
  if(kind==1)spec.route=&other.cast<const bridge::G001&>();
  if(kind==2)spec.boatlee=&other.cast<const boatlee29::Agent&>();
  if(kind==3)spec.kaito=&other.cast<const kaito58::Agent&>();
  if(kind==4)spec.lynn=&other.cast<const lynn5::Agent&>();
  struct Row {double cash=0,enemy=0,seconds=0,maxms=0;int steps=0,missing=0,handoff=-1;std::array<std::array<double,3>,30>days{};std::string error;};
  std::vector<Row>rows(seeds.size());safeopening::Agent prototype(settings);
  {
   py::gil_scoped_release release;
   #pragma omp parallel for num_threads(threads) schedule(dynamic)
   for(size_t n=0;n<seeds.size();n++){
    auto tic=std::chrono::steady_clock::now();auto&r=rows[n];int seat=seats[n];
    try{
     if(seat<0||seat>1)throw std::invalid_argument("seat");fastkag::Simulator env(fastkag::Config{},seeds[n]);
     auto own=prototype;bridge::AuditOpponent enemy(spec);
     while(!env.done()){
      std::array<fastkag::PlayerAction,2>a;auto t=std::chrono::steady_clock::now();a[seat]=own.act(bridge::view(env,seat),seat);
      r.maxms=std::max(r.maxms,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count());
      a[1-seat]=enemy.act(env,1-seat);int day=env.day();env.step(a);
      r.days[day]={env.farms()[seat].money,env.farms()[1-seat].money,double(env.farms()[seat].hands.size())};
     }
     r.cash=env.farms()[seat].money;r.enemy=env.farms()[1-seat].money;r.steps=env.step_count();r.missing=own.missing;r.handoff=own.handoff;
    }catch(const std::exception&e){r.error=e.what();}
    r.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-tic).count();
   }
  }
  py::list out;for(size_t n=0;n<rows.size();n++){auto&r=rows[n];py::dict d;d["seed"]=seeds[n];d["seat"]=seats[n];d["cash"]=r.cash;d["opponent_cash"]=r.enemy;d["steps"]=r.steps;d["seconds"]=r.seconds;d["max_ms"]=r.maxms;d["days"]=r.days;d["error"]=r.error;d["missing_opening_recipe"]=r.missing;d["handoff_day"]=r.handoff;out.append(d);}return out;
 });
}
