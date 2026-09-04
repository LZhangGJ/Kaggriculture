#include "policy.hpp"
#include <iostream>
#include <sstream>
using namespace dp7;
using namespace dp7::portfolio;
static int checks=0,games=0;
static void need(bool ok,const std::string&message){checks++;if(!ok)throw std::runtime_error(message);}
static bool same_tile(const Tile&a,const Tile&b){
 return std::tuple(a.kind,a.crop,a.planted_day,a.yield_units,a.consecutive_unwatered,a.fertilized_until_day,a.max_lifespan_step,a.watered_today)==
        std::tuple(b.kind,b.crop,b.planted_day,b.yield_units,b.consecutive_unwatered,b.fertilized_until_day,b.max_lifespan_step,b.watered_today);
}
static void compare_calendar(const DeliveryCalendar&a,const DeliveryCalendar&b,const std::string&where){
 for(int d=0;d<30;d++){
  need(a.cash.quantity[d]==b.cash.quantity[d],where+" cash quantity day "+std::to_string(d));
  need(a.cash.fixed[d]==b.cash.fixed[d],where+" seed cost day "+std::to_string(d));
  need(std::tuple(a.water[d],a.fert[d],a.harvest[d],a.planting[d],a.dig[d])==std::tuple(b.water[d],b.fert[d],b.harvest[d],b.planting[d],b.dig[d]),where+" work day "+std::to_string(d));
 }
}
// A legal, well-supplied one-plot microfarm. It tests conditional forecast
// semantics, NOT competitive strength. RNG weed events are explicitly disabled.
static void engine_case(int kind,int variant,int startday,int seat){
 Config cfg;cfg.weed_spawn_chance=0.;Simulator env(cfg,77500+variant);std::array<PlayerAction,2>a;
 while(env.day()<startday)env.step(a);
 a[seat].market={action(Op::BUY_SEED,kind,20),action(Op::BUY_PRODUCT,F,8)};
 // Limit expensive seed inventories to preserve the ordinary starting cash.
 if(kind==S||kind==M)a[seat].market[0].quantity=8;
 env.step(a);a={};
 need(env.privates()[seat].shed[F]==8,"legal forecast fixture could not fund fertilizer");
 Params params;params.portfolio_crop_calendar=true;params.efficient_water=variant&1;
 params.renew_ongoing=variant&2;params.finite_fertilizer=variant&4;params.fix_finite_projection=true;
 Controller c(params);c.day=env.day();c.target={{44,kind}};
 // Prices for task decisions are frozen at the observed start, exactly as in
 // the conditional forecast; the authoritative simulator still trades normally.
 Market quotes=env.market();auto shops=env.shops();
 View start{env.step_count(),env.day(),env.hour(),env.farms()[seat],env.farms()[1-seat],env.privates()[seat],quotes,shops};
 auto predicted=crop_delivery(c,start,kind,start.own.tiles[44]);DeliveryCalendar actual;
 Tile model=start.own.tiles[44];int queued_day=-1,last_harvest=-100,harvested=0;Acts todo;size_t cursor=0;
 while(!env.done()){
  int d=env.day(),h=env.hour(),s=env.step_count();
  if(queued_day!=d){
   queued_day=d;cursor=0;todo.clear();c.day=d;
   View view{s,d,h,env.farms()[seat],env.farms()[1-seat],env.privates()[seat],quotes,shops};
   auto work=c.jobs(view);for(const auto&j:work)if(j.needs[F])todo.push_back(action(Op::PICKUP,F,j.needs[F]));
   for(const auto&j:work)todo.insert(todo.end(),j.actions.begin(),j.actions.end());
  }
  Action atom=cursor<todo.size()?todo[cursor++]:action(Op::PASS);
  if(d==29&&cursor>=todo.size()&&atom.op==Op::PASS)atom=action(Op::DROP);
  a={};a[seat].units={atom};
  // Prevent storage overflow without selling private fertilizer or seeds.
  if(h==0)for(int i=0;i<8;i++)if(env.privates()[seat].shed[i])a[seat].market.push_back(action(Op::SELL,i,env.privates()[seat].shed[i]));
  auto unit=env.project_unit_phase(seat,a[seat].units);
  auto&before=env.farms()[seat].tiles[44];auto&after=unit.farms()[seat].tiles[44];
  int quantity=unit.privates()[seat].inventories[0][kind]-env.privates()[seat].inventories[0][kind];
  int expected=crop_atom(model,atom,d);
  need(same_tile(after,model),"unit-phase model mismatch "+name(kind)+" step "+std::to_string(s));
  if(atom.op==Op::HARVEST){need(quantity==expected&&quantity>0,"harvest silently failed");actual.harvest[d]++;harvested+=quantity;last_harvest=h;}
  if(atom.op==Op::PLANT){need(plant(after)&&!plant(before),"plant silently failed");actual.planting[d]++;actual.cash.fixed[d]-=seed_price[kind];}
  if(atom.op==Op::WATER){need(after.watered_today&&!before.watered_today,"water silently failed");actual.water[d]++;}
  if(atom.op==Op::FERTILIZE){need(unit.privates()[seat].inventories[0][F]==env.privates()[seat].inventories[0][F]-1,"fertilize silently failed");actual.fert[d]++;actual.cash.quantity[d][F]--;}
  if(atom.op==Op::DIG)actual.dig[d]++;
  env.step(a);crop_decay(model,s);if(h==23)crop_night(model,d);
  need(same_tile(env.farms()[seat].tiles[44],model),"decay/night mismatch "+name(kind)+" step "+std::to_string(s));
  if(h==23||env.done()){
   if(d<29)actual.cash.quantity[d+1][kind]+=harvested;
   else if(last_harvest<22)actual.cash.quantity[d][kind]+=harvested;
   need(env.last_end_of_day_overflow()[seat]==0,"conditional fixture overflowed");harvested=0;last_harvest=-100;
  }
 }
 compare_calendar(predicted,actual,name(kind)+" variant "+std::to_string(variant)+" start "+std::to_string(startday));games++;
}
int main(){
 try{
  for(int k=0;k<5;k++)for(int variant=0;variant<8;variant++)for(int day:{0,9,23})engine_case(k,variant,day,variant%2);
  Simulator env(Config{},77601);auto own=env.farms()[0];auto priv=env.privates()[0];Controller c;c.day=0;c.p.portfolio_crop_calendar=true;c.p.fix_finite_projection=true;
  for(int k=0;k<5;k++){
   View v{0,0,0,own,env.farms()[1],priv,env.market(),env.shops()};auto fresh=crop_delivery(c,v,k,Tile{});
   Tile planted;crop_atom(planted,action(Op::PLANT,k),0);crop_atom(planted,action(Op::WATER),0);
   View after{2,0,2,own,env.farms()[1],priv,env.market(),env.shops()};auto existing=crop_delivery(c,after,k,planted);
   fresh.cash.fixed[0]+=seed_price[k];fresh.planting[0]--;fresh.water[0]--;
   compare_calendar(fresh,existing,"new-to-owned "+name(k));
   own.tiles[44]=planted;auto context=crop_context(c,after);
   for(int d=0;d<30;d++){need(context.own[d]==existing.cash.quantity[d],"crop context duplicated a contribution");need(context.own_fixed[d]==existing.cash.fixed[d],"owned replant cost missing");}
   own.tiles[44]=Tile{};
  }
  // Late-day promises and already-held ongoing crop output.
  c.day=29;Tile t;t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=10;t.yield_units=4;t.watered_today=true;
  View last{718,29,22,own,env.farms()[1],priv,env.market(),env.shops()};
  auto late=crop_delivery(c,last,S,t);need(late.harvest[29]==1&&late.cash.quantity[29][S]==0,"final action harvest became impossible cash");
  View earlier{717,29,21,own,env.farms()[1],priv,env.market(),env.shops()};
  auto early=crop_delivery(c,earlier,S,t);need(early.cash.quantity[29][S]==4,"held product lost or counted twice");
  auto impossible=crop_delivery(c,last,S,Tile{});need(impossible.planting[29]==0&&impossible.cash.quantity[29][S]==0,"terminal planting invented returns");
  std::cout<<"{\"status\":\"PASS_CONDITIONAL_CROP_CALENDAR\",\"checks\":"<<checks<<",\"controlled_engine_games\":"<<games<<",\"unknown_future_used\":false,\"weed_spawn_chance\":0}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
