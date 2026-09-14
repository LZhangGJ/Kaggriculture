#pragma once
// Conditional, well-supplied single-plot forecast. Shared by a new project and
// a live crop. Its daily semantic work comes from Controller::jobs, not another
// policy. The rules below are a SMALL prediction model checked against the
// unchanged authoritative engine. They never update the real game state.
namespace dp7::portfolio {
inline void crop_decay(Tile&t,int step){
 if(plant(t)&&t.max_lifespan_step>=0&&step>=t.max_lifespan_step&&(step-t.max_lifespan_step)%2==0){
  if(--t.yield_units<=0){t=Tile{};t.kind=TileKind::WEED;}
 }
}
inline void crop_night(Tile&t,int day){
 if(!plant(t))return;
 bool watered=t.watered_today;t.consecutive_unwatered=watered?0:t.consecutive_unwatered+1;t.watered_today=false;
 if(t.consecutive_unwatered>=2){t=Tile{};t.kind=TileKind::WEED;return;}
 int kind=int(t.crop);if(!ongoing(kind))return;
 int ds=day+1-t.planted_day-first[kind];
 if(ds>=0&&ds%interval[kind]==0){int period=ds/interval[kind]+1;
  if(period<=4){bool fertile=watered&&t.fertilized_until_day>=day;t.yield_units=std::min(4,int(t.yield_units)+(fertile?2:1));
   if(period==4)t.max_lifespan_step=(day+2)*24;
  }
 }
}
// Return harvested quantity, leaving cost/obligation recording to the caller.
inline int crop_atom(Tile&t,const Action&a,int day){
 const int kind=int(a.item);
 if(a.op==Op::DIG){t=Tile{};return 0;}
 if(a.op==Op::PLANT&&t.kind==TileKind::EMPTY&&kind>=0&&kind<5){
  t=Tile{};t.kind=TileKind::PLANT;t.crop=Item(kind);t.planted_day=day;t.consecutive_unwatered=1;t.yield_units=ongoing(kind)?0:1;
  t.max_lifespan_step=ongoing(kind)?-1:(day+(kind==W?4:kind==C?3:12)+1)*24;return 0;
 }
 if(!plant(t))return 0;
 int c=int(t.crop),age=day-t.planted_day;
 if(a.op==Op::FERTILIZE)t.fertilized_until_day=std::max(int(t.fertilized_until_day),day+2);
 else if(a.op==Op::WATER&&!t.watered_today){
  t.watered_today=true;int maxday=c==W?4:c==C?3:12;
  if(!ongoing(c)&&age>=(maxday+1)/2&&age<=maxday)t.yield_units=std::min(c==C?4:6,int(t.yield_units)+(t.fertilized_until_day>=day?2:1));
 }else if(a.op==Op::HARVEST&&t.yield_units>0&&age>=first[c]){
  int quantity=t.yield_units;t.yield_units=0;if(!ongoing(c))t=Tile{};return quantity;
 }
 return 0;
}
inline DeliveryCalendar crop_delivery(const Controller&source,const View&observed,int kind,Tile initial){
 if(kind<0||kind>=5)throw std::invalid_argument("crop calendar kind");
 if(source.day<0||source.day>=30||source.day!=observed.day||observed.hour<0||observed.hour>=24)throw std::invalid_argument("crop calendar clock");
 DeliveryCalendar result;Controller policy(source.p);policy.target={{44,kind}};
 // One logical crop work group here; its later assignment may be shared among
 // workers by the real scheduler. This does NOT install a worker-owned chain.
 policy.p.split_service_jobs=policy.p.shared_task_atoms_v2=false;
 Farm farm;farm.tiles.resize(100);farm.tiles[44]=initial;farm.farmer={4,4};
 PrivateState resources;resources.inventories.resize(1);resources.inventory_order.resize(1);
 for(int d=source.day;d<30;d++){
  policy.day=d;int first_hour=d==source.day?observed.hour:0;
  View view{24*d+first_hour,d,first_hour,farm,observed.opponent,resources,observed.market,observed.shops};
  auto jobs=policy.jobs(view);Acts actions;int pickup_steps=0;
  for(const auto&j:jobs){pickup_steps+=j.needs[F]>0;actions.insert(actions.end(),j.actions.begin(),j.actions.end());}
  int harvested=0,last_harvest_hour=-100;size_t cursor=0;
  for(int hour=first_hour;hour<=(d==29?22:23);hour++){
   auto&t=farm.tiles[44];
   if(pickup_steps>0)pickup_steps--;
   else if(cursor<actions.size()){
    Action a=actions[cursor++];bool was_crop=plant(t);int h=crop_atom(t,a,d);
    if(a.op==Op::PLANT&&plant(t)&&!was_crop){result.planting[d]++;result.cash.fixed[d]-=seed_price[kind];}
    if(a.op==Op::FERTILIZE&&was_crop){result.fert[d]++;result.cash.quantity[d][F]--;}
    if(a.op==Op::WATER&&was_crop)result.water[d]++;
    if(a.op==Op::DIG)result.dig[d]++;
    if(h){result.harvest[d]++;harvested+=h;last_harvest_hour=hour;}
   }
   crop_decay(t,24*d+hour);
  }
  // Ordinary-day harvest auto-deposits, so cannot fund earlier purchases.
  // Final-day same-tile DROP requires a remaining action; spatial return time
  // is checked by the joint workforce forecast and current-day scheduler.
  if(d<29)result.cash.quantity[d+1][kind]+=harvested;
  else if(last_harvest_hour<22)result.cash.quantity[d][kind]+=harvested;
  if(d<29)crop_night(farm.tiles[44],d);
 }
 return result;
}
inline Controller::PortfolioContext crop_context(const Controller&source,const View&o){
 auto clock=source;clock.p.own_feed_demand_weight=1.;
 // Retain owned inventories, animals, public rival forecast and expected town
 // demand exactly once. Replace only the old current-crop-only contribution.
 auto without=o.own;for(auto&t:without.tiles)if(plant(t))t=Tile{};
 View other{o.step,o.day,o.hour,without,o.opponent,o.priv,o.market,o.shops};
 auto result=clock.portfolio_context(other);
 for(const auto&t:o.own.tiles)if(plant(t)){
  auto future=crop_delivery(source,o,int(t.crop),t);
  for(int d=source.day;d<30;d++){
   result.own_fixed[d]+=future.cash.fixed[d];
   for(int i=0;i<9;i++)result.own[d][i]+=future.cash.quantity[d][i];
  }
 }
 return result;
}
}
