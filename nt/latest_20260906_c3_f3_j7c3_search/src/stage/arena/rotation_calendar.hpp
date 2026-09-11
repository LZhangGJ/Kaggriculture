#pragma once
// Typed crop-transition calendars, verified by the frozen S5A mechanism suite.
// A maintained, known-state continuation, NOT the real future. No seed,
// opponent controller, private rival inventory or replay is accepted here.
namespace dp7::rotationproto {
using Seeds=std::array<std::array<int,5>,30>;
struct Crop {portfolio::DeliveryCalendar cal;Seeds seeds{};};

// Unlike the old single-kind crop_delivery contract, this explicitly tags the
// old harvest and the replacement seed by their actual product types.
inline Crop crop(const Controller&source,const View&o,int desired,Tile initial,int not_before=-1){
 if(desired<0||desired>=5||source.day!=o.day||o.day<0||o.day>=30||o.hour<0||o.hour>=24)
  throw std::invalid_argument("replacement crop calendar arguments");
 if(not_before<0)not_before=o.day;
 if(not_before<o.day||not_before>30)throw std::invalid_argument("replacement activation day");
 Crop out;Controller ctl(source.p);ctl.target={{44,desired}};
 ctl.p.split_service_jobs=ctl.p.shared_task_atoms_v2=false;
 Farm farm;farm.tiles.resize(100);farm.tiles[44]=initial;farm.farmer={4,4};
 PrivateState pr;pr.inventories.resize(1);pr.inventory_order.resize(1);
 for(int d=o.day;d<30;d++){
  ctl.day=d;int begin=d==o.day?o.hour:0;
  View view{24*d+begin,d,begin,farm,o.opponent,pr,o.market,o.shops};
  Acts actions;int pickups=0;
  for(const auto&j:ctl.jobs(view)){
   pickups+=j.needs[F]>0;
   for(auto a:j.actions){
    // Harvest/maintain the old asset first, then leave the plot fallow until
    // the chosen investment day. This is NOT destroying an old crop to wait.
    if(d<not_before&&a.op==Op::PLANT)break;
    actions.push_back(a);
   }
  }
  Counts harvested{};std::array<int,5>last;last.fill(-100);size_t cursor=0;
  for(int h=begin;h<=(d==29?22:23);h++){
   auto&t=farm.tiles[44];
   if(pickups)pickups--;
   else if(cursor<actions.size()){
    auto a=actions[cursor++];bool before=plant(t);int old_kind=before?int(t.crop):-1;
    int got=portfolio::crop_atom(t,a,d);
    if(a.op==Op::PLANT&&plant(t)&&!before){int k=int(t.crop);out.cal.planting[d]++;out.seeds[d][k]++;out.cal.cash.fixed[d]-=seed_price[k];}
    if(a.op==Op::FERTILIZE&&before){out.cal.fert[d]++;out.cal.cash.quantity[d][F]--;}
    if(a.op==Op::WATER&&before)out.cal.water[d]++;
    if(a.op==Op::DIG)out.cal.dig[d]++;
    if(got){if(old_kind<0)throw std::logic_error("untyped harvest");out.cal.harvest[d]++;harvested[old_kind]+=got;last[old_kind]=h;}
   }
   portfolio::crop_decay(t,24*d+h);
  }
  for(int k=0;k<5;k++)if(d<29)out.cal.cash.quantity[d+1][k]+=harvested[k];
   else if(last[k]<22)out.cal.cash.quantity[d][k]+=harvested[k];
  if(d<29)portfolio::crop_night(farm.tiles[44],d);
 }
 return out;
}
struct Work {int pos,kind;portfolio::DeliveryCalendar cal;Seeds seeds{};double capital=0;};
using Works=std::vector<Work>;
inline Works physical_works(const Controller&c,const View&o){
 Works out;std::map<decltype(dayvalue::tile_key(Tile{})),Crop>crops;
 std::map<decltype(dayvalue::tile_key(Tile{})),portfolio::DeliveryCalendar>animals;
 for(int pos=0;pos<100;pos++){
  auto&t=o.own.tiles[pos];auto key=dayvalue::tile_key(t);
  if(plant(t)){auto it=crops.find(key);if(it==crops.end())it=crops.emplace(key,crop(c,o,int(t.crop),t)).first;out.push_back({pos,int(t.crop),it->second.cal,it->second.seeds});}
  else if(animal(t)){auto it=animals.find(key);if(it==animals.end())it=animals.emplace(key,dayvalue::animal_calendar(c,o,t)).first;out.push_back({pos,int(t.animal),it->second,{}});}
 }
 return out;
}
inline Works replace(const Controller&c,const View&o,const Works&old,int pos,int desired,int not_before=-1){
 auto result=old;int count=0;
 for(auto&w:result)if(w.pos==pos){auto f=crop(c,o,desired,o.own.tiles.at(pos),not_before);w={pos,desired,f.cal,f.seeds};count++;}
 if(count!=1)throw std::invalid_argument("replacement must have exactly one existing plot");
 return result;
}
struct Value {
 double cash=0,sales=0,purchases=0,seeds=0,wages=0,capital=0,funding_gap=0;
 bool schedulable=true;int plots=0;
 std::array<double,9>sale_by_item{},buy_by_item{};
};
inline Value value(const Controller&source,const View&o,const Works&works,bool labor=true){
 Value result;result.plots=int(works.size());result.cash=o.own.money;
 std::array<bool,100>seen{};Controller::Calendar flow{};Seeds seed_need{};
 std::vector<dayvalue::Work>labor_works;
 for(const auto&w:works){
  if(w.pos<0||w.pos>=100||seen[w.pos])throw std::invalid_argument("duplicate plot in portfolio");seen[w.pos]=true;
  result.capital+=w.capital;result.cash-=w.capital;
  labor_works.push_back({w.pos,w.kind,w.cal});
  for(int d=o.day;d<30;d++){
   for(int i=0;i<9;i++)flow[d][i]+=w.cal.cash.quantity[d][i];
   for(int i=0;i<5;i++)seed_need[d][i]+=w.seeds[d][i];
  }
 }
 // Keep rival production and town expectation; do not use context.own, since
 // physical works already own every crop/animal contribution exactly once.
 auto c=source;auto context=c.portfolio_context(o);Counts stock=o.priv.shed;
 for(const auto&bag:o.priv.inventories)add(stock,bag);
 auto owned_seeds=o.priv.seeds;Quantities market{};for(int i=0;i<9;i++)market[i]=o.market.inventory[i];
 for(int d=o.day;d<30;d++){
  double wage=labor?dayvalue::wages_on(c,labor_works,d):0.;
  if(!std::isfinite(wage)){result.schedulable=false;return result;}
  // Existing hires have already been paid. This conditional daily estimate
  // cannot refund them or charge them again; actual current-day packing and
  // financing remain mandatory before any future policy integration.
  if(d==o.day)wage=std::max(0.,wage-Controller::hirecost(o.own.hires_today));
  result.wages+=wage;result.cash-=wage;
  for(int i=0;i<5;i++){
   int reuse=std::min(int(owned_seeds[i]),seed_need[d][i]);owned_seeds[i]-=reuse;
   double paid=(seed_need[d][i]-reuse)*seed_price[i];result.seeds+=paid;result.cash-=paid;
  }
  for(int i=0;i<9;i++){
   market[i]+=context.rival[d][i]-context.dem[d][i];stock[i]+=int(std::nearbyint(flow[d][i]));
   if(stock[i]<0){double paid=-Controller::projected_trade(i,market[i],stock[i]);result.purchases+=paid;result.buy_by_item[i]+=paid;result.cash-=paid;stock[i]=0;}
   int hold=(i==W||i==F)?dayvalue::reserve_after(flow,d,i):0;
   int sell=std::max(0,stock[i]-hold);stock[i]-=sell;
   double got=Controller::projected_trade(i,market[i],sell);result.sales+=got;result.sale_by_item[i]+=got;result.cash+=got;
  }
  result.funding_gap=std::max(result.funding_gap,std::max(0.,-result.cash));
 }
 return result;
}
} // namespace dp7::rotationproto
