#include "../policy/triad.hpp"
#include <iostream>
#include <cstdlib>
using namespace dp7;
static int checks=0;
void ck(bool b,const char*m){++checks;if(!b){std::cerr<<"FAIL "<<m<<"\n";std::exit(1);}}
struct Fixture{
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;triad::Controller c;
 Fixture():c([](){triad::Settings s;s.max_hands=0;s.max_land=4;s.intraday=0;s.a06_r11_recovery=0;s.delay_sale=-1;return s;}()){
  own.tiles.resize(100);rival.tiles.resize(100);own.unlocked_mask=15;own.farmer={4,4};own.money=1000;
  priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(10000);for(int k=0;k<9;k++)market.prices[k]=price(k,10000);
  auto&t=own.tiles[34];t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=7;t.fertilized_until_day=18;t.consecutive_unwatered=1;
  c.core.day=20;c.core.crop_service_day=20;c.core.target={{34,S}};c.core.crop_birth[34]=7;c.core.crop_kind[34]=S;c.core.crop_water[34]=c.core.crop_fertilize[34]=1;
 }
 View view(int h){return{480+h,20,h,own,rival,priv,market,shops};}
 int buys(){int n=0;for(auto a:c.core.queue)if(a.op==Op::BUY_PRODUCT&&int(a.item)==F)n+=a.quantity;return n;}
 bool complete(int h){auto roll=c;roll.previous_step=480+h-1;roll.core.last_step=480+h-1;ObservedDayScenario world(view(h));
  while(!world.finished()){auto v=world.view();auto a=roll.act(v);if(v.hour==23){auto post=world.project_units(a.units,-1);auto&t=post.farms()[0].tiles[34];return plant(t)&&t.watered_today&&t.fertilized_until_day>=20;}world.advance(a);}return false;}
};
int main(){
 for(int hour:{0,17,18,19,20,21,22,23}){
  Fixture f;auto v=f.view(hour);auto old_cash=f.own.money;auto old_shed=f.priv.shed;auto old_crop=f.own.tiles[34];
  f.c.prepare_with_ongoing_supply(v,v,0);
  const bool useful=hour==18||hour==19;
  ck(f.buys()==int(useful),"buy iff the matched execution adds maintenance");
  ck(f.c.ongoing_supply_previews==1&&f.c.ongoing_supply_preview_errors==0,"one finite current-day preview, no errors");
  ck(f.own.money==old_cash&&f.priv.shed==old_shed&&f.own.tiles[34].watered_today==old_crop.watered_today,"preview cannot mutate authoritative facts");
  ck(!f.c.core.suppress_ongoing_fertilizer_supply&&!f.c.ongoing_supply_previewing,"internal comparison flags cannot leak into live state");
  if(useful){ck(f.complete(hour),"funded pickup/travel/fertilizer/water finishes by deadline");Fixture p;p.c.core.suppress_ongoing_fertilizer_supply=true;p.c.core.prepare_orders(v,v,0);ck(!p.complete(hour),"parent preparation cannot finish the same service window");}
 }
 {Fixture f;f.priv.shed[F]=1;auto v=f.view(19);f.c.prepare_with_ongoing_supply(v,v,0);ck(f.buys()==0&&f.c.ongoing_supply_previews==0,"already owned fertilizer: no purchase or scenario");}
 {Fixture f;f.c.core.crop_fertilize[34]=0;auto v=f.view(19);f.c.prepare_with_ongoing_supply(v,v,0);ck(f.buys()==0&&f.c.ongoing_supply_previews==0,"rejected service not recreated");}
 {Fixture f;f.own.money=0;auto v=f.view(19);f.c.prepare_with_ongoing_supply(v,v,0);ck(f.buys()==0&&f.c.ongoing_supply_previews==0,"cashless preparation cannot buy or rely on future produce");}
 {Fixture f;f.priv.inventory_order.clear();auto v=f.view(19);f.c.prepare_with_ongoing_supply(v,v,0);ck(f.buys()==0&&f.c.ongoing_supply_preview_errors==1,"invalid optional preview fails closed to parent preparation");ck(!f.c.core.suppress_ongoing_fertilizer_supply,"exception cannot permanently disable live feature");}
 // Previous r1 calendar planting repair is retained, including negatives.
 {Controller c;c.p.finite_calendar_admission=true;c.p.fix_calendar=true;
  c.day=26;ck(c.planting_window(W),"r1 wheat day26");c.day=27;ck(c.planting_window(W)&&c.planting_window(C),"r1 wheat/carrot day27");
  c.day=28;ck(!c.planting_window(W)&&!c.planting_window(C),"r1 beyond terminal maturity negative");c.day=29;ck(!c.planting_window(W),"terminal new planting negative");
  c.p.finite_calendar_admission=false;c.day=27;ck(!c.planting_window(W),"legacy disabled calendar control");}
 std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"scope\":\"conditional execution admission, no-value/reachable/deadline/stock/cash/failure negatives, retained r1 calendar\"}\n";
}
