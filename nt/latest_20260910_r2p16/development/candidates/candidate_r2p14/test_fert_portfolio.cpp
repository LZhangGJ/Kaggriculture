#define R2_FINITE_FERTILIZER 1
#define R2_FERT_PORTFOLIO_MODE 1
#include "policy/triad.hpp"
#include <cassert>
#include <iostream>
using namespace fastkag;
struct Fixture{
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;
 Fixture(){own.tiles.resize(100);rival.tiles.resize(100);own.money=3000;own.unlocked_mask=1;priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(10000);market.prices.fill(100);}
 dp7::View at(int d)const{return{d*24,d,0,own,rival,priv,market,shops};}
 void plant_at(int pos,int k){Tile x;x.kind=TileKind::PLANT;x.crop=Item(k);x.planted_day=0;x.yield_units=1;x.max_lifespan_step=31*24;own.tiles[pos]=x;}
};
void equal_asset(const competitive::Asset&a,const competitive::Asset&b){for(int d=0;d<30;d++){assert(std::abs(a.fixed[d]-b.fixed[d])<1e-8);assert(std::abs(a.labor[d]-b.labor[d])<1e-8);for(int k=0;k<9;k++)assert(std::abs(a.f[d][k]-b.f[d][k])<1e-8);}}
int main(){int changes=0,tests=0;
 for(int k:{0,1,4})for(bool protected_successor:{false,true})for(bool active_fertilizer:{false,true}){
  int day=k==4?6:2;Fixture f;f.plant_at(0,k);f.plant_at(1,k);f.market.inventory[k]=13000;f.market.inventory[8]=8500;
  triad::Controller c;c.s.crop_fert=1;c.model.day=day;c.model.demand(f.at(day));c.model.public_rival(f.at(day));c.successor.fill(-1);
  for(auto&px:c.prices){px.fill(1000.);px[8]=1.;}
  competitive::Asset other;other.f[day][8]=3;other.f[day][5]=2;c.portfolio=other;
  for(int pos:{0,1}){
   if(active_fertilizer)f.own.tiles[pos].fertilized_until_day=day+2;
   c.paths[pos]=c.crop(k,0,pos,day,c.prices,&f.own.tiles[pos]);competitive::add(c.portfolio,c.paths[pos]);c.finite_first_fertilizer[pos]=c.paths[pos].f[day][8]<0;c.core.triad_crop_age[pos]=day;c.release[pos]=day;c.book[pos]={k,0,day,day,-1,true};
   if(protected_successor)c.successor[pos]=3;
  }
  auto before=c.portfolio;double value=c.model.value(f.at(day),before);
  c.reconcile_fertilizer(f.at(day));assert(c.model.value(f.at(day),c.portfolio)>=value-1e-6);
  if(protected_successor||active_fertilizer){assert(c.fert_comparisons.empty());equal_asset(c.portfolio,before);}
  competitive::Asset expected=other;for(int pos:{0,1}){
   competitive::add(expected,c.paths[pos]);assert(c.paths[pos].end==day&&c.release[pos]==day&&c.core.triad_crop_age[pos]==day&&c.book[pos].length==day);
   if(!c.finite_first_fertilizer[pos])assert(c.paths[pos].f[day][8]==0);
  }equal_asset(expected,c.portfolio);
  for(auto&x:c.fert_comparisons){assert(x.gain>0&&x.finish==day);changes++;}tests++;
 }
 assert(changes>0);std::cout<<"PASS: "<<tests<<" scenarios; "<<changes<<" skipped fertilizer choices; same finish/date/book, no duplicate flow, active fertilizer protected\n";
}
