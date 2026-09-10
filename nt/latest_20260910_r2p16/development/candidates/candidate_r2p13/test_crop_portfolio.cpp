#define R2_FINITE_FERTILIZER 1
#define R2_CROP_PORTFOLIO_MODE 1
#include "policy/triad.hpp"
#include <cassert>
#include <iostream>
using namespace fastkag;
struct Fixture{
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;
 Fixture(){own.tiles.resize(100);rival.tiles.resize(100);own.money=3000;own.unlocked_mask=1;priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(10000);market.prices.fill(100);}
 dp7::View at(int day)const{return{day*24,day,0,own,rival,priv,market,shops};}
 void wheat(int pos){Tile x;x.kind=TileKind::PLANT;x.crop=Item::WHEAT;x.planted_day=0;x.yield_units=1;x.max_lifespan_step=120;own.tiles[pos]=x;}
};
void equal_asset(const competitive::Asset&a,const competitive::Asset&b){
 for(int d=0;d<30;d++){assert(std::abs(a.fixed[d]-b.fixed[d])<1e-8);assert(std::abs(a.labor[d]-b.labor[d])<1e-8);for(int k=0;k<9;k++)assert(std::abs(a.f[d][k]-b.f[d][k])<1e-8);}
}
int main(){int tests=0,changed=0;
 for(int stock:{9850,10000,10150})for(int old_finish:{2,3,4})for(int protected_successor:{0,1}){
  Fixture f;f.market.inventory.fill(stock);f.wheat(0);f.wheat(1);
  triad::Controller c;c.s.rotation=0;c.s.repeat=0;c.model.day=2;c.model.cfg.competition=1.;c.model.demand(f.at(2));c.model.public_rival(f.at(2));
  for(auto&day:c.prices)day.fill(80.);
  competitive::Asset other;other.f[2][competitive::E]=3;other.fixed[2]=-50;c.portfolio=other;
  for(int pos:{0,1}){c.paths[pos]=c.crop(0,0,pos,old_finish,c.prices,&f.own.tiles[pos]);competitive::add(c.portfolio,c.paths[pos]);c.book[pos]={0,0,2,old_finish,protected_successor?3:-1,true};c.core.triad_crop_age[pos]=old_finish;}
  c.model.value(f.at(2),c.portfolio,&c.prices);auto before=c.portfolio;auto value=c.model.value(f.at(2),before);std::vector<int>free=old_finish==2?std::vector<int>{0,1}:std::vector<int>{};
  c.reconcile_finite_crops(f.at(2),free);assert(c.model.value(f.at(2),c.portfolio)>=value-1e-6);
  if(protected_successor){equal_asset(before,c.portfolio);assert(c.crop_comparisons.empty()&&c.crop_compare_trials==0);}
  competitive::Asset expected=other;for(int pos:{0,1}){
   competitive::add(expected,c.paths[pos]);int finish=c.paths[pos].end;assert(finish>=2&&finish<=4);assert(c.core.triad_crop_age[pos]==finish);assert(c.book[pos].length==finish);
   assert(std::count(free.begin(),free.end(),pos)==int(finish==2));
  }equal_asset(expected,c.portfolio);
  for(auto&x:c.crop_comparisons){assert(x.gain>0&&x.new_finish>=2&&x.new_finish<=4);changed++;}
  tests++;
 }
 assert(changed>0);
 Fixture f;triad::Controller c;c.model.day=29;std::vector<int>free;c.reconcile_finite_crops(f.at(29),free);assert(c.crop_comparisons.empty());
 std::cout<<"PASS: "<<tests<<" portfolio cases; "<<changed<<" improvements; conserved flow, legal timing, protected successors, empty endgame\n";
}
