#include "policy/observed_crop_clock.hpp"
#include <cassert>
#include <iostream>
using namespace fastkag;
struct Fixture{
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;
 Fixture(){own.tiles.resize(100);rival.tiles.resize(100);priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(10000);market.prices.fill(100);}
 dp7::View at(int t)const{return{t,t/24,t%24,own,rival,priv,market,shops};}
 void wheat(int t,int age){Tile x;x.kind=TileKind::PLANT;x.crop=Item::WHEAT;x.planted_day=t/24-age;x.yield_units=1;x.max_lifespan_step=(x.planted_day+5)*24;rival.tiles[0]=x;}
};
int main(){
 Fixture f;triad::ObservedCropClock c;
 for(int n=0;n<3;n++){int t=48+2*n;f.wheat(t,2);c.observe(f.at(t));f.rival.tiles[0]={};c.observe(f.at(t+1));c.observe(f.at(t+1));}
 assert(c.count[0]==3&&c.weights()[0][2]==3&&c.weights()[0][4]==1);
 f.own.tiles[0].kind=TileKind::WEED;c.observe(f.at(54));assert(c.count[0]==3);
 c.observe(f.at(0));assert(c.count[0]==0); // new episode resets history
 for(int n=0;n<64;n++){int t=96+2*n;f.wheat(t,n<32?2:3);c.observe(f.at(t));f.rival.tiles[0]={};c.observe(f.at(t+1));}
 assert(c.count[0]==32&&c.weights()[0][2]==0&&c.weights()[0][3]==32);
 c.observe(f.at(500));assert(c.count[0]==0); // missing observation is not a turnover
 for(int which=0;which<3;which++){
  triad::ObservedCropClock a;Fixture g;g.wheat(71,2);
  if(which==0){g.rival.tiles[0].consecutive_unwatered=1;}
  if(which==1){g.rival.tiles[0].max_lifespan_step=72;}
  a.observe(g.at(71));g.rival.tiles[0]={};if(which==2)g.rival.tiles[0].kind=TileKind::WEED;
  a.observe(g.at(72));assert(a.count[0]==0);
 }
 Fixture g;competitive::Planner p;p.day=2;p.cfg.replant=0;
 p.rival_harvest_weights[0][2]=3;p.rival_harvest_weights[0][4]=1;
 g.wheat(48,2);p.demand(g.at(48));p.public_rival(g.at(48));
 assert(std::abs(p.rival[2][0]-1.5)<1e-9&&std::abs(p.rival[4][0]-1.)<1e-9);
 p.day=3;g.wheat(72,3);p.demand(g.at(72));p.public_rival(g.at(72));
 assert(p.rival[3][0]==0&&p.rival[4][0]==3); // past age2 mass discarded
 assert(p.core.p.crop_harvest_age[0]==4); // no own-side calendar mutation
 std::cout<<"PASS: duplicate/reset/window/exclusions/conditional mixture\n";
}
