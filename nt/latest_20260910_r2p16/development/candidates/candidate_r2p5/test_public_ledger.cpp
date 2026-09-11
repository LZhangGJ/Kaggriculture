#include "policy/search.hpp"
#include <cassert>
int main(){
 using namespace fastkag;using namespace dp7;using triad::PublicTradeLedger;
 Farm own{},rival{};own.tiles.resize(100);rival.tiles.resize(100);
 PrivateState priv{};Market market{};market.inventory.fill(10000);std::vector<int8_t>shops;
 auto base=[&](){PublicTradeLedger l;l.previous.step=1;l.previous.rival=rival;l.previous.market=market;return l;};
 auto v=[&](int step){return View{step,step/24,step%24,own,rival,priv,market,shops};};
 {auto c=PublicTradeLedger::consumption(24,{7,7,4});assert(c[WO]==5&&c[C]==3&&c[MI]==1&&c[F]==0);}
 {auto l=base();auto&t=l.previous.rival.tiles[22];t.kind=TileKind::ANIMAL;t.animal=Item::SHEEP;t.yield_units=6;
  rival.tiles[22]=t;rival.tiles[22].yield_units=2;market.inventory[WO]+=3;l.observe(v(2));
  assert(l.valid[WO]&&l.rival_net[WO]==3&&l.upper[WO]==1);market.inventory[WO]-=3;rival.tiles[22]={};}
 {auto l=base();auto&t=l.previous.rival.tiles[22];t.kind=TileKind::ANIMAL;t.animal=Item::SHEEP;t.yield_units=6;
  rival.tiles[22]=t;rival.tiles[22].yield_units=2;l.previous.own_after_units[WO]=10;priv.shed[WO]=7;
  market.inventory[WO]+=5;l.observe(v(2));assert(l.rival_net[WO]==2&&l.upper[WO]==2);
  market.inventory[WO]-=5;priv.shed[WO]=0;rival.tiles[22]={};}
 {auto l=base();auto&t=l.previous.rival.tiles[22];t.kind=TileKind::PLANT;t.crop=Item::MELON;t.planted_day=-10;t.yield_units=4;
  assert(PublicTradeLedger::available_upper(t,0)==6);market.inventory[M]+=6;l.observe(v(2));assert(l.upper[M]==0&&l.rival_net[M]==6);market.inventory[M]-=6;}
 {auto l=base();auto&t=l.previous.rival.tiles[22];t.kind=TileKind::PLANT;t.crop=Item::MELON;t.planted_day=-10;t.yield_units=4;
  l.observe(v(2));assert(l.upper[M]==6);/* DIG is only potential stock, not a fact. */}
 {auto l=base();l.previous.step=23;l.upper[WO]=98;auto&t=l.previous.rival.tiles[22];t.kind=TileKind::ANIMAL;t.animal=Item::SHEEP;t.yield_units=6;
  l.observe(v(24));assert(!l.valid[WO]&&l.upper[WO]==100);}
 {auto l=base();market.inventory[WO]=100000;l.observe(v(2));assert(!l.valid[WO]&&l.skipped_floor>0);market.inventory[WO]=10000;}
 {auto l=base();market.inventory[WO]-=1;l.observe(v(2));assert(!l.valid[WO]&&l.invalid_market==1);market.inventory[WO]+=1;}
}
