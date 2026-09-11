#include "policy/triad.hpp"
#include <cassert>
#include <iostream>

int main(){
 using namespace dp7;
 Farm own,rival;own.tiles.resize(100);rival.tiles.resize(100);
 own.farmer={4,4};rival.farmer={9,9};
 PrivateState priv;priv.inventories.resize(1);
 Market market;market.inventory.fill(10000);
 for(int i=0;i<9;i++)market.prices[i]=price(i,market.inventory[i]);
 std::vector<int8_t>shops;
 View view{241,10,1,own,rival,priv,market,shops};
 dp7::Controller core;core.p.triad_delivery_pressure=1.;
 Counts cargo{};cargo[MI]=10;
 assert(core.earlier_delivery_gain(view,cargo,250,250)==0);
 assert(core.earlier_delivery_gain(view,cargo,242,263)==0);
 auto&t=rival.tiles[44];t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.yield_units=20;
 double pressure=core.earlier_delivery_gain(view,cargo,242,263);
 assert(pressure>0);
 // With ample known consumption and no visible rival stock, delaying can
 // be better. A generic 'always rush to sell' must not pass this check.
 t.kind=TileKind::EMPTY;t.yield_units=0;shops={3,3,3};
 assert(core.earlier_delivery_gain(view,cargo,242,263)<0);
 // Immature crops are not currently harvestable rival output.
 shops.clear();t.kind=TileKind::PLANT;t.crop=Item::MELON;t.planted_day=10;t.yield_units=6;
 cargo={};cargo[M]=10;
 assert(core.earlier_delivery_gain(view,cargo,242,263)==0);
 // Shipping cannot steal wheat already required by this worker's FEED.
 core.day=10;core.phase=3;core.p.midroute_delivery=true;
 priv.inventories[0][MI]=10;priv.inventories[0][W]=1;
 core.plans.resize(1);core.plans[0].a.push_back(action(Op::FEED));core.plans[0].target.push_back(40);
 t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.yield_units=20;
 core.dispatch_midroute(view);
 assert(core.midroute_delivery_insertions==0);
 assert(core.plans[0].a.size()==1&&core.plans[0].a[0].op==Op::FEED);
 std::cout<<"{\"tests\":6,\"status\":\"PASS\",\"synthetic_early_sale_gain\":"<<pressure<<"}\n";
}
