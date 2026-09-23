#define R2_OBSERVE_PUBLIC_TRADES 1
#include "planner.hpp"
#include "public_trade_ledger.hpp"
#include <cassert>

static dp7::View view(int step,fastkag::Farm&own,fastkag::Farm&rival,
                      fastkag::PrivateState&priv,fastkag::Market&market,
                      std::vector<int8_t>&shops){
 return {step,step/24,step%24,own,rival,priv,market,shops};
}

int main(){
 fastkag::Farm own,rival;own.tiles.resize(100);rival.tiles.resize(100);
 rival.tiles[0].kind=fastkag::TileKind::ANIMAL;
 rival.tiles[0].animal=fastkag::Item::SHEEP;rival.tiles[0].yield_units=3;
 fastkag::PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);
 fastkag::Market market;for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
 std::vector<int8_t>shops;triad::PublicTradeLedger ledger;
 ledger.record(view(1,own,rival,priv,market,shops),{});
 rival.tiles[0].yield_units=1;market.inventory[dp7::WO]++;
 ledger.observe(view(2,own,rival,priv,market,shops));
 assert(ledger.certain_added[dp7::WO]==2&&ledger.rival_net[dp7::WO]==1);
 assert(ledger.lower[dp7::WO]==1&&ledger.upper[dp7::WO]==1);
}
