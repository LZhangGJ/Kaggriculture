#pragma once
#include "triad.hpp"
namespace triad {
// A public initial-state prior, never a recovered opponent program or inventory.
inline competitive::Flow startup_supply_prior(const dp7::View&o,Settings base){
 competitive::Flow out{};
 if(o.step!=0||o.day!=0||o.hour!=0)return out;
 for(const auto&t:o.opponent.tiles)if(dp7::plant(t)||dp7::animal(t))return out;
 fastkag::PrivateState initial;
 initial.inventories.resize(o.opponent.hands.size()+1);
 initial.inventory_order.resize(o.opponent.hands.size()+1);
 dp7::View mirror{0,0,0,o.opponent,o.own,initial,o.market,o.shops};
 base.scenario=0;Controller proxy(base);proxy.plan(mirror);
 return proxy.portfolio.f;
}
}
