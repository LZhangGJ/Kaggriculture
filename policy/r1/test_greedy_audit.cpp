#define R2_OPTIMIZER_AUDIT 1
#include "triad.hpp"
#include <cassert>
#include <cmath>

int main(){
 triad::Settings s;s.max_land=3;s.max_animals=40;s.preview=1;s.rotation=0;s.repeat=0;
 triad::Controller c(s);
 fastkag::Farm own,rival;own.money=5000;own.unlocked_mask=7;own.tiles.resize(100);rival.tiles.resize(100);
 for(int pos=75;pos<100;pos++)own.tiles[pos].kind=fastkag::TileKind::LOCKED;
 fastkag::PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);
 fastkag::Market market;market.inventory.fill(10000);for(int i=0;i<9;i++)market.prices[i]=dp7::price(i,10000);
 std::vector<int8_t>shops;dp7::View view{288,12,0,own,rival,priv,market,shops};
 c.plan(view);
 assert(!c.greedy_audit.empty());
 assert(c.greedy_budget_initial>=c.greedy_budget_final);
 for(const auto&a:c.greedy_audit){
  assert(a.candidates>0);
  assert(a.raw_gain+1e-9>=a.picked_gain);
  assert(a.raw_value_gain+1e-9>=a.picked_value_gain);
  if(a.picked_kind>=0)assert(a.picked_rank+1e-9>=a.raw_rank);
  if constexpr(R2_GREEDY_VALUE_ORDER)assert(a.picked_kind==a.value_kind);
 }
#if R2_OUTER_NEIGHBOR_AUDIT
 auto slot=c.greedy_audit.front();triad::Controller forced(s);forced.forced_kind[slot.pos]=slot.picked_kind;forced.plan(view);
 auto target=std::find_if(forced.core.target.begin(),forced.core.target.end(),[&](auto x){return x.first==slot.pos;});
 assert(target!=forced.core.target.end()&&target->second==slot.picked_kind);
#endif
 assert(c.optimizer_json().find("local_regret")!=std::string::npos);
}
