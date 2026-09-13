#include "policy/search.hpp"
#include <cassert>
#include <iostream>
using namespace triad;
int main(){
 for(int scope=0;scope<=2;scope++){
  Settings s;s.a06_reinvest=4;s.a06_roll_scope=scope;
  triad::Controller live(s);assert(!live.prediction());live.a06_received_today=100;
  assert(live.effective_reinvest()==4&&live.reinvest_gate());
  triad::Controller roll=live;roll.call_origin=triad::Controller::CallOrigin::PublicPrediction;
  assert(roll.prediction());assert(roll.effective_reinvest()==(scope==1?0:scope==2?1:4));
  assert(roll.reinvest_gate()==(scope==0));
  Settings common=s;common.scenario=0;roll.configure(common);
  assert(roll.prediction());assert(roll.reinvest_gate()==(scope==0));
  assert(!live.prediction()&&live.reinvest_gate());
  live.a06_received_today=0;assert(!live.reinvest_gate());
 }
 Farm own,other;own.tiles.resize(100);other.tiles.resize(100);
 own.farmer={4,4};other.farmer={4,4};own.money=1000;
 for(int p=0;p<100;p++)if(quad(p)!=0){own.tiles[p].kind=TileKind::LOCKED;other.tiles[p].kind=TileKind::LOCKED;}
 PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);priv.shed[W]=1;priv.shed[CO]=1;
 Market m;for(int i=0;i<9;i++){m.inventory[i]=10000;m.prices[i]=price(i,10000);}std::vector<int8_t> shops;
 View v{24,1,0,own,other,priv,m,shops};
 for(int scope=0;scope<=2;scope++)for(int simulated=0;simulated<=1;simulated++){
  Settings s;s.a06_reinvest=4;s.a06_roll_scope=scope;s.scenario=0;
  triad::Controller c(s);c.core.day=1;c.model.day=1;c.core.phase=3;c.core.plans.resize(1);
  if(simulated)c.call_origin=triad::Controller::CallOrigin::PublicPrediction;
  c.core.pending_admission={true,1,24,0,34,CO};
  PlayerAction a;a.units.resize(1);c.settle_market(v,a,false);
  int sold=0;for(auto x:a.market)if(x.op==Op::SELL&&x.item==Item::WHEAT)sold+=x.quantity;
  assert(sold==((simulated&&scope==1)?1:0));
 }
 // Secondary feed-cover sale reservation. This is synthetic, not game evidence.
 for(int fed:{0,4,16})for(int hour:{2,23})for(int enabled:{0,1}){
  own.unlocked_mask=15;for(auto&t:own.tiles)t=Tile{};
  for(int n=0;n<16;n++){auto&t=own.tiles[n];t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.fed_today=n<fed;}
  int bag=fed==4?12:0,stock=32-fed-bag;
  priv.shed={};priv.shed[W]=stock;priv.inventories[0]={};priv.inventories[0][W]=bag;priv.inventory_order[0].clear();if(bag)priv.inventory_order[0].push_back(W);
  Settings s;s.a06_reinvest=4;s.a06_feed_contract=enabled;
  triad::Controller c(s);c.core.day=1;c.model.day=1;c.core.phase=3;c.core.plans.resize(1);
  c.core.feed_stock_target=32;c.core.daily_need[W]=16;
  View q{24+hour,1,hour,own,other,priv,m,shops};PlayerAction a;a.units.resize(1);c.settle_market(q,a,false);
  int sold=0;for(auto x:a.market)if(x.op==Op::SELL&&x.item==Item::WHEAT)sold+=x.quantity;
  int legacy_reserve=hour<23?std::max(0,16-bag):0;
  int expected=enabled?0:std::max(0,stock-legacy_reserve);
  assert(sold==expected);
 }
 std::cout<<"PASS 12 synthetic feed-cover holdings cases (current feed + future cover, bag offset, fed offset)\n";
 std::cout<<"PASS real/prediction origin scope 0/1/2, configure persistence, real receipt gate, pending retention attribution (6 cases)\n";
}
