#include "three_day_adapter.hpp"
#include <bit>
#include <sstream>
#include <stdexcept>

// The two public agents use different kag::State layouts. Isolate *all* C++
// symbols, not just the four C exports, to prevent silent ODR/ABI collisions.
// Public files are included unchanged. Its global submission Session is NOT
// linked; each match owns one original Context.
#define kag dp7_threeday_kag
#define kag_policy_abi_version dp7_threeday_abi_version
#define kag_policy_create dp7_threeday_create
#define kag_policy_destroy dp7_threeday_destroy
#define kag_policy_act dp7_threeday_act
#include "../opponents/yhay81_three_day/output/source/policy.cpp"
#undef kag_policy_abi_version
#undef kag_policy_create
#undef kag_policy_destroy
#undef kag_policy_act
#undef kag

namespace threeday {
namespace {
namespace k=dp7_threeday_kag;
k::State project(const dp7::View&o,int seat){
  if(seat<0||seat>1)throw std::invalid_argument("Three-Day seat");
  k::State s{};s.step=o.step;s.n_shops=std::min(int(o.shops.size()),k::MAX_SHOP_INSTANCES);
  for(int i=0;i<s.n_shops;i++)s.shops[i]=o.shops[i];
  for(int i=0;i<9;i++){s.market.inventory[i]=o.market.inventory[i];s.market.prices[i]=o.market.prices[i];}
  for(int p=0;p<2;p++){
    auto&f=p==seat?o.own:o.opponent;auto&d=s.farms[p];d.money=f.money;
    d.n_units=std::clamp(1+int(f.hands.size()),1,k::MAX_UNITS);d.n_quadrants=std::popcount(unsigned(f.unlocked_mask));d.hires_today=f.hires_today;
    for(int i=0;i<100;i++){
      auto&t=f.tiles.at(i);auto&tile=d.tiles[i/10][i%10];
      switch(t.kind){
        case fastkag::TileKind::EMPTY:tile.kind=k::T_EMPTY;break;
        case fastkag::TileKind::LOCKED:tile.kind=k::T_LOCKED;break;
        case fastkag::TileKind::WEED:tile.kind=k::T_WEED;break;
        case fastkag::TileKind::COOP:tile.kind=k::T_COOP;break;
        case fastkag::TileKind::PASTURE:tile.kind=k::T_PASTURE;break;
        case fastkag::TileKind::PLANT:tile.kind=k::T_PLANT;break;
        case fastkag::TileKind::ANIMAL:tile.kind=t.animal==fastkag::Item::GOOSE?k::T_COOP:k::T_PASTURE;break;
      }
    }
  }
  // Source Python exposes only own shed/carries. Rival private fields stay 0.
  auto&d=s.farms[seat];for(int i=0;i<12;i++)d.shed[i]=int16_t(o.priv.shed[i]);
  for(int u=0;u<std::min(int(o.priv.inventories.size()),k::MAX_UNITS);u++)for(int i=0;i<12;i++)d.inv[u][i]=int16_t(o.priv.inventories[u][i]);
  return s;
}
fastkag::PlayerAction convert(const k::Action&a){
  using fastkag::Op;using fastkag::Item;
  constexpr Op units[]{Op::PASS,Op::NORTH,Op::SOUTH,Op::EAST,Op::WEST,Op::PICKUP,Op::DROP,Op::PLACE,Op::PLANT,Op::WATER,Op::HARVEST,Op::FERTILIZE,Op::DIG,Op::BUILD_COOP,Op::BUILD_PASTURE,Op::FEED,Op::COLLECT_FERTILIZER,Op::CARE};
  constexpr Op market[]{Op::PASS,Op::HIRE,Op::BUY_LAND,Op::BUY_SEED,Op::BUY_PRODUCT,Op::BUY_ANIMAL,Op::SELL};
  fastkag::PlayerAction out;
  for(int u=0;u<std::clamp(a.n_units,1,k::MAX_UNITS);u++){
    auto x=a.units[u];Op op=x.op<std::size(units)?units[x.op]:Op::PASS;bool has_item=op==Op::PICKUP||op==Op::PLACE||op==Op::PLANT;
    out.units.push_back({op,has_item?Item(x.arg<12?x.arg:0):Item::NONE,has_item?x.n:1});
  }
  for(int i=0;i<std::clamp(a.n_orders,0,16);i++){
    auto x=a.orders[i];Op op=x.op<std::size(market)?market[x.op]:Op::PASS;if(op==Op::PASS)continue;
    bool has_item=op!=Op::HIRE&&op!=Op::BUY_LAND;out.market.push_back({op,has_item?Item(x.item<12?x.item:0):Item::NONE,has_item?x.n:1});
  }return out;
}
}
struct Controller::Impl{
  void*context=nullptr;int last=-1,seat=-1;
  ~Impl(){if(context)dp7_threeday_destroy(context);}
  void reset(int p){if(context)dp7_threeday_destroy(context);context=dp7_threeday_create();seat=p;last=-1;if(!context)throw std::runtime_error("Three-Day allocation");}
};
Controller::Controller():impl_(std::make_unique<Impl>()){if(dp7_threeday_abi_version()!=1)throw std::runtime_error("Three-Day ABI");}
Controller::~Controller()=default;
Controller::Controller(const Controller&other):Controller(){
  impl_->last=other.impl_->last;impl_->seat=other.impl_->seat;
  if(other.impl_->context)impl_->context=new Context(*static_cast<const Context*>(other.impl_->context));
}
fastkag::PlayerAction Controller::act(const dp7::View&o,int seat){
  auto&c=*impl_;if(!c.context||o.step==0||o.step<c.last||seat!=c.seat)c.reset(seat);
  auto s=project(o,seat);k::Config config{};k::Action out{};
  if(dp7_threeday_act(c.context,&s,&config,seat,&out))throw std::runtime_error("Three-Day policy failure");
  c.last=o.step;return convert(out);
}
int Controller::route()const{return impl_->context?static_cast<const Context*>(impl_->context)->selected_route:-1;}
std::vector<double> input_fields(const dp7::View&o,int seat){
  auto s=project(o,seat);std::vector<double>out;auto push=[&](double x){out.push_back(x);};
  push(s.step);push(s.n_shops);for(auto n:s.market.inventory)push(n);for(auto n:s.market.prices)push(n);for(auto n:s.shops)push(n);
  for(auto&f:s.farms){push(f.money);for(auto&row:f.tiles)for(auto&t:row)push(t.kind);push(f.n_units);push(f.n_quadrants);push(f.hires_today);for(auto n:f.shed)push(n);for(auto&row:f.inv)for(auto n:row)push(n);}
  return out;
}
}
