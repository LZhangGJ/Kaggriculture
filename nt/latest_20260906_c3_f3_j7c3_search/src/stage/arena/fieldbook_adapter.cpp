#include "fieldbook_adapter.hpp"

// Compile the frozen public policy unchanged, with private symbol names. Do not
// link submission_bridge.cpp's process-global Session into the batch runtime.
#define kag_policy_abi_version dp7_fieldbook_abi_version
#define kag_policy_create dp7_fieldbook_create
#define kag_policy_destroy dp7_fieldbook_destroy
#define kag_policy_act dp7_fieldbook_act
#include "../opponents/yhay81_six_day/output/sixday_r4_source/policy.cpp"
#undef kag_policy_abi_version
#undef kag_policy_create
#undef kag_policy_destroy
#undef kag_policy_act

namespace fieldbook {
namespace {
kag::State project(const dp7::View& o, int seat) {
  if(seat<0 || seat>1) throw std::invalid_argument("Fieldbook seat");
  kag::State s{};
  s.step=o.step; s.day=o.day; s.hour=o.hour;
  s.n_shops=std::min(int(o.shops.size()),kag::MAX_SHOP_INSTANCES);
  for(int i=0;i<s.n_shops;i++) s.shops[i]=o.shops[i];
  for(int i=0;i<kag::N_PRODUCTS;i++) {
    s.market.inventory[i]=o.market.inventory[i];
    s.market.prices[i]=o.market.prices[i];
  }
  for(int p=0;p<2;p++) {
    const auto& f=p==seat?o.own:o.opponent;
    auto& dst=s.farms[p];
    dst.money=f.money;
    dst.n_units=std::min(1+int(f.hands.size()),kag::MAX_UNITS);
    dst.n_quadrants=std::popcount(unsigned(f.unlocked_mask));
    dst.hires_today=f.hires_today;
    dst.pos_x[0]=f.farmer.x; dst.pos_y[0]=f.farmer.y;
    for(int u=1;u<dst.n_units;u++) {dst.pos_x[u]=f.hands[u-1].x;dst.pos_y[u]=f.hands[u-1].y;}
    for(int i=0;i<100;i++) {
      const auto& t=f.tiles.at(i);auto& d=dst.tiles[i/10][i%10];
      d.max_lifespan_step=-1;d.fertilized_until_day=-1;
      switch(t.kind) {
        case fastkag::TileKind::EMPTY:d.kind=kag::T_EMPTY;break;
        case fastkag::TileKind::LOCKED:d.kind=kag::T_LOCKED;break;
        case fastkag::TileKind::WEED:d.kind=kag::T_WEED;break;
        case fastkag::TileKind::COOP:d.kind=kag::T_COOP;break;
        case fastkag::TileKind::PASTURE:d.kind=kag::T_PASTURE;break;
        case fastkag::TileKind::PLANT:
          d.kind=kag::T_PLANT;d.what=uint8_t(t.crop);d.watered_today=t.watered_today;
          d.consecutive_dry=int8_t(t.consecutive_unwatered);d.yield_units=int8_t(t.yield_units);
          d.planted_day=t.planted_day;d.max_lifespan_step=t.max_lifespan_step;
          d.fertilized_until_day=t.fertilized_until_day;break;
        case fastkag::TileKind::ANIMAL:
          d.kind=t.animal==fastkag::Item::GOOSE?kag::T_COOP:kag::T_PASTURE;
          d.what=uint8_t(t.animal);d.has_animal=true;d.fed_today=t.fed_today;
          d.cared_today=t.cared_today;d.fertilizer_available=t.fertilizer_available;
          d.consecutive_dry=int8_t(t.consecutive_unfed);d.yield_units=int8_t(t.yield_units);
          d.pending_care_bonus=int8_t(t.pending_care_bonus);d.planted_day=t.placed_day;break;
      }
    }
  }
  // Only the acting player's private data exists in a legal observation.
  // The other farm's seeds, shed, carries and instrumentation stay zero.
  auto& own=s.farms[seat];
  for(int i=0;i<kag::N_ITEMS;i++) {own.shed[i]=int16_t(o.priv.shed[i]);own.shed_total+=own.shed[i];}
  for(int i=0;i<kag::N_CROPS;i++) own.seeds[i]=int16_t(o.priv.seeds[i]);
  for(int u=0;u<std::min(int(o.priv.inventories.size()),kag::MAX_UNITS);u++)
    for(int i=0;i<kag::N_ITEMS;i++) own.inv[u][i]=int16_t(o.priv.inventories[u][i]);
  // Original submission bridge does not reconstruct inventory insertion order;
  // preserve its input semantics rather than "improving" the frozen opponent.
  return s;
}
fastkag::PlayerAction convert(const kag::Action& src) {
  using fastkag::Op;using fastkag::Item;
  static constexpr Op units[]{Op::PASS,Op::NORTH,Op::SOUTH,Op::EAST,Op::WEST,Op::PICKUP,Op::DROP,Op::PLACE,Op::PLANT,Op::WATER,Op::HARVEST,Op::FERTILIZE,Op::DIG,Op::BUILD_COOP,Op::BUILD_PASTURE,Op::FEED,Op::COLLECT_FERTILIZER,Op::CARE};
  static constexpr Op market[]{Op::PASS,Op::HIRE,Op::BUY_LAND,Op::BUY_SEED,Op::BUY_PRODUCT,Op::BUY_ANIMAL,Op::SELL};
  fastkag::PlayerAction result;
  for(int u=0;u<std::clamp(src.n_units,1,kag::MAX_UNITS);u++) {
    auto raw=src.units[u];Op op=raw.op<std::size(units)?units[raw.op]:Op::PASS;
    Item item=Item::NONE;int q=1;
    if(op==Op::PICKUP||op==Op::PLACE||op==Op::PLANT) {item=Item(raw.arg<12?raw.arg:0);q=raw.n;}
    result.units.push_back({op,item,q});
  }
  for(int i=0;i<std::clamp(src.n_orders,0,16);i++) {
    auto raw=src.orders[i];Op op=raw.op<std::size(market)?market[raw.op]:Op::PASS;
    if(op==Op::PASS) continue;
    bool item_order=op!=Op::HIRE&&op!=Op::BUY_LAND;
    result.market.push_back({op,item_order?Item(raw.item<12?raw.item:0):Item::NONE,item_order?raw.n:1});
  }
  return result;
}
}

struct Controller::Impl {
  void* context=nullptr;int last=-1;int seat=-1;
  ~Impl(){if(context)dp7_fieldbook_destroy(context);}
  void reset(int player) {
    if(context)dp7_fieldbook_destroy(context);
    context=dp7_fieldbook_create();last=-1;seat=player;
    if(!context)throw std::runtime_error("Fieldbook context allocation");
  }
};
Controller::Controller():impl_(std::make_unique<Impl>()) {
  if(dp7_fieldbook_abi_version()!=1)throw std::runtime_error("Fieldbook ABI");
}
Controller::~Controller()=default;
Controller::Controller(const Controller&other):Controller(){
  impl_->last=other.impl_->last;impl_->seat=other.impl_->seat;
  // Frozen Context consists solely of value arrays and selected indices.
  if(other.impl_->context)impl_->context=new Context(*static_cast<const Context*>(other.impl_->context));
}
fastkag::PlayerAction Controller::act(const dp7::View& o,int seat) {
  auto& c=*impl_;
  if(!c.context||o.step==0||o.step<c.last||seat!=c.seat)c.reset(seat);
  auto state=project(o,seat);kag::Config config{};kag::Action out{};
  if(dp7_fieldbook_act(c.context,&state,&config,seat,&out)!=0)throw std::runtime_error("Fieldbook policy failure");
  c.last=o.step;return convert(out);
}
std::array<int,5> Controller::segments() const {
  std::array<int,5> result{-1,-1,-1,-1,-1};
  if(impl_->context)std::copy_n(static_cast<const Context*>(impl_->context)->selected_,5,result.begin());
  return result;
}
std::vector<double> input_fields(const dp7::View& o,int seat) {
  auto s=project(o,seat);std::vector<double> out;
  out.reserve(6000);auto push=[&](double x){out.push_back(x);};
  push(s.step);push(s.day);push(s.hour);push(s.n_shops);
  for(auto n:s.market.inventory)push(n);for(auto n:s.market.prices)push(n);for(auto n:s.shops)push(n);
  for(auto& f:s.farms) {
    push(f.money);
    for(auto& row:f.tiles)for(auto& t:row) {
      push(t.kind);push(t.what);push(t.has_animal|t.watered_today<<1|t.fed_today<<2|t.cared_today<<3|t.fertilizer_available<<4);
      push(t.consecutive_dry);push(t.yield_units);push(t.pending_care_bonus);push(t.planted_day);push(t.max_lifespan_step);push(t.fertilized_until_day);
    }
    for(auto n:f.pos_x)push(n);for(auto n:f.pos_y)push(n);push(f.n_units);push(f.n_quadrants);push(f.hires_today);
    for(auto n:f.shed)push(n);for(auto n:f.seeds)push(n);for(auto& row:f.inv)for(auto n:row)push(n);
  }
  return out;
}
}
