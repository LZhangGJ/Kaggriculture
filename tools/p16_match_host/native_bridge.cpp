#include "simulator.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
namespace py=pybind11;
using namespace fastkag;
constexpr const char* ops[]{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
Action parse(py::handle h){Action a;auto s=py::reinterpret_borrow<py::sequence>(h);if(!s.size())return a;std::string op=py::cast<std::string>(s[0]);for(int i=0;i<24;i++)if(op==ops[i])a.op=Op(i);if(s.size()>1&&py::isinstance<py::str>(s[1])){std::string n=py::cast<std::string>(s[1]);for(int i=0;i<12;i++)if(n==item_name(i))a.item=Item(i);}if(s.size()>2)a.quantity=py::cast<int>(py::module_::import("builtins").attr("int")(s[2]));return a;}
PlayerAction parse_player(py::handle h){auto d=py::reinterpret_borrow<py::dict>(h);PlayerAction a;a.units.push_back(d.contains("farmer")?parse(d["farmer"]):Action{});if(d.contains("hands"))for(auto x:d["hands"])a.units.push_back(parse(x));if(d.contains("market"))for(auto x:d["market"])a.market.push_back(parse(x));return a;}
std::vector<PlayerAction>tape(py::handle h){std::vector<PlayerAction>x;for(auto a:py::reinterpret_borrow<py::sequence>(h))x.push_back(parse_player(a));return x;}
py::list pack(const Action&a){py::list x;x.append(ops[int(a.op)]);if(a.item!=Item::NONE){x.append(item_name(int(a.item)));if(a.op==Op::PLACE||a.op==Op::PICKUP||a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL||a.op==Op::BUY_PRODUCT||a.op==Op::SELL)x.append(a.quantity);}else if(a.quantity!=1)x.append(a.quantity);return x;}
py::dict pack_player(const PlayerAction&a){py::dict d;d["farmer"]=pack(a.units.empty()?Action{}:a.units[0]);py::list hands,market;for(size_t u=1;u<a.units.size();u++)hands.append(pack(a.units[u]));for(auto x:a.market)market.append(pack(x));d["hands"]=hands;d["market"]=market;return d;}
py::object tile(const Tile&t){if(t.kind==TileKind::EMPTY)return py::none();if(t.kind==TileKind::LOCKED)return py::str("LOCKED");py::dict d;if(t.kind==TileKind::WEED){d["kind"]="WEED";return d;}if(t.kind==TileKind::COOP){d["kind"]="COOP";return d;}if(t.kind==TileKind::PASTURE){d["kind"]="PASTURE";return d;}if(t.kind==TileKind::PLANT){d["kind"]="PLANT";d["crop"]=item_name(int(t.crop));d["planted_day"]=t.planted_day;d["watered_today"]=t.watered_today;d["consecutive_unwatered"]=t.consecutive_unwatered;d["yield_units"]=t.yield_units;d["max_lifespan_step"]=t.max_lifespan_step;d["fertilized_until_day"]=t.fertilized_until_day;return d;}d["kind"]=t.animal==Item::GOOSE?"COOP":"PASTURE";d["animal"]=item_name(int(t.animal));d["placed_day"]=t.placed_day;d["yield_units"]=t.yield_units;d["consecutive_unfed"]=t.consecutive_unfed;d["fed_today"]=t.fed_today;d["cared_today"]=t.cared_today;d["fertilizer_available"]=t.fertilizer_available;d["pending_care_bonus"]=t.pending_care_bonus;return d;}
py::dict observation(const Simulator&s,int seat){py::dict o;o["remainingOverageTime"]=60;o["player"]=seat;o["step"]=s.step_count();o["day"]=s.day();o["hour"]=s.hour();py::list farms;
 for(auto&f:s.farms()){py::dict d;d["money"]=f.money;d["farmer"]=py::make_tuple(f.farmer.x,f.farmer.y);py::list hands,tiles,quads;for(auto p:f.hands)hands.append(py::make_tuple(p.x,p.y));for(int y=0;y<10;y++){py::list row;for(int x=0;x<10;x++)row.append(tile(f.tiles[y*10+x]));tiles.append(row);}const char*q[]{"NW","NE","SW","SE"};for(int i=0;i<4;i++)if(f.unlocked_mask&(1<<i))quads.append(q[i]);d["hands"]=hands;d["tiles"]=tiles;d["unlocked_quadrants"]=quads;d["hires_today"]=f.hires_today;farms.append(d);}o["farms"]=farms;
 auto&p=s.privates()[seat];py::dict priv,shed,seeds;for(int i=0;i<12;i++)shed[item_name(i)]=p.shed[i];for(int i=0;i<5;i++)seeds[item_name(i)]=p.seeds[i];py::list inv;for(size_t u=0;u<p.inventories.size();u++){py::dict d;for(int i:p.inventory_order[u])if(p.inventories[u][i])d[item_name(i)]=p.inventories[u][i];inv.append(d);}priv["shed"]=shed;priv["seeds"]=seeds;priv["inventories"]=inv;o["private"]=priv;py::dict market,mi,mp,town;for(int i=0;i<9;i++){mi[item_name(i)]=s.market().inventory[i];mp[item_name(i)]=s.market().prices[i];}market["inventory"]=mi;market["prices"]=mp;o["market"]=market;py::list shops;for(int sh:s.shops())shops.append(shop_name(sh));town["unlocked_shops"]=shops;o["town"]=town;return o;
}
// Audit projections are discarded. Only the existing compiled step advances play.
py::dict advance(Simulator& s, py::sequence raw, int seat) {
  std::array<PlayerAction,2> acts{parse_player(raw[0]),parse_player(raw[1])};
  const auto& units=acts[seat].units;
  py::list completed, invalid;
  int moves=0, idle=0, pickups=0, drops=0, production=0, daytime_overflow=0;
  int hires_before=s.farms()[seat].hires_today;
  auto before=s.project_unit_phase(seat,units,0);
  for(size_t u=0;u<std::min(units.size(),s.farms()[seat].hands.size()+1);u++) {
    const auto& a=units[u];
    auto after=s.project_unit_phase(seat,units,u+1);
    auto pos=u?before.farms()[seat].hands[u-1]:before.farms()[seat].farmer;
    int cell=pos.x+10*pos.y;
    if(a.op>=Op::NORTH&&a.op<=Op::WEST) moves++;
    else if(a.op==Op::PASS) idle++;
    else {
      if(a.op==Op::PICKUP) pickups++;
      else if(a.op==Op::DROP||(a.op==Op::PLACE&&a.item==Item::FERTILIZER)) drops++;
      else production++;
      const auto& b=before.privates()[seat];const auto& n=after.privates()[seat];
      bool changed=!tile(before.farms()[seat].tiles[cell]).equal(tile(after.farms()[seat].tiles[cell]))
          || b.inventories[u]!=n.inventories[u] || b.shed!=n.shed || b.seeds!=n.seeds;
      if(changed) completed.append(py::make_tuple(cell,int(a.op),int(a.item),a.quantity));
      else invalid.append(py::make_tuple(u,pos.x,pos.y));
      if(a.op==Op::DROP) for(int i=0;i<12;i++)
        daytime_overflow+=std::max(0,b.inventories[u][i]+b.shed[i]-n.inventories[u][i]-n.shed[i]);
    }
    before=std::move(after);
  }
  s.step(acts);
  int hires=0,wages=0;
  for(size_t o=0;o<acts[seat].market.size();o++)
    if(acts[seat].market[o].op==Op::HIRE&&s.last_market_fills()[seat][o]) {
      int a=1,b=1;for(int k=0;k<hires_before+hires;k++){int c=a+b;a=b;b=c;}
      wages+=s.config().farm_hand_cost_mult*a;hires++;
    }
  py::dict d;d["completed"]=completed;d["invalid"]=invalid;d["moves"]=moves;d["idle"]=idle;
  d["pickups"]=pickups;d["drops"]=drops;d["production"]=production;
  d["hires"]=hires;d["wages"]=wages;d["overflow_units"]=s.last_end_of_day_overflow()[seat];
  d["daytime_overflow_units"]=daytime_overflow;d["market_fills"]=s.last_market_fills()[seat];
  d["market_cash_shortfalls"]=s.last_market_cash_shortfalls()[seat];return d;
}
PYBIND11_MODULE(_pool_sim,m) {
  py::class_<Simulator>(m,"FastEnv",py::module_local())
    .def(py::init([](uint64_t seed){return Simulator(Config{},seed);}))
    .def("observation",&observation)
    .def("advance",&advance)
    .def("step",[](Simulator&s,py::sequence a){s.step({parse_player(a[0]),parse_player(a[1])});})
    .def_property_readonly("done",&Simulator::done)
    .def_property_readonly("step_count",&Simulator::step_count);
}
