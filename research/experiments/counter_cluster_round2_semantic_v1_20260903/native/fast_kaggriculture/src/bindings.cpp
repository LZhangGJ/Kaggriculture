// Licensed under the Apache License, Version 2.0.
#include "simulator.hpp"
#include "native_teammate.hpp"
#include "native_adaptive.hpp"
#include "adaptive_candidates.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

namespace py=pybind11;
using namespace fastkag;

namespace {
const std::unordered_map<std::string,Op> OPS={
 {"PASS",Op::PASS},{"NORTH",Op::NORTH},{"SOUTH",Op::SOUTH},{"EAST",Op::EAST},{"WEST",Op::WEST},
 {"DROP",Op::DROP},{"PICKUP",Op::PICKUP},{"PLACE",Op::PLACE},{"PLANT",Op::PLANT},{"WATER",Op::WATER},
 {"HARVEST",Op::HARVEST},{"FERTILIZE",Op::FERTILIZE},{"DIG",Op::DIG},{"BUILD_COOP",Op::BUILD_COOP},
 {"BUILD_PASTURE",Op::BUILD_PASTURE},{"FEED",Op::FEED},{"COLLECT_FERTILIZER",Op::COLLECT_FERTILIZER},
 {"CARE",Op::CARE},{"HIRE",Op::HIRE},{"BUY_LAND",Op::BUY_LAND},{"BUY_SEED",Op::BUY_SEED},
 {"BUY_PRODUCT",Op::BUY_PRODUCT},{"BUY_ANIMAL",Op::BUY_ANIMAL},{"SELL",Op::SELL}
};
const std::unordered_map<std::string,Item> ITEMS=[](){std::unordered_map<std::string,Item>x;for(int i=0;i<N_ITEMS;i++)x[item_name(i)]=(Item)i;return x;}();

Action parse_action(py::handle h){Action a;if(!py::isinstance<py::list>(h)&&!py::isinstance<py::tuple>(h))return a;py::sequence s=py::reinterpret_borrow<py::sequence>(h);if(!s.size())return a;
 try{auto it=OPS.find(py::cast<std::string>(s[0]));if(it==OPS.end())return a;a.op=it->second;if(s.size()>1){auto jt=ITEMS.find(py::cast<std::string>(s[1]));if(jt!=ITEMS.end())a.item=jt->second;}if(s.size()>2)a.quantity=py::cast<int>(s[2]);}catch(...){return Action{};}return a;}
PlayerAction parse_player_action(py::handle h){PlayerAction out;if(!py::isinstance<py::dict>(h))return out;py::dict d=py::reinterpret_borrow<py::dict>(h);out.units.push_back(d.contains("farmer")?parse_action(d["farmer"]):Action{});if(d.contains("hands")&&py::isinstance<py::list>(d["hands"]))for(auto x:d["hands"])out.units.push_back(parse_action(x));if(d.contains("market")&&py::isinstance<py::list>(d["market"]))for(auto x:d["market"])out.market.push_back(parse_action(x));return out;}
std::array<PlayerAction,2> parse_actions(py::handle h){std::array<PlayerAction,2> out;if(!py::isinstance<py::sequence>(h))return out;py::sequence ps=py::reinterpret_borrow<py::sequence>(h);
 for(int p=0;p<2&&p<(int)ps.size();p++)out[p]=parse_player_action(ps[p]);
 return out;}

std::vector<PlayerAction> parse_tape(py::handle h){std::vector<PlayerAction> out;if(!py::isinstance<py::sequence>(h))return out;for(auto x:py::reinterpret_borrow<py::sequence>(h))out.push_back(parse_player_action(x));return out;}

enum AuditMetric : int {
  UNIT_ATTEMPTS, UNIT_VALID, UNIT_NO_ACTOR, PLANT_NO_SEED, PLANT_WEED,
  UNIT_BLOCKED_TILE, PLACE_NO_ANIMAL, PLACE_WRONG_STRUCTURE,
  MARKET_REQUESTED, MARKET_FILLED, SEED_REQUESTED, SEED_FILLED,
  ANIMAL_REQUESTED, ANIMAL_FILLED, HIRE_REQUESTED, HIRE_FILLED,
  LAND_REQUESTED, LAND_FILLED, CASH_SHORTFALL_EVENTS,
  CASH_SHORTFALL_VALUE, END_OF_DAY_OVERFLOW, AUDIT_METRICS
};

using AuditRow = std::array<double, AUDIT_METRICS>;

enum FirstFailure : int {
  FIRST_UNIT_NO_ACTOR, FIRST_PLANT_NO_SEED, FIRST_PLANT_WEED,
  FIRST_UNIT_BLOCKED_TILE, FIRST_PLACE_NO_ANIMAL,
  FIRST_PLACE_WRONG_STRUCTURE, FIRST_MARKET_UNFILLED,
  FIRST_CASH_SHORTFALL, FIRST_FAILURE_METRICS
};
using FirstFailureRow = std::array<int32_t, FIRST_FAILURE_METRICS>;
void mark_first(FirstFailureRow& row, FirstFailure failure, int step) {
  if (row[failure] < 0) row[failure] = step;
}

bool is_macro_unit(const Action& action) {
  return action.op == Op::PLANT || action.op == Op::BUILD_COOP ||
         action.op == Op::BUILD_PASTURE ||
         (action.op == Op::PLACE && int(action.item) >= int(Item::GOOSE) &&
          int(action.item) <= int(Item::SHEEP));
}

void audit_macro_units(const Simulator& env,
                       const std::array<PlayerAction, 2>& actions,
                       std::array<AuditRow, 2>& audit,
                       std::array<FirstFailureRow, 2>& first,
                       int step) {
  for (int player = 0; player < 2; ++player) {
    auto farm = env.farms()[player];
    auto private_state = env.privates()[player];
    std::array<int, N_CROPS> plant_demand{};
    for (const auto& action : actions[player].units)
      if (action.op == Op::PLANT && int(action.item) >= 0 &&
          int(action.item) < N_CROPS)
        plant_demand[int(action.item)]++;
    std::array<bool, N_CROPS> plant_blocked{};
    for (int item = 0; item < N_CROPS; ++item)
      plant_blocked[item] = plant_demand[item] > private_state.seeds[item];
    for (size_t actor = 0; actor < actions[player].units.size(); ++actor) {
      const auto& action = actions[player].units[actor];
      if (!is_macro_unit(action)) continue;
      audit[player][UNIT_ATTEMPTS]++;
      if (actor > farm.hands.size()) {
        audit[player][UNIT_NO_ACTOR]++;
        mark_first(first[player], FIRST_UNIT_NO_ACTOR, step);
        continue;
      }
      const Position position = actor == 0 ? farm.farmer : farm.hands[actor - 1];
      const int tile_index = position.y * env.config().board_size + position.x;
      const Tile& tile = farm.tiles[tile_index];
      if (action.op == Op::PLANT) {
        const int item = int(action.item);
        if (item < 0 || item >= N_CROPS || plant_blocked[item]) {
          audit[player][PLANT_NO_SEED]++;
          mark_first(first[player], FIRST_PLANT_NO_SEED, step);
        } else if (tile.kind == TileKind::WEED) {
          audit[player][PLANT_WEED]++;
          mark_first(first[player], FIRST_PLANT_WEED, step);
        } else if (tile.kind != TileKind::EMPTY) {
          audit[player][UNIT_BLOCKED_TILE]++;
          mark_first(first[player], FIRST_UNIT_BLOCKED_TILE, step);
        } else {
          audit[player][UNIT_VALID]++;
          private_state.seeds[item]--;
          farm.tiles[tile_index].kind = TileKind::PLANT;
        }
        continue;
      }
      if (action.op == Op::BUILD_COOP || action.op == Op::BUILD_PASTURE) {
        if (tile.kind == TileKind::EMPTY) {
          audit[player][UNIT_VALID]++;
          farm.tiles[tile_index].kind = action.op == Op::BUILD_COOP
              ? TileKind::COOP : TileKind::PASTURE;
        }
        else if (tile.kind == TileKind::WEED) {
          audit[player][PLANT_WEED]++;
          mark_first(first[player], FIRST_PLANT_WEED, step);
        } else {
          audit[player][UNIT_BLOCKED_TILE]++;
          mark_first(first[player], FIRST_UNIT_BLOCKED_TILE, step);
        }
        continue;
      }
      const int item = int(action.item);
      const bool valid_animal = item >= int(Item::GOOSE) && item <= int(Item::SHEEP);
      const TileKind required = item == int(Item::GOOSE) ? TileKind::COOP : TileKind::PASTURE;
      if (!valid_animal || tile.kind != required || tile.animal != Item::NONE) {
        audit[player][PLACE_WRONG_STRUCTURE]++;
        mark_first(first[player], FIRST_PLACE_WRONG_STRUCTURE, step);
      } else if (actor >= private_state.inventories.size() ||
                 private_state.inventories[actor][item] <= 0) {
        audit[player][PLACE_NO_ANIMAL]++;
        mark_first(first[player], FIRST_PLACE_NO_ANIMAL, step);
      } else {
        audit[player][UNIT_VALID]++;
        private_state.inventories[actor][item]--;
        farm.tiles[tile_index].kind = TileKind::ANIMAL;
        farm.tiles[tile_index].animal = action.item;
      }
    }
  }
}

void audit_macro_market(const Simulator& env,
                        const std::array<PlayerAction, 2>& actions,
                        std::array<AuditRow, 2>& audit,
                        std::array<FirstFailureRow, 2>& first,
                        int step) {
  for (int player = 0; player < 2; ++player) {
    const auto& fills = env.last_market_fills()[player];
    const auto& shortfalls = env.last_market_cash_shortfalls()[player];
    for (size_t order = 0; order < actions[player].market.size(); ++order) {
      const auto& action = actions[player].market[order];
      int requested = 0;
      AuditMetric requested_metric = MARKET_REQUESTED;
      AuditMetric filled_metric = MARKET_FILLED;
      if (action.op == Op::BUY_SEED) {
        requested = std::max(0, action.quantity);
        requested_metric = SEED_REQUESTED; filled_metric = SEED_FILLED;
      } else if (action.op == Op::BUY_ANIMAL) {
        requested = std::max(0, action.quantity);
        requested_metric = ANIMAL_REQUESTED; filled_metric = ANIMAL_FILLED;
      } else if (action.op == Op::HIRE) {
        requested = action.quantity > 0 ? 1 : 0;
        requested_metric = HIRE_REQUESTED; filled_metric = HIRE_FILLED;
      } else if (action.op == Op::BUY_LAND) {
        requested = action.quantity > 0 ? 1 : 0;
        requested_metric = LAND_REQUESTED; filled_metric = LAND_FILLED;
      } else {
        continue;
      }
      const int filled = order < fills.size() ? fills[order] : 0;
      audit[player][MARKET_REQUESTED] += requested;
      audit[player][MARKET_FILLED] += std::min(requested, filled);
      audit[player][requested_metric] += requested;
      audit[player][filled_metric] += std::min(requested, filled);
      if (filled < requested)
        mark_first(first[player], FIRST_MARKET_UNFILLED, step);
      if (order < shortfalls.size() && shortfalls[order] > 0) {
        audit[player][CASH_SHORTFALL_EVENTS]++;
        audit[player][CASH_SHORTFALL_VALUE] += shortfalls[order];
        mark_first(first[player], FIRST_CASH_SHORTFALL, step);
      }
    }
    audit[player][END_OF_DAY_OVERFLOW] += env.last_end_of_day_overflow()[player];
  }
}

struct RawTapeAudit {
  std::array<AuditRow, 2> metrics{};
  std::array<double, 2> rewards{};
  std::array<FirstFailureRow, 2> first_failure_steps{};
};

RawTapeAudit audit_raw_tapes(
    const std::vector<PlayerAction>& left,
    const std::vector<PlayerAction>& right,
    uint64_t seed,
    const Config& config) {
  Simulator env(config, seed);
  RawTapeAudit result;
  for (auto& row : result.first_failure_steps) row.fill(-1);
  for (int step = 0; step < config.episode_steps - 1 && !env.done(); ++step) {
    std::array<PlayerAction, 2> actions{};
    if (step < int(left.size())) actions[0] = left[step];
    if (step < int(right.size())) actions[1] = right[step];
    audit_macro_units(env, actions, result.metrics, result.first_failure_steps, step);
    env.step(actions);
    audit_macro_market(env, actions, result.metrics, result.first_failure_steps, step);
  }
  result.rewards = {env.farms()[0].money, env.farms()[1].money};
  return result;
}
py::list action_list(const Action&a){py::list x;x.append(std::string(OPS.begin()->first));const char*op="PASS";for(const auto&[name,value]:OPS)if(value==a.op){op=name.c_str();break;}x[0]=op;if(a.item!=Item::NONE)x.append(item_name((int)a.item));if(a.quantity!=1||a.item!=Item::NONE)x.append(a.quantity);return x;}
py::dict player_action_dict(const PlayerAction&a){py::dict d;d["farmer"]=a.units.empty()?action_list(Action{}):action_list(a.units[0]);py::list hands;for(size_t i=1;i<a.units.size();i++)hands.append(action_list(a.units[i]));d["hands"]=hands;py::list market;for(auto&x:a.market)market.append(action_list(x));d["market"]=market;return d;}

py::dict candidate_delta_dict(const AdaptivePlanDelta& delta) {
  py::dict row;
  row["family"] = std::string(candidate_family_name(delta.family));
  row["family_id"] = int(delta.family);
  row["target_delta"] = delta.target_delta;
  row["hand_delta"] = delta.hand_delta;
  row["quadrant_delta"] = delta.quadrant_delta;
  row["effective_delay_days"] = delta.effective_delay_days;
  row["schedule_profile"] = delta.schedule_profile;
  row["market_profile"] = delta.market_profile;
  row["recovery_profile"] = delta.recovery_profile;
  row["suffix_project"] = delta.suffix_project;
  row["market_item"] = delta.market_item;
  row["recovery_issue"] = delta.recovery_issue;
  row["estimated_value"] = delta.estimated_value;
  row["estimated_cash_cost"] = delta.estimated_cash_cost;
  row["estimated_daily_action_load"] = delta.estimated_daily_action_load;
  row["signature"] = delta.signature;
  return row;
}

AdaptivePlanDelta parse_candidate_delta(py::handle raw) {
  if (!py::isinstance<py::dict>(raw))
    throw std::invalid_argument("candidate delta must be a dict");
  const py::dict row = py::reinterpret_borrow<py::dict>(raw);
  const char* required[] = {
      "family_id", "target_delta", "hand_delta", "quadrant_delta",
      "effective_delay_days", "schedule_profile", "market_profile",
      "recovery_profile", "suffix_project", "market_item",
      "recovery_issue", "signature"};
  for (const char* key : required)
    if (!row.contains(key))
      throw std::invalid_argument(
          std::string("missing candidate delta field: ") + key);
  AdaptivePlanDelta delta;
  const int family = py::cast<int>(row["family_id"]);
  if (family < 0 || family > 8)
    throw std::invalid_argument("candidate family_id must be in [0, 8]");
  delta.family = CandidateFamily(family);
  const auto targets = py::cast<std::vector<int>>(row["target_delta"]);
  if (targets.size() != ADAPTIVE_PROJECTS)
    throw std::invalid_argument("candidate target_delta must have length 8");
  for (int i = 0; i < ADAPTIVE_PROJECTS; ++i)
    delta.target_delta[i] = int16_t(targets[i]);
  delta.hand_delta = int8_t(py::cast<int>(row["hand_delta"]));
  delta.quadrant_delta = int8_t(py::cast<int>(row["quadrant_delta"]));
  delta.effective_delay_days =
      int8_t(py::cast<int>(row["effective_delay_days"]));
  delta.schedule_profile = int8_t(py::cast<int>(row["schedule_profile"]));
  delta.market_profile = int8_t(py::cast<int>(row["market_profile"]));
  delta.recovery_profile = int8_t(py::cast<int>(row["recovery_profile"]));
  delta.suffix_project = int8_t(py::cast<int>(row["suffix_project"]));
  delta.market_item = int8_t(py::cast<int>(row["market_item"]));
  delta.recovery_issue = uint8_t(py::cast<int>(row["recovery_issue"]));
  if (row.contains("estimated_value"))
    delta.estimated_value = py::cast<double>(row["estimated_value"]);
  if (row.contains("estimated_cash_cost"))
    delta.estimated_cash_cost = py::cast<double>(row["estimated_cash_cost"]);
  if (row.contains("estimated_daily_action_load"))
    delta.estimated_daily_action_load =
        py::cast<double>(row["estimated_daily_action_load"]);
  delta.signature = py::cast<uint64_t>(row["signature"]);
  return delta;
}

py::object tile_dict(const Tile&t){if(t.kind==TileKind::EMPTY)return py::none();if(t.kind==TileKind::LOCKED)return py::str("LOCKED");py::dict d;
 if(t.kind==TileKind::WEED){d["kind"]="WEED";return std::move(d);}if(t.kind==TileKind::COOP){d["kind"]="COOP";return std::move(d);}if(t.kind==TileKind::PASTURE){d["kind"]="PASTURE";return std::move(d);}
 if(t.kind==TileKind::PLANT){d["kind"]="PLANT";d["crop"]=item_name((int)t.crop);d["planted_day"]=t.planted_day;d["watered_today"]=t.watered_today;d["consecutive_unwatered"]=t.consecutive_unwatered;d["yield_units"]=t.yield_units;d["max_lifespan_step"]=t.max_lifespan_step;d["fertilized_until_day"]=t.fertilized_until_day;return std::move(d);}
 int ai=(int)t.animal;d["kind"]=ai==9?"COOP":"PASTURE";d["animal"]=item_name(ai);d["placed_day"]=t.placed_day;d["yield_units"]=t.yield_units;d["consecutive_unfed"]=t.consecutive_unfed;d["fed_today"]=t.fed_today;d["cared_today"]=t.cared_today;d["fertilizer_available"]=t.fertilizer_available;d["pending_care_bonus"]=t.pending_care_bonus;return std::move(d);
}
py::dict observation(const Simulator&s,int player){py::dict obs;obs["player"]=player;if(player==0)obs["step"]=s.step_count();obs["day"]=s.day();obs["hour"]=s.hour();py::list farms;
 for(auto&f:s.farms()){py::dict fd;fd["money"]=f.money;py::list tiles;int bs=s.config().board_size;for(int y=0;y<bs;y++){py::list row;for(int x=0;x<bs;x++)row.append(tile_dict(f.tiles[y*bs+x]));tiles.append(row);}fd["tiles"]=tiles;fd["farmer"]=py::make_tuple(f.farmer.x,f.farmer.y);py::list hands;for(auto p:f.hands)hands.append(py::make_tuple(p.x,p.y));fd["hands"]=hands;py::list uq;const char*q[4]={"NW","NE","SW","SE"};for(int i=0;i<4;i++)if(f.unlocked_mask&(1<<i))uq.append(q[i]);fd["unlocked_quadrants"]=uq;fd["hires_today"]=f.hires_today;farms.append(fd);}obs["farms"]=farms;
 auto&pr=s.privates()[player];py::dict pd,shed,seeds;for(int i=0;i<N_ITEMS;i++)shed[item_name(i)]=pr.shed[i];for(int i=0;i<N_CROPS;i++)seeds[item_name(i)]=pr.seeds[i];pd["shed"]=shed;pd["seeds"]=seeds;py::list invs;for(size_t u=0;u<pr.inventories.size();u++){py::dict id;for(int i:pr.inventory_order[u])if(pr.inventories[u][i])id[item_name(i)]=pr.inventories[u][i];invs.append(id);}pd["inventories"]=invs;obs["private"]=pd;
 py::dict md,mi,mp;for(int i=0;i<N_PRODUCTS;i++){mi[item_name(i)]=s.market().inventory[i];mp[item_name(i)]=s.market().prices[i];}md["inventory"]=mi;md["prices"]=mp;obs["market"]=md;py::dict td;py::list shops;for(int sh:s.shops())shops.append(shop_name(sh));td["unlocked_shops"]=shops;obs["town"]=td;return obs;}

std::array<PlayerAction,2> packed(py::array_t<int32_t,py::array::c_style|py::array::forcecast> units,py::array_t<int32_t,py::array::c_style|py::array::forcecast> unit_counts,py::array_t<int32_t,py::array::c_style|py::array::forcecast> market,py::array_t<int32_t,py::array::c_style|py::array::forcecast> market_counts){
 auto u=units.unchecked<3>();auto uc=unit_counts.unchecked<1>();auto m=market.unchecked<3>();auto mc=market_counts.unchecked<1>();if(u.shape(0)!=2||u.shape(2)!=3||m.shape(0)!=2||m.shape(2)!=3)throw std::invalid_argument("packed action arrays must be [2,N,3]");std::array<PlayerAction,2>a;
 for(int p=0;p<2;p++){for(int i=0;i<std::min<ssize_t>(uc(p),u.shape(1));i++)a[p].units.push_back(Action{(Op)u(p,i,0),(Item)u(p,i,1),u(p,i,2)});for(int i=0;i<std::min<ssize_t>(mc(p),m.shape(1));i++)a[p].market.push_back(Action{(Op)m(p,i,0),(Item)m(p,i,1),m(p,i,2)});}return a;}

class FastBatch {
 public:
  FastBatch(int n,Config cfg,uint64_t seed0){if(n<=0)throw std::invalid_argument("n must be positive");envs.reserve(n);for(int i=0;i<n;i++)envs.push_back(std::make_unique<Simulator>(cfg,seed0+i));}
  int size()const{return envs.size();}
  void reset(const std::vector<uint64_t>& seeds){if(seeds.size()!=envs.size())throw std::invalid_argument("one seed per environment required");
   py::gil_scoped_release release;
   for(ssize_t i=0;i<(ssize_t)envs.size();i++)envs[i]->reset(seeds[i]);
  }
  py::tuple step_packed(py::array_t<int32_t,py::array::c_style|py::array::forcecast> units,py::array_t<int32_t,py::array::c_style|py::array::forcecast> unit_counts,py::array_t<int32_t,py::array::c_style|py::array::forcecast> market,py::array_t<int32_t,py::array::c_style|py::array::forcecast> market_counts){
   auto u=units.unchecked<4>();auto uc=unit_counts.unchecked<2>();auto m=market.unchecked<4>();auto mc=market_counts.unchecked<2>();ssize_t b=envs.size();if(u.shape(0)!=b||u.shape(1)!=2||u.shape(3)!=3||uc.shape(0)!=b||uc.shape(1)!=2||m.shape(0)!=b||m.shape(1)!=2||m.shape(3)!=3||mc.shape(0)!=b||mc.shape(1)!=2)throw std::invalid_argument("batch shapes: actions [B,2,N,3], counts [B,2]");
   std::vector<std::array<PlayerAction,2>> acts(b);
   for(ssize_t e=0;e<b;e++)for(int p=0;p<2;p++){for(int i=0;i<std::min<ssize_t>(uc(e,p),u.shape(2));i++)acts[e][p].units.push_back(Action{(Op)u(e,p,i,0),(Item)u(e,p,i,1),u(e,p,i,2)});for(int i=0;i<std::min<ssize_t>(mc(e,p),m.shape(2));i++)acts[e][p].market.push_back(Action{(Op)m(e,p,i,0),(Item)m(e,p,i,1),m(e,p,i,2)});}
   {py::gil_scoped_release release;for(ssize_t e=0;e<b;e++)envs[e]->step(acts[e]);}
   py::array_t<double> rewards({b,(ssize_t)2});py::array_t<uint8_t> dones(b);auto rr=rewards.mutable_unchecked<2>();auto dd=dones.mutable_unchecked<1>();for(ssize_t e=0;e<b;e++){dd(e)=envs[e]->done();rr(e,0)=envs[e]->done()?envs[e]->farms()[0].money:0.;rr(e,1)=envs[e]->done()?envs[e]->farms()[1].money:0.;}return py::make_tuple(rewards,dones);
  }
  py::tuple run_packed_segment(py::array_t<int32_t,py::array::c_style|py::array::forcecast> units,py::array_t<int32_t,py::array::c_style|py::array::forcecast> unit_counts,py::array_t<int32_t,py::array::c_style|py::array::forcecast> market,py::array_t<int32_t,py::array::c_style|py::array::forcecast> market_counts){
   auto u=units.unchecked<5>();auto uc=unit_counts.unchecked<3>();auto m=market.unchecked<5>();auto mc=market_counts.unchecked<3>();ssize_t b=envs.size(),turns=u.shape(1);if(u.shape(0)!=b||u.shape(2)!=2||u.shape(4)!=3||uc.shape(0)!=b||uc.shape(1)!=turns||uc.shape(2)!=2||m.shape(0)!=b||m.shape(1)!=turns||m.shape(2)!=2||m.shape(4)!=3||mc.shape(0)!=b||mc.shape(1)!=turns||mc.shape(2)!=2)throw std::invalid_argument("segment shapes: actions [B,T,2,N,3], counts [B,T,2]");
   {py::gil_scoped_release release;
    #pragma omp parallel for schedule(static)
    for(ssize_t e=0;e<b;e++)for(ssize_t t=0;t<turns&&!envs[e]->done();t++){std::array<PlayerAction,2>a;for(int p=0;p<2;p++){for(int i=0;i<std::min<ssize_t>(uc(e,t,p),u.shape(3));i++)a[p].units.push_back(Action{(Op)u(e,t,p,i,0),(Item)u(e,t,p,i,1),u(e,t,p,i,2)});for(int i=0;i<std::min<ssize_t>(mc(e,t,p),m.shape(3));i++)a[p].market.push_back(Action{(Op)m(e,t,p,i,0),(Item)m(e,t,p,i,1),m(e,t,p,i,2)});}envs[e]->step(a);}
   }
   py::array_t<double> rewards({b,(ssize_t)2});py::array_t<uint8_t>dones(b);auto rr=rewards.mutable_unchecked<2>();auto dd=dones.mutable_unchecked<1>();for(ssize_t e=0;e<b;e++){dd(e)=envs[e]->done();rr(e,0)=envs[e]->done()?envs[e]->farms()[0].money:0.;rr(e,1)=envs[e]->done()?envs[e]->farms()[1].money:0.;}return py::make_tuple(rewards,dones);
  }
  py::dict observation_at(int e,int p)const{if(e<0||e>=(int)envs.size())throw py::index_error();return observation(*envs[e],p);}
 private:
  std::vector<std::unique_ptr<Simulator>> envs;
};
}

PYBIND11_MODULE(_fast_kaggriculture,m){m.doc()="Typed C++ Kaggriculture simulator, compatible with kaggle-environments 1.32.7";
 py::enum_<Op>(m,"Op").value("PASS",Op::PASS).value("NORTH",Op::NORTH).value("SOUTH",Op::SOUTH).value("EAST",Op::EAST).value("WEST",Op::WEST).value("DROP",Op::DROP).value("PICKUP",Op::PICKUP).value("PLACE",Op::PLACE).value("PLANT",Op::PLANT).value("WATER",Op::WATER).value("HARVEST",Op::HARVEST).value("FERTILIZE",Op::FERTILIZE).value("DIG",Op::DIG).value("BUILD_COOP",Op::BUILD_COOP).value("BUILD_PASTURE",Op::BUILD_PASTURE).value("FEED",Op::FEED).value("COLLECT_FERTILIZER",Op::COLLECT_FERTILIZER).value("CARE",Op::CARE).value("HIRE",Op::HIRE).value("BUY_LAND",Op::BUY_LAND).value("BUY_SEED",Op::BUY_SEED).value("BUY_PRODUCT",Op::BUY_PRODUCT).value("BUY_ANIMAL",Op::BUY_ANIMAL).value("SELL",Op::SELL);
 py::enum_<Item>(m,"Item").value("NONE",Item::NONE).value("WHEAT",Item::WHEAT).value("CARROT",Item::CARROT).value("TOMATO",Item::TOMATO).value("STRAWBERRY",Item::STRAWBERRY).value("MELON",Item::MELON).value("EGG",Item::EGG).value("MILK",Item::MILK).value("WOOL",Item::WOOL).value("FERTILIZER",Item::FERTILIZER).value("GOOSE",Item::GOOSE).value("COW",Item::COW).value("SHEEP",Item::SHEEP);
 py::class_<Config>(m,"Config").def(py::init<>()).def_readwrite("episode_steps",&Config::episode_steps).def_readwrite("board_size",&Config::board_size).def_readwrite("starting_money",&Config::starting_money).def_readwrite("max_market_orders",&Config::max_market_orders).def_readwrite("turns_per_day",&Config::turns_per_day).def_readwrite("shed_capacity",&Config::shed_capacity).def_readwrite("weed_spawn_chance",&Config::weed_spawn_chance).def_readwrite("town_shop_unlock_interval",&Config::town_shop_unlock_interval).def_readwrite("town_shop_sell_interval",&Config::town_shop_sell_interval).def_readwrite("town_center_sell_interval",&Config::town_center_sell_interval).def_readwrite("farm_hand_cost_mult",&Config::farm_hand_cost_mult);
 py::class_<Simulator>(m,"FastEnv").def(py::init<Config,uint64_t>(),py::arg("config")=Config{},py::arg("seed")=0).def("clone",[](const Simulator&s){return Simulator(s);}).def("reset",[](Simulator&s,uint64_t seed){s.reset(seed);return py::make_tuple(observation(s,0),observation(s,1));}).def("reset_raw",&Simulator::reset).def("step",[](Simulator&s,py::object a){s.step(parse_actions(a));return py::make_tuple(observation(s,0),observation(s,1));}).def("step_raw",[](Simulator&s,py::object a){auto actions=parse_actions(a);py::gil_scoped_release release;s.step(actions);}).def("step_packed",[](Simulator&s,py::array_t<int32_t>u,py::array_t<int32_t>uc,py::array_t<int32_t>ma,py::array_t<int32_t>mc){s.step(packed(u,uc,ma,mc));return py::make_tuple(observation(s,0),observation(s,1));}).def("step_packed_raw",[](Simulator&s,py::array_t<int32_t>u,py::array_t<int32_t>uc,py::array_t<int32_t>ma,py::array_t<int32_t>mc){auto actions=packed(u,uc,ma,mc);py::gil_scoped_release release;s.step(actions);}).def("observation",&observation).def_property_readonly("last_market_fills",[](const Simulator&s){return py::make_tuple(s.last_market_fills()[0],s.last_market_fills()[1]);}).def_property_readonly("last_market_cash_shortfalls",[](const Simulator&s){return py::make_tuple(s.last_market_cash_shortfalls()[0],s.last_market_cash_shortfalls()[1]);}).def_property_readonly("last_end_of_day_overflow",[](const Simulator&s){return py::make_tuple(s.last_end_of_day_overflow()[0],s.last_end_of_day_overflow()[1]);}).def_property_readonly("done",&Simulator::done).def_property_readonly("step_count",&Simulator::step_count).def_property_readonly("rewards",[](const Simulator&s){return py::make_tuple(s.done()?s.farms()[0].money:0.0,s.done()?s.farms()[1].money:0.0);});
 py::class_<FastBatch>(m,"FastBatchEnv").def(py::init<int,Config,uint64_t>(),py::arg("num_envs"),py::arg("config")=Config{},py::arg("seed0")=0).def("reset",&FastBatch::reset).def("step_packed",&FastBatch::step_packed).def("run_packed_segment",&FastBatch::run_packed_segment).def("observation",&FastBatch::observation_at).def_property_readonly("num_envs",&FastBatch::size);
 m.def("raw_tape_audit_metric_names",[](){return std::vector<std::string>{
   "unit_attempts", "unit_valid", "unit_no_actor", "plant_no_seed",
   "plant_weed", "unit_blocked_tile", "place_no_animal",
   "place_wrong_structure", "market_requested", "market_filled",
   "seed_requested", "seed_filled", "animal_requested", "animal_filled",
   "hire_requested", "hire_filled", "land_requested", "land_filled",
   "cash_shortfall_events", "cash_shortfall_value", "end_of_day_overflow"};});
 m.def("raw_tape_first_failure_names",[](){return std::vector<std::string>{
   "unit_no_actor", "plant_no_seed", "plant_weed", "unit_blocked_tile",
   "place_no_animal", "place_wrong_structure", "market_unfilled",
   "cash_shortfall"};});
 m.def("audit_raw_tapes",[](py::object left,py::object right,uint64_t seed,const Config& config){
   auto left_tape=parse_tape(left),right_tape=parse_tape(right);RawTapeAudit result;
   {py::gil_scoped_release release;result=audit_raw_tapes(left_tape,right_tape,seed,config);}
   py::array_t<double> metrics({(ssize_t)2,(ssize_t)AUDIT_METRICS});auto out=metrics.mutable_unchecked<2>();
   for(int player=0;player<2;player++)for(int metric=0;metric<AUDIT_METRICS;metric++)out(player,metric)=result.metrics[player][metric];
   return py::make_tuple(metrics,py::make_tuple(result.rewards[0],result.rewards[1]));
 },py::arg("left"),py::arg("right"),py::arg("seed"),py::arg("config")=Config{});
 m.def("audit_raw_tapes_detailed",[](py::object left,py::object right,uint64_t seed,const Config& config){
   auto left_tape=parse_tape(left),right_tape=parse_tape(right);RawTapeAudit result;
   {py::gil_scoped_release release;result=audit_raw_tapes(left_tape,right_tape,seed,config);}
   py::array_t<double> metrics({(ssize_t)2,(ssize_t)AUDIT_METRICS});auto out=metrics.mutable_unchecked<2>();
   py::array_t<int32_t> first({(ssize_t)2,(ssize_t)FIRST_FAILURE_METRICS});auto ff=first.mutable_unchecked<2>();
   for(int player=0;player<2;player++){
     for(int metric=0;metric<AUDIT_METRICS;metric++)out(player,metric)=result.metrics[player][metric];
     for(int metric=0;metric<FIRST_FAILURE_METRICS;metric++)ff(player,metric)=result.first_failure_steps[player][metric];
   }
   return py::make_tuple(metrics,py::make_tuple(result.rewards[0],result.rewards[1]),first);
 },py::arg("left"),py::arg("right"),py::arg("seed"),py::arg("config")=Config{});
 py::class_<NativeTeammateExecutor>(m,"NativeTeammateExecutor")
  .def(py::init([](py::sequence routes,py::object r5,py::object md,py::sequence moon,py::sequence moon_legacy){
    NativeTapeLibrary lib;for(auto tape:routes)lib.routes.push_back(parse_tape(tape));lib.r5_reference=parse_tape(r5);lib.md_reference=parse_tape(md);
    if(moon.size()!=5||moon_legacy.size()!=5)throw std::invalid_argument("moon and moon_legacy must each contain five tapes");
    for(int i=0;i<5;i++){lib.moon[i]=parse_tape(moon[i]);lib.moon_legacy[i]=parse_tape(moon_legacy[i]);}
    return NativeTeammateExecutor(std::move(lib));
  }),py::arg("routes"),py::arg("r5_reference"),py::arg("md_reference"),py::arg("moon"),py::arg("moon_legacy"))
  .def("play",[](const NativeTeammateExecutor&x,int route0,int route1,uint64_t seed,int switch_step0,int switch_route0,int switch_step1,int switch_route1,bool capture_trace){
    NativeMatchResult result;{py::gil_scoped_release release;result=x.play(route0,route1,seed,switch_step0,switch_route0,switch_step1,switch_route1,capture_trace,true);}py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;}return out;
  },py::arg("route0"),py::arg("route1"),py::arg("seed"),py::arg("switch_step0")=-1,py::arg("switch_route0")=-1,py::arg("switch_step1")=-1,py::arg("switch_route1")=-1,py::arg("capture_trace")=false)
  .def("play_tape_suffix",[](const NativeTeammateExecutor&x,int opening,int opponent,uint64_t seed,int candidate_seat,int prefix_steps,py::sequence raw_candidate_tape,bool capture_trace){
    std::vector<PlayerAction> candidate_tape=parse_tape(raw_candidate_tape);
    NativeMatchResult result;
    {py::gil_scoped_release release;result=x.play_tape_suffix(opening,opponent,candidate_tape,seed,candidate_seat,prefix_steps,capture_trace);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;}return out;
  },py::arg("opening"),py::arg("opponent"),py::arg("seed"),py::arg("candidate_seat"),py::arg("prefix_steps"),py::arg("candidate_tape"),py::arg("capture_trace")=false)
  .def("play_tape_suffix_batch",[](const NativeTeammateExecutor&x,py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto tt=tasks.unchecked<2>();
    if(tt.shape(1)!=6)throw std::invalid_argument("tasks must have columns opening,opponent,tape_route,seed,seat,prefix_steps");
    const int action_steps=Config{}.episode_steps-1;
    for(ssize_t i=0;i<tt.shape(0);i++){
      const int opening=(int)tt(i,0),opponent=(int)tt(i,1),tape_route=(int)tt(i,2),seat=(int)tt(i,4),prefix=(int)tt(i,5);
      if(opening<0||opening>=x.route_count()||opponent<0||opponent>=x.route_count()||x.route_action_count(tape_route)!=action_steps||seat<0||seat>1||prefix<0||prefix>action_steps)
        throw std::invalid_argument("invalid play_tape_suffix_batch task");
    }
    py::array_t<double> rewards({tt.shape(0),(ssize_t)2});auto rr=rewards.mutable_unchecked<2>();
    {py::gil_scoped_release release;
      #pragma omp parallel for schedule(dynamic,1)
      for(ssize_t i=0;i<tt.shape(0);i++){
        const auto result=x.play_route_suffix((int)tt(i,0),(int)tt(i,1),(int)tt(i,2),(uint64_t)tt(i,3),(int)tt(i,4),(int)tt(i,5),false);
        rr(i,0)=result.rewards[0];rr(i,1)=result.rewards[1];
      }
    }
    return rewards;
  },py::arg("tasks"))
  .def("features_at",[](const NativeTeammateExecutor&x,int route0,int route1,uint64_t seed,int checkpoint,int player,int feature_route){
    std::array<float,147> values;{py::gil_scoped_release release;values=x.features_at(route0,route1,seed,checkpoint,player,feature_route);}py::array_t<float> out(147);auto dst=out.mutable_unchecked<1>();for(int i=0;i<147;i++)dst(i)=values[i];return out;
  },py::arg("route0"),py::arg("route1"),py::arg("seed"),py::arg("checkpoint"),py::arg("player"),py::arg("feature_route"))
  .def("features_batch",[](const NativeTeammateExecutor&x,py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto in=tasks.unchecked<2>();if(in.shape(1)!=6)throw std::invalid_argument("feature tasks must have columns route0,route1,seed,checkpoint,player,feature_route");
    py::array_t<float> features({in.shape(0),(ssize_t)147});auto out=features.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<in.shape(0);i++){auto row=x.features_at((int)in(i,0),(int)in(i,1),(uint64_t)in(i,2),(int)in(i,3),(int)in(i,4),(int)in(i,5));for(int j=0;j<147;j++)out(i,j)=row[j];}}
    return features;
  },py::arg("tasks"))
  .def("play_batch",[](const NativeTeammateExecutor&x,py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto in=tasks.unchecked<2>();if(in.shape(1)!=7)throw std::invalid_argument("tasks must have columns route0,route1,seed,switch_step0,switch_route0,switch_step1,switch_route1");
    py::array_t<double> rewards({in.shape(0),(ssize_t)2});auto out=rewards.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<in.shape(0);i++){auto r=x.play((int)in(i,0),(int)in(i,1),(uint64_t)in(i,2),(int)in(i,3),(int)in(i,4),(int)in(i,5),(int)in(i,6),false,false);out(i,0)=r.rewards[0];out(i,1)=r.rewards[1];}}
    return rewards;
  },py::arg("tasks"))
  .def("play_audit_batch",[](const NativeTeammateExecutor&x,py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto in=tasks.unchecked<2>();if(in.shape(1)!=7)throw std::invalid_argument("tasks must have columns route0,route1,seed,switch_step0,switch_route0,switch_step1,switch_route1");
    py::array_t<double> rewards({in.shape(0),(ssize_t)2});auto rw=rewards.mutable_unchecked<2>();
    py::array_t<int32_t> audit({in.shape(0),(ssize_t)2,(ssize_t)3});auto au=audit.mutable_unchecked<3>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<in.shape(0);i++){auto r=x.play((int)in(i,0),(int)in(i,1),(uint64_t)in(i,2),(int)in(i,3),(int)in(i,4),(int)in(i,5),(int)in(i,6),false);for(int p=0;p<2;p++){rw(i,p)=r.rewards[p];au(i,p,0)=r.macro_unit_failures[p];au(i,p,1)=r.macro_market_failures[p];au(i,p,2)=r.first_macro_failure_step[p];}}}
    return py::make_tuple(rewards,audit);
  },py::arg("tasks"))
  .def("round_robin",[](const NativeTeammateExecutor&x,py::sequence raw_seeds){
    std::vector<uint64_t> seeds;seeds.reserve(raw_seeds.size());for(auto value:raw_seeds)seeds.push_back(py::cast<uint64_t>(value));
    if(seeds.empty())throw std::invalid_argument("at least one seed is required");const ssize_t n=x.route_count();
    py::array_t<double> score({n,n}),margin({n,n}),unit_fail({n,n}),market_fail({n,n});
    py::array_t<int64_t> games({n,n});auto ss=score.mutable_unchecked<2>(),mm=margin.mutable_unchecked<2>();
    auto uu=unit_fail.mutable_unchecked<2>(),mf=market_fail.mutable_unchecked<2>();auto gg=games.mutable_unchecked<2>();
    for(ssize_t i=0;i<n;i++)for(ssize_t j=0;j<n;j++){ss(i,j)=mm(i,j)=uu(i,j)=mf(i,j)=0.;gg(i,j)=0;}
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t left=0;left<n;left++)for(ssize_t right=left+1;right<n;right++){
       double left_score=0.,left_margin=0.,left_unit=0.,right_unit=0.,left_market=0.,right_market=0.;
       for(uint64_t seed:seeds){
         auto a=x.play((int)left,(int)right,seed);double d=a.rewards[0]-a.rewards[1];left_score+=d>0?1.:d==0?.5:0.;left_margin+=d;
         left_unit+=a.macro_unit_failures[0];right_unit+=a.macro_unit_failures[1];left_market+=a.macro_market_failures[0];right_market+=a.macro_market_failures[1];
         auto b=x.play((int)right,(int)left,seed);d=b.rewards[1]-b.rewards[0];left_score+=d>0?1.:d==0?.5:0.;left_margin+=d;
         left_unit+=b.macro_unit_failures[1];right_unit+=b.macro_unit_failures[0];left_market+=b.macro_market_failures[1];right_market+=b.macro_market_failures[0];
       }
       const double count=2.*seeds.size();ss(left,right)=left_score/count;ss(right,left)=1.-ss(left,right);
       mm(left,right)=left_margin/count;mm(right,left)=-mm(left,right);gg(left,right)=gg(right,left)=(int64_t)count;
       uu(left,right)=left_unit/count;uu(right,left)=right_unit/count;mf(left,right)=left_market/count;mf(right,left)=right_market/count;
     }}
    py::dict out;out["score"]=score;out["margin"]=margin;out["games"]=games;out["mean_unit_failures"]=unit_fail;out["mean_market_failures"]=market_fail;return out;
  },py::arg("seeds"))
  .def("switch_search",[](const NativeTeammateExecutor&x,py::sequence raw_openings,
       py::sequence raw_targets,py::sequence raw_checkpoints,py::sequence raw_seeds){
    auto integers=[](py::sequence raw){std::vector<int64_t> out;out.reserve(raw.size());for(auto value:raw)out.push_back(py::cast<int64_t>(value));return out;};
    const auto openings=integers(raw_openings),targets=integers(raw_targets),checkpoints=integers(raw_checkpoints),seed_values=integers(raw_seeds);
    if(openings.empty()||targets.empty()||checkpoints.empty()||seed_values.empty())throw std::invalid_argument("switch search dimensions must be non-empty");
    const ssize_t no=openings.size(),nc=checkpoints.size(),nt=targets.size(),nr=x.route_count(),ns=seed_values.size();
    for(auto route:openings)if(route<0||route>=nr)throw std::invalid_argument("invalid opening route");
    for(auto route:targets)if(route<0||route>=nr)throw std::invalid_argument("invalid target route");
    py::array_t<uint8_t> outcome({no,nc,nt,nr,ns,(ssize_t)2});
    py::array_t<float> margin({no,nc,nt,nr,ns,(ssize_t)2});
    py::array_t<float> states({no,nc,nr,ns,(ssize_t)2,(ssize_t)147});
    auto yy=outcome.mutable_unchecked<6>();
    auto mm=margin.mutable_unchecked<6>();
    auto xx=states.mutable_unchecked<6>();
    std::vector<int> target_values;target_values.reserve(targets.size());for(auto value:targets)target_values.push_back((int)value);
    const ssize_t state_tasks=no*nc*nr*ns*2;
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t flat=0;flat<state_tasks;flat++){
       ssize_t value=flat;const int seat=value%2;value/=2;const ssize_t si=value%ns;value/=ns;
       const ssize_t opponent=value%nr;value/=nr;const ssize_t ci=value%nc;value/=nc;const ssize_t oi=value;
       const int opening=(int)openings[oi],checkpoint=(int)checkpoints[ci];const uint64_t seed=(uint64_t)seed_values[si];
       const auto result=x.switch_case(opening,(int)opponent,seed,checkpoint,seat,target_values);
       for(int f=0;f<147;f++)xx(oi,ci,opponent,si,seat,f)=result.features[f];
       for(ssize_t ti=0;ti<nt;ti++){
         yy(oi,ci,ti,opponent,si,seat)=result.outcome[ti];
         mm(oi,ci,ti,opponent,si,seat)=result.margin[ti];
       }
     }}
    py::dict out;out["outcome"]=outcome;out["margin"]=margin;out["states"]=states;return out;
  },py::arg("openings"),py::arg("targets"),py::arg("checkpoints"),py::arg("seeds"));
 py::class_<NativeAdaptiveExecutor>(m,"NativeAdaptiveExecutor")
  .def(py::init([](py::sequence routes,py::object r5,py::object md,py::sequence moon,py::sequence moon_legacy,py::object raw_backbone){
    NativeTapeLibrary lib;for(auto tape:routes)lib.routes.push_back(parse_tape(tape));lib.r5_reference=parse_tape(r5);lib.md_reference=parse_tape(md);
    if(moon.size()!=5||moon_legacy.size()!=5)throw std::invalid_argument("moon and moon_legacy must each contain five tapes");
    for(int i=0;i<5;i++){lib.moon[i]=parse_tape(moon[i]);lib.moon_legacy[i]=parse_tape(moon_legacy[i]);}
    std::vector<AdaptiveBackbone> backbones;
    if(!raw_backbone.is_none()){
      py::dict d=py::cast<py::dict>(raw_backbone);
      const auto hands=py::cast<std::vector<std::vector<int>>>(d["hand_target_by_day"]);
      const auto lands=py::cast<std::vector<std::vector<int>>>(d["unlocked_target_by_day"]);
      const auto animals=py::cast<std::vector<std::vector<std::vector<int>>>>(d["animal_owned_target_by_day"]);
      const auto animal_service=py::cast<std::vector<std::vector<std::vector<int>>>>(d["animal_service_target_by_day"]);
      const auto crops=py::cast<std::vector<std::vector<std::vector<int>>>>(d["crop_target_by_day"]);
      const auto wheat=py::cast<std::vector<std::vector<int>>>(d["wheat_buffer_by_day"]);
      const auto sell_first=py::cast<std::vector<std::vector<int>>>(d["sell_first_hour_by_day"]);
      const auto sell_last=py::cast<std::vector<std::vector<int>>>(d["sell_last_hour_by_day"]);
      const size_t plans=hands.size();
      if((plans!=1&&plans!=4)||lands.size()!=plans||animals.size()!=plans||
         animal_service.size()!=plans||crops.size()!=plans||wheat.size()!=plans||
         sell_first.size()!=plans||sell_last.size()!=plans)
        throw std::invalid_argument("adaptive backbone must contain one or four complete 30-day plans");
      backbones.resize(plans);
      for(size_t plan=0;plan<plans;plan++){
        if(hands[plan].size()!=30||lands[plan].size()!=30||animals[plan].size()!=30||
           animal_service[plan].size()!=30||crops[plan].size()!=30||wheat[plan].size()!=30||
           sell_first[plan].size()!=30||sell_last[plan].size()!=30)
          throw std::invalid_argument("adaptive backbone day dimensions are invalid");
        auto& backbone=backbones[plan];
        for(int day=0;day<30;day++){
          if(animals[plan][day].size()!=N_ANIMALS||animal_service[plan][day].size()!=N_ANIMALS||crops[plan][day].size()!=N_CROPS)
            throw std::invalid_argument("adaptive backbone item dimensions are invalid");
          backbone.hand_target[day]=int16_t(hands[plan][day]);backbone.quadrant_target[day]=int16_t(lands[plan][day]);
          backbone.wheat_buffer[day]=int16_t(wheat[plan][day]);
          backbone.sell_first_hour[day]=int16_t(sell_first[plan][day]);backbone.sell_last_hour[day]=int16_t(sell_last[plan][day]);
          for(int a=0;a<N_ANIMALS;a++){
            backbone.animal_targets[day][a]=int16_t(animals[plan][day][a]);
            backbone.animal_service_targets[day][a]=int16_t(animal_service[plan][day][a]);
          }
          for(int c=0;c<N_CROPS;c++)backbone.crop_targets[day][c]=int16_t(crops[plan][day][c]);
        }
      }
      auto load_crop_flow=[&](const char*key,auto member){
        if(!d.contains(key))return;
        const auto values=py::cast<std::vector<std::vector<std::vector<int>>>>(d[key]);
        if(values.size()!=plans)throw std::invalid_argument(std::string(key)+" plan count is invalid");
        for(size_t plan=0;plan<plans;plan++)for(int day=0;day<30;day++){
          if(values[plan].size()!=30||values[plan][day].size()!=N_CROPS)throw std::invalid_argument(std::string(key)+" crop dimension is invalid");
          for(int crop=0;crop<N_CROPS;crop++)(backbones[plan].*member)[day][crop]=int16_t(values[plan][day][crop]);
        }
      };
      auto load_animal_flow=[&](const char*key,auto member){
        if(!d.contains(key))return;
        const auto values=py::cast<std::vector<std::vector<std::vector<int>>>>(d[key]);
        if(values.size()!=plans)throw std::invalid_argument(std::string(key)+" plan count is invalid");
        for(size_t plan=0;plan<plans;plan++)for(int day=0;day<30;day++){
          if(values[plan].size()!=30||values[plan][day].size()!=N_ANIMALS)throw std::invalid_argument(std::string(key)+" animal dimension is invalid");
          for(int animal=0;animal<N_ANIMALS;animal++)(backbones[plan].*member)[day][animal]=int16_t(values[plan][day][animal]);
        }
      };
      auto load_day_flow=[&](const char*key,auto member){
        if(!d.contains(key))return;
        const auto values=py::cast<std::vector<std::vector<int>>>(d[key]);
        if(values.size()!=plans)throw std::invalid_argument(std::string(key)+" plan count is invalid");
        for(size_t plan=0;plan<plans;plan++){
          if(values[plan].size()!=30)throw std::invalid_argument(std::string(key)+" day dimension is invalid");
          for(int day=0;day<30;day++)(backbones[plan].*member)[day]=int16_t(values[plan][day]);
        }
      };
      load_crop_flow("crop_plant_by_day",&AdaptiveBackbone::crop_plant);
      load_crop_flow("crop_water_by_day",&AdaptiveBackbone::crop_water);
      load_crop_flow("crop_harvest_by_day",&AdaptiveBackbone::crop_harvest);
      load_crop_flow("crop_fertilize_by_day",&AdaptiveBackbone::crop_fertilize);
      load_crop_flow("crop_harvest_min_yield_by_day",&AdaptiveBackbone::crop_harvest_min_yield);
      load_day_flow("crop_clear_by_day",&AdaptiveBackbone::crop_clear);
      load_animal_flow("animal_feed_by_day",&AdaptiveBackbone::animal_feed);
      load_animal_flow("animal_care_by_day",&AdaptiveBackbone::animal_care);
      load_animal_flow("animal_product_by_day",&AdaptiveBackbone::animal_product);
      load_animal_flow("animal_fertilizer_by_day",&AdaptiveBackbone::animal_fertilizer);
      load_day_flow("buy_first_hour_by_day",&AdaptiveBackbone::buy_first_hour);
      load_day_flow("buy_last_hour_by_day",&AdaptiveBackbone::buy_last_hour);
      if(d.contains("terminal_liquidation_start_step")){
        const auto terminal=py::cast<std::vector<int>>(d["terminal_liquidation_start_step"]);
        if(terminal.size()!=plans)throw std::invalid_argument("terminal liquidation plan count is invalid");
        for(size_t plan=0;plan<plans;plan++)backbones[plan].liquidation_start_step=int16_t(terminal[plan]);
      }
      for(auto& backbone:backbones)backbone.enabled=true;
    }
    return NativeAdaptiveExecutor(std::move(lib),std::move(backbones));
  }),py::arg("routes"),py::arg("r5_reference"),py::arg("md_reference"),py::arg("moon"),py::arg("moon_legacy"),py::arg("backbone")=py::none())
  .def("play",[](const NativeAdaptiveExecutor&x,py::sequence raw_genome,int opponent_route,uint64_t seed,int candidate_seat,bool capture_trace){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    AdaptiveMatchResult result;{py::gil_scoped_release release;result=x.play(AdaptiveGenome::from_vector(values),opponent_route,seed,candidate_seat,capture_trace);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;
    out["replans"]=result.replans;out["override_actions"]=result.override_actions;out["avoidable_crop_losses"]=result.avoidable_crop_losses;
    out["avoidable_animal_losses"]=result.avoidable_animal_losses;out["end_overflow"]=result.end_overflow;
    out["final_operating_prior_index"]=result.final_operating_prior_index;out["operating_prior_switches"]=result.operating_prior_switches;
    out["bundle_switches"]=result.bundle_switches;out["bundle_predicted_gain"]=result.bundle_predicted_gain;out["first_bundle_switch_day"]=result.first_bundle_switch_day;
    out["candidate8_decisions"]=result.candidate8_decisions;out["first_candidate8_day"]=result.first_candidate8_day;out["first_candidate8_family"]=result.first_candidate8_family;out["first_candidate8_signature"]=result.first_candidate8_signature;out["first_candidate8_raw_count"]=result.first_candidate8_raw_count;out["first_candidate8_feasible_count"]=result.first_candidate8_feasible_count;out["first_candidate8_shortlist_count"]=result.first_candidate8_shortlist_count;
    py::list first_switch_features;for(auto value:result.first_bundle_switch_features)first_switch_features.append(value);out["first_bundle_switch_features"]=first_switch_features;
    py::list max_animals,max_crops,final_animals;for(auto v:result.max_animal_targets)max_animals.append(v);for(auto v:result.max_crop_targets)max_crops.append(v);for(auto v:result.final_animals)final_animals.append(v);out["max_animal_targets"]=max_animals;out["max_crop_targets"]=max_crops;out["final_animals"]=final_animals;
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;
      py::list plans;for(const auto&entry:result.plan_trace){py::dict p;py::list animals,animal_service,crops,deferred_animals,deferred_crops,animal_modes,crop_modes;for(auto value:entry.second.animal_targets)animals.append(value);for(auto value:entry.second.animal_service_targets)animal_service.append(value);for(auto value:entry.second.crop_targets)crops.append(value);for(auto value:entry.second.deferred_animal_targets)deferred_animals.append(value);for(auto value:entry.second.deferred_crop_targets)deferred_crops.append(value);for(auto value:entry.second.animal_project_modes)animal_modes.append(value);for(auto value:entry.second.crop_project_modes)crop_modes.append(value);p["step"]=entry.first;p["day"]=entry.second.generated_day;p["animal_targets"]=animals;p["animal_service_targets"]=animal_service;p["crop_targets"]=crops;p["deferred_animal_targets"]=deferred_animals;p["deferred_crop_targets"]=deferred_crops;p["animal_project_modes"]=animal_modes;p["crop_project_modes"]=crop_modes;p["quadrant_target"]=entry.second.quadrant_target;p["deferred_quadrant_target"]=entry.second.deferred_quadrant_target;p["quadrant_project_mode"]=entry.second.quadrant_project_mode;p["hand_target"]=entry.second.hand_target;p["wheat_buffer"]=entry.second.wheat_buffer;p["expected_incremental_value"]=entry.second.expected_incremental_value;p["bundle_switch_applied"]=entry.second.bundle_switch_applied;p["bundle_predicted_gain"]=entry.second.bundle_predicted_gain;p["candidate8_family"]=entry.second.candidate8_family;p["candidate8_schedule_profile"]=entry.second.candidate8_schedule_profile;p["candidate8_market_profile"]=entry.second.candidate8_market_profile;p["candidate8_recovery_profile"]=entry.second.candidate8_recovery_profile;p["candidate8_market_item"]=entry.second.candidate8_market_item;p["candidate8_recovery_issue"]=entry.second.candidate8_recovery_issue;p["candidate8_signature"]=entry.second.candidate8_signature;p["candidate8_raw_count"]=entry.second.candidate8_raw_count;p["candidate8_feasible_count"]=entry.second.candidate8_feasible_count;p["candidate8_shortlist_count"]=entry.second.candidate8_shortlist_count;plans.append(p);}out["plans"]=plans;py::list prior_trace;for(const auto&entry:result.operating_prior_trace){py::dict p;p["step"]=entry.first;p["day"]=entry.first/24;p["prior_index"]=entry.second;prior_trace.append(p);}out["operating_prior_trace"]=prior_trace;}
    return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("seed"),py::arg("candidate_seat")=0,py::arg("capture_trace")=false)
  .def("play_forced_backbone",[](const NativeAdaptiveExecutor&x,py::sequence raw_genome,int opponent_route,uint64_t seed,int candidate_seat,bool capture_trace){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    AdaptiveMatchResult result;{py::gil_scoped_release release;result=x.play_forced_backbone(AdaptiveGenome::from_vector(values),opponent_route,seed,candidate_seat,capture_trace);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;
    out["replans"]=result.replans;out["override_actions"]=result.override_actions;out["avoidable_crop_losses"]=result.avoidable_crop_losses;out["avoidable_animal_losses"]=result.avoidable_animal_losses;out["end_overflow"]=result.end_overflow;
    py::list max_animals,max_crops,final_animals;for(auto v:result.max_animal_targets)max_animals.append(v);for(auto v:result.max_crop_targets)max_crops.append(v);for(auto v:result.final_animals)final_animals.append(v);out["max_animal_targets"]=max_animals;out["max_crop_targets"]=max_crops;out["final_animals"]=final_animals;
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;}
    return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("seed"),py::arg("candidate_seat")=0,py::arg("capture_trace")=false)
  .def("play_market_preempt",[](const NativeAdaptiveExecutor&x,py::sequence raw_genome,int opponent_route,uint64_t seed,int candidate_seat,double fraction,int min_ready,double min_price_ratio,int max_quantity,bool capture_trace){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    AdaptiveGenome genome=AdaptiveGenome::from_vector(values);genome.runtime_market_preempt_fraction=std::clamp(fraction,0.0,1.0);genome.runtime_market_preempt_min_ready=std::clamp(min_ready,1,1000);genome.runtime_market_preempt_min_price_ratio=std::clamp(min_price_ratio,0.0,4.0);genome.runtime_market_preempt_max_quantity=std::clamp(max_quantity,1,1000);
    AdaptiveMatchResult result;{py::gil_scoped_release release;result=x.play(genome,opponent_route,seed,candidate_seat,capture_trace);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;out["replans"]=result.replans;out["override_actions"]=result.override_actions;out["avoidable_crop_losses"]=result.avoidable_crop_losses;out["avoidable_animal_losses"]=result.avoidable_animal_losses;out["end_overflow"]=result.end_overflow;
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;}
    return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("seed"),py::arg("candidate_seat")=0,py::arg("fraction")=0.5,py::arg("min_ready")=4,py::arg("min_price_ratio")=0.0,py::arg("max_quantity")=30,py::arg("capture_trace")=false)
  .def("play_r8",[](const NativeAdaptiveExecutor&x,py::sequence raw_genome,int opponent_route,uint64_t seed,int candidate_seat,int day_horizon_steps,int joint_horizon_steps,int joint_candidate_limit,int feature_level,double lookahead_scale,bool capture_trace,int sequence_solver,int beam_width){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    AdaptiveGenome genome=AdaptiveGenome::from_vector(values);genome.r8_execution_enabled=true;genome.r8_day_horizon_steps=std::clamp(day_horizon_steps,1,24);genome.r8_joint_horizon_steps=std::clamp(joint_horizon_steps,4,8);genome.r8_joint_candidate_limit=std::clamp(joint_candidate_limit,4,32);genome.r8_feature_level=std::clamp(feature_level,1,4);genome.r8_lookahead_scale=std::clamp(lookahead_scale,0.0,2.0);genome.r8_sequence_solver=std::clamp(sequence_solver,0,1);genome.r8_beam_width=std::clamp(beam_width,1,64);
    AdaptiveMatchResult result;{py::gil_scoped_release release;result=x.play(genome,opponent_route,seed,candidate_seat,capture_trace);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;
    out["replans"]=result.replans;out["override_actions"]=result.override_actions;out["avoidable_crop_losses"]=result.avoidable_crop_losses;out["avoidable_animal_losses"]=result.avoidable_animal_losses;out["end_overflow"]=result.end_overflow;
    out["r8_day_plan_rebuilds"]=result.r8_day_plan_rebuilds;out["r8_rolling_updates"]=result.r8_rolling_updates;out["r8_task_nodes_created"]=result.r8_task_nodes_created;out["r8_task_reassignments"]=result.r8_task_reassignments;out["r8_joint_matches"]=result.r8_joint_matches;out["r8_lookahead_evaluations"]=result.r8_lookahead_evaluations;out["r8_idle_unit_actions"]=result.r8_idle_unit_actions;out["r8_peak_active_tasks"]=result.r8_peak_active_tasks;out["r8_move_unit_actions"]=result.r8_move_unit_actions;out["r8_resolved_task_nodes"]=result.r8_resolved_task_nodes;out["r8_resolution_latency_p50"]=result.r8_resolution_latency_p50;out["r8_resolution_latency_p95"]=result.r8_resolution_latency_p95;out["r8_unresolved_hard_day_tasks"]=result.r8_unresolved_hard_day_tasks;out["r8_duplicate_reservation_violations"]=result.r8_duplicate_reservation_violations;
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;}
    return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("seed"),py::arg("candidate_seat")=0,py::arg("day_horizon_steps")=24,py::arg("joint_horizon_steps")=8,py::arg("joint_candidate_limit")=8,py::arg("feature_level")=4,py::arg("lookahead_scale")=0.01,py::arg("capture_trace")=false,py::arg("sequence_solver")=1,py::arg("beam_width")=8)
  .def("play_r8_batch",[](const NativeAdaptiveExecutor&x,
       py::array_t<double,py::array::c_style|py::array::forcecast> genomes,
       py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks,
       int day_horizon_steps,int joint_horizon_steps,int joint_candidate_limit,int feature_level,double lookahead_scale,int sequence_solver,int beam_width){
    auto gg=genomes.unchecked<2>();auto tt=tasks.unchecked<2>();
    if(gg.shape(1)!=AdaptiveGenome::DIM)throw std::invalid_argument("genomes must have AdaptiveGenome::DIM columns");
    if(tt.shape(1)!=4)throw std::invalid_argument("R8 tasks must have columns genome_index,opponent_route,seed,candidate_seat");
    std::vector<AdaptiveGenome> parsed;parsed.reserve(gg.shape(0));
    for(ssize_t i=0;i<gg.shape(0);i++){std::vector<double> row(AdaptiveGenome::DIM);for(int j=0;j<AdaptiveGenome::DIM;j++)row[j]=gg(i,j);auto genome=AdaptiveGenome::from_vector(row);genome.r8_execution_enabled=true;genome.r8_day_horizon_steps=std::clamp(day_horizon_steps,1,24);genome.r8_joint_horizon_steps=std::clamp(joint_horizon_steps,4,8);genome.r8_joint_candidate_limit=std::clamp(joint_candidate_limit,4,32);genome.r8_feature_level=std::clamp(feature_level,1,4);genome.r8_lookahead_scale=std::clamp(lookahead_scale,0.0,2.0);genome.r8_sequence_solver=std::clamp(sequence_solver,0,1);genome.r8_beam_width=std::clamp(beam_width,1,64);parsed.push_back(genome);}
    py::array_t<double> rewards({tt.shape(0),(ssize_t)2});auto rr=rewards.mutable_unchecked<2>();
    constexpr ssize_t R8_DIAGNOSTIC_DIM=19;
    py::array_t<int64_t> diagnostics({tt.shape(0),R8_DIAGNOSTIC_DIM});auto dd=diagnostics.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<tt.shape(0);i++){
       const int genome_index=(int)tt(i,0);if(genome_index<0||genome_index>=(int)parsed.size())continue;
       const auto r=x.play(parsed[genome_index],(int)tt(i,1),(uint64_t)tt(i,2),(int)tt(i,3),false);
       rr(i,0)=r.rewards[0];rr(i,1)=r.rewards[1];dd(i,0)=r.avoidable_crop_losses;dd(i,1)=r.avoidable_animal_losses;dd(i,2)=r.end_overflow;dd(i,3)=r.replans;dd(i,4)=r.override_actions;dd(i,5)=r.r8_day_plan_rebuilds;dd(i,6)=r.r8_rolling_updates;dd(i,7)=r.r8_task_nodes_created;dd(i,8)=r.r8_task_reassignments;dd(i,9)=r.r8_joint_matches;dd(i,10)=r.r8_lookahead_evaluations;dd(i,11)=r.r8_idle_unit_actions;dd(i,12)=r.r8_peak_active_tasks;dd(i,13)=r.r8_move_unit_actions;dd(i,14)=r.r8_resolved_task_nodes;dd(i,15)=r.r8_resolution_latency_p50;dd(i,16)=r.r8_resolution_latency_p95;dd(i,17)=r.r8_unresolved_hard_day_tasks;dd(i,18)=r.r8_duplicate_reservation_violations;
     }}
    return py::make_tuple(rewards,diagnostics);
  },py::arg("genomes"),py::arg("tasks"),py::arg("day_horizon_steps")=24,py::arg("joint_horizon_steps")=8,py::arg("joint_candidate_limit")=8,py::arg("feature_level")=4,py::arg("lookahead_scale")=0.01,py::arg("sequence_solver")=1,py::arg("beam_width")=8)
  .def("play_blend_trace",[](const NativeAdaptiveExecutor&x,py::sequence raw_genome,int base_route,int opponent_route,uint64_t seed,int candidate_seat,int blend_mode,int prefix_steps){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    AdaptiveMatchResult result;{py::gil_scoped_release release;result=x.play_blend(AdaptiveGenome::from_vector(values),base_route,opponent_route,seed,candidate_seat,blend_mode,true,prefix_steps);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;
    py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;return out;
  },py::arg("genome"),py::arg("base_route"),py::arg("opponent_route"),py::arg("seed"),py::arg("candidate_seat")=0,py::arg("blend_mode")=0,py::arg("prefix_steps")=1)
  .def("play_candidate8",[](const NativeAdaptiveExecutor&x,py::sequence raw_genome,int opponent_route,uint64_t seed,int candidate_seat,int candidate_rank,int minimum_decision_day,bool use_feasible_pool,bool capture_trace){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    AdaptiveMatchResult result;{py::gil_scoped_release release;result=x.play_candidate8(AdaptiveGenome::from_vector(values),opponent_route,seed,candidate_seat,candidate_rank,minimum_decision_day,use_feasible_pool,capture_trace);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;
    out["replans"]=result.replans;out["override_actions"]=result.override_actions;out["avoidable_crop_losses"]=result.avoidable_crop_losses;out["avoidable_animal_losses"]=result.avoidable_animal_losses;out["end_overflow"]=result.end_overflow;
    out["candidate8_decisions"]=result.candidate8_decisions;out["first_candidate8_day"]=result.first_candidate8_day;out["first_candidate8_family"]=result.first_candidate8_family;out["first_candidate8_signature"]=result.first_candidate8_signature;out["first_candidate8_raw_count"]=result.first_candidate8_raw_count;out["first_candidate8_feasible_count"]=result.first_candidate8_feasible_count;out["first_candidate8_shortlist_count"]=result.first_candidate8_shortlist_count;
    py::list max_animals,max_crops,final_animals;for(auto v:result.max_animal_targets)max_animals.append(v);for(auto v:result.max_crop_targets)max_crops.append(v);for(auto v:result.final_animals)final_animals.append(v);out["max_animal_targets"]=max_animals;out["max_crop_targets"]=max_crops;out["final_animals"]=final_animals;
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;py::list plans;for(const auto&entry:result.plan_trace){py::dict p;p["step"]=entry.first;p["day"]=entry.second.generated_day;p["candidate8_family"]=entry.second.candidate8_family;p["candidate8_schedule_profile"]=entry.second.candidate8_schedule_profile;p["candidate8_market_profile"]=entry.second.candidate8_market_profile;p["candidate8_recovery_profile"]=entry.second.candidate8_recovery_profile;p["candidate8_market_item"]=entry.second.candidate8_market_item;p["candidate8_signature"]=entry.second.candidate8_signature;p["candidate8_raw_count"]=entry.second.candidate8_raw_count;p["candidate8_feasible_count"]=entry.second.candidate8_feasible_count;p["candidate8_shortlist_count"]=entry.second.candidate8_shortlist_count;py::list crops,animals;for(auto value:entry.second.crop_targets)crops.append(value);for(auto value:entry.second.animal_targets)animals.append(value);p["crop_targets"]=crops;p["animal_targets"]=animals;plans.append(p);}out["plans"]=plans;}
    return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("seed"),py::arg("candidate_seat")=0,py::arg("candidate_rank")=0,py::arg("minimum_decision_day")=0,py::arg("use_feasible_pool")=false,py::arg("capture_trace")=false)
  .def("play_candidate8_batch",[](const NativeAdaptiveExecutor&x,
       py::array_t<double,py::array::c_style|py::array::forcecast> genomes,
       py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto gg=genomes.unchecked<2>();auto tt=tasks.unchecked<2>();
    if(gg.shape(1)!=AdaptiveGenome::DIM)throw std::invalid_argument("genomes must have AdaptiveGenome::DIM columns");
    if(tt.shape(1)!=7)throw std::invalid_argument("candidate8 tasks must have columns genome_index,opponent_route,seed,candidate_seat,candidate_rank,minimum_decision_day,use_feasible_pool");
    std::vector<AdaptiveGenome> parsed;parsed.reserve(gg.shape(0));
    for(ssize_t i=0;i<gg.shape(0);i++){std::vector<double> row(AdaptiveGenome::DIM);for(int j=0;j<AdaptiveGenome::DIM;j++)row[j]=gg(i,j);parsed.push_back(AdaptiveGenome::from_vector(row));}
    py::array_t<double> rewards({tt.shape(0),(ssize_t)2});auto rr=rewards.mutable_unchecked<2>();
    py::array_t<int64_t> diagnostics({tt.shape(0),(ssize_t)(34+AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM)});auto dd=diagnostics.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<tt.shape(0);i++){
       const int genome=(int)tt(i,0);if(genome<0||genome>=(int)parsed.size())continue;
       const auto r=x.play_candidate8(parsed[genome],(int)tt(i,1),(uint64_t)tt(i,2),(int)tt(i,3),(int)tt(i,4),(int)tt(i,5),tt(i,6)!=0,false);
       rr(i,0)=r.rewards[0];rr(i,1)=r.rewards[1];
       dd(i,0)=r.replans;dd(i,1)=r.override_actions;dd(i,2)=r.avoidable_crop_losses;dd(i,3)=r.avoidable_animal_losses;dd(i,4)=r.end_overflow;dd(i,5)=r.candidate_seat;
       dd(i,6)=r.candidate8_decisions;dd(i,7)=r.first_candidate8_day;dd(i,8)=r.first_candidate8_family;dd(i,9)=(int64_t)r.first_candidate8_signature;dd(i,10)=r.first_candidate8_raw_count;dd(i,11)=r.first_candidate8_feasible_count;dd(i,12)=r.first_candidate8_shortlist_count;dd(i,13)=tt(i,4);
       for(int project=0;project<ADAPTIVE_PROJECTS;project++)dd(i,14+project)=r.first_candidate8_target_delta[project];
       dd(i,22)=r.first_candidate8_hand_delta;dd(i,23)=r.first_candidate8_quadrant_delta;dd(i,24)=r.first_candidate8_effective_delay_days;dd(i,25)=r.first_candidate8_schedule_profile;dd(i,26)=r.first_candidate8_market_profile;dd(i,27)=r.first_candidate8_recovery_profile;dd(i,28)=r.first_candidate8_suffix_project;dd(i,29)=r.first_candidate8_market_item;dd(i,30)=r.first_candidate8_recovery_issue;dd(i,31)=(int64_t)std::nearbyint(100.0*r.first_candidate8_estimated_value);dd(i,32)=(int64_t)std::nearbyint(100.0*r.first_candidate8_estimated_cash_cost);dd(i,33)=(int64_t)std::nearbyint(100.0*r.first_candidate8_estimated_daily_action_load);
       for(int feature=0;feature<AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM;feature++)dd(i,34+feature)=r.first_candidate8_context_features[feature];
     }}
    return py::make_tuple(rewards,diagnostics);
  },py::arg("genomes"),py::arg("tasks"))
  .def("candidate8_counterfactual",[](const NativeAdaptiveExecutor&x,
       py::sequence raw_genome,int opponent_route,uint64_t prefix_seed,
       py::sequence raw_future_seeds,int candidate_seat,
       int minimum_decision_day,bool use_feasible_pool,int maximum_arms,
       py::sequence raw_committed_days,py::sequence raw_committed_ranks,
       bool include_response_scenarios){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    std::vector<uint64_t> future_seeds;future_seeds.reserve(raw_future_seeds.size());for(auto value:raw_future_seeds)future_seeds.push_back(py::cast<uint64_t>(value));
    std::vector<int> committed_days;committed_days.reserve(raw_committed_days.size());for(auto value:raw_committed_days)committed_days.push_back(py::cast<int>(value));
    std::vector<int> committed_ranks;committed_ranks.reserve(raw_committed_ranks.size());for(auto value:raw_committed_ranks)committed_ranks.push_back(py::cast<int>(value));
    AdaptiveCandidate8CounterfactualResult result;{py::gil_scoped_release release;result=x.candidate8_counterfactual(AdaptiveGenome::from_vector(values),opponent_route,prefix_seed,future_seeds,candidate_seat,minimum_decision_day,use_feasible_pool,maximum_arms,committed_days,committed_ranks,include_response_scenarios);}
    py::dict out;out["decision_found"]=result.decision_found;out["decision_step"]=result.decision_step;
    py::array_t<double> rewards({(ssize_t)result.future_count,(ssize_t)result.arms});auto rr=rewards.mutable_unchecked<2>();
    py::array_t<double> opponent_rewards({(ssize_t)result.future_count,(ssize_t)result.arms});auto orr=opponent_rewards.mutable_unchecked<2>();
    py::array_t<int32_t> overflow({(ssize_t)result.future_count,(ssize_t)result.arms});auto oo=overflow.mutable_unchecked<2>();
    for(int f=0;f<result.future_count;f++)for(int a=0;a<result.arms;a++){const int index=f*result.arms+a;rr(f,a)=result.own_rewards[index];orr(f,a)=result.opponent_rewards[index];oo(f,a)=result.end_overflow[index];}
    py::array_t<int8_t> family(result.arms);auto fa=family.mutable_unchecked<1>();for(int a=0;a<result.arms;a++)fa(a)=result.arm_family[a];
    py::array_t<uint64_t> signature(result.arms);auto ss=signature.mutable_unchecked<1>();for(int a=0;a<result.arms;a++)ss(a)=result.arm_signature[a];
    py::array_t<int32_t> features({(ssize_t)result.arms,(ssize_t)AdaptiveCandidate8CounterfactualResult::CANDIDATE_FEATURE_DIM});auto ff=features.mutable_unchecked<2>();for(int a=0;a<result.arms;a++)for(int f=0;f<AdaptiveCandidate8CounterfactualResult::CANDIDATE_FEATURE_DIM;f++)ff(a,f)=result.arm_features[a][f];
    py::array_t<int32_t> consequence_features({(ssize_t)result.arms,(ssize_t)AdaptiveCandidate8CounterfactualResult::CONSEQUENCE_FEATURE_DIM});auto cff=consequence_features.mutable_unchecked<2>();for(int a=0;a<result.arms;a++)for(int f=0;f<AdaptiveCandidate8CounterfactualResult::CONSEQUENCE_FEATURE_DIM;f++)cff(a,f)=result.arm_consequence_features[a][f];
    py::array_t<int32_t> response_scenario_features({(ssize_t)result.arms,(ssize_t)AdaptiveCandidate8CounterfactualResult::RESPONSE_SCENARIO_COUNT,(ssize_t)AdaptiveCandidate8CounterfactualResult::CONSEQUENCE_FEATURE_DIM});auto rsf=response_scenario_features.mutable_unchecked<3>();
    py::array_t<int32_t> response_scenario_outcomes({(ssize_t)result.arms,(ssize_t)AdaptiveCandidate8CounterfactualResult::RESPONSE_SCENARIO_COUNT,(ssize_t)AdaptiveCandidate8CounterfactualResult::RESPONSE_OUTCOME_DIM});auto rso=response_scenario_outcomes.mutable_unchecked<3>();
    if(include_response_scenarios)for(int a=0;a<result.arms;a++)for(int s=0;s<AdaptiveCandidate8CounterfactualResult::RESPONSE_SCENARIO_COUNT;s++){const int row=a*AdaptiveCandidate8CounterfactualResult::RESPONSE_SCENARIO_COUNT+s;for(int f=0;f<AdaptiveCandidate8CounterfactualResult::CONSEQUENCE_FEATURE_DIM;f++)rsf(a,s,f)=result.arm_response_scenario_features[row][f];for(int f=0;f<AdaptiveCandidate8CounterfactualResult::RESPONSE_OUTCOME_DIM;f++)rso(a,s,f)=result.arm_response_scenario_outcomes[row][f];}
    py::array_t<int32_t> context(AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM);auto cc=context.mutable_unchecked<1>();for(int f=0;f<AdaptivePlan::CANDIDATE8_CONTEXT_FEATURE_DIM;f++)cc(f)=result.context_features[f];
    py::array_t<int16_t> first_action_change(result.arms),action_change_24(result.arms),action_change_full(result.arms),first_state_change(result.arms),state_change_24(result.arms),state_change_full(result.arms);
    auto fac=first_action_change.mutable_unchecked<1>();auto ac24=action_change_24.mutable_unchecked<1>();auto acf=action_change_full.mutable_unchecked<1>();auto fsc=first_state_change.mutable_unchecked<1>();auto sc24=state_change_24.mutable_unchecked<1>();auto scf=state_change_full.mutable_unchecked<1>();
    for(int a=0;a<result.arms;a++){fac(a)=result.arm_first_action_change_offset[a];ac24(a)=result.arm_action_change_count_24[a];acf(a)=result.arm_action_change_count_full[a];fsc(a)=result.arm_first_state_change_offset[a];sc24(a)=result.arm_state_change_count_24[a];scf(a)=result.arm_state_change_count_full[a];}
    out["rewards"]=rewards;out["opponent_rewards"]=opponent_rewards;out["end_overflow"]=overflow;out["arm_family"]=family;out["arm_signature"]=signature;out["arm_features"]=features;out["consequence_features"]=consequence_features;out["response_scenario_features"]=response_scenario_features;out["response_scenario_outcomes"]=response_scenario_outcomes;out["context_features"]=context;out["first_action_change_offset"]=first_action_change;out["action_change_count_24"]=action_change_24;out["action_change_count_full"]=action_change_full;out["first_state_change_offset"]=first_state_change;out["state_change_count_24"]=state_change_24;out["state_change_count_full"]=state_change_full;return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("prefix_seed"),py::arg("future_seeds"),py::arg("candidate_seat")=0,py::arg("minimum_decision_day")=0,py::arg("use_feasible_pool")=true,py::arg("maximum_arms")=4096,py::arg("committed_days")=py::tuple(),py::arg("committed_ranks")=py::tuple(),py::arg("include_response_scenarios")=false)
  .def("candidate8_committed_sequence",[](const NativeAdaptiveExecutor&x,
       py::sequence raw_genome,int opponent_route,uint64_t actual_seed,
       py::sequence raw_decision_days,py::sequence raw_selected_ranks,
       int candidate_seat,bool use_feasible_pool,int prefix_route,
       int prefix_steps,bool capture_trace,py::sequence raw_selected_deltas){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    std::vector<int> decision_days;decision_days.reserve(raw_decision_days.size());for(auto value:raw_decision_days)decision_days.push_back(py::cast<int>(value));
    std::vector<int> selected_ranks;selected_ranks.reserve(raw_selected_ranks.size());for(auto value:raw_selected_ranks)selected_ranks.push_back(py::cast<int>(value));
    std::vector<AdaptivePlanDelta> selected_deltas;selected_deltas.reserve(raw_selected_deltas.size());for(auto value:raw_selected_deltas)selected_deltas.push_back(parse_candidate_delta(value));
    AdaptiveCandidate8CommittedSequenceResult result;{py::gil_scoped_release release;result=x.candidate8_committed_sequence(AdaptiveGenome::from_vector(values),opponent_route,actual_seed,decision_days,selected_ranks,candidate_seat,use_feasible_pool,prefix_route,prefix_steps,capture_trace,selected_deltas);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;out["end_overflow"]=result.end_overflow;out["avoidable_crop_losses"]=result.avoidable_crop_losses;out["avoidable_animal_losses"]=result.avoidable_animal_losses;
    py::list days,ranks,families,signatures,matches,deltas;for(size_t i=0;i<result.decision_day.size();i++){days.append(result.decision_day[i]);ranks.append(result.selected_rank[i]);families.append(result.selected_family[i]);signatures.append(result.selected_signature[i]);matches.append(bool(result.selected_matched[i]));deltas.append(candidate_delta_dict(result.selected_delta[i]));}
    out["decision_day"]=days;out["selected_rank"]=ranks;out["selected_family"]=families;out["selected_signature"]=signatures;out["selected_matched"]=matches;out["selected_deltas"]=deltas;if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;}return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("actual_seed"),py::arg("decision_days"),py::arg("selected_ranks"),py::arg("candidate_seat")=0,py::arg("use_feasible_pool")=false,py::arg("prefix_route")=-1,py::arg("prefix_steps")=0,py::arg("capture_trace")=false,py::arg("selected_deltas")=py::tuple())
  .def("candidate8_rolling_oracle",[](const NativeAdaptiveExecutor&x,
       py::sequence raw_genome,int opponent_route,uint64_t actual_seed,
       py::sequence raw_decision_days,uint64_t future_seed_base,
       int future_count,int candidate_seat,bool use_feasible_pool,
       int maximum_arms,bool clairvoyant_actual_future,
       bool competitive_objective,bool r8_execution_enabled,
       int r8_sequence_solver,int r8_beam_width,
       int r8_day_horizon_steps,int r8_joint_horizon_steps,
       int r8_joint_candidate_limit,int r8_feature_level,
       double r8_lookahead_scale){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    std::vector<int> decision_days;decision_days.reserve(raw_decision_days.size());for(auto value:raw_decision_days)decision_days.push_back(py::cast<int>(value));
    AdaptiveGenome genome=AdaptiveGenome::from_vector(values);genome.r8_execution_enabled=r8_execution_enabled;genome.r8_sequence_solver=std::clamp(r8_sequence_solver,0,1);genome.r8_beam_width=std::clamp(r8_beam_width,1,64);genome.r8_day_horizon_steps=std::clamp(r8_day_horizon_steps,1,24);genome.r8_joint_horizon_steps=std::clamp(r8_joint_horizon_steps,4,8);genome.r8_joint_candidate_limit=std::clamp(r8_joint_candidate_limit,4,32);genome.r8_feature_level=std::clamp(r8_feature_level,1,4);genome.r8_lookahead_scale=std::clamp(r8_lookahead_scale,0.0,2.0);
    AdaptiveCandidate8RollingOracleResult result;{py::gil_scoped_release release;result=x.candidate8_rolling_oracle(genome,opponent_route,actual_seed,decision_days,future_seed_base,future_count,candidate_seat,use_feasible_pool,maximum_arms,clairvoyant_actual_future,competitive_objective);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;out["end_overflow"]=result.end_overflow;out["avoidable_crop_losses"]=result.avoidable_crop_losses;out["avoidable_animal_losses"]=result.avoidable_animal_losses;out["complete_continuations"]=result.complete_continuations;
    py::list days,steps,counts,ranks,families,signatures,selected_rewards,keep_rewards,gains,selected_win_rates,keep_win_rates,selected_margins,keep_margins;
    for(size_t i=0;i<result.decision_day.size();i++){days.append(result.decision_day[i]);steps.append(result.decision_step[i]);counts.append(result.feasible_count[i]);ranks.append(result.selected_rank[i]);families.append(result.selected_family[i]);signatures.append(result.selected_signature[i]);selected_rewards.append(result.selected_expected_reward[i]);keep_rewards.append(result.keep_expected_reward[i]);gains.append(result.stage_expected_gain[i]);selected_win_rates.append(result.selected_expected_win_rate[i]);keep_win_rates.append(result.keep_expected_win_rate[i]);selected_margins.append(result.selected_expected_margin[i]);keep_margins.append(result.keep_expected_margin[i]);}
    out["decision_day"]=days;out["decision_step"]=steps;out["feasible_count"]=counts;out["selected_rank"]=ranks;out["selected_family"]=families;out["selected_signature"]=signatures;out["selected_expected_reward"]=selected_rewards;out["keep_expected_reward"]=keep_rewards;out["stage_expected_gain"]=gains;out["selected_expected_win_rate"]=selected_win_rates;out["keep_expected_win_rate"]=keep_win_rates;out["selected_expected_margin"]=selected_margins;out["keep_expected_margin"]=keep_margins;return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("actual_seed"),py::arg("decision_days"),py::arg("future_seed_base"),py::arg("future_count")=8,py::arg("candidate_seat")=0,py::arg("use_feasible_pool")=true,py::arg("maximum_arms")=4096,py::arg("clairvoyant_actual_future")=false,py::arg("competitive_objective")=false,py::arg("r8_execution_enabled")=false,py::arg("r8_sequence_solver")=1,py::arg("r8_beam_width")=8,py::arg("r8_day_horizon_steps")=24,py::arg("r8_joint_horizon_steps")=8,py::arg("r8_joint_candidate_limit")=8,py::arg("r8_feature_level")=4,py::arg("r8_lookahead_scale")=0.01)
  .def("candidate8_sequence_oracle",[](const NativeAdaptiveExecutor&x,
       py::sequence raw_genome,int opponent_route,uint64_t actual_seed,
       py::sequence raw_decision_days,int candidate_seat,int beam_width,
       int per_node_arms,bool use_feasible_pool,bool competitive_objective,
       py::sequence raw_committed_days,py::sequence raw_committed_rank_sequences,
       int prefix_route,int prefix_steps){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    std::vector<int> decision_days;decision_days.reserve(raw_decision_days.size());for(auto value:raw_decision_days)decision_days.push_back(py::cast<int>(value));
    std::vector<int> committed_days;committed_days.reserve(raw_committed_days.size());for(auto value:raw_committed_days)committed_days.push_back(py::cast<int>(value));
    std::vector<std::vector<int>> committed_rank_sequences;committed_rank_sequences.reserve(raw_committed_rank_sequences.size());for(auto raw_sequence:raw_committed_rank_sequences){std::vector<int> sequence;for(auto value:py::reinterpret_borrow<py::sequence>(raw_sequence))sequence.push_back(py::cast<int>(value));committed_rank_sequences.push_back(std::move(sequence));}
    AdaptiveCandidate8SequenceOracleResult result;{py::gil_scoped_release release;result=x.candidate8_sequence_oracle(AdaptiveGenome::from_vector(values),opponent_route,actual_seed,decision_days,candidate_seat,beam_width,per_node_arms,use_feasible_pool,competitive_objective,committed_days,committed_rank_sequences,prefix_route,prefix_steps);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;out["complete_continuations"]=result.complete_continuations;out["expanded_nodes"]=result.expanded_nodes;out["maximum_live_beam"]=result.maximum_live_beam;out["final_path_candidate_rewards"]=result.final_path_candidate_rewards;out["final_path_opponent_rewards"]=result.final_path_opponent_rewards;out["final_path_sequence_length"]=result.final_path_sequence_length;out["final_path_selected_ranks"]=result.final_path_selected_ranks;out["final_path_selected_families"]=result.final_path_selected_families;
    py::list days,ranks,families,signatures,counts;for(size_t i=0;i<result.decision_day.size();i++){days.append(result.decision_day[i]);ranks.append(result.selected_rank[i]);families.append(result.selected_family[i]);signatures.append(result.selected_signature[i]);counts.append(result.feasible_count[i]);}
    out["decision_day"]=days;out["selected_rank"]=ranks;out["selected_family"]=families;out["selected_signature"]=signatures;out["feasible_count"]=counts;return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("actual_seed"),py::arg("decision_days"),py::arg("candidate_seat")=0,py::arg("beam_width")=32,py::arg("per_node_arms")=64,py::arg("use_feasible_pool")=true,py::arg("competitive_objective")=true,py::arg("committed_days")=py::tuple(),py::arg("committed_rank_sequences")=py::tuple(),py::arg("prefix_route")=-1,py::arg("prefix_steps")=0)
  .def("candidate8_mcts_oracle",[](const NativeAdaptiveExecutor&x,
       py::sequence raw_genome,int opponent_route,uint64_t actual_seed,
       py::sequence raw_decision_days,int candidate_seat,
       int simulation_budget,int per_node_arms,bool use_feasible_pool,
       bool competitive_objective,double exploration_constant,
       double progressive_widening_constant,
       double progressive_widening_alpha,int rollout_arms,
       uint64_t search_seed,int prefix_route,int prefix_steps){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    std::vector<int> decision_days;decision_days.reserve(raw_decision_days.size());for(auto value:raw_decision_days)decision_days.push_back(py::cast<int>(value));
    AdaptiveCandidate8MctsResult result;{py::gil_scoped_release release;result=x.candidate8_mcts_oracle(AdaptiveGenome::from_vector(values),opponent_route,actual_seed,decision_days,candidate_seat,simulation_budget,per_node_arms,use_feasible_pool,competitive_objective,exploration_constant,progressive_widening_constant,progressive_widening_alpha,rollout_arms,search_seed,prefix_route,prefix_steps);}
    py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);out["candidate_seat"]=result.candidate_seat;out["simulations"]=result.simulations;out["tree_nodes"]=result.tree_nodes;out["maximum_depth"]=result.maximum_depth;out["first_win_simulation"]=result.first_win_simulation;out["best_found_simulation"]=result.best_found_simulation;out["winning_simulations"]=result.winning_simulations;out["unique_sampled_paths"]=result.unique_sampled_paths;out["sampled_candidate_rewards"]=result.sampled_candidate_rewards;out["sampled_opponent_rewards"]=result.sampled_opponent_rewards;
    py::list days,ranks,families,signatures,counts;for(size_t i=0;i<result.decision_day.size();i++){days.append(result.decision_day[i]);ranks.append(result.selected_rank[i]);families.append(result.selected_family[i]);signatures.append(result.selected_signature[i]);counts.append(result.feasible_count[i]);}
    out["decision_day"]=days;out["selected_rank"]=ranks;out["selected_family"]=families;out["selected_signature"]=signatures;out["feasible_count"]=counts;out["root_rank"]=result.root_rank;out["root_visits"]=result.root_visits;out["root_mean_value"]=result.root_mean_value;return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("actual_seed"),py::arg("decision_days"),py::arg("candidate_seat")=0,py::arg("simulation_budget")=12000,py::arg("per_node_arms")=64,py::arg("use_feasible_pool")=false,py::arg("competitive_objective")=true,py::arg("exploration_constant")=1.25,py::arg("progressive_widening_constant")=2.0,py::arg("progressive_widening_alpha")=0.5,py::arg("rollout_arms")=8,py::arg("search_seed")=1,py::arg("prefix_route")=-1,py::arg("prefix_steps")=0)
  .def("portfolio_counterfactual",[](const NativeAdaptiveExecutor&x,
       py::sequence raw_genome,int opponent_route,uint64_t prefix_seed,
       py::sequence raw_future_seeds,int candidate_seat,int candidate_ranks,
       double switch_margin,int minimum_decision_day){
    std::vector<double> values;values.reserve(raw_genome.size());for(auto value:raw_genome)values.push_back(py::cast<double>(value));
    std::vector<uint64_t> future_seeds;future_seeds.reserve(raw_future_seeds.size());for(auto value:raw_future_seeds)future_seeds.push_back(py::cast<uint64_t>(value));
    AdaptivePortfolioCounterfactualResult result;{py::gil_scoped_release release;result=x.portfolio_counterfactual(AdaptiveGenome::from_vector(values),opponent_route,prefix_seed,future_seeds,candidate_seat,candidate_ranks,switch_margin,minimum_decision_day);}
    py::dict out;out["decision_found"]=result.decision_found;out["decision_step"]=result.decision_step;
    py::array_t<double> rewards({(ssize_t)result.future_count,(ssize_t)result.arms});auto rr=rewards.mutable_unchecked<2>();
    py::array_t<int32_t> overflow({(ssize_t)result.future_count,(ssize_t)result.arms});auto oo=overflow.mutable_unchecked<2>();
    for(int f=0;f<result.future_count;f++)for(int a=0;a<result.arms;a++){const int index=f*result.arms+a;rr(f,a)=result.own_rewards[index];oo(f,a)=result.end_overflow[index];}
    py::array_t<int8_t> available(result.arms);auto aa=available.mutable_unchecked<1>();for(int a=0;a<result.arms;a++)aa(a)=result.arm_available[a];
    py::array_t<int32_t> features({(ssize_t)result.arms,(ssize_t)AdaptivePlan::SWITCH_FEATURE_DIM});auto ff=features.mutable_unchecked<2>();for(int a=0;a<result.arms;a++)for(int f=0;f<AdaptivePlan::SWITCH_FEATURE_DIM;f++)ff(a,f)=result.arm_features[a][f];
    out["rewards"]=rewards;out["end_overflow"]=overflow;out["arm_available"]=available;out["arm_features"]=features;return out;
  },py::arg("genome"),py::arg("opponent_route"),py::arg("prefix_seed"),py::arg("future_seeds"),py::arg("candidate_seat")=0,py::arg("candidate_ranks")=4,py::arg("switch_margin")=1e-6,py::arg("minimum_decision_day")=0)
  .def("play_batch",[](const NativeAdaptiveExecutor&x,
       py::array_t<double,py::array::c_style|py::array::forcecast> genomes,
       py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto gg=genomes.unchecked<2>();auto tt=tasks.unchecked<2>();
    if(gg.shape(1)!=AdaptiveGenome::DIM)throw std::invalid_argument("genomes must have AdaptiveGenome::DIM columns");
    if(tt.shape(1)!=4)throw std::invalid_argument("tasks must have columns genome_index,opponent_route,seed,candidate_seat");
    std::vector<AdaptiveGenome> parsed;parsed.reserve(gg.shape(0));
    for(ssize_t i=0;i<gg.shape(0);i++){std::vector<double> row(AdaptiveGenome::DIM);for(int j=0;j<AdaptiveGenome::DIM;j++)row[j]=gg(i,j);parsed.push_back(AdaptiveGenome::from_vector(row));}
    py::array_t<double> rewards({tt.shape(0),(ssize_t)2});auto rr=rewards.mutable_unchecked<2>();
    constexpr ssize_t ADAPTIVE_DIAGNOSTIC_DIM=22+AdaptivePlan::SWITCH_FEATURE_DIM;
    py::array_t<int32_t> diagnostics({tt.shape(0),ADAPTIVE_DIAGNOSTIC_DIM});auto dd=diagnostics.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<tt.shape(0);i++){
       const int genome=(int)tt(i,0);if(genome<0||genome>=(int)parsed.size())continue;
       const auto r=x.play(parsed[genome],(int)tt(i,1),(uint64_t)tt(i,2),(int)tt(i,3),false);
       rr(i,0)=r.rewards[0];rr(i,1)=r.rewards[1];dd(i,0)=r.replans;dd(i,1)=r.override_actions;
       dd(i,2)=r.avoidable_crop_losses;dd(i,3)=r.avoidable_animal_losses;dd(i,4)=r.end_overflow;dd(i,5)=r.candidate_seat;
        for(int a=0;a<N_ANIMALS;a++){dd(i,6+a)=r.max_animal_targets[a];dd(i,14+a)=r.final_animals[a];}
        for(int c=0;c<N_CROPS;c++)dd(i,9+c)=r.max_crop_targets[c];
        dd(i,17)=r.final_operating_prior_index;dd(i,18)=r.operating_prior_switches;dd(i,19)=r.bundle_switches;dd(i,20)=(int32_t)std::nearbyint(r.bundle_predicted_gain);dd(i,21)=r.first_bundle_switch_day;
        for(int f=0;f<AdaptivePlan::SWITCH_FEATURE_DIM;f++)dd(i,22+f)=r.first_bundle_switch_features[f];
     }}
    return py::make_tuple(rewards,diagnostics);
  },py::arg("genomes"),py::arg("tasks"))
  .def("play_market_preempt_batch",[](const NativeAdaptiveExecutor&x,
       py::array_t<double,py::array::c_style|py::array::forcecast> genomes,
       py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto gg=genomes.unchecked<2>();auto tt=tasks.unchecked<2>();
    if(gg.shape(1)!=AdaptiveGenome::DIM)throw std::invalid_argument("genomes must have AdaptiveGenome::DIM columns");
    if(tt.shape(1)!=8)throw std::invalid_argument("market-preempt tasks must have columns genome_index,opponent_route,seed,candidate_seat,fraction_x1000,min_ready,min_price_ratio_x1000,max_quantity");
    std::vector<AdaptiveGenome> parsed;parsed.reserve(gg.shape(0));
    for(ssize_t i=0;i<gg.shape(0);i++){std::vector<double> row(AdaptiveGenome::DIM);for(int j=0;j<AdaptiveGenome::DIM;j++)row[j]=gg(i,j);parsed.push_back(AdaptiveGenome::from_vector(row));}
    py::array_t<double> rewards({tt.shape(0),(ssize_t)2});auto rr=rewards.mutable_unchecked<2>();
    py::array_t<int32_t> diagnostics({tt.shape(0),(ssize_t)5});auto dd=diagnostics.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<tt.shape(0);i++){
       const int genome_index=(int)tt(i,0);if(genome_index<0||genome_index>=(int)parsed.size())continue;
       AdaptiveGenome genome=parsed[genome_index];genome.runtime_market_preempt_fraction=std::clamp(double(tt(i,4))/1000.0,0.0,1.0);genome.runtime_market_preempt_min_ready=std::clamp((int)tt(i,5),1,1000);genome.runtime_market_preempt_min_price_ratio=std::clamp(double(tt(i,6))/1000.0,0.0,4.0);genome.runtime_market_preempt_max_quantity=std::clamp((int)tt(i,7),1,1000);
       const auto r=x.play(genome,(int)tt(i,1),(uint64_t)tt(i,2),(int)tt(i,3),false);
       rr(i,0)=r.rewards[0];rr(i,1)=r.rewards[1];dd(i,0)=r.avoidable_crop_losses;dd(i,1)=r.avoidable_animal_losses;dd(i,2)=r.end_overflow;dd(i,3)=r.replans;dd(i,4)=r.override_actions;
     }}
    return py::make_tuple(rewards,diagnostics);
  },py::arg("genomes"),py::arg("tasks"))
  .def("play_blend_batch",[](const NativeAdaptiveExecutor&x,
       py::array_t<double,py::array::c_style|py::array::forcecast> genomes,
       py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto gg=genomes.unchecked<2>();auto tt=tasks.unchecked<2>();
    if(gg.shape(1)!=AdaptiveGenome::DIM)throw std::invalid_argument("genomes must have AdaptiveGenome::DIM columns");
    if(tt.shape(1)!=6)throw std::invalid_argument("blend tasks must have columns genome_index,base_route,opponent_route,seed,candidate_seat,blend_mode");
    std::vector<AdaptiveGenome> parsed;parsed.reserve(gg.shape(0));
    for(ssize_t i=0;i<gg.shape(0);i++){std::vector<double> row(AdaptiveGenome::DIM);for(int j=0;j<AdaptiveGenome::DIM;j++)row[j]=gg(i,j);parsed.push_back(AdaptiveGenome::from_vector(row));}
    py::array_t<double> rewards({tt.shape(0),(ssize_t)2});auto rr=rewards.mutable_unchecked<2>();
    constexpr ssize_t ADAPTIVE_DIAGNOSTIC_DIM=22+AdaptivePlan::SWITCH_FEATURE_DIM;
    py::array_t<int32_t> diagnostics({tt.shape(0),ADAPTIVE_DIAGNOSTIC_DIM});auto dd=diagnostics.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<tt.shape(0);i++){
       const int genome=(int)tt(i,0);if(genome<0||genome>=(int)parsed.size())continue;
       const auto r=x.play_blend(parsed[genome],(int)tt(i,1),(int)tt(i,2),(uint64_t)tt(i,3),(int)tt(i,4),(int)tt(i,5),false);
       rr(i,0)=r.rewards[0];rr(i,1)=r.rewards[1];dd(i,0)=r.replans;dd(i,1)=r.override_actions;
       dd(i,2)=r.avoidable_crop_losses;dd(i,3)=r.avoidable_animal_losses;dd(i,4)=r.end_overflow;dd(i,5)=r.candidate_seat;
        for(int a=0;a<N_ANIMALS;a++){dd(i,6+a)=r.max_animal_targets[a];dd(i,14+a)=r.final_animals[a];}
        for(int c=0;c<N_CROPS;c++)dd(i,9+c)=r.max_crop_targets[c];
        dd(i,17)=r.final_operating_prior_index;dd(i,18)=r.operating_prior_switches;dd(i,19)=r.bundle_switches;dd(i,20)=(int32_t)std::nearbyint(r.bundle_predicted_gain);dd(i,21)=r.first_bundle_switch_day;
        for(int f=0;f<AdaptivePlan::SWITCH_FEATURE_DIM;f++)dd(i,22+f)=r.first_bundle_switch_features[f];
     }}
    return py::make_tuple(rewards,diagnostics);
  },py::arg("genomes"),py::arg("tasks"))
  .def("play_prefix_batch",[](const NativeAdaptiveExecutor&x,
       py::array_t<double,py::array::c_style|py::array::forcecast> genomes,
       py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto gg=genomes.unchecked<2>();auto tt=tasks.unchecked<2>();
    if(gg.shape(1)!=AdaptiveGenome::DIM)throw std::invalid_argument("genomes must have AdaptiveGenome::DIM columns");
    if(tt.shape(1)!=6)throw std::invalid_argument("prefix tasks must have columns genome_index,prefix_route,opponent_route,seed,candidate_seat,prefix_steps");
    std::vector<AdaptiveGenome> parsed;parsed.reserve(gg.shape(0));
    for(ssize_t i=0;i<gg.shape(0);i++){std::vector<double> row(AdaptiveGenome::DIM);for(int j=0;j<AdaptiveGenome::DIM;j++)row[j]=gg(i,j);parsed.push_back(AdaptiveGenome::from_vector(row));}
    py::array_t<double> rewards({tt.shape(0),(ssize_t)2});auto rr=rewards.mutable_unchecked<2>();
    constexpr ssize_t ADAPTIVE_DIAGNOSTIC_DIM=5;
    py::array_t<int32_t> diagnostics({tt.shape(0),ADAPTIVE_DIAGNOSTIC_DIM});auto dd=diagnostics.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<tt.shape(0);i++){
       const int genome=(int)tt(i,0);if(genome<0||genome>=(int)parsed.size())continue;
       const auto r=x.play_blend(parsed[genome],(int)tt(i,1),(int)tt(i,2),(uint64_t)tt(i,3),(int)tt(i,4),6,false,(int)tt(i,5));
       rr(i,0)=r.rewards[0];rr(i,1)=r.rewards[1];dd(i,0)=r.avoidable_crop_losses;dd(i,1)=r.avoidable_animal_losses;dd(i,2)=r.end_overflow;dd(i,3)=r.replans;dd(i,4)=r.override_actions;
     }}
    return py::make_tuple(rewards,diagnostics);
  },py::arg("genomes"),py::arg("tasks"));
 m.def("generate_adaptive_candidates",[](py::dict raw_context,int max_shortlist,int max_raw,bool include_feasible){
    AdaptiveCandidateContext c;
    auto get_int=[&](const char*key,int fallback){return raw_context.contains(key)?py::cast<int>(raw_context[key]):fallback;};
    auto get_double=[&](const char*key,double fallback){return raw_context.contains(key)?py::cast<double>(raw_context[key]):fallback;};
    auto load_i16=[&](const char*key,auto&target,int expected,bool required){
      if(!raw_context.contains(key)){if(required)throw std::invalid_argument(std::string("missing candidate context field: ")+key);return;}
      const auto values=py::cast<std::vector<int>>(raw_context[key]);if(int(values.size())!=expected)throw std::invalid_argument(std::string(key)+" has invalid length");
      for(int i=0;i<expected;i++)target[i]=int16_t(values[i]);
    };
    auto load_double=[&](const char*key,auto&target,int expected,bool required){
      if(!raw_context.contains(key)){if(required)throw std::invalid_argument(std::string("missing candidate context field: ")+key);return;}
      const auto values=py::cast<std::vector<double>>(raw_context[key]);if(int(values.size())!=expected)throw std::invalid_argument(std::string(key)+" has invalid length");
      for(int i=0;i<expected;i++)target[i]=values[i];
    };
    c.day=get_int("day",0);
    load_i16("targets",c.targets,ADAPTIVE_PROJECTS,true);
    load_i16("irreversible_floor",c.irreversible_floor,ADAPTIVE_PROJECTS,true);
    load_i16("caps",c.caps,ADAPTIVE_PROJECTS,true);
    load_double("marginal_value",c.marginal_value,ADAPTIVE_PROJECTS,true);
    load_double("purchase_cost",c.purchase_cost,ADAPTIVE_PROJECTS,true);
    load_double("daily_action_load",c.daily_action_load,ADAPTIVE_PROJECTS,true);
    load_i16("first_cash_lag_days",c.first_cash_lag_days,ADAPTIVE_PROJECTS,true);
    c.liquid_cash=get_double("liquid_cash",0.0);c.protected_cash=get_double("protected_cash",0.0);
    c.financeable_inventory_value=get_double("financeable_inventory_value",0.0);
    c.unlocked_quadrants=get_int("unlocked_quadrants",1);c.maximum_quadrants=get_int("maximum_quadrants",4);
    c.productive_tiles=get_int("productive_tiles",0);c.hands=get_int("hands",0);c.maximum_hands=get_int("maximum_hands",12);
    c.next_hand_cost=get_double("next_hand_cost",0.0);c.next_quadrant_cost=get_double("next_quadrant_cost",0.0);
    c.tiles_per_quadrant=get_int("tiles_per_quadrant",25);
    c.current_daily_action_load=get_double("current_daily_action_load",0.0);
    c.hard_deadline_load=get_double("hard_deadline_load",0.0);c.estimated_travel_load=get_double("estimated_travel_load",0.0);
    c.delayed_loss=get_double("delayed_loss",0.0);c.market_slots_available=get_int("market_slots_available",10);
    c.recovery_issues=uint8_t(get_int("recovery_issues",0));
    load_i16("sellable_inventory",c.sellable_inventory,ADAPTIVE_PRODUCTS,false);
    load_i16("market_prices",c.market_prices,ADAPTIVE_PRODUCTS,false);
    load_i16("demand_within_day",c.demand_within_day,ADAPTIVE_PRODUCTS,false);
    AdaptiveCandidateSet result;{py::gil_scoped_release release;result=generate_adaptive_candidates(c,max_shortlist,max_raw);}
    auto counts_dict=[](const std::array<int16_t,9>&values){py::dict out;for(int family=0;family<9;family++)out[py::str(candidate_family_name(CandidateFamily(family)))]=values[family];return out;};
    py::dict out;out["raw_count"]=result.raw_count;out["feasible_count"]=result.feasible_count;out["duplicate_count"]=result.duplicate_count;
    out["raw_by_family"]=counts_dict(result.raw_by_family);out["feasible_by_family"]=counts_dict(result.feasible_by_family);
    out["shortlisted_by_family"]=counts_dict(result.shortlisted_by_family);py::list shortlist;for(const auto&candidate:result.shortlist)shortlist.append(candidate_delta_dict(candidate));out["shortlist"]=shortlist;
    if(include_feasible){py::list feasible;for(const auto&candidate:result.feasible)feasible.append(candidate_delta_dict(candidate));out["feasible"]=feasible;}
    return out;
  },py::arg("context"),py::arg("max_shortlist")=64,py::arg("max_raw")=1000,py::arg("include_feasible")=true);
 m.def("adaptive_candidate_family_names",[](){py::list out;for(int family=0;family<9;family++)out.append(std::string(candidate_family_name(CandidateFamily(family))));return out;});
 m.def("adaptive_genome_names",[](){py::list out;for(const char* name:AdaptiveGenome::names())out.append(name);return out;});
 m.def("adaptive_default_genome",[](){AdaptiveGenome g;return std::vector<double>{
   g.cash_reserve,g.action_cost,g.move_cost,g.risk_multiplier,g.market_impact_weight,g.opponent_supply_weight,
   g.demand_drift_weight,g.sell_drop_limit,g.price_replan_fraction,g.task_stickiness,g.deadline_weight,
   g.fertilizer_value_fraction,double(g.max_hands),double(g.max_quadrants),double(g.max_total_animals),
   double(g.max_cows),double(g.max_sheep),double(g.max_geese),double(g.max_wheat),double(g.max_carrot),
   double(g.max_tomato),double(g.max_strawberry),double(g.max_melon),double(g.min_wheat_buffer),
   double(g.liquidation_day),double(g.stop_new_animals_day),double(g.stop_new_crops_day),double(g.replan_interval_steps),
   g.routine_priority_scale,g.task_value_scale,g.plant_priority,
   g.drop_value_threshold,double(g.preempt_quantity),
   g.wheat_relay_capacity_fraction,g.wheat_relay_opponent_risk,
   g.industry_affinity_scale,g.route_density_scale,g.flow_pace_priority,
   g.prerequisite_chain_strength,g.local_candidate_order,
   g.region_ownership_scale,g.animal_flow_control,g.workload_region_mode,
   g.routine_cell_exclusivity,g.hard_travel_slack_scale,
   g.future_structure_reservation_fraction,
   g.yarn_animal_flex,g.yarn_wool_price_threshold,
   g.yarn_opponent_sheep_gate,g.pet_crop_flex,
   g.pet_carrot_price_threshold,g.weed_capacity_priority,
   g.animal_flow_pressure_gate,g.animal_flow_executable_gate,
   g.terminal_animal_economics,g.crop_lane_layout_mode,
   g.region_assignment_mode,g.hard_latest_start_reservation,
   g.enroute_task_reservation,g.global_routine_matching,
   g.backbone_crop_flex,g.region_owner_first,
   g.quadrant_affinity_scale,g.opening_cash_reserve_fraction,
   g.region_anchor_distance_scale,g.town_demand_sell_timing,
   g.repeating_crop_economics,g.proactive_land_investment,
   g.capital_lockup_weight,g.backbone_animal_flex,
   g.animal_branch_economic_selector,
   double(g.animal_branch_observation_days),
   g.seed_transaction_value_order,
   double(g.animal_counter_crowding_gate),
    g.due_flow_cash_release,
    g.market_race_acceleration,
    g.operating_prior_selection,
    g.operating_prior_switch_margin,
    g.autonomous_commitment_persistence,
    g.land_capacity_trigger_fraction,
    g.crop_specific_terminal_horizon,
    g.autonomous_project_state_machine,
   g.future_shop_expectation_weight,
   g.portfolio_supply_impact_weight,
    g.live_commitment_projection_weight,
    g.portfolio_bundle_switch_margin,
    double(g.portfolio_switch_cooldown_days),
    double(g.portfolio_switch_candidate_rank),
    double(g.portfolio_local_edit_hold_days)};});
 m.def("native_threshold_variants",[](py::array_t<int32_t,py::array::c_style|py::array::forcecast> left,
      py::array_t<int32_t,py::array::c_style|py::array::forcecast> right,
      py::array_t<int32_t,py::array::c_style|py::array::forcecast> feature,
      py::array_t<double,py::array::c_style|py::array::forcecast> threshold,
      py::array_t<int32_t,py::array::c_style|py::array::forcecast> leaf_class,
      py::array_t<float,py::array::c_style|py::array::forcecast> matrix,
      py::array_t<float,py::array::c_style|py::array::forcecast> train_matrix,
      uint64_t random_seed){
    auto ll=left.unchecked<1>();auto rr=right.unchecked<1>();auto ff=feature.unchecked<1>();
    auto tt=threshold.unchecked<1>();auto cc=leaf_class.unchecked<1>();
    auto xx=matrix.unchecked<2>(),train=train_matrix.unchecked<2>();const ssize_t nodes=ll.shape(0),samples=xx.shape(0);
    if(rr.shape(0)!=nodes||ff.shape(0)!=nodes||tt.shape(0)!=nodes||cc.shape(0)!=nodes||xx.shape(1)!=train.shape(1))throw std::invalid_argument("incompatible native threshold arrays");
    std::vector<double>deltas(nodes,0.);auto quantile=[](std::vector<float>&v,double q){if(v.empty())return 0.;std::sort(v.begin(),v.end());double at=(v.size()-1)*q;size_t lo=(size_t)std::floor(at),hi=(size_t)std::ceil(at);return double(v[lo])+(double(v[hi])-v[lo])*(at-lo);};
    for(ssize_t node=0;node<nodes;node++)if(ll(node)>=0){int f=ff(node);std::vector<float>values;values.reserve(train.shape(0));for(ssize_t i=0;i<train.shape(0);i++)values.push_back(train(i,f));double low=quantile(values,.05),high=quantile(values,.95);deltas[node]=std::max(1e-6,.05*(high-low));}
    py::array_t<int32_t>out({(ssize_t)10,samples});auto yy=out.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for collapse(2) schedule(static)
     for(int variant=0;variant<10;variant++)for(ssize_t i=0;i<samples;i++){int node=0;while(ll(node)>=0){int sign;if(variant==0)sign=-1;else if(variant==1)sign=1;else{uint64_t z=random_seed^uint64_t(variant+1)*0x9e3779b97f4a7c15ULL^uint64_t(node+1)*0xbf58476d1ce4e5b9ULL;z^=z>>30;z*=0xbf58476d1ce4e5b9ULL;z^=z>>27;z*=0x94d049bb133111ebULL;z^=z>>31;sign=(z&1)?1:-1;}double split=tt(node)+sign*deltas[node];node=xx(i,ff(node))<=split?ll(node):rr(node);}yy(variant,i)=cc(node);}}
    return out;
  },py::arg("left"),py::arg("right"),py::arg("feature"),py::arg("threshold"),py::arg("leaf_class"),py::arg("matrix"),py::arg("train_matrix"),py::arg("random_seed"));
 m.def("native_tree_predict",[](py::array_t<int32_t,py::array::c_style|py::array::forcecast> left,
      py::array_t<int32_t,py::array::c_style|py::array::forcecast> right,
      py::array_t<int32_t,py::array::c_style|py::array::forcecast> feature,
      py::array_t<double,py::array::c_style|py::array::forcecast> threshold,
      py::array_t<int32_t,py::array::c_style|py::array::forcecast> leaf_class,
      py::array_t<float,py::array::c_style|py::array::forcecast> matrix){
    auto ll=left.unchecked<1>();auto rr=right.unchecked<1>();auto ff=feature.unchecked<1>();auto tt=threshold.unchecked<1>();auto cc=leaf_class.unchecked<1>();auto xx=matrix.unchecked<2>();
    py::array_t<int32_t>out(xx.shape(0));auto yy=out.mutable_unchecked<1>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(static)
     for(ssize_t i=0;i<xx.shape(0);i++){int node=0;while(ll(node)>=0)node=xx(i,ff(node))<=tt(node)?ll(node):rr(node);yy(i)=cc(node);}}
    return out;
  },py::arg("left"),py::arg("right"),py::arg("feature"),py::arg("threshold"),py::arg("leaf_class"),py::arg("matrix"));
}
