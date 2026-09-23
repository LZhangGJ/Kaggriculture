// Licensed under the Apache License, Version 2.0.
#include "simulator.hpp"
#include "native_teammate.hpp"
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

py::dict repair_audit_dict(const NativeRepairAudit& a) {
  py::dict out;
  out["weed_triggers"] = a.weed_triggers;
  out["weed_absorbed_pass"] = a.weed_absorbed_pass;
  out["weed_absorbed_productive"] = a.weed_absorbed_productive;
  out["weed_legacy_would_drop_move"] = a.weed_legacy_would_drop_move;
  out["weed_no_safe_alignment"] = a.weed_no_safe_alignment;
  out["movement_edits"] = a.movement_edits;
  out["animal_original_attempted"] = a.animal_original_attempted;
  out["animal_inferred_filled"] = a.animal_inferred_filled;
  out["animal_inferred_partial"] = a.animal_inferred_partial;
  out["animal_inferred_zero"] = a.animal_inferred_zero;
  out["animal_retries_emitted"] = a.animal_retries_emitted;
  out["animal_retry_pickup_realign"] = a.animal_retry_pickup_realign;
  out["animal_retry_blocked_no_path"] = a.animal_retry_blocked_no_path;
  out["empty_stall_reused"] = a.empty_stall_reused;
  out["empty_stall_unresolved"] = a.empty_stall_unresolved;
  out["local_repair_decisions"] = a.local_repair_decisions;
  out["local_repair_commits"] = a.local_repair_commits;
  out["local_repair_receipts_confirmed"] =
      a.local_repair_receipts_confirmed;
  out["local_repair_receipts_failed"] = a.local_repair_receipts_failed;
  out["local_repair_fail_closed"] = a.local_repair_fail_closed;
  out["local_repair_seed_orders"] = a.local_repair_seed_orders;
  out["local_repair_seed_units"] = a.local_repair_seed_units;
  out["local_repair_seed_fills"] = a.local_repair_seed_fills;
  out["local_repair_seed_zero_fills"] =
      a.local_repair_seed_zero_fills;
  out["local_repair_transaction_claims"] =
      a.local_repair_transaction_claims;
  out["local_repair_transaction_claim_rejections"] =
      a.local_repair_transaction_claim_rejections;
  out["day_horizon_plans"] = a.day_horizon_plans;
  out["day_horizon_commits"] = a.day_horizon_commits;
  out["day_horizon_recompiles"] = a.day_horizon_recompiles;
  out["day_horizon_fail_closed"] = a.day_horizon_fail_closed;
  out["day_horizon_assignments"] = a.day_horizon_assignments;
  out["day_horizon_terminal_raw"] = a.day_horizon_terminal_raw;
  out["day_horizon_v2_plans"] = a.day_horizon_v2_plans;
  out["day_horizon_v2_replans"] = a.day_horizon_v2_replans;
  out["day_horizon_v2_receipts_confirmed"] =
      a.day_horizon_v2_receipts_confirmed;
  out["day_horizon_v2_receipts_failed"] =
      a.day_horizon_v2_receipts_failed;
  out["day_horizon_v2_fail_closed"] = a.day_horizon_v2_fail_closed;
  out["day_horizon_v2_assignments"] = a.day_horizon_v2_assignments;
  out["day_horizon_v2_exact_plans"] = a.day_horizon_v2_exact_plans;
  out["day_horizon_v2_fallback_plans"] =
      a.day_horizon_v2_fallback_plans;
  out["day_horizon_v2_budget_exhausted"] =
      a.day_horizon_v2_budget_exhausted;
  out["day_horizon_v2_objectives_completed"] =
      a.day_horizon_v2_objectives_completed;
  out["day_horizon_v2_objectives_carried"] =
      a.day_horizon_v2_objectives_carried;
  out["day_horizon_v2_maturity_waits"] =
      a.day_horizon_v2_maturity_waits;
  out["day_horizon_v2_maturity_tokens_consumed"] =
      a.day_horizon_v2_maturity_tokens_consumed;
  out["day_horizon_v2_seed_unscheduled"] =
      a.day_horizon_v2_seed_unscheduled;
  out["day_horizon_v2_seed_orders"] = a.day_horizon_v2_seed_orders;
  out["day_horizon_v2_seed_zero_fills"] =
      a.day_horizon_v2_seed_zero_fills;
  out["day_horizon_v2_seed_fills"] = a.day_horizon_v2_seed_fills;
  out["day_horizon_v2_terminal_raw"] = a.day_horizon_v2_terminal_raw;
  out["day_horizon_v2_fail_reasons"] = a.day_horizon_v2_fail_reasons;
  out["route_skeleton_v3_plans"] = a.route_skeleton_v3_plans;
  out["route_skeleton_v3_replans"] = a.route_skeleton_v3_replans;
  out["route_skeleton_v3_rebases"] = a.route_skeleton_v3_rebases;
  out["route_skeleton_v3_lcs_kept"] = a.route_skeleton_v3_lcs_kept;
  out["route_skeleton_v3_receipts_confirmed"] =
      a.route_skeleton_v3_receipts_confirmed;
  out["route_skeleton_v3_receipts_failed"] =
      a.route_skeleton_v3_receipts_failed;
  out["route_skeleton_v3_fail_closed"] =
      a.route_skeleton_v3_fail_closed;
  out["route_skeleton_v3_assignments"] =
      a.route_skeleton_v3_assignments;
  out["route_skeleton_v3_objectives_completed"] =
      a.route_skeleton_v3_objectives_completed;
  out["route_skeleton_v3_objectives_carried"] =
      a.route_skeleton_v3_objectives_carried;
  out["route_skeleton_v3_moves_emitted"] =
      a.route_skeleton_v3_moves_emitted;
  out["route_skeleton_v3_terminal_moves"] =
      a.route_skeleton_v3_terminal_moves;
  out["route_skeleton_v3_fallback_plans"] =
      a.route_skeleton_v3_fallback_plans;
  out["route_skeleton_v3_budget_exhausted"] =
      a.route_skeleton_v3_budget_exhausted;
  out["route_skeleton_v3_fail_reasons"] =
      a.route_skeleton_v3_fail_reasons;
  return out;
}

py::dict economy_audit_dict(const NativeGeneralMarketAudit& a) {
  py::dict out;
  out["obligation_nodes"] = a.obligation_nodes;
  out["purchase_orders_due"] = a.purchase_orders_due;
  out["funding_sell_orders"] = a.funding_sell_orders;
  out["funding_sell_units"] = a.funding_sell_units;
  out["allocated_orders"] = a.allocated_orders;
  out["compiler_feasible"] = a.compiler_feasible;
  out["allocator_feasible"] = a.allocator_feasible;
  out["diagnostic_code"] = a.diagnostic_code;
  out["diagnostic_step"] = a.diagnostic_step;
  out["diagnostic_actor"] = a.diagnostic_actor;
  out["diagnostic_item"] = a.diagnostic_item;
  out["diagnostic_quantity"] = a.diagnostic_quantity;
  out["soft_misses"] = a.soft_misses;
  out["soft_pickup_replacements"] = a.soft_pickup_replacements;
  out["soft_pass_replacements"] = a.soft_pass_replacements;
  out["soft_downstream_loss_proxy"] = a.soft_downstream_loss_proxy;
  out["production_protection_soft_misses"] =
      a.production_protection_soft_misses;
  out["first_due_kind"] = a.first_due_kind;
  out["first_due_item"] = a.first_due_item;
  out["first_due_quantity"] = a.first_due_quantity;
  out["first_due_cash_quote"] = a.first_due_cash_quote;
  out["first_due_free_capacity"] = a.first_due_free_capacity;
  out["reason"] = a.reason;
  return out;
}

py::dict phased_market_audit_dict(const NativePhasedMarketAudit& a) {
  py::dict out;
  out["arm"] = a.arm;
  out["calls"] = a.calls;
  out["protected_steps"] = a.protected_steps;
  out["selective_steps"] = a.selective_steps;
  out["full_steps"] = a.full_steps;
  out["protected_queue_calls"] = a.protected_queue_calls;
  out["phase_gate_calls"] = a.phase_gate_calls;
  out["no_candidate_steps"] = a.no_candidate_steps;
  out["uncertified_steps"] = a.uncertified_steps;
  out["exact_legacy_fallbacks"] = a.exact_legacy_fallbacks;
  out["production_dag_certified_steps"] = a.production_dag_certified_steps;
  out["v2_exact_legacy_fallbacks"] = a.v2_exact_legacy_fallbacks;
  out["production_obligation_nodes"] = a.production_obligation_nodes;
  out["due_legacy_bindings"] = a.due_legacy_bindings;
  out["protected_non_sell_orders"] = a.protected_non_sell_orders;
  out["required_funding_sell_orders"] = a.required_funding_sell_orders;
  out["required_funding_sell_units"] = a.required_funding_sell_units;
  out["optional_replacement_orders"] = a.optional_replacement_orders;
  out["optional_replacement_units"] = a.optional_replacement_units;
  out["search_states"] = a.search_states;
  out["observation_resets"] = a.observation_resets;
  out["observation_updates"] = a.observation_updates;
  out["observation_stages"] = a.observation_stages;
  out["observation_failures"] = a.observation_failures;
  out["builder_calls"] = a.builder_calls;
  out["builder_rejections"] = a.builder_rejections;
  out["runtime_calls"] = a.runtime_calls;
  out["runtime_selected_steps"] = a.runtime_selected_steps;
  out["sell_compaction_selected_steps"] =
      a.sell_compaction_selected_steps;
  out["exact_terminal_liquidation_selected_steps"] =
      a.exact_terminal_liquidation_selected_steps;
  out["general_robust_selected_steps"] =
      a.general_robust_selected_steps;
  out["runtime_fallback_steps"] = a.runtime_fallback_steps;
  out["market_changed_steps"] = a.market_changed_steps;
  out["market_changed_slots"] = a.market_changed_slots;
  out["maximum_opponent_total_upper_units"] =
      a.maximum_opponent_total_upper_units;
  out["maximum_scenario_dump_units"] = a.maximum_scenario_dump_units;
  out["runtime_total_us"] = a.runtime_total_us;
  out["runtime_planner_us"] = a.runtime_planner_us;
  out["observation_us"] = a.observation_us;
  out["builder_us"] = a.builder_us;
  out["native_overlay_total_us"] = a.native_overlay_total_us;
  py::dict fallback_counts;
  for (int fallback = 0; fallback < kNativeRuntimeFallbackKinds; ++fallback) {
    const auto value = static_cast<selective_runtime::Fallback>(fallback);
    fallback_counts[selective_runtime::fallback_name(value)] =
        a.runtime_fallback_counts[fallback];
  }
  out["runtime_fallback_counts"] = std::move(fallback_counts);
  py::dict zero_target_fallback_counts;
  py::dict virtual_no_sell_fallback_counts;
  for (int fallback = 0; fallback < kNativeRuntimeFallbackKinds; ++fallback) {
    const auto value = static_cast<selective_runtime::Fallback>(fallback);
    const char* name = selective_runtime::fallback_name(value);
    zero_target_fallback_counts[name] =
        a.zero_target_fallback_counts[fallback];
    virtual_no_sell_fallback_counts[name] =
        a.virtual_no_sell_fallback_counts[fallback];
  }
  out["zero_target_fallback_counts"] =
      std::move(zero_target_fallback_counts);
  out["virtual_no_sell_fallback_counts"] =
      std::move(virtual_no_sell_fallback_counts);
  py::dict bridge_reason_counts;
  py::dict virtual_no_sell_bridge_reason_counts;
  py::dict bridge_reason_fallback_counts;
  for (int reason = 0; reason < kNativeBridgeReasonKinds; ++reason) {
    const auto value = static_cast<economic_intent_bridge::Reason>(reason);
    const char* name = economic_intent_bridge::reason_name(value);
    bridge_reason_counts[name] = a.bridge_reason_counts[reason];
    virtual_no_sell_bridge_reason_counts[name] =
        a.virtual_no_sell_bridge_reason_counts[reason];
    py::dict by_fallback;
    for (int fallback = 0; fallback < kNativeRuntimeFallbackKinds; ++fallback) {
      const int count = a.bridge_reason_fallback_counts[reason][fallback];
      if (count == 0) continue;
      by_fallback[selective_runtime::fallback_name(
          static_cast<selective_runtime::Fallback>(fallback))] = count;
    }
    bridge_reason_fallback_counts[name] = std::move(by_fallback);
  }
  out["bridge_reason_counts"] = std::move(bridge_reason_counts);
  out["virtual_no_sell_bridge_reason_counts"] =
      std::move(virtual_no_sell_bridge_reason_counts);
  out["bridge_reason_fallback_counts"] =
      std::move(bridge_reason_fallback_counts);
  py::dict intent_mode_counts;
  for (int mode = 0; mode < kNativeIntentModeKinds; ++mode) {
    const auto value = static_cast<economic_intent_bridge::IntentMode>(mode);
    intent_mode_counts[economic_intent_bridge::intent_mode_name(value)] =
        a.intent_mode_counts[mode];
  }
  out["intent_mode_counts"] = std::move(intent_mode_counts);
  out["zero_target_steps"] = a.zero_target_steps;
  out["state_only_continuation_steps"] = a.state_only_continuation_steps;
  out["virtual_no_sell_steps"] = a.virtual_no_sell_steps;
  out["first_virtual_no_sell_rejection_step"] =
      a.first_virtual_no_sell_rejection_step;
  out["first_virtual_no_sell_rejection_reason"] =
      a.first_virtual_no_sell_rejection_reason;
  out["first_sell_intent_rejection_step"] =
      a.first_sell_intent_rejection_step;
  out["first_sell_intent_rejection_reason"] =
      a.first_sell_intent_rejection_reason;
  out["full_ever_selected"] = a.full_ever_selected;
  out["last_mode"] = a.last_mode;
  out["last_runtime_fallback"] = a.last_runtime_fallback;
  out["last_runtime_fallback_name"] = a.last_runtime_fallback_name;
  out["last_observation_reason"] = a.last_observation_reason;
  out["last_reason"] = a.last_reason;
  return out;
}

NativeRepairOptions repair_options_from_mask(int mask) {
  return native_repair_options_from_mask(mask);
}

class NativeReplayOpponent {
 public:
  NativeReplayOpponent(const NativeTeammateExecutor& executor, int route,
                       bool neutral_special_economy, int repair_mask)
      : executor_(&executor),
        neutral_special_economy_(neutral_special_economy) {
    if (repair_mask < 0 || repair_mask > 127)
      throw std::invalid_argument("repair_mask must be in [0,127]");
    repair_options_ = repair_options_from_mask(repair_mask);
    reset(route);
  }

  void reset(int route) {
    if (route < 0 || route >= executor_->route_count())
      throw std::out_of_range("native replay route is outside library");
    route_ = route;
    state_.reset();
    last_action_step_ = -1;
  }

  PlayerAction action(const Simulator& env, int player, int step) {
    if (player < 0 || player > 1)
      throw std::invalid_argument("player must be 0 or 1");
    if (step != env.step_count())
      throw std::invalid_argument("action step does not match FastEnv");
    if (step != last_action_step_ + 1)
      throw std::logic_error(
          "native replay opponent must be reset and called once per step");
    auto result = executor_->action_external(
        env, player, route_, state_, NativeMarketArm::LegacyDefault, nullptr,
        nullptr, neutral_special_economy_, repair_options_, nullptr);
    last_action_step_ = step;
    return result;
  }

  int route() const { return route_; }

 private:
  const NativeTeammateExecutor* executor_;
  NativeAgentState state_;
  bool neutral_special_economy_;
  NativeRepairOptions repair_options_;
  int route_{-1};
  int last_action_step_{-1};
};
}

PYBIND11_MODULE(_fast_kaggriculture,m){m.doc()="Typed C++ Kaggriculture simulator, compatible with kaggle-environments 1.32.7";
 py::enum_<Op>(m,"Op").value("PASS",Op::PASS).value("NORTH",Op::NORTH).value("SOUTH",Op::SOUTH).value("EAST",Op::EAST).value("WEST",Op::WEST).value("DROP",Op::DROP).value("PICKUP",Op::PICKUP).value("PLACE",Op::PLACE).value("PLANT",Op::PLANT).value("WATER",Op::WATER).value("HARVEST",Op::HARVEST).value("FERTILIZE",Op::FERTILIZE).value("DIG",Op::DIG).value("BUILD_COOP",Op::BUILD_COOP).value("BUILD_PASTURE",Op::BUILD_PASTURE).value("FEED",Op::FEED).value("COLLECT_FERTILIZER",Op::COLLECT_FERTILIZER).value("CARE",Op::CARE).value("HIRE",Op::HIRE).value("BUY_LAND",Op::BUY_LAND).value("BUY_SEED",Op::BUY_SEED).value("BUY_PRODUCT",Op::BUY_PRODUCT).value("BUY_ANIMAL",Op::BUY_ANIMAL).value("SELL",Op::SELL);
 py::enum_<Item>(m,"Item").value("NONE",Item::NONE).value("WHEAT",Item::WHEAT).value("CARROT",Item::CARROT).value("TOMATO",Item::TOMATO).value("STRAWBERRY",Item::STRAWBERRY).value("MELON",Item::MELON).value("EGG",Item::EGG).value("MILK",Item::MILK).value("WOOL",Item::WOOL).value("FERTILIZER",Item::FERTILIZER).value("GOOSE",Item::GOOSE).value("COW",Item::COW).value("SHEEP",Item::SHEEP);
 py::class_<Config>(m,"Config").def(py::init<>()).def_readwrite("episode_steps",&Config::episode_steps).def_readwrite("board_size",&Config::board_size).def_readwrite("starting_money",&Config::starting_money).def_readwrite("max_market_orders",&Config::max_market_orders).def_readwrite("turns_per_day",&Config::turns_per_day).def_readwrite("shed_capacity",&Config::shed_capacity).def_readwrite("weed_spawn_chance",&Config::weed_spawn_chance).def_readwrite("town_shop_unlock_interval",&Config::town_shop_unlock_interval).def_readwrite("town_shop_sell_interval",&Config::town_shop_sell_interval).def_readwrite("town_center_sell_interval",&Config::town_center_sell_interval).def_readwrite("farm_hand_cost_mult",&Config::farm_hand_cost_mult);
 py::class_<Simulator>(m,"FastEnv").def(py::init<Config,uint64_t>(),py::arg("config")=Config{},py::arg("seed")=0).def("clone",[](const Simulator&s){return Simulator(s);}).def("reset",[](Simulator&s,uint64_t seed){s.reset(seed);return py::make_tuple(observation(s,0),observation(s,1));}).def("reset_raw",&Simulator::reset).def("reseed_future",&Simulator::reseed_future).def("step",[](Simulator&s,py::object a){s.step(parse_actions(a));return py::make_tuple(observation(s,0),observation(s,1));}).def("step_raw",[](Simulator&s,py::object a){auto actions=parse_actions(a);py::gil_scoped_release release;s.step(actions);}).def("step_packed",[](Simulator&s,py::array_t<int32_t>u,py::array_t<int32_t>uc,py::array_t<int32_t>ma,py::array_t<int32_t>mc){s.step(packed(u,uc,ma,mc));return py::make_tuple(observation(s,0),observation(s,1));}).def("step_packed_raw",[](Simulator&s,py::array_t<int32_t>u,py::array_t<int32_t>uc,py::array_t<int32_t>ma,py::array_t<int32_t>mc){auto actions=packed(u,uc,ma,mc);py::gil_scoped_release release;s.step(actions);}).def("observation",&observation).def_property_readonly("last_market_fills",[](const Simulator&s){return py::make_tuple(s.last_market_fills()[0],s.last_market_fills()[1]);}).def_property_readonly("last_market_cash_shortfalls",[](const Simulator&s){return py::make_tuple(s.last_market_cash_shortfalls()[0],s.last_market_cash_shortfalls()[1]);}).def_property_readonly("last_end_of_day_overflow",[](const Simulator&s){return py::make_tuple(s.last_end_of_day_overflow()[0],s.last_end_of_day_overflow()[1]);}).def_property_readonly("done",&Simulator::done).def_property_readonly("step_count",&Simulator::step_count).def_property_readonly("rewards",[](const Simulator&s){return s.done()?py::make_tuple(s.farms()[0].money,s.farms()[1].money):py::make_tuple(0.,0.);});
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
  .def(py::init([](py::sequence routes,py::object r5,py::object md,py::sequence moon,py::sequence moon_legacy,py::object thomas_predict_pairs){
    NativeTapeLibrary lib;for(auto tape:routes)lib.routes.push_back(parse_tape(tape));lib.r5_reference=parse_tape(r5);lib.md_reference=parse_tape(md);
    if(moon.size()!=5||moon_legacy.size()!=5)throw std::invalid_argument("moon and moon_legacy must each contain five tapes");
    for(int i=0;i<5;i++){lib.moon[i]=parse_tape(moon[i]);lib.moon_legacy[i]=parse_tape(moon_legacy[i]);}
    if(!thomas_predict_pairs.is_none())for(auto entry:thomas_predict_pairs.cast<py::dict>()){
      const int pair=py::cast<int>(entry.first);if(pair<0||pair>=64)throw std::invalid_argument("Thomas predictor pair must be in [0,63]");
      for(auto stream:py::cast<py::sequence>(entry.second)){
        std::vector<NativeTapeLibrary::ThomasPredictEvent> parsed;
        for(auto raw:py::cast<py::sequence>(stream)){
          auto event=py::cast<py::sequence>(raw);if(event.size()!=3)throw std::invalid_argument("Thomas predictor event must be [step,item,quantity]");
          parsed.push_back({py::cast<int16_t>(event[0]),py::cast<int8_t>(event[1]),py::cast<int16_t>(event[2])});
        }lib.thomas_predict_pairs[pair].push_back(std::move(parsed));
      }
    }
    return NativeTeammateExecutor(std::move(lib));
  }),py::arg("routes"),py::arg("r5_reference"),py::arg("md_reference"),py::arg("moon"),py::arg("moon_legacy"),py::arg("thomas_predict_pairs")=py::none())
  .def("play",[](const NativeTeammateExecutor&x,int route0,int route1,uint64_t seed,int switch_step0,int switch_route0,int switch_step1,int switch_route1,bool capture_trace,bool neutral_special_economy,bool experimental_repair,int experimental_repair_player,int experimental_repair_mask,bool experimental_general_takeover,int experimental_general_takeover_player,int experimental_market_arm,int experimental_market_player,int stop_after_steps,int evaluation_compaction_minimum_step,int evaluation_compaction_evidence_mode,int thomas_prefix_player){
    const auto repair=experimental_repair?repair_options_from_mask(experimental_repair_mask):NativeRepairOptions{};
    NativeMatchResult result;{py::gil_scoped_release release;result=x.play(route0,route1,seed,switch_step0,switch_route0,switch_step1,switch_route1,capture_trace,true,neutral_special_economy,repair,experimental_repair_player,experimental_general_takeover,experimental_general_takeover_player,experimental_market_arm,experimental_market_player,stop_after_steps,evaluation_compaction_minimum_step,evaluation_compaction_evidence_mode,thomas_prefix_player);}py::dict out;out["rewards"]=py::make_tuple(result.rewards[0],result.rewards[1]);
    out["macro_unit_failures"]=py::make_tuple(result.macro_unit_failures[0],result.macro_unit_failures[1]);
    out["macro_market_failures"]=py::make_tuple(result.macro_market_failures[0],result.macro_market_failures[1]);
    out["first_macro_failure_step"]=py::make_tuple(result.first_macro_failure_step[0],result.first_macro_failure_step[1]);
    out["first200_unit_failures"]=py::make_tuple(result.first200_unit_failures[0],result.first200_unit_failures[1]);
    out["first200_market_failures"]=py::make_tuple(result.first200_market_failures[0],result.first200_market_failures[1]);
    if(experimental_repair)out["repair_audit"]=py::make_tuple(repair_audit_dict(result.repair_audit[0]),repair_audit_dict(result.repair_audit[1]));
    if(experimental_general_takeover)out["economy_audit"]=py::make_tuple(economy_audit_dict(result.economy_audit[0]),economy_audit_dict(result.economy_audit[1]));
    if(experimental_market_arm)out["phased_market_audit"]=py::make_tuple(phased_market_audit_dict(result.phased_market_audit[0]),phased_market_audit_dict(result.phased_market_audit[1]));
    if(capture_trace){py::list trace;for(auto&joint:result.trace)trace.append(py::make_tuple(player_action_dict(joint[0]),player_action_dict(joint[1])));out["trace"]=trace;}return out;
  },py::arg("route0"),py::arg("route1"),py::arg("seed"),py::arg("switch_step0")=-1,py::arg("switch_route0")=-1,py::arg("switch_step1")=-1,py::arg("switch_route1")=-1,py::arg("capture_trace")=false,py::arg("neutral_special_economy")=false,py::arg("experimental_repair")=false,py::arg("experimental_repair_player")=-1,py::arg("experimental_repair_mask")=7,py::arg("experimental_general_takeover")=false,py::arg("experimental_general_takeover_player")=-1,py::arg("experimental_market_arm")=0,py::arg("experimental_market_player")=-1,py::arg("stop_after_steps")=-1,py::arg("evaluation_compaction_minimum_step")=-1,py::arg("evaluation_compaction_evidence_mode")=0,py::arg("thomas_prefix_player")=-2)
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
  .def("play_batch",[](const NativeTeammateExecutor&x,py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks,bool neutral_special_economy,int repair_mask){
    auto in=tasks.unchecked<2>();if(in.shape(1)!=7)throw std::invalid_argument("tasks must have columns route0,route1,seed,switch_step0,switch_route0,switch_step1,switch_route1");
    py::array_t<double> rewards({in.shape(0),(ssize_t)2});auto out=rewards.mutable_unchecked<2>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<in.shape(0);i++){auto r=x.play((int)in(i,0),(int)in(i,1),(uint64_t)in(i,2),(int)in(i,3),(int)in(i,4),(int)in(i,5),(int)in(i,6),false,false,neutral_special_economy,native_repair_options_from_mask(repair_mask),-1);out(i,0)=r.rewards[0];out(i,1)=r.rewards[1];}}
    return rewards;
  },py::arg("tasks"),py::arg("neutral_special_economy")=false,py::arg("repair_mask")=0)
  .def("play_audit_batch",[](const NativeTeammateExecutor&x,py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks,bool neutral_special_economy,int repair_mask){
    auto in=tasks.unchecked<2>();if(in.shape(1)!=7)throw std::invalid_argument("tasks must have columns route0,route1,seed,switch_step0,switch_route0,switch_step1,switch_route1");
    py::array_t<double> rewards({in.shape(0),(ssize_t)2});auto rw=rewards.mutable_unchecked<2>();
    py::array_t<int32_t> audit({in.shape(0),(ssize_t)2,(ssize_t)3});auto au=audit.mutable_unchecked<3>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<in.shape(0);i++){auto r=x.play((int)in(i,0),(int)in(i,1),(uint64_t)in(i,2),(int)in(i,3),(int)in(i,4),(int)in(i,5),(int)in(i,6),false,true,neutral_special_economy,native_repair_options_from_mask(repair_mask),-1);for(int p=0;p<2;p++){rw(i,p)=r.rewards[p];au(i,p,0)=r.macro_unit_failures[p];au(i,p,1)=r.macro_market_failures[p];au(i,p,2)=r.first_macro_failure_step[p];}}}
    return py::make_tuple(rewards,audit);
  },py::arg("tasks"),py::arg("neutral_special_economy")=false,py::arg("repair_mask")=0)
  .def("play_repair_audit_batch",[](const NativeTeammateExecutor&x,py::array_t<int64_t,py::array::c_style|py::array::forcecast> tasks){
    auto in=tasks.unchecked<2>();if(in.shape(1)!=9)throw std::invalid_argument("tasks must have columns route0,route1,seed,switch_step0,switch_route0,switch_step1,switch_route1,repair_player,repair_mask");
    py::array_t<double> rewards({in.shape(0),(ssize_t)2});auto rw=rewards.mutable_unchecked<2>();
    py::array_t<int32_t> macro({in.shape(0),(ssize_t)2,(ssize_t)3});auto ma=macro.mutable_unchecked<3>();
    py::array_t<int32_t> repair({in.shape(0),(ssize_t)2,(ssize_t)26});auto ra=repair.mutable_unchecked<3>();
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t i=0;i<in.shape(0);i++){auto r=x.play((int)in(i,0),(int)in(i,1),(uint64_t)in(i,2),(int)in(i,3),(int)in(i,4),(int)in(i,5),(int)in(i,6),false,true,false,repair_options_from_mask((int)in(i,8)),(int)in(i,7));for(int p=0;p<2;p++){rw(i,p)=r.rewards[p];ma(i,p,0)=r.macro_unit_failures[p];ma(i,p,1)=r.macro_market_failures[p];ma(i,p,2)=r.first_macro_failure_step[p];const auto&a=r.repair_audit[p];const int values[26]={a.weed_triggers,a.weed_absorbed_pass,a.weed_absorbed_productive,a.weed_legacy_would_drop_move,a.weed_no_safe_alignment,a.movement_edits,a.animal_original_attempted,a.animal_inferred_filled,a.animal_inferred_partial,a.animal_inferred_zero,a.animal_retries_emitted,a.animal_retry_pickup_realign,a.animal_retry_blocked_no_path,a.empty_stall_reused,a.empty_stall_unresolved,a.local_repair_decisions,a.local_repair_commits,a.local_repair_receipts_confirmed,a.local_repair_receipts_failed,a.local_repair_fail_closed,a.local_repair_seed_orders,a.local_repair_seed_units,a.local_repair_seed_fills,a.local_repair_seed_zero_fills,a.local_repair_transaction_claims,a.local_repair_transaction_claim_rejections};for(int j=0;j<26;j++)ra(i,p,j)=values[j];}}}
    return py::make_tuple(rewards,macro,repair);
  },py::arg("tasks"))
  .def_static("repair_audit_metric_names",[](){return std::vector<std::string>{"weed_triggers","weed_absorbed_pass","weed_absorbed_productive","weed_legacy_would_drop_move","weed_no_safe_alignment","movement_edits","animal_original_attempted","animal_inferred_filled","animal_inferred_partial","animal_inferred_zero","animal_retries_emitted","animal_retry_pickup_realign","animal_retry_blocked_no_path","empty_stall_reused","empty_stall_unresolved","local_repair_decisions","local_repair_commits","local_repair_receipts_confirmed","local_repair_receipts_failed","local_repair_fail_closed","local_repair_seed_orders","local_repair_seed_units","local_repair_seed_fills","local_repair_seed_zero_fills","local_repair_transaction_claims","local_repair_transaction_claim_rejections"};})
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
       py::sequence raw_targets,py::sequence raw_checkpoints,py::sequence raw_seeds,
       int stop_after_steps){
    auto integers=[](py::sequence raw){std::vector<int64_t> out;out.reserve(raw.size());for(auto value:raw)out.push_back(py::cast<int64_t>(value));return out;};
    const auto openings=integers(raw_openings),targets=integers(raw_targets),checkpoints=integers(raw_checkpoints),seed_values=integers(raw_seeds);
    if(openings.empty()||targets.empty()||checkpoints.empty()||seed_values.empty())throw std::invalid_argument("switch search dimensions must be non-empty");
    if(stop_after_steps==0||stop_after_steps<-1)throw std::invalid_argument("switch search stop_after_steps must be -1 or positive");
    const ssize_t no=openings.size(),nc=checkpoints.size(),nt=targets.size(),nr=x.route_count(),ns=seed_values.size();
    for(auto route:openings)if(route<0||route>=nr)throw std::invalid_argument("invalid opening route");
    for(auto route:targets)if(route<0||route>=nr)throw std::invalid_argument("invalid target route");
    py::array_t<uint8_t> outcome({no,nc,nt,nr,ns,(ssize_t)2});
    py::array_t<float> margin({no,nc,nt,nr,ns,(ssize_t)2});
    py::array_t<float> states({no,nc,nr,ns,(ssize_t)2,(ssize_t)147});
    auto yy=outcome.mutable_unchecked<6>();
    auto mm=margin.mutable_unchecked<6>();
    auto xx=states.mutable_unchecked<6>();
    const ssize_t state_tasks=no*nc*nr*ns*2;
    {py::gil_scoped_release release;
     #pragma omp parallel for schedule(dynamic,1)
     for(ssize_t flat=0;flat<state_tasks;flat++){
       ssize_t value=flat;const int seat=value%2;value/=2;const ssize_t si=value%ns;value/=ns;
       const ssize_t opponent=value%nr;value/=nr;const ssize_t ci=value%nc;value/=nc;const ssize_t oi=value;
       const int opening=(int)openings[oi],checkpoint=(int)checkpoints[ci];const uint64_t seed=(uint64_t)seed_values[si];
       const int route0=seat==0?opening:(int)opponent,route1=seat==0?(int)opponent:opening;
       const auto features=x.features_at(route0,route1,seed,checkpoint,seat,opening);
       for(int f=0;f<147;f++)xx(oi,ci,opponent,si,seat,f)=features[f];
       for(ssize_t ti=0;ti<nt;ti++){
         const int target=(int)targets[ti];
         // Parameters 9..16 below repeat play()'s own defaults; only
         // stop_after_steps is forwarded, so a -1 keeps the legacy full game.
         const auto result=seat==0
             ?x.play(opening,(int)opponent,seed,checkpoint,target,-1,-1,false,true,false,
                     NativeRepairOptions{},-1,false,-1,0,-1,stop_after_steps)
             :x.play((int)opponent,opening,seed,-1,-1,checkpoint,target,false,true,false,
                     NativeRepairOptions{},-1,false,-1,0,-1,stop_after_steps);
         const double own=result.rewards[seat],other=result.rewards[1-seat],difference=own-other;
         yy(oi,ci,ti,opponent,si,seat)=difference>0?2:difference==0?1:0;
         mm(oi,ci,ti,opponent,si,seat)=(float)difference;
       }
     }}
    py::dict out;out["outcome"]=outcome;out["margin"]=margin;out["states"]=states;return out;
  },py::arg("openings"),py::arg("targets"),py::arg("checkpoints"),py::arg("seeds"),py::arg("stop_after_steps")=-1);
 py::class_<NativeReplayOpponent>(m,"NativeReplayOpponent")
  .def(py::init<const NativeTeammateExecutor&,int,bool,int>(),
       py::keep_alive<1,2>(),py::arg("executor"),py::arg("route"),
       py::arg("neutral_special_economy")=true,py::arg("repair_mask")=0)
  .def("reset",&NativeReplayOpponent::reset,py::arg("route"))
  .def("action",[](NativeReplayOpponent&x,const Simulator&env,int player,int step){
    return player_action_dict(x.action(env,player,step));
  },py::arg("env"),py::arg("player"),py::arg("step"))
  .def_property_readonly("route",&NativeReplayOpponent::route);
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
