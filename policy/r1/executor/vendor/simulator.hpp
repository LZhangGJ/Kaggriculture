// Licensed under the Apache License, Version 2.0.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>
#ifndef R2_FLOW_AUDIT
#define R2_FLOW_AUDIT 0
#endif

namespace fastkag {

constexpr int N_PRODUCTS = 9;
constexpr int N_CROPS = 5;
constexpr int N_ANIMALS = 3;
constexpr int N_ITEMS = 12;

enum class Item : int8_t {
  NONE=-1, WHEAT=0, CARROT=1, TOMATO=2, STRAWBERRY=3, MELON=4,
  EGG=5, MILK=6, WOOL=7, FERTILIZER=8, GOOSE=9, COW=10, SHEEP=11
};
enum class Op : int8_t {
  PASS=0, NORTH, SOUTH, EAST, WEST, DROP, PICKUP, PLACE, PLANT, WATER,
  HARVEST, FERTILIZE, DIG, BUILD_COOP, BUILD_PASTURE, FEED,
  COLLECT_FERTILIZER, CARE, HIRE, BUY_LAND, BUY_SEED, BUY_PRODUCT,
  BUY_ANIMAL, SELL
};
enum class TileKind : int8_t { EMPTY=0, LOCKED, WEED, PLANT, COOP, PASTURE, ANIMAL };

struct Action { Op op=Op::PASS; Item item=Item::NONE; int32_t quantity=1; };
struct PlayerAction { std::vector<Action> units; std::vector<Action> market; };
struct Position { int16_t x=0, y=0; };

struct Tile {
  TileKind kind=TileKind::EMPTY;
  Item crop=Item::NONE, animal=Item::NONE;
  int16_t planted_day=0, placed_day=0;
  int16_t yield_units=0, consecutive_unwatered=0, consecutive_unfed=0;
  int16_t fertilized_until_day=-1, pending_care_bonus=0;
  int32_t max_lifespan_step=-1;
  bool watered_today=false, fed_today=false, cared_today=false;
  bool fertilizer_available=false;
};

struct Farm {
  double money=3000;
  std::vector<Tile> tiles;
  Position farmer;
  std::vector<Position> hands;
  uint8_t unlocked_mask=1;
  int16_t hires_today=0;
};
struct PrivateState {
  std::array<int32_t,N_ITEMS> shed{};
  std::array<int32_t,N_CROPS> seeds{};
  std::vector<std::array<int32_t,N_ITEMS>> inventories;
  // Python inventories are dicts. Capacity-limited deposits iterate dict
  // insertion order, so the counts alone are not enough for exact fidelity.
  std::vector<std::vector<int8_t>> inventory_order;
};
struct Market {
  std::array<int32_t,N_PRODUCTS> inventory{};
  std::array<int32_t,N_PRODUCTS> prices{};
};

struct Config {
  int episode_steps=720, board_size=10, starting_money=3000;
  int max_market_orders=10, turns_per_day=24, shed_capacity=100;
  double weed_spawn_chance=0.005;
  int town_shop_unlock_interval=3, town_shop_sell_interval=4;
  int town_center_sell_interval=24, farm_hand_cost_mult=1;
};

class PythonRandom {
 public:
  explicit PythonRandom(uint64_t seed=0) { reseed(seed); }
  void reseed(uint64_t seed);
  uint32_t genrand_uint32();
  double random();
  uint32_t getrandbits(int k);
  uint32_t randbelow(uint32_t n);
 private:
  std::array<uint32_t,624> mt_{};
  int index_=624;
  void init_genrand(uint32_t s);
  void init_by_array(const uint32_t* key, int len);
};

class Simulator {
  // Capability-restricted, observation-only intraday scenario. The friend
  // reconstructs a fresh synthetic state; it never takes a live Simulator.
  friend class ObservedDayScenario;
  friend class PublicFlowScenario;
 public:
  explicit Simulator(Config config={}, uint64_t seed=0);
  void reset(uint64_t seed);
  void step(const std::array<PlayerAction,2>& actions);
  // Exact, read-only projection of one player's ordered unit phase. The full
  // action list is used for the official aggregate seed-demand precheck even
  // when only a prefix is projected. No market, clock, decay or RNG advances.
  Simulator project_unit_phase(int player, const std::vector<Action>& actions,
                               int prefix=-1) const;
  // Read-only own-order scenario using the unchanged official market kernel.
  // Rival orders are absent: quotes are conditional, never promised fills.
  // No units, clock, town demand, decay, day end or future RNG is executed.
  Simulator project_own_market(int player, const std::vector<Action>& orders) const;
  const Config& config() const { return cfg_; }
  int step_count() const { return step_; }
  int day() const { return step_/cfg_.turns_per_day; }
  int hour() const { return step_%cfg_.turns_per_day; }
  bool done() const { return done_; }
  uint64_t seed() const { return seed_; }
  // Offline counterfactual validation only: preserve the complete current
  // state while drawing future day-level shops/weeds from another official
  // seed.  Runtime agents never receive or call this method.
  void reseed_future(uint64_t seed) { seed_ = seed; }
  // Offline public-belief counterfactuals only.  A deployable agent cannot
  // inspect the rival shed, seeds or carried inventory, so response-scenario
  // previews must replace those fields with a state reconstructed solely from
  // public history before allowing a synthetic rival policy to continue.
  // This setter is never called by the game loop or a submitted agent.
  void replace_private_for_public_counterfactual(
      int player, const PrivateState& state);
  const std::array<Farm,2>& farms() const { return farms_; }
  const std::array<PrivateState,2>& privates() const { return privates_; }
  const Market& market() const { return market_; }
  const std::vector<int8_t>& shops() const { return shops_; }
  const std::array<std::vector<int32_t>,2>& last_market_fills() const {
    return last_market_fills_;
  }
  const std::array<std::vector<double>,2>& last_market_cash_shortfalls() const {
    return last_market_cash_shortfalls_;
  }
  const std::array<std::vector<double>,2>& last_market_cash_deltas() const {
    return last_market_cash_deltas_;
  }
  const std::array<std::vector<int32_t>,2>& last_market_inventory_deltas() const {
    return last_market_inventory_deltas_;
  }
  const std::array<int32_t,2>& last_end_of_day_overflow() const {
    return last_end_of_day_overflow_;
  }

 private:
  Config cfg_;
  uint64_t seed_=0;
  int step_=0;
  bool done_=false;
  std::array<Farm,2> farms_;
  std::array<PrivateState,2> privates_;
  Market market_;
  std::vector<int8_t> shops_;
  std::array<std::vector<int32_t>,2> last_market_fills_;
  std::array<std::vector<double>,2> last_market_cash_shortfalls_;
  std::array<std::vector<double>,2> last_market_cash_deltas_;
  std::array<std::vector<int32_t>,2> last_market_inventory_deltas_;
  std::array<int32_t,2> last_end_of_day_overflow_{};

  int tile_index(int x,int y) const { return y*cfg_.board_size+x; }
  int quadrant(int x,int y) const;
  bool shed_adjacent(Position p) const;
  Position default_spawn() const;
  Position spawn_hand(const Farm& farm) const;
  void apply_unit(int player,int idx,const Action& a,int day);
  void apply_unit_phase(int player,const std::vector<Action>& actions,int prefix);
  void process_market(const std::array<PlayerAction,2>& actions);
  void town_consume(int step);
  void decay_plants(int step);
  void end_of_day(int day);
  void refresh_prices();
  int market_price(int item,int inventory) const;
  bool commit_unit(Op op,int item,int price,int player);
};

const char* item_name(int item);
const char* shop_name(int shop);

} // namespace fastkag
