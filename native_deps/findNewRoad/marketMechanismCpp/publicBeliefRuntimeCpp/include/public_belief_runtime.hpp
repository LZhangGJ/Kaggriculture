#pragma once

#include "belief.hpp"
#include "dump_scenarios.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <array>
#include <vector>

namespace public_belief_runtime {

using Inventory = g001::market::Inventory;

struct InventoryInterval {
  Inventory lower{};
  Inventory point{};
  Inventory upper{};
};

struct Position {
  int x{-1};
  int y{-1};
  friend bool operator==(const Position&, const Position&) = default;
};

enum class TileKind : std::uint8_t { Other, Plant, Animal };

// A normalized view of official public tile fields.  Plant.product is one of
// Wheat..Melon; Animal.product is Egg/Milk/Wool.
struct Tile {
  TileKind kind{TileKind::Other};
  int product{-1};
  int yield_units{};
  bool fertilizer_available{};
  bool fed_today{};
  int fertilized_until_day{};
  int planted_day{};
  // All fields below are present on the official public tile surface.  They
  // are lifecycle evidence, not opponent-private inventory.
  int placed_day{};
  int pending_care_bonus{};
  int consecutive_unwatered{};
  int consecutive_unfed{};
  int max_lifespan_step{-1};
  bool watered_today{};
  bool cared_today{};
};

struct FarmObservation {
  std::int64_t public_money{};
  Position farmer{};
  std::vector<Position> hands;
  int hires_today{};
  int unlocked_quadrants{1};
  std::vector<Tile> tiles;
};

// One official public observation plus the focal player's own total products
// (shed + carried).  There is deliberately no opponent shed/private state,
// action, fill field, route identity, or simulator handle.
struct Observation {
  int step{};
  int day{};
  int turns_per_day{24};
  int board_size{10};
  int episode_steps{720};
  int farm_hand_cost_multiplier{1};
  int town_shop_unlock_interval{3};
  int town_shop_interval{4};
  int town_center_interval{24};
  int town_unlocked_shop_count{};
  Inventory town_shop_demand_per_tick{};
  Inventory market_inventory{};
  Inventory market_price{};
  Inventory own_total{};
  FarmObservation own_farm;
  FarmObservation opponent_farm;
};

// Causal evidence for the transition from the preceding observation to the
// current one.  Fill bounds must come from observed own inventory
// conservation; BUY bounds come only from the focal player's submitted queue.
struct TransitionEvidence {
  Inventory own_sell_requested{};
  InventoryInterval confirmed_own_sell_fill{};
  InventoryInterval own_buy_product{};
  Inventory public_town_drain{};
};

struct Config {
  int shed_capacity{100};
  std::size_t history_capacity{64};
  int opponent_supply_horizon_steps{24};
};

struct FeedProfitabilityRisk {
  // Public price support after deterministic town demand and the complete
  // public-only opponent supply upper.  This is a discontinuity warning, not
  // a claim that a rational opponent must feed.
  int product_price_lower{};
  int product_price_upper{};
  int wheat_replacement_price_lower{};
  int wheat_replacement_price_upper{};
  bool crosses_one_feed_one_output_threshold{};
};

struct Snapshot {
  g001::market::InventoryBelief belief{};
  // Exact public feature used by the frozen late Strawberry H1 policy.  This
  // deliberately matches the historical SharedBeliefTracker definition for
  // crops:
  //   sum(tile.yield_units) +
  //   sum(min(100, age_days * 100 / first_production_days)) / 100.
  // It is not the broader H-step production upper below.
  Inventory opponent_visible_standing_production{};
  // Public products already standing on the opponent's tiles.  They are not
  // private inventory yet, but may become sellable after a causal HARVEST /
  // COLLECT_FERTILIZER, so a zero shed belief alone cannot certify a safe
  // delayed sale.
  Inventory opponent_harvestable_now_upper{};
  // Public-only H-step supply exposure.  The total includes the existing
  // private-stock upper, currently standing public yield/fertilizer, and the
  // most production those already-visible producers can create.  H is capped
  // at 24, so newly bought/planted crops and newly placed animals cannot reach
  // their official first-production day and contribute zero.
  int opponent_supply_horizon_steps{};
  Inventory opponent_new_production_within_horizon_upper{};
  Inventory opponent_sellable_within_horizon_upper{};
  std::array<FeedProfitabilityRisk, 3> feed_profitability_risk{};
  // Chronological bounded copy, directly consumable by general::Input.
  std::vector<g001::dump::PublicMarketFrame> public_market_history;
};

// Mutable per-game state.  Instances share no globals and are intended to be
// owned by one seat/game thread; separate instances are fully isolated.
class Runtime {
 public:
  explicit Runtime(Config config = {});

  [[nodiscard]] Snapshot reset(const Observation& observation);
  [[nodiscard]] Snapshot update(const Observation& observation,
                                const TransitionEvidence& evidence);
  [[nodiscard]] Snapshot snapshot() const;
  [[nodiscard]] bool initialized() const noexcept { return initialized_; }
  [[nodiscard]] std::size_t history_size() const noexcept { return history_.size(); }

 private:
  [[nodiscard]] g001::market::PublicStateSummary base_summary(
      const Observation& observation) const;
  void append_frame(g001::dump::PublicMarketFrame frame);

  Config config_;
  g001::market::OpponentInventoryBelief filter_;
  bool initialized_{};
  Observation previous_{};
  g001::market::InventoryBelief belief_{};
  std::deque<g001::dump::PublicMarketFrame> history_;
};

}  // namespace public_belief_runtime
