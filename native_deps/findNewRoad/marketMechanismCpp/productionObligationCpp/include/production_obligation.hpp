#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace production_obligation {

constexpr int kProducts = 9;
constexpr int kCrops = 5;
constexpr int kAnimals = 3;
constexpr int kItems = 12;

enum class Item : std::int8_t {
  None = -1,
  Wheat = 0,
  Carrot = 1,
  Tomato = 2,
  Strawberry = 3,
  Melon = 4,
  Egg = 5,
  Milk = 6,
  Wool = 7,
  Fertilizer = 8,
  Goose = 9,
  Cow = 10,
  Sheep = 11,
};

// Deliberately contains no market operation.  This type boundary is the
// compiler's old-market-tape independence guarantee.
enum class UnitOp : std::int8_t {
  Pass = 0,
  North,
  South,
  East,
  West,
  Drop,
  Pickup,
  Place,
  Plant,
  Water,
  Harvest,
  Fertilize,
  Dig,
  BuildCoop,
  BuildPasture,
  Feed,
  CollectFertilizer,
  Care,
};

enum class TileKind : std::int8_t {
  Empty = 0,
  Locked,
  Weed,
  Plant,
  Coop,
  Pasture,
  Animal,
};

struct Position {
  int x = 0;
  int y = 0;
};

struct Inventory {
  std::array<int, kItems> quantity{};
  // Official end-of-day deposits iterate each actor's dict insertion order.
  // Missing items are appended in numeric order during validation.
  std::vector<Item> insertion_order;
};

struct TileState {
  TileKind kind = TileKind::Empty;
  Item item = Item::None;  // crop for Plant, animal for Animal
  int planted_day = 0;
  int yield_units = 0;
  int consecutive_unfed = 0;
  bool fed_today = false;
  bool fertilizer_available = false;
};

struct CurrentState {
  int step = 0;
  int turns_per_day = 24;
  int board_size = 10;
  int shed_capacity = 100;
  int max_market_orders = 10;
  int farm_hand_cost_mult = 1;
  double money = 3000.0;

  // Bit 0 is NW, then NE, SW, SE. Official play always unlocks a prefix.
  std::uint8_t unlocked_mask = 1;
  int hires_today = 0;

  std::array<int, kItems> shed{};
  std::array<int, kCrops> seeds{};
  std::array<int, kProducts> product_price{};

  // Actor 0 is farmer; actors 1.. are today's hands.
  std::vector<Position> actor_positions;
  std::vector<Inventory> carried;
  std::vector<TileState> tiles;
};

struct UnitAction {
  UnitOp op = UnitOp::Pass;
  Item item = Item::None;
  int quantity = 1;
};

// actor_actions cardinality is the actor count required by this frame. It is
// intentionally explicit: sparse action lists cannot prove HIRE obligations.
struct UnitFrame {
  int step = 0;
  std::vector<UnitAction> actor_actions;
};

struct CompilerInput {
  CurrentState current;
  std::vector<UnitFrame> future_units;
  // Zero keeps the exhaustive offline compiler behavior. Native deployment
  // supplies a positive fail-closed budget so malformed or resource-dense
  // plans cannot monopolize a competition turn.
  int maximum_resource_derivation_iterations = 0;
};

enum class NodeKind : std::int8_t {
  BuySeed,
  BuyProduct,
  BuyAnimal,
  Hire,
  BuyLand,
  Pickup,
  Consume,
  LandAccess,
  CashReserve,
  CapacityRelease,
  SlotBudget,
  DayBoundary,
  SoftMiss,
  DownstreamAction,
};

struct ObligationNode {
  int id = -1;
  NodeKind kind = NodeKind::Consume;
  Item item = Item::None;
  int quantity = 0;
  // For acquisition nodes: total units of the same operation/item whose
  // deadlines are no later than this node's deadline.
  int cumulative_quantity = 0;
  int actor = -1;
  int consumer_step = -1;
  int deadline_step = -1;
  int earliest_step = -1;
  int execution_step = -1;
  int order_slot = -1;
  int quadrant = -1;
  int unit_cost_quote = 0;
  int cash_quote = 0;
  int cumulative_cash_quote = 0;
  int cash_shortfall_quote = 0;
  int free_capacity_required = 0;
  Position position{-1, -1};
  UnitOp original_unit_op = UnitOp::Pass;
  UnitOp suggested_unit_op = UnitOp::Pass;
  int downstream_dependents = 0;
  // Structural proxy only: one missed action plus the number of unit actions
  // whose lifecycle precondition may be invalidated. It is not money/reward.
  int downstream_loss_proxy = 0;
  std::string reason;
};

struct Edge {
  int before = -1;
  int after = -1;
  std::string reason;
};

enum class DiagnosticCode : std::int8_t {
  InvalidInput,
  MissedAcquisitionDeadline,
  MissingPickupPath,
  ActorUnavailableAtDayStart,
  MarketSlotInfeasible,
  InvalidLandPrefix,
  UnitWouldNotExecute,
  EndOfDayOverflow,
};

struct Diagnostic {
  DiagnosticCode code = DiagnosticCode::InvalidInput;
  int step = -1;
  int actor = -1;
  Item item = Item::None;
  int quantity = 0;
  std::string message;
};

struct CompileResult {
  bool feasible = true;
  std::vector<ObligationNode> nodes;
  std::vector<Edge> edges;
  std::vector<Diagnostic> diagnostics;

  int purchase_units = 0;
  int market_order_nodes = 0;
  int quoted_cash = 0;
  int peak_orders_in_step = 0;
  int soft_misses = 0;

  std::string to_json() const;
  std::uint64_t fingerprint() const;
};

CompileResult compile(const CompilerInput& input);

struct UnitReplacement {
  int step = -1;
  int actor = -1;
  UnitAction original;
  UnitAction replacement;
  Position position{-1, -1};
  int soft_miss_node = -1;
};

struct SoftCompileResult {
  CompileResult dag;
  // Contains current-step replacements only. MOVE is never present here.
  std::vector<UnitReplacement> current_unit_replacements;
};

// Softens only current-step resource misses that market cannot repair because
// unit execution precedes market. All later unit frames remain hard inputs and
// are compiled normally.
SoftCompileResult compile_soft_current(const CompilerInput& input);

const char* item_name(Item item);
const char* node_kind_name(NodeKind kind);
const char* diagnostic_code_name(DiagnosticCode code);
const char* unit_op_name(UnitOp op);

}  // namespace production_obligation
