#include "prefix_batch.hpp"

#include "../../fast_kaggriculture/src/simulator.hpp"
#include "../../fast_kaggriculture/src/native_teammate.hpp"
#include "../native_opponents/metav4_2965/metav4_2965.hpp"
#include "../native_opponents/fieldcraft_2887/fieldcraft_2887.hpp"
#include "../native_opponents/salemali7_2900/salemali7_2900.hpp"
#include "../native_opponents/thomas_2945_cpp/thomas_2945.hpp"
#include "../native_student_v3/actor.hpp"
#include "../native_student_v3/tokenizer.hpp"
#include "economic_features_v1.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <dlfcn.h>
#include <openssl/sha.h>
#include <pthread.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::array<char, 8> kMagic{'K', 'G', 'P', 'F', 'X', 'C', '1', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kPrefixSteps = 288;
constexpr int kContextWidth = 2233;
constexpr std::size_t kDefaultStackBytes = 2u << 20;

using Digest = std::array<unsigned char, SHA256_DIGEST_LENGTH>;

std::string hex(const Digest& value) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const auto byte : value) output << std::setw(2) << int(byte);
  return output.str();
}

Digest file_digest(const std::string& path) {
  std::ifstream source(path, std::ios::binary);
  if (!source) throw std::runtime_error("cannot open artifact: " + path);
  SHA256_CTX context;
  SHA256_Init(&context);
  std::array<char, 1 << 16> buffer{};
  while (source) {
    source.read(buffer.data(), buffer.size());
    if (source.gcount()) SHA256_Update(&context, buffer.data(), source.gcount());
  }
  if (!source.eof()) throw std::runtime_error("cannot read artifact: " + path);
  Digest output{};
  SHA256_Final(output.data(), &context);
  return output;
}

Digest double_digest(const std::vector<double>& values) {
  SHA256_CTX context;
  SHA256_Init(&context);
  for (const double value : values) {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    std::array<unsigned char, 8> bytes{};
    for (int i = 0; i < 8; ++i) bytes[i] = (bits >> (8 * i)) & 0xff;
    SHA256_Update(&context, bytes.data(), bytes.size());
  }
  Digest output{};
  SHA256_Final(output.data(), &context);
  return output;
}

class Reader {
 public:
  explicit Reader(const std::string& path)
      : source_(path, std::ios::binary), path_(path) {
    if (!source_) throw std::runtime_error("cannot open prefix cache: " + path);
  }

  template <class Unsigned>
  Unsigned unsigned_value() {
    static_assert(std::is_unsigned_v<Unsigned>);
    std::array<unsigned char, sizeof(Unsigned)> bytes{};
    read(bytes.data(), bytes.size());
    Unsigned value{};
    for (std::size_t i = 0; i < bytes.size(); ++i)
      value |= Unsigned(bytes[i]) << (8 * i);
    return value;
  }

  std::int8_t i8() { return std::bit_cast<std::int8_t>(unsigned_value<std::uint8_t>()); }
  std::int32_t i32() { return std::bit_cast<std::int32_t>(unsigned_value<std::uint32_t>()); }
  double f64() { return std::bit_cast<double>(unsigned_value<std::uint64_t>()); }

  Digest digest() {
    Digest result{};
    read(result.data(), result.size());
    return result;
  }

  void read(void* output, std::size_t count) {
    source_.read(static_cast<char*>(output), count);
    if (source_.gcount() != static_cast<std::streamsize>(count))
      throw std::runtime_error("truncated prefix cache: " + path_);
  }

  void require_eof() {
    if (source_.peek() != std::ifstream::traits_type::eof())
      throw std::runtime_error("trailing prefix cache bytes: " + path_);
  }

 private:
  std::ifstream source_;
  std::string path_;
};

fastkag::PlayerAction read_action(Reader& input) {
  const std::uint16_t units = input.unsigned_value<std::uint16_t>();
  const std::uint16_t market = input.unsigned_value<std::uint16_t>();
  if (units > 16 || market > 10) throw std::runtime_error("prefix action count");
  fastkag::PlayerAction output;
  auto atom = [&]() {
    const int op = input.i8(), item = input.i8();
    const int quantity = input.i32();
    if (op < 0 || op >= 24 || item < -1 || item >= fastkag::N_ITEMS)
      throw std::runtime_error("prefix action atom");
    return fastkag::Action{fastkag::Op(op), fastkag::Item(item), quantity};
  };
  output.units.reserve(units);
  output.market.reserve(market);
  for (int i = 0; i < units; ++i) output.units.push_back(atom());
  for (int i = 0; i < market; ++i) output.market.push_back(atom());
  return output;
}

struct Cache {
  std::string path;
  std::uint64_t seed{};
  int seat{};
  int opponent{};
  int route{};
  std::uint64_t policy_seed{};
  std::uint32_t steps{};
  std::uint32_t packed_count{};
  std::array<Digest, 7> hashes{};
  std::vector<double> settings;
  std::vector<std::array<fastkag::PlayerAction, 2>> actions;
};

Cache read_cache(const std::string& path) {
  Reader input(path);
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  if (magic != kMagic || input.unsigned_value<std::uint32_t>() != kVersion)
    throw std::runtime_error("unsupported prefix cache: " + path);
  Cache cache;
  cache.path = path;
  cache.seed = input.unsigned_value<std::uint64_t>();
  cache.seat = input.unsigned_value<std::uint8_t>();
  cache.opponent = input.unsigned_value<std::uint8_t>();
  (void)input.unsigned_value<std::uint16_t>();
  cache.route = input.i32();
  cache.steps = input.unsigned_value<std::uint32_t>();
  const auto settings_count = input.unsigned_value<std::uint32_t>();
  cache.packed_count = input.unsigned_value<std::uint32_t>();
  if (cache.seat < 0 || cache.seat > 1 ||
      (cache.opponent != 1 && cache.opponent != 2) ||
      cache.steps != kPrefixSteps || settings_count == 0 || settings_count > 128 ||
      cache.packed_count < 3000 || cache.packed_count > 4096)
    throw std::runtime_error("invalid prefix cache header: " + path);
  for (auto& digest : cache.hashes) digest = input.digest();
  cache.settings.resize(settings_count);
  for (auto& value : cache.settings) value = input.f64();
  cache.actions.resize(cache.steps);
  for (auto& step : cache.actions) {
    step[0] = read_action(input);  // opening policy
    step[1] = read_action(input);  // expected native opponent
  }
  input.require_eof();
  return cache;
}

bool same_action(const fastkag::PlayerAction& left,
                 const fastkag::PlayerAction& right) {
  auto same = [](const fastkag::Action& a, const fastkag::Action& b) {
    return a.op == b.op && a.item == b.item && a.quantity == b.quantity;
  };
  return left.units.size() == right.units.size() &&
         left.market.size() == right.market.size() &&
         std::equal(left.units.begin(), left.units.end(), right.units.begin(), same) &&
         std::equal(left.market.begin(), left.market.end(), right.market.begin(), same);
}

std::vector<int32_t> encode_action(const fastkag::PlayerAction& action) {
  std::vector<int32_t> output;
  output.reserve(2 + 3 * (action.units.size() + action.market.size()));
  output.push_back(action.units.size());
  output.push_back(action.market.size());
  auto append = [&](const fastkag::Action& value) {
    output.push_back(int(value.op));
    output.push_back(int(value.item));
    output.push_back(value.quantity);
  };
  for (const auto& value : action.units) append(value);
  for (const auto& value : action.market) append(value);
  return output;
}

fastkag::PlayerAction decode_action(const int32_t* values, int count) {
  if (count < 2) throw std::runtime_error("short td_observe action");
  const int units = values[0], market = values[1];
  if (units < 0 || units > 16 || market < 0 || market > 10 ||
      count != 2 + 3 * (units + market))
    throw std::runtime_error("malformed td_observe action");
  fastkag::PlayerAction output;
  int at = 2;
  auto take = [&]() {
    const int op = values[at++], item = values[at++], quantity = values[at++];
    if (op < 0 || op >= 24 || item < -1 || item >= fastkag::N_ITEMS)
      throw std::runtime_error("invalid td_observe atom");
    return fastkag::Action{fastkag::Op(op), fastkag::Item(item), quantity};
  };
  for (int i = 0; i < units; ++i) output.units.push_back(take());
  for (int i = 0; i < market; ++i) output.market.push_back(take());
  return output;
}

std::vector<double> pack_observation(const fastkag::Simulator& env, int seat,
                                     bool canonical = false) {
  std::vector<double> output;
  output.reserve(3200);
  auto add = [&](auto value) { output.push_back(double(value)); };
  add(env.step_count());
  add(env.day());
  add(env.hour());
  add(canonical ? 0 : seat);
  for (int logical = 0; logical < 2; ++logical) {
    const int farm_index = canonical ? (logical == 0 ? seat : 1 - seat) : logical;
    const auto& farm = env.farms()[farm_index];
    add(farm.money);
    add(farm.farmer.x);
    add(farm.farmer.y);
    add(farm.hands.size());
    for (const auto hand : farm.hands) {
      add(hand.x);
      add(hand.y);
    }
    add(farm.unlocked_mask);
    add(farm.hires_today);
    if (farm.tiles.size() != 100) throw std::runtime_error("native board width");
    for (const auto& tile : farm.tiles) {
      add(int(tile.kind));
      add(int(tile.crop));
      add(int(tile.animal));
      add(tile.planted_day);
      add(tile.placed_day);
      add(tile.yield_units);
      add(tile.consecutive_unwatered);
      add(tile.consecutive_unfed);
      add(tile.fertilized_until_day);
      add(tile.pending_care_bonus);
      add(tile.max_lifespan_step);
      add(tile.watered_today);
      add(tile.fed_today);
      add(tile.cared_today);
      add(tile.fertilizer_available);
    }
  }
  const auto& priv = env.privates()[seat];
  for (const auto value : priv.shed) add(value);
  for (const auto value : priv.seeds) add(value);
  add(priv.inventories.size());
  if (priv.inventory_order.size() != priv.inventories.size())
    throw std::runtime_error("native inventory order width");
  for (std::size_t bag = 0; bag < priv.inventories.size(); ++bag) {
    for (const auto value : priv.inventories[bag]) add(value);
    std::vector<int> order;
    for (const int item : priv.inventory_order[bag])
      if (item >= 0 && item < fastkag::N_ITEMS && priv.inventories[bag][item])
        order.push_back(item);
    add(order.size());
    for (const int item : order) add(item);
  }
  for (const auto value : env.market().inventory) add(value);
  for (const auto value : env.market().prices) add(value);
  add(env.shops().size());
  for (const auto value : env.shops()) add(value);
  return output;
}

const char* op_name(fastkag::Op op) {
  static constexpr const char* names[] = {
      "PASS", "NORTH", "SOUTH", "EAST", "WEST", "DROP", "PICKUP",
      "PLACE", "PLANT", "WATER", "HARVEST", "FERTILIZE", "DIG",
      "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
      "CARE", "HIRE", "BUY_LAND", "BUY_SEED", "BUY_PRODUCT",
      "BUY_ANIMAL", "SELL"};
  const int index = int(op);
  return index >= 0 && index < int(std::size(names)) ? names[index] : "PASS";
}

py::list action_list(const fastkag::Action& action) {
  py::list output;
  output.append(op_name(action.op));
  if (action.item != fastkag::Item::NONE)
    output.append(fastkag::item_name(int(action.item)));
  if (action.quantity != 1 || action.item != fastkag::Item::NONE)
    output.append(action.quantity);
  return output;
}

py::dict tile_dict(const fastkag::Tile& tile) {
  py::dict output;
  if (tile.kind == fastkag::TileKind::WEED) {
    output["kind"] = "WEED";
  } else if (tile.kind == fastkag::TileKind::COOP) {
    output["kind"] = "COOP";
  } else if (tile.kind == fastkag::TileKind::PASTURE) {
    output["kind"] = "PASTURE";
  } else if (tile.kind == fastkag::TileKind::PLANT) {
    output["kind"] = "PLANT";
    output["crop"] = fastkag::item_name(int(tile.crop));
    output["planted_day"] = tile.planted_day;
    output["watered_today"] = tile.watered_today;
    output["consecutive_unwatered"] = tile.consecutive_unwatered;
    output["yield_units"] = tile.yield_units;
    output["max_lifespan_step"] = tile.max_lifespan_step;
    output["fertilized_until_day"] = tile.fertilized_until_day;
  } else if (tile.kind == fastkag::TileKind::ANIMAL) {
    output["kind"] = int(tile.animal) == 9 ? "COOP" : "PASTURE";
    output["animal"] = fastkag::item_name(int(tile.animal));
    output["placed_day"] = tile.placed_day;
    output["yield_units"] = tile.yield_units;
    output["consecutive_unfed"] = tile.consecutive_unfed;
    output["fed_today"] = tile.fed_today;
    output["cared_today"] = tile.cared_today;
    output["fertilizer_available"] = tile.fertilizer_available;
    output["pending_care_bonus"] = tile.pending_care_bonus;
  }
  return output;
}

py::dict observation_dict(const fastkag::Simulator& env, int seat) {
  py::dict observation;
  observation["player"] = seat;
  observation["step"] = env.step_count();
  observation["day"] = env.day();
  observation["hour"] = env.hour();
  py::list farms;
  for (const auto& farm : env.farms()) {
    py::dict value;
    value["money"] = farm.money;
    value["farmer"] = py::make_tuple(farm.farmer.x, farm.farmer.y);
    py::list hands;
    for (const auto hand : farm.hands)
      hands.append(py::make_tuple(hand.x, hand.y));
    value["hands"] = std::move(hands);
    py::list quadrants;
    static constexpr const char* names[] = {"NW", "NE", "SW", "SE"};
    for (int i = 0; i < 4; ++i)
      if (farm.unlocked_mask & (1 << i)) quadrants.append(names[i]);
    value["unlocked_quadrants"] = std::move(quadrants);
    value["hires_today"] = farm.hires_today;
    py::list rows;
    for (int y = 0; y < 10; ++y) {
      py::list row;
      for (int x = 0; x < 10; ++x) {
        const auto& tile = farm.tiles[y * 10 + x];
        if (tile.kind == fastkag::TileKind::EMPTY) row.append(py::none());
        else if (tile.kind == fastkag::TileKind::LOCKED) row.append("LOCKED");
        else row.append(tile_dict(tile));
      }
      rows.append(std::move(row));
    }
    value["tiles"] = std::move(rows);
    farms.append(std::move(value));
  }
  observation["farms"] = std::move(farms);
  const auto& priv = env.privates()[seat];
  py::dict private_state, shed, seeds;
  for (int item = 0; item < fastkag::N_ITEMS; ++item)
    shed[fastkag::item_name(item)] = priv.shed[item];
  for (int item = 0; item < fastkag::N_CROPS; ++item)
    seeds[fastkag::item_name(item)] = priv.seeds[item];
  private_state["shed"] = std::move(shed);
  private_state["seeds"] = std::move(seeds);
  py::list inventories;
  for (std::size_t bag = 0; bag < priv.inventories.size(); ++bag) {
    py::dict values;
    for (const int item : priv.inventory_order[bag])
      if (item >= 0 && item < fastkag::N_ITEMS && priv.inventories[bag][item])
        values[fastkag::item_name(item)] = priv.inventories[bag][item];
    inventories.append(std::move(values));
  }
  private_state["inventories"] = std::move(inventories);
  observation["private"] = std::move(private_state);
  py::dict market, inventory, prices;
  for (int item = 0; item < fastkag::N_PRODUCTS; ++item) {
    inventory[fastkag::item_name(item)] = env.market().inventory[item];
    prices[fastkag::item_name(item)] = env.market().prices[item];
  }
  market["inventory"] = std::move(inventory);
  market["prices"] = std::move(prices);
  observation["market"] = std::move(market);
  py::dict town;
  py::list shops;
  for (const auto shop : env.shops()) shops.append(fastkag::shop_name(shop));
  town["unlocked_shops"] = std::move(shops);
  observation["town"] = std::move(town);
  return observation;
}

using StudentCallback = int (*)(void*, int32_t, int32_t, int32_t,
                                const double*, std::size_t);

class Runtime {
 public:
  explicit Runtime(const std::string& path) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(std::string("dlopen: ") + dlerror());
    try {
      create = symbol<Create>("td_new");
      destroy = symbol<Destroy>("td_delete");
      settings_count = symbol<SettingsCount>("td_settings_count");
      observe_external = symbol<ObserveExternal>("td_observe_external");
      observe = symbol<Observe>("td_observe");
      activate = symbol<Activate>("td_activate_external");
      pre_context = symbol<PreContext>("td_student_pre_context_observation");
      plan = symbol<Plan>("td_student_plan_v3_callback_observation");
      install = symbol<Install>("td_student_install_prepared");
      debug = symbol<Debug>("td_debug");
    } catch (...) {
      dlclose(library_);
      library_ = nullptr;
      throw;
    }
  }
  ~Runtime() { if (library_) dlclose(library_); }

  using Create = void* (*)(const double*, std::size_t);
  using Destroy = void (*)(void*);
  using SettingsCount = std::size_t (*)();
  using ObserveExternal = int (*)(void*, const double*, std::size_t,
                                  const int32_t*, std::size_t);
  using Observe = int (*)(void*, const double*, std::size_t, int32_t*,
                          std::size_t);
  using Activate = int (*)(void*, const double*, std::size_t);
  using PreContext = int (*)(void*, const double*, std::size_t, double*,
                             std::size_t);
  using Plan = int (*)(void*, const double*, std::size_t, StudentCallback,
                       StudentCallback, void*, const double*, std::size_t);
  using Install = int (*)(void*);
  using Debug = const char* (*)(void*);
  Create create{};
  Destroy destroy{};
  SettingsCount settings_count{};
  ObserveExternal observe_external{};
  Observe observe{};
  Activate activate{};
  PreContext pre_context{};
  Plan plan{};
  Install install{};
  Debug debug{};

 private:
  template <class T>
  T symbol(const char* name) {
    dlerror();
    void* value = dlsym(library_, name);
    if (const char* error = dlerror())
      throw std::runtime_error(std::string("dlsym ") + name + ": " + error);
    return reinterpret_cast<T>(value);
  }
  void* library_{};
};

struct OwnedHandle {
  std::shared_ptr<Runtime> runtime;
  void* value{};
  ~OwnedHandle() { if (value) runtime->destroy(value); }
};

struct ActorEvent {
  int step{};
  std::uint64_t counter{};
  int stage{};
  int cell{};
  int suggested{};
  std::uint32_t legal_mask{};
  int action{};
  float log_probability{};
  float entropy{};
  std::vector<float> resources;
};

struct ActorDayCapture {
  int step{};
  std::vector<float> context;
  std::vector<float> observation;
  float observation_length{};
  std::vector<float> token_continuous;
  std::vector<std::uint32_t> token_categories;
  std::uint32_t token_count{};
  std::vector<ActorEvent> events;
};

struct Result {
  Cache cache;
  std::unique_ptr<OwnedHandle> handle;
  std::unique_ptr<fastkag::Simulator> env;
  std::unique_ptr<thomas_2945::Opponent> thomas;
  std::unique_ptr<metav4_2965::Opponent> meta;
  std::unique_ptr<metav4_2965::Opponent> soil;
  std::unique_ptr<salemali7_2900::Opponent> salemali;
  std::unique_ptr<fieldcraft_2887::Opponent> fieldcraft;
  fastkag::NativeAgentState opening_state;
  fastkag::NativeAgentState replay_opponent_state;
  std::vector<double> packed;
  std::vector<double> context;
  std::vector<std::vector<int32_t>> own_trace;
  std::vector<std::vector<int32_t>> rival_trace;
  std::uint64_t action_hash{1469598103934665603ull};
  std::uint64_t actor_hash{1469598103934665603ull};
  std::uint64_t actor_seed{};
  std::uint64_t actor_counter{};
  std::uint64_t branch_prefix_action_hash{};
  int forced_cell{-1};
  int forced_from{-1};
  int forced_to{-1};
  std::vector<ActorEvent> actor_trace;
  std::vector<ActorDayCapture> actor_day_captures;
  std::uint64_t prefix_action_hash{1469598103934665603ull};
  int opening_route{-1};
  int opening_switch_step{-1};
  int actor_days{};
  int validated_steps{};
  int suffix_steps{};
  std::string error;
};

void hash_actor_int(std::uint64_t& hash, std::int32_t value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  for (int i = 0; i < 4; ++i) {
    hash ^= (bits >> (8 * i)) & 0xff;
    hash *= 1099511628211ull;
  }
}

struct ActorCallbackState {
  const student_v3::Actor& actor;
  Result& result;
  int step;
  bool capture;
  bool greedy;
  bool force_alternative;
  ActorDayCapture* day_capture;
  std::vector<float> hidden;
  std::vector<float> resources;
  std::vector<float> logits;
  std::vector<float> next_hidden;
  std::uint32_t previous;
  int events{};
  bool forced{};

  static int release(void* state, int32_t cell, int32_t suggested,
                     int32_t mask, const double* resources,
                     std::size_t width) {
    return static_cast<ActorCallbackState*>(state)->choose(
        0, cell, suggested, mask, resources, width);
  }

  static int slot(void* state, int32_t cell, int32_t suggested,
                  int32_t mask, const double* resources,
                  std::size_t width) {
    return static_cast<ActorCallbackState*>(state)->choose(
        1, cell, suggested, mask, resources, width);
  }

  int choose(int stage, int cell, int suggested, int mask,
             const double* values, std::size_t width) {
    const auto& dimensions = actor.dimensions();
    const bool valid_stage_mask =
        (stage == 0 && (mask & ~0b110) == 0 && (mask & 0b110)) ||
        (stage == 1 && !(mask & (1 << 2)) && (mask & (1 << 1)));
    if (cell < 0 || cell >= int(dimensions.cells) || suggested < 0 ||
        suggested >= int(dimensions.classes) || !mask ||
        (std::uint32_t(mask) >> dimensions.classes) ||
        !(mask & (1 << suggested)) ||
        width != economic_v1::kOldResource ||
        (dimensions.resources != economic_v1::kOldResource &&
         dimensions.resources != economic_v1::kOldResource +
                                 economic_v1::kResourceExtra &&
         dimensions.resources != economic_v1::kOldResource +
                                 economic_v1::kResourceExtra +
                                 economic_v1::kShopResourceExtra) ||
        !values || !valid_stage_mask)
      throw std::runtime_error("invalid native actor callback event");
    for (std::size_t i = 0; i < width; ++i)
      resources[i] = static_cast<float>(values[i]);
    if (dimensions.resources > width)
      economic_v1::prefix_flow(values, step / 24, resources.data() + width);
    if (dimensions.resources > width + economic_v1::kResourceExtra)
      economic_v1::shop_rate(*result.env,
          resources.data() + width + economic_v1::kResourceExtra);
    actor.step(hidden.data(), resources.data(), cell, stage, previous,
               std::uint32_t(mask), logits.data(), next_hidden.data());
    const std::uint64_t counter = result.actor_counter++;
    auto distribution = actor.sample_policy(
        logits.data(), std::uint32_t(mask), result.actor_seed, counter,
        greedy ? -1.0f : 1.0f);
    if (force_alternative && !forced && stage == 1 &&
        (mask & (mask - 1))) {
      const int original = distribution.action;
      distribution.action = actor.sample_policy(
          logits.data(), std::uint32_t(mask) & ~(1u << original),
          result.actor_seed, counter, -1.0f).action;
      result.forced_cell = cell;
      result.forced_from = original;
      result.forced_to = distribution.action;
      forced = true;
    }
    if (!(mask & (1 << distribution.action)))
      throw std::runtime_error("native actor sampled illegal action");
    hash_actor_int(result.actor_hash, step);
    hash_actor_int(result.actor_hash, stage);
    hash_actor_int(result.actor_hash, cell);
    hash_actor_int(result.actor_hash, mask);
    hash_actor_int(result.actor_hash, distribution.action);
    if (capture) {
      result.actor_trace.push_back({
          step, counter, stage, cell, suggested, std::uint32_t(mask),
          distribution.action, distribution.log_probability,
          distribution.entropy, {}});
      std::vector<float> normalized(dimensions.resources);
      actor.normalize_resources(this->resources.data(), normalized.data());
      day_capture->events.push_back({
          step, counter, stage, cell, suggested, std::uint32_t(mask),
          distribution.action, distribution.log_probability,
          distribution.entropy, std::move(normalized)});
    }
    hidden.swap(next_hidden);
    previous = std::uint32_t(distribution.action);
    ++events;
    return distribution.action;
  }
};

struct ActorDayMetric {
  int step{};
  int events{};
  std::vector<int> event_counts;
  double seconds{};
};

int fib(int index) {
  int left = 1, right = 1;
  for (int i = 0; i < std::max(0, index); ++i) {
    const int next = left + right;
    left = right;
    right = next;
  }
  return left;
}

bool unlocked_shop(const fastkag::Simulator& env, int shop) {
  return std::find(env.shops().begin(), env.shops().end(), shop) !=
         env.shops().end();
}

double route_plan_requirement(const fastkag::Simulator& env, int seat,
                              const std::vector<fastkag::PlayerAction>& route,
                              int horizon) {
  static constexpr int seed_cost[] = {10, 20, 50, 100, 80};
  static constexpr int animal_cost[] = {300, 400, 500};
  static constexpr int land_cost[] = {1000, 2000, 4000};
  const int step = env.step_count();
  int hires = env.farms()[seat].hires_today;
  int land = std::popcount(unsigned(env.farms()[seat].unlocked_mask));
  int previous_day = step / 24;
  double cumulative = 0.0, requirement = 0.0;
  const int stop = std::min<int>(route.size(), step + horizon);
  for (int future = step; future < stop; ++future) {
    const int day = future / 24;
    if (day != previous_day) {
      hires = 0;
      previous_day = day;
    }
    for (const auto& order : route[future].market) {
      const int quantity = std::max(1, order.quantity);
      double spend = 0.0, revenue = 0.0;
      const int item = int(order.item);
      switch (order.op) {
        case fastkag::Op::HIRE:
          spend = fib(hires++);
          break;
        case fastkag::Op::BUY_LAND: {
          const int extra = std::max(0, land - 1);
          if (extra < int(std::size(land_cost))) {
            spend = land_cost[extra];
            ++land;
          }
          break;
        }
        case fastkag::Op::BUY_SEED:
          if (item >= 0 && item < int(std::size(seed_cost)))
            spend = quantity * seed_cost[item];
          break;
        case fastkag::Op::BUY_ANIMAL:
          if (item >= int(fastkag::Item::GOOSE) &&
              item <= int(fastkag::Item::SHEEP))
            spend = quantity * animal_cost[item - int(fastkag::Item::GOOSE)];
          break;
        case fastkag::Op::BUY_PRODUCT:
          if (item >= 0 && item < fastkag::N_PRODUCTS)
            spend = quantity * env.market().prices[item];
          break;
        case fastkag::Op::SELL:
          if (item >= 0 && item < fastkag::N_PRODUCTS)
            revenue = quantity * env.market().prices[item];
          break;
        default:
          break;
      }
      cumulative += spend - revenue;
      requirement = std::max(requirement, cumulative);
    }
  }
  return requirement;
}

int choose_deployment_route(const fastkag::Simulator& env, int seat,
                            const fastkag::NativeTeammateExecutor& replay,
                            const std::array<int, 5>& routes) {
  // routes = G275, G195, G024, G316, G267. These are the two accepted
  // depth-3 deployment trees, reduced to only the fields they actually read.
  const int step = env.step_count();
  const auto& opening = replay.route_tape(routes[0]);
  if (step == 144) {
    if (!unlocked_shop(env, 7)) {  // YARN_STORE
      if (float(env.market().prices[int(fastkag::Item::MILK)]) <= 192.0f)
        return unlocked_shop(env, 5) ? routes[1] : routes[0];  // PIZZA_SHOP
      return unlocked_shop(env, 2) ? routes[0] : routes[1];   // FARMERS_MARKET
    }
    const float slack = float(env.farms()[seat].money -
                              route_plan_requirement(env, seat, opening, 48));
    if (slack <= 639.0f)
      return unlocked_shop(env, 3) ? routes[2] : routes[3];  // ICE_CREAM_SHOP
    return unlocked_shop(env, 4) ? routes[3] : routes[2];    // PET_CAFE
  }
  if (step == 168) {
    if (!unlocked_shop(env, 7)) {
      if (float(env.market().inventory[int(fastkag::Item::MILK)]) <= 9978.0f) {
        const float requirement = float(route_plan_requirement(
            env, seat, opening, 72));
        // G019 is projected through the frozen fallback to G195.
        return requirement <= 2930.0f ? routes[1] : routes[1];
      }
      // Both leaves are G275 after the frozen G114 -> G275 fallback.
      return routes[0];
    }
    const float requirement = float(route_plan_requirement(
        env, seat, opening, 72));
    if (requirement <= 1434.5f) return routes[4];
    // Both strawberry-price leaves select G024.
    return routes[2];
  }
  throw std::invalid_argument("deployment tree called outside checkpoint");
}

double market_shape(int shape, double value, double scale) {
  value = std::max(0.0, value);
  switch (shape) {
    case 0: return value;
    case 1: return value * value;
    case 2: return std::sqrt(value);
    case 3: return std::log1p(value);
    case 4: {
      const double ratio = value / scale;
      return ratio + 8.0 * std::pow(std::max(0.0, ratio - 1.0), 2);
    }
  }
  return value;
}

int projected_market_price(int item, int inventory) {
  struct Parameters { int base, equilibrium, scale, below, above; double bt, at; };
  static constexpr Parameters values[] = {
      {25,10000,400,2,3,.8,.2}, {35,10000,450,4,2,1.,.7},
      {60,10000,200,4,2,.4,.6}, {120,10000,100,2,0,.7,1.6},
      {250,10000,300,3,1,.2,3.6}, {50,10000,332,4,3,.4,.2},
      {160,10000,122,2,0,.6,1.6}, {200,10000,105,3,1,.2,3.2},
      {100,10000,200,0,0,.4,.4}};
  const auto& p = values[item];
  double price;
  if (inventory < p.equilibrium) {
    const double amplitude = p.bt * p.base /
        market_shape(p.below, p.scale, p.scale);
    price = p.base + amplitude *
        market_shape(p.below, p.equilibrium - inventory, p.scale);
  } else {
    const double amplitude = p.at * p.base /
        market_shape(p.above, p.scale, p.scale);
    price = p.base - amplitude *
        market_shape(p.above, inventory - p.equilibrium, p.scale);
  }
  return std::max(1, int(std::round(price)));
}

void sell_before_unfunded_land(const fastkag::Simulator& env, int seat,
                               fastkag::PlayerAction& action) {
  if (env.step_count() < 168 || action.market.empty() ||
      action.market.front().op != fastkag::Op::BUY_LAND)
    return;
  const auto& farm = env.farms()[seat];
  const int extra = std::popcount(unsigned(farm.unlocked_mask)) - 1;
  static constexpr int land_cost[] = {1000, 2000, 4000};
  if (extra < 0 || extra >= int(std::size(land_cost)) ||
      farm.money >= land_cost[extra])
    return;
  std::array<fastkag::PlayerAction, 2> preview_actions;
  preview_actions[seat] = action;
  const auto preview = env.preview_unit_phase(preview_actions);
  auto shed = preview.privates()[seat].shed;
  for (std::size_t index = 1; index < action.market.size(); ++index) {
    const auto& order = action.market[index];
    const int item = int(order.item);
    if (order.op != fastkag::Op::SELL || item < 0 ||
        item >= fastkag::N_PRODUCTS)
      continue;
    const int quantity = std::min(std::max(0, order.quantity),
                                  std::max(0, shed[item]));
    if (!quantity) continue;
    int revenue = 0;
    for (int unit = 0; unit < quantity; ++unit)
      revenue += projected_market_price(
          item, env.market().inventory[item] + unit);
    if (farm.money + revenue < land_cost[extra]) return;
    const auto moved = order;
    action.market.erase(action.market.begin() + index);
    action.market.insert(action.market.begin(), moved);
    return;
  }
}

class PrefixBatch {
 public:
  PrefixBatch(const std::string& library_path,
              const std::vector<std::string>& cache_paths,
              const std::string& thomas_asset,
              const std::string& meta_asset,
              const std::string& deployment_path)
      : runtime_(std::make_shared<Runtime>(library_path)),
        thomas_asset_(thomas_asset), meta_asset_(meta_asset) {
    if (cache_paths.empty() || cache_paths.size() > 256)
      throw std::invalid_argument("cache count must be in [1,256]");
    const Digest binary = file_digest(library_path);
    const Digest deployment = file_digest(deployment_path);
    const Digest thomas = thomas_asset.empty() ? Digest{} : file_digest(thomas_asset);
    const Digest meta = meta_asset.empty() ? Digest{} : file_digest(meta_asset);
    results_.reserve(cache_paths.size());
    for (const auto& path : cache_paths) {
      Result result;
      result.cache = read_cache(path);
      if (result.cache.hashes[0] != binary)
        throw std::runtime_error("R1 binary hash mismatch: " + path);
      if (result.cache.hashes[1] != deployment)
        throw std::runtime_error("deployment hash mismatch: " + path);
      const Digest& expected_asset = result.cache.opponent == 1 ? thomas : meta;
      if (expected_asset == Digest{} || result.cache.hashes[3] != expected_asset)
        throw std::runtime_error("opponent asset hash mismatch: " + path);
      if (result.cache.settings.size() != runtime_->settings_count())
        throw std::runtime_error("R1 settings width mismatch: " + path);
      results_.push_back(std::move(result));
    }
  }

  PrefixBatch(const std::string& library_path,
              const fastkag::NativeTeammateExecutor& replay,
              const std::vector<std::uint64_t>& seeds,
              const std::vector<int>& seats,
              const std::vector<int>& opponents,
              const std::vector<int>& routes,
              const std::vector<std::uint64_t>& policy_seeds,
              const std::vector<double>& settings,
              const std::vector<int>& deployment_routes,
              const std::string& thomas_asset,
              const std::string& meta_asset,
              const std::string& salemali_asset,
              const std::string& fieldcraft_asset,
              const std::string& soil_asset)
      : runtime_(std::make_shared<Runtime>(library_path)), replay_(&replay),
        thomas_asset_(thomas_asset), meta_asset_(meta_asset),
        salemali_asset_(salemali_asset), fieldcraft_asset_(fieldcraft_asset),
        soil_asset_(soil_asset),
        arbitrary_jobs_(true), per_job_actor_seeds_(true) {
    const std::size_t count = seeds.size();
    if (!count || count > 2048 || seats.size() != count ||
        opponents.size() != count || routes.size() != count ||
        policy_seeds.size() != count)
      throw std::invalid_argument("job columns must have equal size in [1,2048]");
    if (settings.size() != runtime_->settings_count())
      throw std::invalid_argument("R1 settings width mismatch");
    if (deployment_routes.size() != deployment_routes_.size())
      throw std::invalid_argument("deployment_routes must be G275,G195,G024,G316,G267");
    for (std::size_t i = 0; i < deployment_routes_.size(); ++i) {
      const int route = deployment_routes[i];
      if (route < 0 || route >= replay.route_count())
        throw std::out_of_range("deployment route is outside replay library");
      deployment_routes_[i] = route;
    }
    results_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
      const bool replay_opponent = opponents[i] == 3;
      const bool public_opponent =
          opponents[i] == 1 || opponents[i] == 2 || opponents[i] == 4 ||
          opponents[i] == 5 || opponents[i] == 6;
      if (opponents[i] == 6 && soil_asset_.empty())
        throw std::invalid_argument("Soil current requires soil_asset");
      if (seats[i] < 0 || seats[i] > 1 ||
          (!public_opponent && !replay_opponent) ||
          (replay_opponent
               ? routes[i] < 0 || routes[i] >= replay.route_count()
               : routes[i] != -1))
        throw std::invalid_argument(
            "JobBatch needs a public C++ opponent route=-1 or replay route");
      Result result;
      result.cache.path = "job[" + std::to_string(i) + "]";
      result.cache.seed = seeds[i];
      result.cache.seat = seats[i];
      result.cache.opponent = opponents[i];
      result.cache.route = routes[i];
      result.cache.policy_seed = policy_seeds[i];
      result.cache.steps = kPrefixSteps;
      result.cache.settings = settings;
      results_.push_back(std::move(result));
    }
  }

  void run(int threads, std::size_t stack_bytes, bool capture_prefix = false) {
    if (ran_) throw std::logic_error("prefix batch already ran");
    ran_ = true;
    if (threads <= 0)
      threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<int>(threads, results_.size());
    next_.store(0);
    struct Worker { PrefixBatch* self; bool capture; } worker{this, capture_prefix};
    auto entry = [](void* raw) -> void* {
      auto* self = static_cast<Worker*>(raw)->self;
      for (;;) {
        const std::size_t index = self->next_.fetch_add(1);
        if (index >= self->results_.size()) break;
        try { self->run_case(self->results_[index], static_cast<Worker*>(raw)->capture); }
        catch (const std::exception& error) { self->results_[index].error = error.what(); }
        catch (...) { self->results_[index].error = "unknown prefix runner error"; }
      }
      return nullptr;
    };
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) throw std::runtime_error("pthread_attr_init failed");
    stack_bytes = std::max<std::size_t>(stack_bytes, PTHREAD_STACK_MIN);
    if (pthread_attr_setstacksize(&attr, stack_bytes)) {
      pthread_attr_destroy(&attr);
      throw std::runtime_error("pthread_attr_setstacksize failed");
    }
    std::vector<pthread_t> workers(threads);
    int created = 0;
    for (; created < threads; ++created)
      if (pthread_create(&workers[created], &attr, entry, &worker)) break;
    pthread_attr_destroy(&attr);
    {
      py::gil_scoped_release release;
      for (int i = 0; i < created; ++i) pthread_join(workers[i], nullptr);
    }
    if (created != threads) throw std::runtime_error("pthread_create failed");
    for (const auto& result : results_)
      if (!result.error.empty())
        throw std::runtime_error(result.cache.path + ": " + result.error);
  }

  std::vector<std::uintptr_t> handles() const {
    require_complete();
    std::vector<std::uintptr_t> output;
    output.reserve(results_.size());
    for (const auto& result : results_)
      output.push_back(reinterpret_cast<std::uintptr_t>(result.handle->value));
    return output;
  }

  py::list packed() const {
    require_complete();
    py::list output;
    for (const auto& result : results_) {
      py::array_t<double> array(result.packed.size());
      std::copy(result.packed.begin(), result.packed.end(), array.mutable_data());
      output.append(std::move(array));
    }
    return output;
  }

  py::array_t<double> contexts() const {
    require_complete();
    py::array_t<double> output({static_cast<py::ssize_t>(results_.size()),
                                static_cast<py::ssize_t>(kContextWidth)});
    auto view = output.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(results_.size()); ++i)
      for (py::ssize_t j = 0; j < kContextWidth; ++j)
        view(i, j) = results_[i].context[j];
    return output;
  }

  py::list observations() const {
    require_complete();
    py::list output;
    for (const auto& result : results_)
      output.append(observation_dict(*result.env, result.cache.seat));
    return output;
  }

  int current_step() const {
    require_complete();
    const int step = results_.front().env->step_count();
    for (const auto& result : results_)
      if (result.env->step_count() != step)
        throw std::logic_error("prefix batch clocks diverged");
    return step;
  }

  void advance_to(int target_step, int threads, std::size_t stack_bytes,
                  bool capture_actions) {
    require_complete();
    const int start = current_step();
    if (target_step <= start || target_step > 719)
      throw std::invalid_argument("target_step must advance within the episode");
    if (threads <= 0)
      threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<int>(threads, results_.size());
    next_.store(0);
    struct Worker {
      PrefixBatch* self;
      int target;
      bool capture;
    } worker{this, target_step, capture_actions};
    auto entry = [](void* raw) -> void* {
      auto* worker = static_cast<Worker*>(raw);
      for (;;) {
        const std::size_t index = worker->self->next_.fetch_add(1);
        if (index >= worker->self->results_.size()) break;
        auto& result = worker->self->results_[index];
        try {
          worker->self->advance_case(result, worker->target, worker->capture);
        } catch (const std::exception& error) {
          result.error = error.what();
        } catch (...) {
          result.error = "unknown native suffix error";
        }
      }
      return nullptr;
    };
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) throw std::runtime_error("pthread_attr_init failed");
    stack_bytes = std::max<std::size_t>(stack_bytes, PTHREAD_STACK_MIN);
    if (pthread_attr_setstacksize(&attr, stack_bytes)) {
      pthread_attr_destroy(&attr);
      throw std::runtime_error("pthread_attr_setstacksize failed");
    }
    std::vector<pthread_t> workers(threads);
    int created = 0;
    for (; created < threads; ++created)
      if (pthread_create(&workers[created], &attr, entry, &worker)) break;
    pthread_attr_destroy(&attr);
    {
      py::gil_scoped_release release;
      for (int i = 0; i < created; ++i) pthread_join(workers[i], nullptr);
    }
    if (created != threads) throw std::runtime_error("pthread_create failed");
    for (const auto& result : results_) {
      if (!result.error.empty())
        throw std::runtime_error(result.cache.path + ": " + result.error);
      if (result.env->step_count() != target_step)
        throw std::runtime_error(result.cache.path + ": suffix clock drift");
    }
  }

  py::dict action_traces() const {
    require_complete();
    py::list own, rival;
    for (const auto& result : results_) {
      own.append(py::cast(result.own_trace));
      rival.append(py::cast(result.rival_trace));
    }
    py::dict output;
    output["own"] = std::move(own);
    output["rival"] = std::move(rival);
    return output;
  }

  void clear_action_traces() {
    for (auto& result : results_) {
      result.own_trace.clear();
      result.rival_trace.clear();
    }
  }

  py::dict actor_initial_state(const std::string& weights_path) {
    require_complete();
    ensure_actor(weights_path);
    const auto& dimensions = actor_->dimensions();
    py::array_t<float> hidden({static_cast<py::ssize_t>(results_.size()),
                               static_cast<py::ssize_t>(dimensions.hidden)});
    py::array_t<std::uint32_t> token_counts(results_.size());
    py::array_t<std::uint32_t> observation_lengths(results_.size());
    auto output = hidden.mutable_unchecked<2>();
    auto counts = token_counts.mutable_unchecked<1>();
    auto lengths = observation_lengths.mutable_unchecked<1>();
    for (std::size_t row = 0; row < results_.size(); ++row) {
      auto& result = results_[row];
      prepare_actor_state(result);
      const auto canonical = pack_observation(
          *result.env, result.cache.seat, true);
      if (canonical.size() > economic_v1::kOldObservation)
        throw std::runtime_error("canonical observation exceeds actor capacity");
      const auto tokens = student_v3::tokenize(
          *result.env, result.cache.seat, dimensions.token_capacity);
      std::vector<float> context(dimensions.context);
      std::vector<float> observation(dimensions.observation, 0.0f);
      for (std::size_t i = 0; i < context.size(); ++i)
        context[i] = static_cast<float>(result.context[i]);
      for (std::size_t i = 0; i < canonical.size(); ++i)
        observation[i] = static_cast<float>(canonical[i]);
      if (dimensions.observation > economic_v1::kOldObservation) {
        const auto extra = economic_v1::observation(
            *result.env, result.cache.seat);
        std::copy(extra.begin(), extra.end(),
                  observation.begin() + economic_v1::kOldObservation);
      }
      std::vector<float> state(dimensions.hidden);
      actor_->initial_hidden(
          context.data(), observation.data(), float(canonical.size()),
          tokens.continuous.data(), tokens.categories.data(), tokens.count,
          state.data());
      for (std::size_t column = 0; column < state.size(); ++column)
        output(row, column) = state[column];
      counts(row) = tokens.count;
      lengths(row) = canonical.size();
    }
    py::dict result;
    result["hidden"] = std::move(hidden);
    result["token_counts"] = std::move(token_counts);
    result["observation_lengths"] = std::move(observation_lengths);
    return result;
  }

  py::dict plan_native_actor(const std::string& weights_path,
                             std::uint64_t policy_seed, int threads,
                             std::size_t stack_bytes, bool capture) {
    require_complete();
    configure_actor(weights_path, policy_seed);
    const auto metric = plan_actor_day(threads, stack_bytes, capture);
    return actor_day_dict(metric);
  }

  py::dict run_native_actor_suffix(const std::string& weights_path,
                                   std::uint64_t policy_seed, int threads,
                                   std::size_t stack_bytes, bool capture,
                                   bool greedy = false,
                                   bool capture_action_traces = true,
                                   int branch_step = -1,
                                   std::uint64_t branch_salt = 0,
                                   bool branch_force_alternative = false,
                                   std::uint64_t branch_future_seed = 0) {
    require_complete();
    if (current_step() != 288)
      throw std::logic_error("native actor suffix must start at step 288");
    if (branch_step != -1 &&
        (branch_step < 288 || branch_step > 672 || branch_step % 24 || greedy))
      throw std::invalid_argument("sampled branch requires a student day");
    if (branch_step == -1 && (branch_salt || branch_future_seed))
      throw std::invalid_argument("branch RNG change requires a branch day");
    if (branch_force_alternative &&
        (branch_step == -1 || branch_salt || capture))
      throw std::invalid_argument("forced branch is diagnostic-only and needs a day");
    actor_greedy_ = greedy;
    actor_force_alt_step_ = branch_force_alternative ? branch_step : -1;
    configure_actor(weights_path, policy_seed);
    const auto started = std::chrono::steady_clock::now();
    double plan_seconds = 0.0, environment_seconds = 0.0;
    int events = 0;
    py::list days;
    for (int step = 288; step <= 672; step += 24) {
      if (current_step() != step)
        throw std::runtime_error("native actor suffix clock drift");
      if (step == branch_step && branch_future_seed)
        for (auto& result : results_)
          result.env->reseed_future(branch_future_seed);
      if (step == branch_step)
        for (auto& result : results_) {
          result.branch_prefix_action_hash = result.action_hash;
          result.actor_seed ^= branch_salt;
        }
      const auto metric = plan_actor_day(threads, stack_bytes, capture);
      if (step == branch_step)
        for (auto& result : results_) result.actor_seed ^= branch_salt;
      plan_seconds += metric.seconds;
      events += metric.events;
      days.append(actor_day_dict(metric));
      const auto advance_started = std::chrono::steady_clock::now();
      advance_to(step + 24, threads, stack_bytes,
                 capture && capture_action_traces);
      environment_seconds += std::chrono::duration<double>(
          std::chrono::steady_clock::now() - advance_started).count();
    }
    const auto tail_started = std::chrono::steady_clock::now();
    advance_to(719, threads, stack_bytes, capture && capture_action_traces);
    environment_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - tail_started).count();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    py::dict output;
    output["sessions"] = results_.size();
    output["events"] = events;
    output["days"] = std::move(days);
    output["plan_seconds"] = plan_seconds;
    output["environment_seconds"] = environment_seconds;
    output["wall_seconds"] = seconds;
    output["events_per_second"] = events / seconds;
    output["current_step"] = current_step();
    return output;
  }

  py::list actor_traces() const {
    require_complete();
    py::list sessions;
    for (const auto& result : results_) {
      py::list events;
      for (const auto& event : result.actor_trace)
        events.append(py::make_tuple(
            event.step, event.counter, event.stage, event.cell,
            event.suggested, event.legal_mask, event.action,
            event.log_probability, event.entropy));
      sessions.append(std::move(events));
    }
    return sessions;
  }

  py::dict ppo_arrays() const {
    require_complete();
    if (!actor_ || current_step() != 719)
      throw std::logic_error("PPO arrays require a completed native actor rollout");
    const auto& d = actor_->dimensions();
    std::size_t day_count = 0, event_count = 0;
    for (const auto& result : results_) {
      day_count += result.actor_day_captures.size();
      for (const auto& day : result.actor_day_captures)
        event_count += day.events.size();
    }
    if (day_count != results_.size() * 17)
      throw std::logic_error("PPO capture requires all 17 actor days");
    const auto nday = static_cast<py::ssize_t>(day_count);
    const auto nevent = static_cast<py::ssize_t>(event_count);
    py::array_t<std::int32_t> day_session(nday), day_step(nday);
    py::array_t<std::int64_t> day_event_offsets(day_count + 1);
    py::array_t<float> context({nday, py::ssize_t(d.context)});
    py::array_t<float> observation({nday, py::ssize_t(d.observation)});
    py::array_t<float> observation_length(nday);
    py::array_t<float> token_continuous(
        {nday, py::ssize_t(d.token_capacity), py::ssize_t(d.token_continuous)});
    py::array_t<std::uint32_t> token_categories(
        {nday, py::ssize_t(d.categories), py::ssize_t(d.token_capacity)});
    py::array_t<std::uint32_t> token_count(nday);
    py::array_t<float> event_resources({nevent, py::ssize_t(d.resources)});
    py::array_t<std::int32_t> event_day(nevent), event_session(nevent),
        event_cell(nevent), event_stage(nevent), event_previous(nevent),
        event_action(nevent);
    py::array_t<std::uint32_t> event_legal_mask(nevent);
    py::array_t<bool> event_legal({nevent, py::ssize_t(d.classes)});
    py::array_t<float> old_logprob(nevent), old_entropy(nevent);
    py::array_t<std::uint64_t> seeds(results_.size()),
        policy_seeds(results_.size());
    py::array_t<std::int32_t> seats(results_.size()),
        opponents(results_.size()), routes(results_.size());
    py::array_t<double> own_cash(results_.size()), rival_cash(results_.size());

    auto ds = day_session.mutable_unchecked<1>();
    auto dst = day_step.mutable_unchecked<1>();
    auto offsets = day_event_offsets.mutable_unchecked<1>();
    auto contexts = context.mutable_unchecked<2>();
    auto observations = observation.mutable_unchecked<2>();
    auto lengths = observation_length.mutable_unchecked<1>();
    auto continuous = token_continuous.mutable_unchecked<3>();
    auto categories = token_categories.mutable_unchecked<3>();
    auto counts = token_count.mutable_unchecked<1>();
    auto resources = event_resources.mutable_unchecked<2>();
    auto eday = event_day.mutable_unchecked<1>();
    auto esession = event_session.mutable_unchecked<1>();
    auto cells = event_cell.mutable_unchecked<1>();
    auto stages = event_stage.mutable_unchecked<1>();
    auto previous = event_previous.mutable_unchecked<1>();
    auto actions = event_action.mutable_unchecked<1>();
    auto masks = event_legal_mask.mutable_unchecked<1>();
    auto legal = event_legal.mutable_unchecked<2>();
    auto logprobs = old_logprob.mutable_unchecked<1>();
    auto entropies = old_entropy.mutable_unchecked<1>();
    auto seed_values = seeds.mutable_unchecked<1>();
    auto policy_values = policy_seeds.mutable_unchecked<1>();
    auto seat_values = seats.mutable_unchecked<1>();
    auto opponent_values = opponents.mutable_unchecked<1>();
    auto route_values = routes.mutable_unchecked<1>();
    auto own_values = own_cash.mutable_unchecked<1>();
    auto rival_values = rival_cash.mutable_unchecked<1>();
    std::size_t day_index = 0, event_index = 0;
    offsets(0) = 0;
    for (std::size_t session = 0; session < results_.size(); ++session) {
      const auto& result = results_[session];
      seed_values(session) = result.cache.seed;
      policy_values(session) = result.cache.policy_seed;
      seat_values(session) = result.cache.seat;
      opponent_values(session) = result.cache.opponent;
      route_values(session) = result.cache.route;
      own_values(session) = result.env->farms()[result.cache.seat].money;
      rival_values(session) = result.env->farms()[1 - result.cache.seat].money;
      for (const auto& day : result.actor_day_captures) {
        ds(day_index) = session;
        dst(day_index) = day.step;
        lengths(day_index) = day.observation_length;
        counts(day_index) = day.token_count;
        for (std::size_t i = 0; i < day.context.size(); ++i)
          contexts(day_index, i) = day.context[i];
        for (std::size_t i = 0; i < day.observation.size(); ++i)
          observations(day_index, i) = day.observation[i];
        for (std::size_t token = 0; token < d.token_capacity; ++token)
          for (std::size_t column = 0; column < d.token_continuous; ++column)
            continuous(day_index, token, column) =
                day.token_continuous[token * d.token_continuous + column];
        for (std::size_t category = 0; category < d.categories; ++category)
          for (std::size_t token = 0; token < d.token_capacity; ++token)
            categories(day_index, category, token) =
                day.token_categories[category * d.token_capacity + token];
        std::uint32_t prior = d.classes;
        for (const auto& event : day.events) {
          eday(event_index) = day_index;
          esession(event_index) = session;
          cells(event_index) = event.cell;
          stages(event_index) = event.stage;
          previous(event_index) = prior;
          actions(event_index) = event.action;
          masks(event_index) = event.legal_mask;
          logprobs(event_index) = event.log_probability;
          entropies(event_index) = event.entropy;
          for (std::size_t i = 0; i < event.resources.size(); ++i)
            resources(event_index, i) = event.resources[i];
          for (std::size_t i = 0; i < d.classes; ++i)
            legal(event_index, i) = event.legal_mask & (1u << i);
          prior = event.action;
          ++event_index;
        }
        offsets(day_index + 1) = event_index;
        ++day_index;
      }
    }
    py::dict output;
    output["seed"] = std::move(seeds);
    output["seat"] = std::move(seats);
    output["opponent"] = std::move(opponents);
    output["route"] = std::move(routes);
    output["policy_seed"] = std::move(policy_seeds);
    output["own_cash"] = std::move(own_cash);
    output["rival_cash"] = std::move(rival_cash);
    output["day_session_index"] = std::move(day_session);
    output["day_step"] = std::move(day_step);
    output["day_event_offsets"] = std::move(day_event_offsets);
    output["context"] = std::move(context);
    output["observation"] = std::move(observation);
    output["observation_length"] = std::move(observation_length);
    output["token_continuous"] = std::move(token_continuous);
    output["token_categories"] = std::move(token_categories);
    output["token_count"] = std::move(token_count);
    output["event_day_index"] = std::move(event_day);
    output["event_session_index"] = std::move(event_session);
    output["event_resources"] = std::move(event_resources);
    output["event_cell"] = std::move(event_cell);
    output["event_stage"] = std::move(event_stage);
    output["event_previous"] = std::move(event_previous);
    output["event_legal_mask"] = std::move(event_legal_mask);
    output["event_legal"] = std::move(event_legal);
    output["event_action"] = std::move(event_action);
    output["old_logprob"] = std::move(old_logprob);
    output["old_entropy"] = std::move(old_entropy);
    return output;
  }

  py::dict summary() const {
    py::list rows;
    for (const auto& result : results_) {
      py::dict row;
      row["cache"] = result.cache.path;
      row["seed"] = result.cache.seed;
      row["seat"] = result.cache.seat;
      row["opponent"] = result.cache.opponent == 1 ? "thomas" :
                        result.cache.opponent == 2 ? "meta" :
                        result.cache.opponent == 4 ? "salemali7" :
                        result.cache.opponent == 5 ? "fieldcraft" :
                        result.cache.opponent == 6 ? "soil_current" :
                        "replay_clean";
      row["route"] = result.cache.route;
      row["validated_steps"] = result.validated_steps;
      row["packed_count"] = result.packed.size();
      row["packed_sha256"] = result.packed.empty() ? "" : hex(double_digest(result.packed));
      row["context_sha256"] = result.context.empty() ? "" : hex(double_digest(result.context));
      row["step"] = result.env ? result.env->step_count() : -1;
      row["done"] = result.env && result.env->done();
      row["own_cash"] = result.env ? result.env->farms()[result.cache.seat].money : 0.;
      row["rival_cash"] = result.env ? result.env->farms()[1 - result.cache.seat].money : 0.;
      row["suffix_steps"] = result.suffix_steps;
      row["captured_actions"] = result.own_trace.size();
      row["action_hash"] = py::int_(result.action_hash);
      row["actor_days"] = result.actor_days;
      row["actor_events"] = py::int_(result.actor_counter);
      row["actor_seed"] = py::int_(result.actor_seed);
      row["actor_hash"] = py::int_(result.actor_hash);
      row["branch_prefix_action_hash"] = py::int_(result.branch_prefix_action_hash);
      row["forced_cell"] = result.forced_cell;
      row["forced_from"] = result.forced_from;
      row["forced_to"] = result.forced_to;
      row["policy_seed"] = py::int_(result.cache.policy_seed);
      row["prefix_action_hash"] = py::int_(result.prefix_action_hash);
      row["opening_route"] = result.opening_route;
      row["opening_switch_step"] = result.opening_switch_step;
      row["error"] = result.error;
      rows.append(std::move(row));
    }
    py::dict output;
    output["ran"] = ran_;
    output["complete"] = complete();
    output["cases"] = std::move(rows);
    return output;
  }

 private:
  void ensure_actor(const std::string& weights_path) {
    if (actor_) {
      if (weights_path != actor_path_)
        throw std::logic_error("native actor weights cannot change in-place");
      return;
    }
    auto actor = std::make_shared<student_v3::Actor>(
        student_v3::Actor::load(weights_path));
    const auto& dimensions = actor->dimensions();
    const bool original = dimensions.observation == economic_v1::kOldObservation &&
        dimensions.resources == economic_v1::kOldResource;
    const bool economic = dimensions.observation ==
            economic_v1::kOldObservation + economic_v1::kObservationExtra &&
        (dimensions.resources ==
            economic_v1::kOldResource + economic_v1::kResourceExtra ||
         dimensions.resources == economic_v1::kOldResource +
            economic_v1::kResourceExtra + economic_v1::kShopResourceExtra);
    if (dimensions.context != kContextWidth || !(original || economic) ||
        dimensions.classes != 11 || dimensions.previous != 12)
      throw std::runtime_error("native actor dimensions do not match planner ABI");
    actor_ = std::move(actor);
    actor_path_ = weights_path;
  }

  void configure_actor(const std::string& weights_path,
                       std::uint64_t policy_seed) {
    ensure_actor(weights_path);
    if (actor_configured_) {
      if (policy_seed != actor_policy_seed_)
        throw std::logic_error("native actor policy seed cannot change in-place");
      return;
    }
    actor_policy_seed_ = policy_seed;
    for (std::size_t index = 0; index < results_.size(); ++index)
      results_[index].actor_seed = per_job_actor_seeds_
          ? results_[index].cache.policy_seed
          : policy_seed + index;
    actor_configured_ = true;
  }

  void prepare_actor_state(Result& result) {
    result.packed = pack_observation(*result.env, result.cache.seat);
    result.context.resize(kContextWidth);
    if (runtime_->pre_context(
            result.handle->value, result.packed.data(), result.packed.size(),
            result.context.data(), result.context.size()) != kContextWidth) {
      const char* debug = runtime_->debug(result.handle->value);
      throw std::runtime_error(std::string("native actor pre-context: ") +
                               (debug ? debug : "unknown"));
    }
  }

  int plan_actor_case(Result& result, bool capture) {
    prepare_actor_state(result);
    const auto& dimensions = actor_->dimensions();
    const auto canonical = pack_observation(
        *result.env, result.cache.seat, true);
    if (canonical.size() > economic_v1::kOldObservation)
      throw std::runtime_error("canonical observation exceeds actor capacity");
    const auto tokens = student_v3::tokenize(
        *result.env, result.cache.seat, dimensions.token_capacity);
    std::vector<float> context(dimensions.context);
    std::vector<float> observation(dimensions.observation, 0.0f);
    for (std::size_t i = 0; i < context.size(); ++i)
      context[i] = static_cast<float>(result.context[i]);
    for (std::size_t i = 0; i < canonical.size(); ++i)
      observation[i] = static_cast<float>(canonical[i]);
    if (dimensions.observation > economic_v1::kOldObservation) {
      const auto extra = economic_v1::observation(
          *result.env, result.cache.seat);
      std::copy(extra.begin(), extra.end(),
                observation.begin() + economic_v1::kOldObservation);
    }
    ActorDayCapture* day_capture = nullptr;
    if (capture) {
      result.actor_day_captures.emplace_back();
      day_capture = &result.actor_day_captures.back();
      day_capture->step = result.env->step_count();
      day_capture->context.resize(dimensions.context);
      day_capture->observation.resize(dimensions.observation);
      actor_->normalize_context(context.data(), day_capture->context.data());
      actor_->normalize_observation(
          observation.data(), day_capture->observation.data());
      day_capture->observation_length = actor_->normalize_observation_length(
          float(canonical.size()));
      day_capture->token_continuous = tokens.continuous;
      day_capture->token_categories = tokens.categories;
      day_capture->token_count = tokens.count;
    }
    ActorCallbackState state{
        *actor_, result, result.env->step_count(), capture, actor_greedy_,
        actor_force_alt_step_ == result.env->step_count(),
        day_capture,
        std::vector<float>(dimensions.hidden),
        std::vector<float>(dimensions.resources),
        std::vector<float>(dimensions.classes),
        std::vector<float>(dimensions.hidden), dimensions.classes};
    actor_->initial_hidden(
        context.data(), observation.data(), float(canonical.size()),
        tokens.continuous.data(), tokens.categories.data(), tokens.count,
        state.hidden.data());
    const int count = runtime_->plan(
        result.handle->value, result.packed.data(), result.packed.size(),
        &ActorCallbackState::release, &ActorCallbackState::slot, &state,
        nullptr, 0);
    if (count < 0 || count != state.events) {
      const char* debug = runtime_->debug(result.handle->value);
      throw std::runtime_error(std::string("native actor plan: ") +
                               (debug ? debug : "event count mismatch"));
    }
    if (runtime_->install(result.handle->value)) {
      const char* debug = runtime_->debug(result.handle->value);
      throw std::runtime_error(std::string("native actor install: ") +
                               (debug ? debug : "unknown"));
    }
    ++result.actor_days;
    return state.events;
  }

  ActorDayMetric plan_actor_day(int threads, std::size_t stack_bytes,
                                bool capture) {
    if (!actor_configured_)
      throw std::logic_error("native actor is not configured");
    const int step = current_step();
    if (step < 288 || step > 672 || step % 24)
      throw std::logic_error("native actor plan requires a full day boundary");
    if (threads <= 0)
      threads = std::max(1u, std::thread::hardware_concurrency());
    threads = std::min<int>(threads, results_.size());
    std::vector<int> counts(results_.size());
    next_.store(0);
    struct Worker {
      PrefixBatch* self;
      bool capture;
      std::vector<int>* counts;
    } worker{this, capture, &counts};
    auto entry = [](void* raw) -> void* {
      auto* worker = static_cast<Worker*>(raw);
      for (;;) {
        const std::size_t index = worker->self->next_.fetch_add(1);
        if (index >= worker->self->results_.size()) break;
        auto& result = worker->self->results_[index];
        try {
          (*worker->counts)[index] = worker->self->plan_actor_case(
              result, worker->capture);
        } catch (const std::exception& error) {
          result.error = error.what();
        } catch (...) {
          result.error = "unknown native actor error";
        }
      }
      return nullptr;
    };
    pthread_attr_t attr;
    if (pthread_attr_init(&attr))
      throw std::runtime_error("pthread_attr_init failed");
    stack_bytes = std::max<std::size_t>(stack_bytes, PTHREAD_STACK_MIN);
    if (pthread_attr_setstacksize(&attr, stack_bytes)) {
      pthread_attr_destroy(&attr);
      throw std::runtime_error("pthread_attr_setstacksize failed");
    }
    std::vector<pthread_t> workers(threads);
    const auto started = std::chrono::steady_clock::now();
    int created = 0;
    for (; created < threads; ++created)
      if (pthread_create(&workers[created], &attr, entry, &worker)) break;
    pthread_attr_destroy(&attr);
    {
      py::gil_scoped_release release;
      for (int i = 0; i < created; ++i) pthread_join(workers[i], nullptr);
    }
    if (created != threads) throw std::runtime_error("pthread_create failed");
    for (const auto& result : results_)
      if (!result.error.empty())
        throw std::runtime_error(result.cache.path + ": " + result.error);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    int events = 0;
    for (const int count : counts) events += count;
    return {step, events, std::move(counts), seconds};
  }

  static py::dict actor_day_dict(const ActorDayMetric& metric) {
    py::dict output;
    output["step"] = metric.step;
    output["events"] = metric.events;
    output["event_counts"] = metric.event_counts;
    output["seconds"] = metric.seconds;
    output["events_per_second"] = metric.events / metric.seconds;
    return output;
  }

  fastkag::PlayerAction opponent_action(Result& result) const {
    const int player = 1 - result.cache.seat;
    if (result.cache.opponent == 1)
      return result.thomas->action(*result.env, player);
    if (result.cache.opponent == 2)
      return result.meta->action(*result.env, player);
    if (result.cache.opponent == 4)
      return result.salemali->action(*result.env, player);
    if (result.cache.opponent == 5)
      return result.fieldcraft->action(*result.env, player);
    if (result.cache.opponent == 6)
      return result.soil->action(*result.env, player);
    auto action = replay_->action_external(
        *result.env, player, result.cache.route,
        result.replay_opponent_state,
        fastkag::NativeMarketArm::LegacyDefault, nullptr, nullptr, true);
    if (fastkag::native_macro_unit_failures(*result.env, player, action))
      throw std::runtime_error("replay opponent unit macro failure at step " +
                               std::to_string(result.env->step_count()));
    return action;
  }

  void check_opponent_market(Result& result,
                             const fastkag::PlayerAction& action) const {
    if (result.cache.opponent == 3 && fastkag::native_macro_market_failures(
            *result.env, 1 - result.cache.seat, action))
      throw std::runtime_error("replay opponent market macro failure at step " +
                               std::to_string(result.env->step_count() - 1));
  }

  void run_case(Result& result, bool capture) {
    const auto& cache = result.cache;
    auto handle = std::make_unique<OwnedHandle>();
    handle->runtime = runtime_;
    handle->value = runtime_->create(cache.settings.data(), cache.settings.size());
    if (!handle->value) throw std::runtime_error("td_new failed");
    result.env = std::make_unique<fastkag::Simulator>(fastkag::Config{}, 0);
    result.env->reset(cache.seed);
    if (cache.opponent == 1)
      result.thomas = std::make_unique<thomas_2945::Opponent>(thomas_asset_);
    else if (cache.opponent == 2)
      result.meta = std::make_unique<metav4_2965::Opponent>(meta_asset_);
    else if (cache.opponent == 4)
      result.salemali = std::make_unique<salemali7_2900::Opponent>(
          salemali_asset_);
    else if (cache.opponent == 5)
      result.fieldcraft = std::make_unique<fieldcraft_2887::Opponent>(
          fieldcraft_asset_);
    else if (cache.opponent == 6)
      result.soil = std::make_unique<metav4_2965::Opponent>(soil_asset_, true);
    auto& env = *result.env;
    result.opening_route = arbitrary_jobs_ ? deployment_routes_[0] : -1;
    for (std::uint32_t step = 0; step < cache.steps; ++step) {
      if (env.step_count() != int(step)) throw std::runtime_error("prefix clock drift");
      if (arbitrary_jobs_ && result.opening_switch_step < 0 &&
          (step == 144 || step == 168)) {
        const int selected = choose_deployment_route(
            env, cache.seat, *replay_, deployment_routes_);
        if (selected != result.opening_route) {
          result.opening_route = selected;
          result.opening_switch_step = step;
        }
      }
      auto own_action = arbitrary_jobs_
          ? replay_->action_external(
                env, cache.seat, result.opening_route, result.opening_state)
          : cache.actions[step][0];
      if (arbitrary_jobs_)
        sell_before_unfunded_land(env, cache.seat, own_action);
      const auto opponent_action = this->opponent_action(result);
      if (!arbitrary_jobs_ && !same_action(opponent_action, cache.actions[step][1]))
        throw std::runtime_error("native opponent action mismatch at step " +
                                 std::to_string(step));
      const auto packed = pack_observation(env, cache.seat);
      const auto encoded = encode_action(own_action);
      if (runtime_->observe_external(handle->value, packed.data(), packed.size(),
                                     encoded.data(), encoded.size())) {
        const char* debug = runtime_->debug(handle->value);
        throw std::runtime_error(std::string("td_observe_external step ") +
                                 std::to_string(step) + ": " +
                                 (debug ? debug : "unknown"));
      }
      const auto rival_encoded = encode_action(opponent_action);
      hash_action(result.prefix_action_hash, step, 0, encoded);
      hash_action(result.prefix_action_hash, step, 1, rival_encoded);
      if (capture) {
        result.own_trace.push_back(encoded);
        result.rival_trace.push_back(rival_encoded);
      }
      std::array<fastkag::PlayerAction, 2> actions;
      actions[cache.seat] = std::move(own_action);
      actions[1 - cache.seat] = opponent_action;
      env.step(actions);
      check_opponent_market(result, opponent_action);
      ++result.validated_steps;
    }
    if (!arbitrary_jobs_) {
      const int actual_route = result.thomas
          ? result.thomas->route(1 - cache.seat)
          : result.meta->route(1 - cache.seat);
      if (actual_route != cache.route)
        throw std::runtime_error("native route mismatch");
    }
    result.packed = pack_observation(env, cache.seat);
    if (!arbitrary_jobs_ && (result.packed.size() != cache.packed_count ||
        double_digest(result.packed) != cache.hashes[5]))
      throw std::runtime_error("step288 packed hash mismatch");
    if (runtime_->activate(handle->value, result.packed.data(), result.packed.size()))
      throw std::runtime_error("td_activate_external failed");
    result.context.resize(kContextWidth);
    if (runtime_->pre_context(handle->value, result.packed.data(), result.packed.size(),
                              result.context.data(), result.context.size()) != kContextWidth)
      throw std::runtime_error("td_student_pre_context failed");
    if (!arbitrary_jobs_ && double_digest(result.context) != cache.hashes[6])
      throw std::runtime_error("step288 context hash mismatch");
    result.handle = std::move(handle);
  }

  static void hash_int(std::uint64_t& hash, std::int32_t value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    for (int i = 0; i < 4; ++i) {
      hash ^= (bits >> (8 * i)) & 0xff;
      hash *= 1099511628211ull;
    }
  }

  static void hash_action(std::uint64_t& hash, int step, int side,
                          const std::vector<int32_t>& action) {
    hash_int(hash, step);
    hash_int(hash, side);
    hash_int(hash, action.size());
    for (const int32_t value : action) hash_int(hash, value);
  }

  void advance_case(Result& result, int target_step, bool capture) {
    auto& env = *result.env;
    while (env.step_count() < target_step) {
      if (env.done()) throw std::runtime_error("episode ended before target step");
      const int step = env.step_count();
      result.packed = pack_observation(env, result.cache.seat);
      std::array<int32_t, 256> output{};
      const int count = runtime_->observe(
          result.handle->value, result.packed.data(), result.packed.size(),
          output.data(), output.size());
      if (count < 0) {
        const char* debug = runtime_->debug(result.handle->value);
        throw std::runtime_error(std::string("td_observe step ") +
                                 std::to_string(step) + ": " +
                                 (debug ? debug : "unknown"));
      }
      const auto own_action = decode_action(output.data(), count);
      const auto rival_action = opponent_action(result);
      const auto own_encoded = encode_action(own_action);
      const auto rival_encoded = encode_action(rival_action);
      hash_action(result.action_hash, step, 0, own_encoded);
      hash_action(result.action_hash, step, 1, rival_encoded);
      if (capture) {
        result.own_trace.push_back(own_encoded);
        result.rival_trace.push_back(rival_encoded);
      }
      std::array<fastkag::PlayerAction, 2> actions;
      actions[result.cache.seat] = own_action;
      actions[1 - result.cache.seat] = rival_action;
      env.step(actions);
      check_opponent_market(result, rival_action);
      ++result.suffix_steps;
    }
    result.packed = pack_observation(env, result.cache.seat);
  }

  bool complete() const {
    if (!ran_) return false;
    return std::all_of(results_.begin(), results_.end(), [](const auto& result) {
      return result.error.empty() && result.handle &&
             result.validated_steps == int(kPrefixSteps);
    });
  }

  void require_complete() const {
    if (!complete()) throw std::logic_error("prefix batch is not complete");
  }

  std::shared_ptr<Runtime> runtime_;
  std::shared_ptr<const student_v3::Actor> actor_;
  const fastkag::NativeTeammateExecutor* replay_{};
  std::array<int, 5> deployment_routes_{};
  std::string actor_path_;
  std::uint64_t actor_policy_seed_{};
  bool actor_configured_{};
  bool actor_greedy_{};
  int actor_force_alt_step_{-1};
  std::string thomas_asset_;
  std::string meta_asset_;
  std::string salemali_asset_;
  std::string fieldcraft_asset_;
  std::string soil_asset_;
  std::vector<Result> results_;
  std::atomic<std::size_t> next_{};
  bool ran_{};
  bool arbitrary_jobs_{};
  bool per_job_actor_seeds_{};
};

class JobBatch : public PrefixBatch {
 public:
  using PrefixBatch::PrefixBatch;
};

}  // namespace

void bind_prefix_batch(py::module_& module) {
  py::class_<PrefixBatch>(module, "PrefixBatch")
      .def(py::init<const std::string&, const std::vector<std::string>&,
                    const std::string&, const std::string&, const std::string&>(),
           py::arg("library_path"), py::arg("cache_paths"),
           py::arg("thomas_asset"), py::arg("meta_asset"),
           py::arg("deployment_path"))
      .def("run", &PrefixBatch::run, py::arg("threads") = 0,
           py::arg("stack_bytes") = kDefaultStackBytes,
           py::arg("capture_prefix") = false)
      .def_property_readonly("handles", &PrefixBatch::handles)
      .def_property_readonly("packed", &PrefixBatch::packed)
      .def_property_readonly("contexts", &PrefixBatch::contexts)
      .def_property_readonly("observations", &PrefixBatch::observations)
      .def_property_readonly("current_step", &PrefixBatch::current_step)
      .def("advance_to", &PrefixBatch::advance_to,
           py::arg("target_step"), py::arg("threads") = 0,
           py::arg("stack_bytes") = kDefaultStackBytes,
           py::arg("capture_actions") = false)
      .def("action_traces", &PrefixBatch::action_traces)
      .def("clear_action_traces", &PrefixBatch::clear_action_traces)
      .def("actor_initial_state", &PrefixBatch::actor_initial_state,
           py::arg("weights_path"))
      .def("plan_native_actor", &PrefixBatch::plan_native_actor,
           py::arg("weights_path"), py::arg("policy_seed"),
           py::arg("threads") = 0,
           py::arg("stack_bytes") = kDefaultStackBytes,
           py::arg("capture") = false)
      .def("run_native_actor_suffix", &PrefixBatch::run_native_actor_suffix,
           py::arg("weights_path"), py::arg("policy_seed"),
           py::arg("threads") = 0,
           py::arg("stack_bytes") = kDefaultStackBytes,
           py::arg("capture") = false,
           py::arg("greedy") = false,
           py::arg("capture_action_traces") = true,
           py::arg("branch_step") = -1,
           py::arg("branch_salt") = 0,
           py::arg("branch_force_alternative") = false,
           py::arg("branch_future_seed") = 0)
      .def("actor_traces", &PrefixBatch::actor_traces)
      .def("ppo_arrays", &PrefixBatch::ppo_arrays)
      .def("summary", &PrefixBatch::summary);

  py::class_<JobBatch, PrefixBatch>(module, "JobBatch")
      .def(py::init<const std::string&,
                    const fastkag::NativeTeammateExecutor&,
                    const std::vector<std::uint64_t>&,
                    const std::vector<int>&, const std::vector<int>&,
                    const std::vector<int>&,
                    const std::vector<std::uint64_t>&,
                    const std::vector<double>&, const std::vector<int>&,
                    const std::string&, const std::string&,
                    const std::string&, const std::string&,
                    const std::string&>(),
           py::arg("library_path"), py::arg("replay_executor"),
           py::arg("seeds"), py::arg("seats"), py::arg("opponents"),
           py::arg("routes"), py::arg("policy_seeds"), py::arg("settings"),
           py::arg("deployment_routes"), py::arg("thomas_asset"),
           py::arg("meta_asset"), py::arg("salemali_asset") = "",
           py::arg("fieldcraft_asset") = "",
           py::arg("soil_asset") = "",
           py::keep_alive<1, 3>());
}
