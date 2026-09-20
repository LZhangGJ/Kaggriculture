#pragma once

#include "replay_format.hpp"

#include <cstddef>
#include <cstdint>

namespace g001::option {

constexpr std::size_t option_count = 7;

enum class Kind : std::uint8_t {
    Baseline = 0,
    Hold = 1,
    Drip = 2,
    PriceTarget = 3,
    InventoryTarget = 4,
    PreDump = 5,
    Clear = 6,
};

inline constexpr std::uint64_t schema_hash = 0xc11d66e7a47f9284ULL;

[[nodiscard]] const char* name(Kind kind);

struct DripState {
    int remaining_quota{};
    int remaining_windows{};
    int debt{};
};

struct DripTransition {
    int base_due{};
    int requested{};
    DripState next{};
};

[[nodiscard]] DripTransition settle_drip(
    DripState state, int available, int filled
);

#pragma pack(push, 1)
struct Header {
    char magic[8];                    // "MCFOPTN1"
    std::uint32_t version;
    std::uint32_t row_size;
    std::uint64_t row_count;
    std::uint64_t schema_hash_value{schema_hash};
    std::uint8_t reserved[32];
};
#pragma pack(pop)

struct Continuation {
    Kind kind{Kind::Baseline};
    std::uint8_t product_id{};
    std::uint8_t active{};
    std::uint8_t reserved{};
    std::int16_t remaining_quota{};
    std::int16_t remaining_windows{};
    std::int16_t debt{};
    std::uint16_t horizon_remaining{};
};

// One row selects one product, avoiding a nine-product quantity Cartesian
// product. Options persist across later raw-tape market visits.
struct Row {
    std::uint64_t episode_id{};
    std::uint64_t split_group{};
    std::uint16_t turn{};
    std::uint8_t seat{};
    std::uint8_t product_id{};
    std::uint8_t valid_mask{1};
    std::uint8_t done_mask{};
    std::uint16_t horizon{24};
    std::int16_t initial_remaining_quota{};
    std::int16_t initial_remaining_windows{};
    std::int16_t initial_debt{};
    std::int16_t price_target{};
    std::int16_t inventory_target{};
    float terminal_own_money[option_count]{};
    float terminal_opponent_money[option_count]{};
    float terminal_margin[option_count]{};
    float transition_margin_reward[option_count]{};
    std::uint16_t next_turn[option_count]{};
    Continuation continuation[option_count]{};
    // Public next state plus focal player's own private state only. No opponent
    // private inventory is serialized.
    g001::replay::Record next_state[option_count]{};
};

static_assert(sizeof(Header) == 64);
static_assert(alignof(Row) >= alignof(float));

}  // namespace g001::option
