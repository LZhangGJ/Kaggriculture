#pragma once

#include <cstdint>

namespace g001::replay {

inline constexpr char file_magic[8] = {'M', 'R', 'A', 'U', 'D', 'I', 'T', '1'};
inline constexpr std::uint32_t format_version = 1;

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t record_size;
    std::uint64_t replay_count;
    std::uint64_t record_count;
    std::uint8_t reserved[32];
};

struct Record {
    std::uint16_t turn{};
    std::uint8_t seat{};
    std::uint8_t day{};
    std::uint8_t hour{};
    std::uint8_t hire_count{};
    std::uint8_t buy_land_count{};
    std::uint8_t reserved{};
    float money{};
    float farm_money[2]{};
    std::int32_t market_inventory[9]{};
    std::int16_t market_price[9]{};
    std::int16_t own_stock[12]{};
    std::int16_t own_seeds[5]{};
    std::int16_t farm_ready[2][9]{};
    std::int16_t farm_producers[2][9]{};
    std::int16_t sell[9]{};
    std::int16_t buy_product[9]{};
    std::int16_t buy_seed[5]{};
    std::int16_t buy_animal[3]{};
};

struct ReplayHeader {
    std::uint32_t chunk_bytes{};
    std::uint32_t record_count{};
    std::uint64_t episode_id{};
    std::uint64_t seed{};
    double reward[2]{};
    std::uint16_t team_name_bytes[2]{};
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 64);
static_assert(sizeof(Record) == 232);
static_assert(sizeof(ReplayHeader) == 44);

}  // namespace g001::replay
