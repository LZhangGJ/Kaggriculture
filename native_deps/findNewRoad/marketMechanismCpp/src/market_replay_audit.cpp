#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "replay_format.hpp"

namespace fs = std::filesystem;

namespace {

using g001::replay::FileHeader;
using g001::replay::Record;
using g001::replay::ReplayHeader;

constexpr std::array<std::string_view, 9> kMarketItems{
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER"};
constexpr std::array<std::string_view, 12> kStockItems{
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER", "COW", "SHEEP", "GOOSE"};
constexpr std::array<std::string_view, 5> kSeedItems{
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"};
constexpr std::array<std::string_view, 3> kAnimals{"COW", "SHEEP", "GOOSE"};

template <std::size_t N>
int index_of(std::string_view s, const std::array<std::string_view, N>& names) {
  for (std::size_t i = 0; i < N; ++i) {
    if (s == names[i]) return static_cast<int>(i);
  }
  return -1;
}

int product_for_animal(std::string_view animal) {
  if (animal == "COW") return index_of("MILK", kMarketItems);
  if (animal == "SHEEP") return index_of("WOOL", kMarketItems);
  if (animal == "GOOSE") return index_of("EGG", kMarketItems);
  return -1;
}

template <typename T>
T clamp_cast(long long value) {
  const auto lo = static_cast<long long>(std::numeric_limits<T>::min());
  const auto hi = static_cast<long long>(std::numeric_limits<T>::max());
  return static_cast<T>(std::clamp(value, lo, hi));
}

class MappedFile {
 public:
  explicit MappedFile(const fs::path& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path.string());
    struct stat st {};
    if (::fstat(fd_, &st) != 0) {
      ::close(fd_);
      throw std::runtime_error("fstat failed: " + path.string());
    }
    size_ = static_cast<std::size_t>(st.st_size);
    if (size_ == 0) {
      ::close(fd_);
      throw std::runtime_error("empty input: " + path.string());
    }
    data_ = static_cast<const char*>(::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (data_ == MAP_FAILED) {
      ::close(fd_);
      data_ = nullptr;
      throw std::runtime_error("mmap failed: " + path.string());
    }
  }
  ~MappedFile() {
    if (data_) ::munmap(const_cast<char*>(data_), size_);
    if (fd_ >= 0) ::close(fd_);
  }
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  const char* data() const { return data_; }
  std::size_t size() const { return size_; }

 private:
  int fd_{-1};
  const char* data_{nullptr};
  std::size_t size_{0};
};

class JsonCursor {
 public:
  JsonCursor(const char* begin, const char* end) : p_(begin), end_(end), begin_(begin) {}

  void ws() {
    while (p_ != end_ && (*p_ == ' ' || *p_ == '\n' || *p_ == '\r' || *p_ == '\t')) ++p_;
  }
  bool take(char c) {
    ws();
    if (p_ != end_ && *p_ == c) { ++p_; return true; }
    return false;
  }
  void expect(char c) {
    if (!take(c)) fail(std::string("expected '") + c + "'");
  }
  std::string_view string() {
    ws();
    if (p_ == end_ || *p_ != '"') fail("expected string");
    const char* start = ++p_;
    bool escaped = false;
    while (p_ != end_) {
      if (*p_ == '\\') {
        escaped = true;
        p_ += 2;
        continue;
      }
      if (*p_ == '"') {
        const char* stop = p_++;
        if (escaped) fail("escaped strings are unsupported in schema keys/items");
        return {start, static_cast<std::size_t>(stop - start)};
      }
      ++p_;
    }
    fail("unterminated string");
  }
  std::string decoded_string() {
    ws();
    if (p_ == end_ || *p_ != '"') fail("expected string");
    ++p_;
    std::string out;
    while (p_ != end_) {
      char c = *p_++;
      if (c == '"') return out;
      if (c != '\\') { out.push_back(c); continue; }
      if (p_ == end_) fail("unterminated escape");
      char e = *p_++;
      switch (e) {
        case '"': case '\\': case '/': out.push_back(e); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u':
          // Team names are metadata only. Keep non-ASCII escapes losslessly readable.
          out.append("\\u");
          for (int i = 0; i < 4; ++i) {
            if (p_ == end_) fail("short unicode escape");
            out.push_back(*p_++);
          }
          break;
        default: fail("invalid escape");
      }
    }
    fail("unterminated string");
  }
  double number() {
    ws();
    const char* start = p_;
    while (p_ != end_ && ((*p_ >= '0' && *p_ <= '9') || *p_ == '-' || *p_ == '+' ||
                          *p_ == '.' || *p_ == 'e' || *p_ == 'E')) ++p_;
    if (start == p_) fail("expected number");
    double value = 0;
    auto result = std::from_chars(start, p_, value);
    if (result.ec != std::errc{}) fail("invalid number");
    return value;
  }
  long long integer() { return static_cast<long long>(std::llround(number())); }

  void skip() {
    ws();
    if (p_ == end_) fail("unexpected EOF");
    if (*p_ == '"') { skip_string(); return; }
    if (*p_ == '{') {
      ++p_; ws();
      if (take('}')) return;
      do { skip_string(); expect(':'); skip(); } while (take(','));
      expect('}'); return;
    }
    if (*p_ == '[') {
      ++p_; ws();
      if (take(']')) return;
      do { skip(); } while (take(','));
      expect(']'); return;
    }
    while (p_ != end_ && *p_ != ',' && *p_ != '}' && *p_ != ']' &&
           *p_ != ' ' && *p_ != '\n' && *p_ != '\r' && *p_ != '\t') ++p_;
  }
  bool eof() { ws(); return p_ == end_; }

 private:
  [[noreturn]] void fail(const std::string& what) const {
    throw std::runtime_error(what + " at byte " + std::to_string(p_ - begin_));
  }
  void skip_string() {
    ws();
    if (p_ == end_ || *p_++ != '"') fail("expected string");
    while (p_ != end_) {
      char c = *p_++;
      if (c == '"') return;
      if (c == '\\') {
        if (p_ == end_) fail("unterminated escape");
        ++p_;
      }
    }
    fail("unterminated string");
  }
  const char* p_;
  const char* end_;
  const char* begin_;
};

struct ReplayData {
  std::uint64_t episode_id{};
  std::uint64_t seed{};
  std::array<double, 2> reward{};
  std::array<std::string, 2> names{};
  std::vector<Record> records;
  std::uint64_t input_bytes{};
};

template <typename Fn>
void object_fields(JsonCursor& j, Fn&& fn) {
  j.expect('{');
  if (j.take('}')) return;
  do {
    auto key = j.string();
    j.expect(':');
    fn(key);
  } while (j.take(','));
  j.expect('}');
}

template <typename Fn>
void array_values(JsonCursor& j, Fn&& fn) {
  j.expect('[');
  if (j.take(']')) return;
  std::size_t index = 0;
  do { fn(index++); } while (j.take(','));
  j.expect(']');
}

template <std::size_t N>
void parse_named_numbers(JsonCursor& j, const std::array<std::string_view, N>& names,
                         std::int16_t (&out)[N], bool add = false) {
  object_fields(j, [&](std::string_view key) {
    const auto value = j.integer();
    const int idx = index_of(key, names);
    if (idx >= 0) {
      const long long merged = add ? static_cast<long long>(out[idx]) + value : value;
      out[idx] = clamp_cast<std::int16_t>(merged);
    }
  });
}

void parse_market_named(JsonCursor& j, std::int32_t (&out)[9]) {
  object_fields(j, [&](std::string_view key) {
    const auto value = j.integer();
    const int idx = index_of(key, kMarketItems);
    if (idx >= 0) out[idx] = clamp_cast<std::int32_t>(value);
  });
}

void parse_market_price(JsonCursor& j, std::int16_t (&out)[9]) {
  parse_named_numbers(j, kMarketItems, out);
}

void parse_action_order(JsonCursor& j, Record& rec) {
  std::string_view op;
  std::string_view item;
  long long quantity = 1;
  array_values(j, [&](std::size_t i) {
    if (i == 0) op = j.string();
    else if (i == 1) item = j.string();
    else if (i == 2) quantity = j.integer();
    else j.skip();
  });
  if (op == "SELL") {
    int idx = index_of(item, kMarketItems);
    if (idx >= 0) rec.sell[idx] = clamp_cast<std::int16_t>(rec.sell[idx] + quantity);
  } else if (op == "BUY_PRODUCT") {
    int idx = index_of(item, kMarketItems);
    if (idx >= 0) rec.buy_product[idx] = clamp_cast<std::int16_t>(rec.buy_product[idx] + quantity);
  } else if (op == "BUY_SEED") {
    int idx = index_of(item, kSeedItems);
    if (idx >= 0) rec.buy_seed[idx] = clamp_cast<std::int16_t>(rec.buy_seed[idx] + quantity);
  } else if (op == "BUY_ANIMAL") {
    int idx = index_of(item, kAnimals);
    if (idx >= 0) rec.buy_animal[idx] = clamp_cast<std::int16_t>(rec.buy_animal[idx] + quantity);
  } else if (op == "HIRE") {
    rec.hire_count = clamp_cast<std::uint8_t>(rec.hire_count + 1);
  } else if (op == "BUY_LAND") {
    rec.buy_land_count = clamp_cast<std::uint8_t>(rec.buy_land_count + 1);
  }
}

void parse_action(JsonCursor& j, Record& rec) {
  object_fields(j, [&](std::string_view key) {
    if (key == "market") {
      array_values(j, [&](std::size_t) { parse_action_order(j, rec); });
    } else {
      j.skip();
    }
  });
}

void parse_tile(JsonCursor& j, Record& rec, std::size_t farm) {
  // null, "LOCKED", and empty strings have no public production information.
  j.ws();
  if (!j.take('{')) { j.skip(); return; }
  std::string_view kind;
  std::string_view crop;
  std::string_view animal;
  long long yield = 0;
  if (!j.take('}')) {
    do {
      auto key = j.string();
      j.expect(':');
      if (key == "kind") kind = j.string();
      else if (key == "crop") crop = j.string();
      else if (key == "animal") animal = j.string();
      else if (key == "yield_units") yield = j.integer();
      else j.skip();
    } while (j.take(','));
    j.expect('}');
  }
  int product = -1;
  if (kind == "PLANT") product = index_of(crop, kMarketItems);
  else if (kind == "PASTURE" || kind == "COOP") product = product_for_animal(animal);
  if (product >= 0 && farm < 2) {
    rec.farm_producers[farm][product] = clamp_cast<std::int16_t>(rec.farm_producers[farm][product] + 1);
    rec.farm_ready[farm][product] = clamp_cast<std::int16_t>(rec.farm_ready[farm][product] + yield);
  }
}

void parse_farm(JsonCursor& j, Record& rec, std::size_t farm) {
  object_fields(j, [&](std::string_view key) {
    if (key == "money") {
      const auto value = j.number();
      if (farm < 2) rec.farm_money[farm] = static_cast<float>(value);
    } else if (key == "tiles") {
      array_values(j, [&](std::size_t) {
        array_values(j, [&](std::size_t) { parse_tile(j, rec, farm); });
      });
    } else {
      j.skip();
    }
  });
}

void parse_private(JsonCursor& j, Record& rec) {
  object_fields(j, [&](std::string_view key) {
    if (key == "shed") {
      parse_named_numbers(j, kStockItems, rec.own_stock, true);
    } else if (key == "seeds") {
      parse_named_numbers(j, kSeedItems, rec.own_seeds);
    } else if (key == "inventories") {
      array_values(j, [&](std::size_t) {
        parse_named_numbers(j, kStockItems, rec.own_stock, true);
      });
    } else {
      j.skip();
    }
  });
}

void parse_observation(JsonCursor& j, Record& rec) {
  object_fields(j, [&](std::string_view key) {
    if (key == "day") rec.day = clamp_cast<std::uint8_t>(j.integer());
    else if (key == "hour") rec.hour = clamp_cast<std::uint8_t>(j.integer());
    else if (key == "step") rec.turn = clamp_cast<std::uint16_t>(j.integer());
    else if (key == "player") rec.seat = clamp_cast<std::uint8_t>(j.integer());
    else if (key == "farms") {
      array_values(j, [&](std::size_t farm) { parse_farm(j, rec, farm); });
      if (rec.seat < 2) rec.money = rec.farm_money[rec.seat];
    } else if (key == "market") {
      object_fields(j, [&](std::string_view market_key) {
        if (market_key == "inventory") parse_market_named(j, rec.market_inventory);
        else if (market_key == "prices") parse_market_price(j, rec.market_price);
        else j.skip();
      });
    } else if (key == "private") {
      parse_private(j, rec);
    } else {
      j.skip();
    }
  });
}

void parse_agent_step(JsonCursor& j, ReplayData& out, std::size_t outer_turn,
                      std::size_t seat_hint) {
  Record rec{};
  rec.turn = clamp_cast<std::uint16_t>(outer_turn);
  rec.seat = clamp_cast<std::uint8_t>(seat_hint);
  object_fields(j, [&](std::string_view key) {
    if (key == "action") parse_action(j, rec);
    else if (key == "observation") parse_observation(j, rec);
    else j.skip();
  });
  out.records.push_back(rec);
}

void parse_info(JsonCursor& j, ReplayData& out) {
  object_fields(j, [&](std::string_view key) {
    if (key == "EpisodeId") out.episode_id = static_cast<std::uint64_t>(j.integer());
    else if (key == "seed") out.seed = static_cast<std::uint64_t>(j.integer());
    else if (key == "TeamNames") {
      array_values(j, [&](std::size_t i) {
        auto value = j.decoded_string();
        if (i < 2) out.names[i] = std::move(value);
      });
    } else j.skip();
  });
}

ReplayData parse_replay(const fs::path& path) {
  MappedFile file(path);
  JsonCursor j(file.data(), file.data() + file.size());
  ReplayData out;
  out.input_bytes = file.size();
  object_fields(j, [&](std::string_view key) {
    if (key == "info") parse_info(j, out);
    else if (key == "rewards") {
      array_values(j, [&](std::size_t i) {
        const auto value = j.number();
        if (i < 2) out.reward[i] = value;
      });
    } else if (key == "steps") {
      array_values(j, [&](std::size_t turn) {
        array_values(j, [&](std::size_t seat) { parse_agent_step(j, out, turn, seat); });
      });
    } else j.skip();
  });
  if (!j.eof()) throw std::runtime_error("trailing JSON data");
  if (out.records.empty()) throw std::runtime_error("no step records");
  return out;
}

void append_bytes(std::vector<char>& dst, const void* data, std::size_t size) {
  const auto* p = static_cast<const char*>(data);
  dst.insert(dst.end(), p, p + size);
}

std::vector<char> serialize_replay(const ReplayData& replay) {
  ReplayHeader header{};
  header.record_count = static_cast<std::uint32_t>(replay.records.size());
  header.episode_id = replay.episode_id;
  header.seed = replay.seed;
  header.reward[0] = replay.reward[0];
  header.reward[1] = replay.reward[1];
  for (int i = 0; i < 2; ++i) {
    header.team_name_bytes[i] = static_cast<std::uint16_t>(
        std::min<std::size_t>(replay.names[i].size(), std::numeric_limits<std::uint16_t>::max()));
  }
  header.chunk_bytes = static_cast<std::uint32_t>(
      sizeof(header) + header.team_name_bytes[0] + header.team_name_bytes[1] +
      replay.records.size() * sizeof(Record));
  std::vector<char> bytes;
  bytes.reserve(header.chunk_bytes);
  append_bytes(bytes, &header, sizeof(header));
  append_bytes(bytes, replay.names[0].data(), header.team_name_bytes[0]);
  append_bytes(bytes, replay.names[1].data(), header.team_name_bytes[1]);
  append_bytes(bytes, replay.records.data(), replay.records.size() * sizeof(Record));
  return bytes;
}

struct Options {
  std::vector<fs::path> inputs;
  fs::path output;
  std::size_t threads = std::max(1u, std::thread::hardware_concurrency());
  std::size_t limit = 0;
  bool verify = false;
};

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --input FILE_OR_DIR [--input ...] --output audit.mra"
               " [--threads N] [--limit N] [--verify]\n";
}

Options parse_options(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string_view arg = argv[i];
    auto value = [&](const char* name) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[i];
    };
    if (arg == "--input") o.inputs.emplace_back(value("--input"));
    else if (arg == "--output") o.output = value("--output");
    else if (arg == "--threads") o.threads = std::stoull(value("--threads"));
    else if (arg == "--limit") o.limit = std::stoull(value("--limit"));
    else if (arg == "--verify") o.verify = true;
    else if (arg == "--help" || arg == "-h") { usage(argv[0]); std::exit(0); }
    else throw std::runtime_error("unknown argument: " + std::string(arg));
  }
  if (o.inputs.empty() || o.output.empty()) throw std::runtime_error("--input and --output are required");
  if (o.threads == 0) throw std::runtime_error("--threads must be positive");
  return o;
}

std::vector<fs::path> discover(const Options& options) {
  std::vector<fs::path> paths;
  for (const auto& input : options.inputs) {
    if (fs::is_regular_file(input)) {
      paths.push_back(input);
    } else if (fs::is_directory(input)) {
      for (const auto& entry : fs::recursive_directory_iterator(input)) {
        if (!entry.is_regular_file()) continue;
        const auto name = entry.path().filename().string();
        if (name.size() >= 12 && name.ends_with("-replay.json")) paths.push_back(entry.path());
      }
    } else {
      throw std::runtime_error("input does not exist: " + input.string());
    }
  }
  std::sort(paths.begin(), paths.end());
  paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
  if (options.limit && paths.size() > options.limit) paths.resize(options.limit);
  if (paths.empty()) throw std::runtime_error("no replay JSON files found");
  return paths;
}

void verify_output(const fs::path& path, std::uint64_t expected_replays,
                   std::uint64_t expected_records) {
  std::ifstream in(path, std::ios::binary);
  FileHeader file{};
  in.read(reinterpret_cast<char*>(&file), sizeof(file));
  if (!in || std::memcmp(file.magic, "MRAUDIT1", 8) != 0 || file.version != 1 ||
      file.record_size != sizeof(Record) || file.replay_count != expected_replays ||
      file.record_count != expected_records) {
    throw std::runtime_error("binary header verification failed");
  }
  std::uint64_t replays = 0;
  std::uint64_t records = 0;
  while (in.peek() != EOF) {
    ReplayHeader chunk{};
    in.read(reinterpret_cast<char*>(&chunk), sizeof(chunk));
    if (!in || chunk.chunk_bytes < sizeof(chunk) || chunk.record_count == 0) {
      throw std::runtime_error("invalid replay chunk");
    }
    const std::uint64_t expected_chunk_bytes = sizeof(chunk) + chunk.team_name_bytes[0] +
        chunk.team_name_bytes[1] + static_cast<std::uint64_t>(chunk.record_count) * sizeof(Record);
    if (chunk.chunk_bytes != expected_chunk_bytes) throw std::runtime_error("bad replay chunk size");
    in.seekg(chunk.team_name_bytes[0] + chunk.team_name_bytes[1], std::ios::cur);
    bool checked_fixture = false;
    for (std::uint32_t i = 0; i < chunk.record_count; ++i) {
      Record rec{};
      in.read(reinterpret_cast<char*>(&rec), sizeof(rec));
      if (!in || rec.seat >= 2 || rec.day >= 30 || rec.hour >= 24 || rec.turn >= 720) {
        throw std::runtime_error("invalid or truncated turn record");
      }
      if (chunk.episode_id == 42 && rec.seat == 0) {
        const int strawberry_market = index_of("STRAWBERRY", kMarketItems);
        const int strawberry_stock = index_of("STRAWBERRY", kStockItems);
        const int melon_seed = index_of("MELON", kSeedItems);
        const int milk = index_of("MILK", kMarketItems);
        if (chunk.reward[0] != 123.0 || rec.turn != 719 || rec.sell[strawberry_market] != 3 ||
            rec.buy_product[0] != 2 || rec.buy_seed[melon_seed] != 4 ||
            rec.own_stock[strawberry_stock] != 3 || rec.farm_ready[0][strawberry_market] != 5 ||
            rec.farm_ready[1][milk] != 2) {
          throw std::runtime_error("fixture semantic verification failed");
        }
        checked_fixture = true;
      }
    }
    if (chunk.episode_id == 42 && !checked_fixture) {
      throw std::runtime_error("fixture seat 0 record missing");
    }
    ++replays;
    records += chunk.record_count;
  }
  if (replays != expected_replays || records != expected_records) {
    throw std::runtime_error("binary chunk verification failed");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto paths = discover(options);
    fs::create_directories(options.output.parent_path().empty() ? "." : options.output.parent_path());
    std::ofstream output(options.output, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create output: " + options.output.string());
    FileHeader file_header{};
    std::memcpy(file_header.magic, g001::replay::file_magic, 8);
    file_header.version = g001::replay::format_version;
    file_header.record_size = sizeof(Record);
    output.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));

    std::mutex output_mutex;
    std::mutex error_mutex;
    std::vector<std::string> errors;
    std::atomic<std::size_t> next{0};
    std::atomic<std::uint64_t> replay_count{0};
    std::atomic<std::uint64_t> record_count{0};
    std::atomic<std::uint64_t> input_bytes{0};
    const auto started = std::chrono::steady_clock::now();
    const auto worker_count = std::min<std::size_t>(options.threads, paths.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&] {
        for (;;) {
          const auto index = next.fetch_add(1);
          if (index >= paths.size()) break;
          try {
            auto replay = parse_replay(paths[index]);
            auto bytes = serialize_replay(replay);
            {
              std::lock_guard lock(output_mutex);
              output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
              if (!output) throw std::runtime_error("output write failed");
            }
            input_bytes.fetch_add(replay.input_bytes);
            record_count.fetch_add(replay.records.size());
            replay_count.fetch_add(1);
          } catch (const std::exception& e) {
            std::lock_guard lock(error_mutex);
            errors.push_back(paths[index].string() + ": " + e.what());
          }
        }
      });
    }
    for (auto& worker : workers) worker.join();
    file_header.replay_count = replay_count.load();
    file_header.record_count = record_count.load();
    output.seekp(0);
    output.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
    output.close();

    if (options.verify) verify_output(options.output, file_header.replay_count, file_header.record_count);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const double mib = static_cast<double>(input_bytes.load()) / (1024.0 * 1024.0);
    std::cout << "replays=" << file_header.replay_count
              << " records=" << file_header.record_count
              << " input_mib=" << std::llround(mib)
              << " output_mib=" << std::llround(static_cast<double>(fs::file_size(options.output)) / (1024.0 * 1024.0))
              << " seconds=" << elapsed
              << " throughput_mib_s=" << (elapsed > 0 ? mib / elapsed : 0)
              << " threads=" << worker_count
              << " errors=" << errors.size() << '\n';
    for (const auto& error : errors) std::cerr << "error: " << error << '\n';
    return errors.empty() ? 0 : 2;
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << '\n';
    usage(argv[0]);
    return 1;
  }
}
