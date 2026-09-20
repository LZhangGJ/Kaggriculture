#include "market.hpp"
#include "replay_format.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

using g001::market::Product;
using g001::market::price;
using g001::replay::FileHeader;
using g001::replay::Record;
using g001::replay::ReplayHeader;

constexpr std::size_t kProducts = 9;
constexpr std::size_t kPhases = 4;
constexpr std::size_t kMaxTurns = 720;
constexpr std::array<std::string_view, kProducts> kProductNames{
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER"};
constexpr std::array<std::string_view, kPhases> kPhaseNames{
    "ALL", "EARLY", "MIDDLE", "END"};

struct Options {
    fs::path input;
    fs::path csv;
    fs::path text;
    std::size_t threads = 192;
    int early_end = 240;
    int end_start = 624;
};

void usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " --input FILE.mra --csv metrics.csv --text report.txt"
                 " [--threads 192] [--early-end 240] [--end-start 624]\n";
}

Options parse_options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&](std::string_view name) -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + std::string(name));
            return argv[i];
        };
        if (arg == "--input") result.input = value(arg);
        else if (arg == "--csv") result.csv = value(arg);
        else if (arg == "--text") result.text = value(arg);
        else if (arg == "--threads") result.threads = std::stoull(value(arg));
        else if (arg == "--early-end") result.early_end = std::stoi(value(arg));
        else if (arg == "--end-start") result.end_start = std::stoi(value(arg));
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(arg));
        }
    }
    if (result.input.empty() || result.csv.empty() || result.text.empty()) {
        throw std::runtime_error("--input, --csv, and --text are required");
    }
    if (result.threads == 0) throw std::runtime_error("--threads must be positive");
    if (result.early_end <= 0 || result.end_start <= result.early_end
        || result.end_start >= static_cast<int>(kMaxTurns)) {
        throw std::runtime_error("invalid phase boundaries");
    }
    return result;
}

class MappedFile {
public:
    explicit MappedFile(const fs::path& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path.string());
        struct stat status {};
        if (::fstat(fd_, &status) != 0 || status.st_size <= 0) {
            ::close(fd_);
            throw std::runtime_error("cannot stat or empty input " + path.string());
        }
        size_ = static_cast<std::size_t>(status.st_size);
        data_ = static_cast<const char*>(
            ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0)
        );
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("mmap failed for " + path.string());
        }
    }
    ~MappedFile() {
        if (data_) ::munmap(const_cast<char*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
    }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    [[nodiscard]] const char* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }

private:
    int fd_ = -1;
    const char* data_ = nullptr;
    std::size_t size_ = 0;
};

template <typename T>
T load(const char* source) {
    T result{};
    std::memcpy(&result, source, sizeof(result));
    return result;
}

struct ReplayView {
    ReplayHeader header{};
    const char* records = nullptr;
};

struct Corpus {
    FileHeader header{};
    std::vector<ReplayView> replays;
};

Corpus parse_corpus(const MappedFile& file) {
    if (file.size() < sizeof(FileHeader)) throw std::runtime_error("truncated file header");
    Corpus result;
    result.header = load<FileHeader>(file.data());
    if (std::memcmp(result.header.magic, g001::replay::file_magic, 8) != 0
        || result.header.version != g001::replay::format_version
        || result.header.record_size != sizeof(Record)) {
        throw std::runtime_error("unsupported or corrupt MRA header");
    }

    std::size_t offset = sizeof(FileHeader);
    std::uint64_t records = 0;
    while (offset < file.size()) {
        if (file.size() - offset < sizeof(ReplayHeader)) {
            throw std::runtime_error("truncated replay header");
        }
        const auto replay = load<ReplayHeader>(file.data() + offset);
        const std::uint64_t expected = sizeof(ReplayHeader)
            + replay.team_name_bytes[0] + replay.team_name_bytes[1]
            + static_cast<std::uint64_t>(replay.record_count) * sizeof(Record);
        if (replay.record_count == 0 || replay.chunk_bytes != expected
            || expected > file.size() - offset) {
            throw std::runtime_error("invalid replay chunk at byte " + std::to_string(offset));
        }
        const auto names_bytes = static_cast<std::size_t>(
            replay.team_name_bytes[0] + replay.team_name_bytes[1]
        );
        result.replays.push_back({
            replay, file.data() + offset + sizeof(ReplayHeader) + names_bytes
        });
        records += replay.record_count;
        offset += replay.chunk_bytes;
    }
    if (offset != file.size() || result.replays.size() != result.header.replay_count
        || records != result.header.record_count) {
        throw std::runtime_error("MRA aggregate counts do not match payload");
    }
    return result;
}

Record replay_record(const ReplayView& replay, std::size_t index) {
    return load<Record>(replay.records + index * sizeof(Record));
}

struct Bucket {
    int lower = 0;
    int upper = 0;
    bool finite = false;
    bool valid = false;
};

int first_inventory_with_price_at_most(Product product, int target) {
    int high = 1;
    while (price(product, high) > target) {
        if (high > 1'000'000 / 2) throw std::runtime_error("price inversion did not converge");
        high *= 2;
    }
    int low = -1;
    while (high - low > 1) {
        const int middle = low + (high - low) / 2;
        if (price(product, middle) <= target) high = middle;
        else low = middle;
    }
    return high;
}

int first_inventory_with_price_below(Product product, int target) {
    int high = 1;
    while (price(product, high) >= target) {
        if (high > 1'000'000 / 2) throw std::runtime_error("price inversion did not converge");
        high *= 2;
    }
    int low = -1;
    while (high - low > 1) {
        const int middle = low + (high - low) / 2;
        if (price(product, middle) < target) high = middle;
        else low = middle;
    }
    return high;
}

Bucket invert_price(Product product, int observed_price) {
    if (observed_price < 1 || observed_price > price(product, 0)) return {};
    Bucket result;
    result.lower = first_inventory_with_price_at_most(product, observed_price);
    if (price(product, result.lower) != observed_price) return {};
    result.valid = true;
    if (observed_price == 1) {
        result.finite = false;
        return result;
    }
    result.upper = first_inventory_with_price_below(product, observed_price) - 1;
    result.finite = true;
    return result;
}

using BucketMap = std::array<std::map<int, Bucket>, kProducts>;

BucketMap build_bucket_cache(const Corpus& corpus) {
    std::array<std::map<int, bool>, kProducts> observed;
    for (const auto& replay : corpus.replays) {
        for (std::size_t r = 0; r < replay.header.record_count; ++r) {
            const auto record = replay_record(replay, r);
            if (record.seat != 0) continue;  // one copy of the shared public market
            for (std::size_t product = 0; product < kProducts; ++product) {
                observed[product].emplace(record.market_price[product], true);
            }
        }
    }
    BucketMap result;
    for (std::size_t product = 0; product < kProducts; ++product) {
        for (const auto& [observed_price, unused] : observed[product]) {
            (void)unused;
            result[product].emplace(
                observed_price,
                invert_price(static_cast<Product>(product), observed_price)
            );
        }
    }
    return result;
}

struct Metrics {
    std::uint64_t states = 0;
    std::uint64_t valid_price_states = 0;
    std::uint64_t finite_bucket_states = 0;
    std::uint64_t floor_states = 0;
    std::uint64_t price_bucket_covered = 0;
    long double bucket_span_sum = 0;
    long double bucket_cardinality_sum = 0;
    std::uint64_t max_bucket_span = 0;

    std::uint64_t transitions = 0;
    std::uint64_t valid_delta_intervals = 0;
    std::uint64_t finite_delta_intervals = 0;
    std::uint64_t delta_covered = 0;
    std::uint64_t floor_unidentifiable = 0;
    long double delta_midpoint_absolute_error_sum = 0;
    long double delta_interval_span_sum = 0;
    long double delta_interval_cardinality_sum = 0;
    std::uint64_t exact_inventory_changes = 0;
    std::uint64_t same_price_transitions = 0;
    std::uint64_t same_price_inventory_changes = 0;

    std::uint64_t opponent_sell_request_events = 0;
    std::uint64_t opponent_sell_requested_units = 0;
    std::uint64_t same_price_changes_with_sell_request = 0;
    std::uint64_t floor_transitions_with_sell_request = 0;

    Metrics& operator+=(const Metrics& other) {
#define ADD(field) field += other.field
        ADD(states); ADD(valid_price_states); ADD(finite_bucket_states); ADD(floor_states);
        ADD(price_bucket_covered); ADD(bucket_span_sum); ADD(bucket_cardinality_sum);
        max_bucket_span = std::max(max_bucket_span, other.max_bucket_span);
        ADD(transitions); ADD(valid_delta_intervals); ADD(finite_delta_intervals);
        ADD(delta_covered); ADD(floor_unidentifiable);
        ADD(delta_midpoint_absolute_error_sum); ADD(delta_interval_span_sum);
        ADD(delta_interval_cardinality_sum); ADD(exact_inventory_changes);
        ADD(same_price_transitions); ADD(same_price_inventory_changes);
        ADD(opponent_sell_request_events); ADD(opponent_sell_requested_units);
        ADD(same_price_changes_with_sell_request); ADD(floor_transitions_with_sell_request);
#undef ADD
        return *this;
    }
};

using MetricGrid = std::array<std::array<Metrics, kProducts>, kPhases>;

std::size_t phase_for_turn(int turn, const Options& options) {
    if (turn < options.early_end) return 1;
    if (turn < options.end_start) return 2;
    return 3;
}

void update_state(Metrics& metric, const Record& record, std::size_t product,
                  const Bucket& bucket) {
    ++metric.states;
    if (!bucket.valid) return;
    ++metric.valid_price_states;
    const auto exact = record.market_inventory[product];
    const bool covered = exact >= bucket.lower && (!bucket.finite || exact <= bucket.upper);
    metric.price_bucket_covered += static_cast<std::uint64_t>(covered);
    if (!bucket.finite) {
        ++metric.floor_states;
        return;
    }
    ++metric.finite_bucket_states;
    const auto span = static_cast<std::uint64_t>(bucket.upper - bucket.lower);
    metric.bucket_span_sum += span;
    metric.bucket_cardinality_sum += span + 1;
    metric.max_bucket_span = std::max(metric.max_bucket_span, span);
}

void update_transition(Metrics& metric, const Record& previous, const Record& current,
                       const Record* opponent_request, std::size_t product,
                       const Bucket& before, const Bucket& after) {
    ++metric.transitions;
    const auto true_delta = static_cast<std::int64_t>(current.market_inventory[product])
        - previous.market_inventory[product];
    const bool changed = true_delta != 0;
    metric.exact_inventory_changes += static_cast<std::uint64_t>(changed);
    const bool same_price = previous.market_price[product] == current.market_price[product];
    metric.same_price_transitions += static_cast<std::uint64_t>(same_price);
    metric.same_price_inventory_changes += static_cast<std::uint64_t>(same_price && changed);

    const int requested = opponent_request
        ? std::max(0, static_cast<int>(opponent_request->sell[product])) : 0;
    metric.opponent_sell_request_events += static_cast<std::uint64_t>(requested > 0);
    metric.opponent_sell_requested_units += static_cast<std::uint64_t>(requested);
    metric.same_price_changes_with_sell_request += static_cast<std::uint64_t>(
        same_price && changed && requested > 0
    );

    if (!before.valid || !after.valid) return;
    ++metric.valid_delta_intervals;
    const bool unbounded = !before.finite || !after.finite;
    metric.floor_unidentifiable += static_cast<std::uint64_t>(unbounded);
    metric.floor_transitions_with_sell_request += static_cast<std::uint64_t>(
        unbounded && requested > 0
    );

    // Price-only possible delta: [after.lower-before.upper,
    // after.upper-before.lower].  A floor endpoint contributes infinity.
    const bool lower_infinite = !before.finite;
    const bool upper_infinite = !after.finite;
    const std::int64_t lower = lower_infinite ? 0
        : static_cast<std::int64_t>(after.lower) - before.upper;
    const std::int64_t upper = upper_infinite ? 0
        : static_cast<std::int64_t>(after.upper) - before.lower;
    const bool covered = (lower_infinite || true_delta >= lower)
        && (upper_infinite || true_delta <= upper);
    metric.delta_covered += static_cast<std::uint64_t>(covered);

    if (unbounded) return;
    ++metric.finite_delta_intervals;
    const long double midpoint = (
        static_cast<long double>(after.lower) + after.upper
        - before.lower - before.upper
    ) / 2.0L;
    metric.delta_midpoint_absolute_error_sum += std::fabs(midpoint - true_delta);
    const auto span = static_cast<std::uint64_t>(upper - lower);
    metric.delta_interval_span_sum += span;
    metric.delta_interval_cardinality_sum += span + 1;
}

void process_replay(const ReplayView& replay, const BucketMap& buckets,
                    const Options& options, MetricGrid& output) {
    std::array<Record, kMaxTurns> observer{};
    std::array<Record, kMaxTurns> opponent{};
    std::array<bool, kMaxTurns> have_observer{};
    std::array<bool, kMaxTurns> have_opponent{};
    for (std::size_t index = 0; index < replay.header.record_count; ++index) {
        const auto record = replay_record(replay, index);
        if (record.turn >= kMaxTurns || record.seat >= 2) continue;
        if (record.seat == 0) {
            observer[record.turn] = record;
            have_observer[record.turn] = true;
        } else {
            opponent[record.turn] = record;
            have_opponent[record.turn] = true;
        }
    }

    for (std::size_t turn = 0; turn < kMaxTurns; ++turn) {
        if (!have_observer[turn]) continue;
        const auto& current = observer[turn];
        const auto phase = phase_for_turn(current.turn, options);
        for (std::size_t product = 0; product < kProducts; ++product) {
            const auto found = buckets[product].find(current.market_price[product]);
            if (found == buckets[product].end()) throw std::runtime_error("missing bucket cache");
            update_state(output[0][product], current, product, found->second);
            update_state(output[phase][product], current, product, found->second);
        }

        if (turn == 0 || !have_observer[turn - 1]) continue;
        const auto& previous = observer[turn - 1];
        // MRA convention: observation[t-1] -- requested action[t] -->
        // observation[t].  This is only an event label; requested SELL is not
        // assumed to have committed and is never used as inventory truth.
        const Record* request = have_opponent[turn] ? &opponent[turn] : nullptr;
        for (std::size_t product = 0; product < kProducts; ++product) {
            const auto& before = buckets[product].at(previous.market_price[product]);
            const auto& after = buckets[product].at(current.market_price[product]);
            update_transition(output[0][product], previous, current, request,
                              product, before, after);
            update_transition(output[phase][product], previous, current, request,
                              product, before, after);
        }
    }
}

double ratio(std::uint64_t numerator, std::uint64_t denominator) {
    return denominator == 0 ? std::numeric_limits<double>::quiet_NaN()
                            : static_cast<double>(numerator) / denominator;
}

double average(long double total, std::uint64_t count) {
    return count == 0 ? std::numeric_limits<double>::quiet_NaN()
                      : static_cast<double>(total / count);
}

Metrics product_sum(const MetricGrid& metrics, std::size_t phase) {
    Metrics result;
    for (const auto& metric : metrics[phase]) result += metric;
    return result;
}

void write_csv_row(std::ostream& out, std::string_view phase, std::string_view product,
                   const Metrics& m) {
    out << phase << ',' << product << ',' << m.states << ',' << m.valid_price_states << ','
        << m.finite_bucket_states << ',' << m.floor_states << ','
        << ratio(m.floor_states, m.valid_price_states) << ','
        << average(m.bucket_span_sum, m.finite_bucket_states) << ','
        << average(m.bucket_cardinality_sum, m.finite_bucket_states) << ','
        << m.max_bucket_span << ',' << ratio(m.price_bucket_covered, m.valid_price_states) << ','
        << m.transitions << ',' << m.valid_delta_intervals << ','
        << m.finite_delta_intervals << ',' << m.floor_unidentifiable << ','
        << ratio(m.floor_unidentifiable, m.valid_delta_intervals) << ','
        << average(m.delta_midpoint_absolute_error_sum, m.finite_delta_intervals) << ','
        << average(m.delta_interval_span_sum, m.finite_delta_intervals) << ','
        << average(m.delta_interval_cardinality_sum, m.finite_delta_intervals) << ','
        << ratio(m.delta_covered, m.valid_delta_intervals) << ','
        << m.exact_inventory_changes << ',' << m.same_price_transitions << ','
        << m.same_price_inventory_changes << ','
        << ratio(m.same_price_inventory_changes, m.same_price_transitions) << ','
        << ratio(m.same_price_inventory_changes, m.exact_inventory_changes) << ','
        << m.opponent_sell_request_events << ',' << m.opponent_sell_requested_units << ','
        << m.same_price_changes_with_sell_request << ','
        << m.floor_transitions_with_sell_request << '\n';
}

void write_csv(const fs::path& path, const MetricGrid& metrics) {
    fs::create_directories(path.parent_path().empty() ? "." : path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create " + path.string());
    out << std::setprecision(10);
    out << "phase,product,state_count,valid_price_state_count,finite_bucket_state_count,"
           "floor_state_count,floor_state_rate,mean_price_bucket_span,"
           "mean_price_bucket_cardinality,max_price_bucket_span,price_bucket_coverage_rate,"
           "transition_count,valid_delta_interval_count,finite_delta_interval_count,"
           "floor_unidentifiable_transition_count,floor_unidentifiable_rate,"
           "price_only_delta_midpoint_mae,mean_delta_interval_span,"
           "mean_delta_interval_cardinality,delta_interval_coverage_rate,"
           "exact_inventory_change_count,same_price_transition_count,"
           "same_price_inventory_change_count,same_price_change_rate,"
           "share_of_inventory_changes_hidden_by_same_price,opponent_sell_request_events,"
           "opponent_sell_requested_units,same_price_changes_with_sell_request,"
           "floor_transitions_with_sell_request\n";
    for (std::size_t phase = 0; phase < kPhases; ++phase) {
        write_csv_row(out, kPhaseNames[phase], "ALL_PRODUCTS", product_sum(metrics, phase));
        for (std::size_t product = 0; product < kProducts; ++product) {
            write_csv_row(out, kPhaseNames[phase], kProductNames[product],
                          metrics[phase][product]);
        }
    }
}

void write_text(const fs::path& path, const Corpus& corpus, const MetricGrid& metrics,
                const Options& options, double seconds, double input_mib) {
    fs::create_directories(path.parent_path().empty() ? "." : path.parent_path());
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot create " + path.string());
    out << std::fixed << std::setprecision(4);
    out << "Price-only market information audit\n"
        << "replays=" << corpus.header.replay_count
        << " records=" << corpus.header.record_count
        << " threads=" << std::min(options.threads, corpus.replays.size())
        << " seconds=" << seconds
        << " input_mib=" << input_mib
        << " throughput_mib_s=" << input_mib / seconds << '\n'
        << "phases: EARLY=[0," << options.early_end - 1 << "] MIDDLE=["
        << options.early_end << ',' << options.end_start - 1 << "] END=["
        << options.end_start << ",719]\n"
        << "Exact public market.inventory is offline truth. Rounded price is the only"
           " input to the reconstructed bucket/delta. A $1 bucket has no finite upper"
           " bound, so midpoint MAE and width exclude those transitions while coverage"
           " retains their unbounded interval.\n"
        << "Opponent SELL is requested-action metadata only: requested != committed."
           " No other-seat private field is read.\n\n";

    const auto aggregate = product_sum(metrics, 0);
    out << "ALL_PRODUCTS finite-delta midpoint MAE="
        << average(aggregate.delta_midpoint_absolute_error_sum,
                   aggregate.finite_delta_intervals)
        << " mean interval span="
        << average(aggregate.delta_interval_span_sum,
                   aggregate.finite_delta_intervals)
        << " coverage=" << 100.0 * ratio(aggregate.delta_covered,
                                          aggregate.valid_delta_intervals)
        << "% same-price inventory changes=" << aggregate.same_price_inventory_changes
        << " floor-unidentifiable transitions=" << aggregate.floor_unidentifiable
        << "\n\n";

    out << "Exact-inventory information gain over price-only:\n"
        << "- finite price bucket has "
        << average(aggregate.bucket_cardinality_sum,
                   aggregate.finite_bucket_states)
        << " candidate integer inventories on average; exact inventory selects one.\n"
        << "- finite price-only delta interval has "
        << average(aggregate.delta_interval_cardinality_sum,
                   aggregate.finite_delta_intervals)
        << " candidate integer deltas on average; exact inventory selects one.\n"
        << "- " << 100.0 * ratio(aggregate.same_price_inventory_changes,
                                  aggregate.exact_inventory_changes)
        << "% of exact inventory-change product-events leave rounded price unchanged.\n"
        << "- " << 100.0 * ratio(aggregate.floor_unidentifiable,
                                  aggregate.valid_delta_intervals)
        << "% of price-only delta intervals are unbounded because at least one endpoint"
           " is at the $1 floor; exact inventory remains finite there.\n\n";

    out << "Aggregate by phase (each observation is one product-transition):\n"
        << "phase   states       floor%   bucket_span  delta_MAE  delta_span  "
           "same_price_changed  floor_delta\n";
    for (std::size_t phase = 0; phase < kPhases; ++phase) {
        const auto m = product_sum(metrics, phase);
        out << std::left << std::setw(8) << kPhaseNames[phase] << std::right
            << std::setw(12) << m.states
            << std::setw(10) << 100.0 * ratio(m.floor_states, m.valid_price_states)
            << std::setw(14) << average(m.bucket_span_sum, m.finite_bucket_states)
            << std::setw(11) << average(
                   m.delta_midpoint_absolute_error_sum, m.finite_delta_intervals)
            << std::setw(12) << average(m.delta_interval_span_sum,
                                         m.finite_delta_intervals)
            << std::setw(20) << m.same_price_inventory_changes
            << std::setw(13) << m.floor_unidentifiable << '\n';
    }
    out << '\n';

    out << "phase product       states  floor%  bucket_span  delta_MAE  delta_span  cover% "
           "same_price_changed floor_delta sell_req\n";
    for (std::size_t phase = 0; phase < kPhases; ++phase) {
        for (std::size_t product = 0; product < kProducts; ++product) {
            const auto& m = metrics[phase][product];
            out << std::left << std::setw(7) << kPhaseNames[phase]
                << std::setw(14) << kProductNames[product] << std::right
                << ' ' << std::setw(8) << m.states
                << ' ' << std::setw(8) << 100.0 * ratio(m.floor_states, m.valid_price_states)
                << ' ' << std::setw(12) << average(m.bucket_span_sum, m.finite_bucket_states)
                << ' ' << std::setw(10) << average(
                       m.delta_midpoint_absolute_error_sum, m.finite_delta_intervals)
                << ' ' << std::setw(12) << average(m.delta_interval_span_sum,
                                             m.finite_delta_intervals)
                << ' ' << std::setw(8) << 100.0 * ratio(m.delta_covered, m.valid_delta_intervals)
                << ' ' << std::setw(19) << m.same_price_inventory_changes
                << ' ' << std::setw(12) << m.floor_unidentifiable
                << ' ' << std::setw(10) << m.opponent_sell_request_events << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        const auto started = std::chrono::steady_clock::now();
        const MappedFile file(options.input);
        const auto corpus = parse_corpus(file);
        const auto buckets = build_bucket_cache(corpus);

        const auto worker_count = std::min(options.threads, corpus.replays.size());
        std::vector<MetricGrid> local(worker_count);
        std::atomic<std::size_t> next{0};
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            workers.emplace_back([&, worker] {
                for (;;) {
                    const auto index = next.fetch_add(1, std::memory_order_relaxed);
                    if (index >= corpus.replays.size()) break;
                    process_replay(corpus.replays[index], buckets, options, local[worker]);
                }
            });
        }
        for (auto& worker : workers) worker.join();

        MetricGrid totals{};
        for (const auto& grid : local) {
            for (std::size_t phase = 0; phase < kPhases; ++phase) {
                for (std::size_t product = 0; product < kProducts; ++product) {
                    totals[phase][product] += grid[phase][product];
                }
            }
        }
        const auto seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started
        ).count();
        const double input_mib = static_cast<double>(file.size()) / (1024.0 * 1024.0);
        write_csv(options.csv, totals);
        write_text(options.text, corpus, totals, options, seconds, input_mib);

        const auto all = product_sum(totals, 0);
        std::cout << std::fixed << std::setprecision(4)
                  << "replays=" << corpus.header.replay_count
                  << " records=" << corpus.header.record_count
                  << " threads=" << worker_count
                  << " seconds=" << seconds
                  << " throughput_mib_s=" << input_mib / seconds
                  << " finite_delta_midpoint_mae="
                  << average(all.delta_midpoint_absolute_error_sum,
                             all.finite_delta_intervals)
                  << " finite_delta_span="
                  << average(all.delta_interval_span_sum, all.finite_delta_intervals)
                  << " interval_coverage="
                  << ratio(all.delta_covered, all.valid_delta_intervals)
                  << " same_price_inventory_changes=" << all.same_price_inventory_changes
                  << " floor_unidentifiable=" << all.floor_unidentifiable << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
