#include "tape_runner.hpp"

#include <charconv>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

template <class Integer>
Integer number(std::string_view value) {
    Integer result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid integer: " + std::string(value));
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        g001::market::tape::RunnerOptions options;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument = argv[index];
            auto next = [&]() -> std::string_view {
                if (++index >= argc) throw std::runtime_error("missing option value");
                return argv[index];
            };
            if (argument == "--tapes") options.compressed_tapes_path = next();
            else if (argument == "--library") options.route_library_path = next();
            else if (argument == "--route") options.route = next();
            else if (argument == "--opponent-route") options.opponent_route = next();
            else if (argument == "--seed-begin") options.seed_begin = number<std::uint64_t>(next());
            else if (argument == "--seeds") options.seed_count = number<std::uint64_t>(next());
            else if (argument == "--threads") options.threads = number<std::size_t>(next());
            else if (argument == "--mechanism-mask") {
                options.mechanisms = number<g001::market::MechanismMask>(next());
            } else if (argument == "--help") {
                std::cout << "tape_market_experiment --tapes FILE --library FILE "
                             "[--route G001|ID] [--opponent-route G001|ID] "
                             "[--seed-begin N] [--seeds 192] [--threads 192] "
                             "[--mechanism-mask N]\n";
                return 0;
            } else throw std::runtime_error("unknown argument: " + std::string(argument));
        }
        if (options.compressed_tapes_path.empty() || options.route_library_path.empty()) {
            throw std::runtime_error("--tapes and --library are required");
        }
        const auto summary = g001::market::tape::run_paired(options);
        std::cout << g001::market::tape::summary_json(summary, options);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "tape_market_experiment: " << error.what() << '\n';
        return 2;
    }
}
