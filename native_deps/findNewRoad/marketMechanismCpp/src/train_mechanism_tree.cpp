#include "mechanism_tree.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Options {
    std::string mra;
    std::string labels;
    std::string output;
    g001::tree::TrainConfig train;
};

void usage(const char* program) {
    std::cerr
        << "Usage: " << program << " --mra DATA.mra --labels VALUES.mcf --output REPORT.json\n"
        << "  [--depth 3..5] [--min-leaf N] [--max-thresholds N]\n"
        << "  [--min-gain X] [--ccp-alpha X] [--holdout X] [--bootstrap N] [--seed N] [--threads N]\n";
}

Options options(int argc, char** argv) {
    Options out;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::runtime_error("missing value for " + std::string(arg));
            return argv[i];
        };
        if (arg == "--mra") out.mra = value();
        else if (arg == "--labels") out.labels = value();
        else if (arg == "--output") out.output = value();
        else if (arg == "--depth") out.train.max_depth = std::stoi(value());
        else if (arg == "--min-leaf") out.train.min_leaf = std::stoull(value());
        else if (arg == "--max-thresholds") out.train.max_thresholds = std::stoull(value());
        else if (arg == "--min-gain") out.train.min_gain_per_sample = std::stod(value());
        else if (arg == "--ccp-alpha") out.train.ccp_alpha = std::stod(value());
        else if (arg == "--holdout") out.train.holdout_fraction = std::stod(value());
        else if (arg == "--bootstrap") out.train.bootstrap_rounds = std::stoull(value());
        else if (arg == "--seed") out.train.seed = std::stoull(value());
        else if (arg == "--threads") out.train.threads = std::stoull(value());
        else if (arg == "--help" || arg == "-h") { usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + std::string(arg));
    }
    if (out.mra.empty() || out.labels.empty() || out.output.empty())
        throw std::runtime_error("--mra, --labels and --output are required");
    if (out.train.max_depth < 3 || out.train.max_depth > 5)
        throw std::runtime_error("production tree depth must be 3..5");
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto config = options(argc, argv);
        auto examples = g001::tree::load_joined_examples(config.mra, config.labels);
        auto report = g001::tree::train_grouped_tree(examples, config.train);
        const auto parent = std::filesystem::path(config.output).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        g001::tree::write_json_report(config.output, report, config.train, examples.size());
        std::cout << "examples=" << examples.size()
                  << " train=" << report.train_samples
                  << " holdout=" << report.holdout_samples
                  << " components=" << report.split_components
                  << " nodes=" << report.model.nodes.size()
                  << " holdout_gain=" << report.holdout.gain_over_baseline
                  << " bootstrap_gain95=[" << report.bootstrap_holdout_gain_q025
                  << ',' << report.bootstrap_holdout_gain_q975 << ']'
                  << " leaf_agreement=" << report.bootstrap_leaf_action_agreement
                  << " report=" << config.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
