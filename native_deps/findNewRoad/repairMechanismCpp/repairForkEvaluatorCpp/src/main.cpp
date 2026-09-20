#include "repair_fork_evaluator.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace {

template <class T> T number(std::string_view text) {
  T value{};
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
    throw std::invalid_argument("invalid number: " + std::string(text));
  return value;
}

struct Cli {
  g001::repair_fork::EvaluatorOptions eval;
  std::string output;
};

Cli parse(int argc, char** argv) {
  Cli out;
  out.eval.threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  out.eval.panels.clear();
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    auto next = [&]() -> std::string_view {
      if (++i >= argc) throw std::invalid_argument("missing argument value");
      return argv[i];
    };
    if (arg == "--tapes") out.eval.tapes = next();
    else if (arg == "--library") out.eval.library = next();
    else if (arg == "--refs") out.eval.references = next();
    else if (arg == "--opponent") out.eval.opponent = next();
    else if (arg == "--seed-begin") out.eval.seed_begin = number<std::uint64_t>(next());
    else if (arg == "--seeds") out.eval.seeds = number<int>(next());
    else if (arg == "--threads") out.eval.threads = number<int>(next());
    else if (arg == "--repair-mask")
      out.eval.candidate_repair_mask = number<int>(next());
    else if (arg == "--output") out.output = next();
    else if (arg == "--panel") {
      const auto value = next();
      if (value == "normal") out.eval.panels.push_back(g001::repair_fork::Panel::Normal);
      else if (value == "forced-weed")
        out.eval.panels.push_back(g001::repair_fork::Panel::ForcedWeed);
      else throw std::invalid_argument("panel must be normal or forced-weed");
    } else throw std::invalid_argument("unknown argument: " + std::string(arg));
  }
  if (out.eval.panels.empty()) out.eval.panels.push_back(g001::repair_fork::Panel::Normal);
  if (out.eval.tapes.empty() || out.eval.library.empty() ||
      out.eval.references.empty() || out.output.empty())
    throw std::invalid_argument("--tapes --library --refs --output are required");
  return out;
}

}  // namespace

int main(int argc, char** argv) try {
  const auto cli = parse(argc, argv);
  const auto report = g001::repair_fork::evaluate(
      cli.eval, [] { return std::make_unique<g001::repair_fork::PassThroughOwner>(); });
  std::filesystem::create_directories(cli.output);
  const auto root = std::filesystem::path(cli.output);
  std::ofstream(root / "cases.jsonl") << report.cases_jsonl;
  std::ofstream(root / "summary.json") << report.summary_json;
  std::ofstream(root / "report.sha256") << report.deterministic_payload_sha256
                                         << "  summary.json+cases.jsonl\n";
  std::cout << "games=" << report.games.size()
            << " sha256=" << report.deterministic_payload_sha256
            << " output=" << cli.output << '\n';
  return 0;
} catch (const std::exception& error) {
  std::cerr << "error: " << error.what() << '\n';
  return 2;
}
