#include "actor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::uint32_t u32(const unsigned char* p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}

std::uint64_t u64(const unsigned char* p) {
  return std::uint64_t(u32(p)) | (std::uint64_t(u32(p + 4)) << 32);
}

struct Reader {
  const unsigned char* current;
  const unsigned char* end;

  std::vector<float> floats(std::size_t count) {
    if (std::size_t(end - current) < count * sizeof(float))
      throw std::runtime_error("fixture float array is truncated");
    std::vector<float> result(count);
    std::memcpy(result.data(), current, count * sizeof(float));
    current += count * sizeof(float);
    return result;
  }

  std::vector<std::uint32_t> integers(std::size_t count) {
    if (std::size_t(end - current) < count * sizeof(std::uint32_t))
      throw std::runtime_error("fixture integer array is truncated");
    std::vector<std::uint32_t> result(count);
    for (std::size_t i = 0; i < count; ++i) result[i] = u32(current + 4 * i);
    current += count * sizeof(std::uint32_t);
    return result;
  }
};

struct Fixture {
  std::uint32_t events, token_capacity, token_continuous, categories;
  std::uint32_t context_width, observation_width, resource_width;
  std::uint32_t hidden_width, classes;
  std::uint64_t rng_seed;
  std::vector<float> context, observation, token_values, expected_initial;
  float observation_length;
  std::vector<std::uint32_t> token_categories;
  std::uint32_t token_count;
  std::vector<float> hidden, resources, logits, next_hidden, log_probability;
  std::vector<float> entropy;
  std::vector<std::uint32_t> cells, stages, previous, legal_masks, actions;
  std::vector<std::uint32_t> greedy, counter_actions;
};

Fixture load_fixture(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open fixture: " + path);
  const auto length = stream.tellg();
  if (length < 64) throw std::runtime_error("fixture is truncated");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), length);
  if (!stream) throw std::runtime_error("cannot read fixture");
  const auto* h = bytes.data();
  if (std::memcmp(h, "KSV3FX1\0", 8) || u32(h + 8) != 1 || u32(h + 12) != 64 ||
      u32(h + 52) != 1)
    throw std::runtime_error("fixture contract mismatch");
  Fixture f{};
  f.events = u32(h + 16); f.token_capacity = u32(h + 20);
  f.token_continuous = u32(h + 24); f.categories = u32(h + 28);
  f.context_width = u32(h + 32); f.observation_width = u32(h + 36);
  f.resource_width = u32(h + 40); f.hidden_width = u32(h + 44);
  f.classes = u32(h + 48); f.rng_seed = u64(h + 56);
  Reader reader{h + 64, bytes.data() + bytes.size()};
  f.context = reader.floats(f.context_width);
  f.observation = reader.floats(f.observation_width);
  f.observation_length = reader.floats(1)[0];
  f.token_values = reader.floats(
      std::size_t(f.token_capacity) * f.token_continuous);
  f.token_categories = reader.integers(
      std::size_t(f.categories) * f.token_capacity);
  f.token_count = reader.integers(1)[0];
  f.expected_initial = reader.floats(f.hidden_width);
  f.hidden = reader.floats(std::size_t(f.events) * f.hidden_width);
  f.resources = reader.floats(std::size_t(f.events) * f.resource_width);
  f.cells = reader.integers(f.events); f.stages = reader.integers(f.events);
  f.previous = reader.integers(f.events);
  f.legal_masks = reader.integers(f.events); f.actions = reader.integers(f.events);
  f.greedy = reader.integers(f.events);
  f.logits = reader.floats(std::size_t(f.events) * f.classes);
  f.next_hidden = reader.floats(std::size_t(f.events) * f.hidden_width);
  f.log_probability = reader.floats(f.events); f.entropy = reader.floats(f.events);
  f.counter_actions = reader.integers(f.events);
  if (reader.current != reader.end) throw std::runtime_error("fixture has trailing bytes");
  return f;
}

float maximum_error(const float* actual, const float* expected,
                    std::size_t count) {
  float result = 0.0f;
  for (std::size_t i = 0; i < count; ++i)
    result = std::max(result, std::abs(actual[i] - expected[i]));
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: " << argv[0]
                << " WEIGHTS.bin FIXTURE.bin [iterations]\n";
      return 2;
    }
    const std::uint64_t iterations = argc == 4 ? std::stoull(argv[3]) : 10000;
    if (!iterations) throw std::invalid_argument("iterations must be positive");
    const auto actor = student_v3::Actor::load(argv[1]);
    const auto fixture = load_fixture(argv[2]);
    const auto& d = actor.dimensions();
    if (fixture.token_capacity != d.token_capacity ||
        fixture.token_continuous != d.token_continuous ||
        fixture.categories != d.categories || fixture.context_width != d.context ||
        fixture.observation_width != d.observation ||
        fixture.resource_width != d.resources || fixture.hidden_width != d.hidden ||
        fixture.classes != d.classes)
      throw std::runtime_error("fixture and actor dimensions differ");

    std::vector<float> initial(d.hidden), logits(d.classes), hidden(d.hidden);
    actor.initial_hidden_normalized(
        fixture.context.data(), fixture.observation.data(),
        fixture.observation_length, fixture.token_values.data(),
        fixture.token_categories.data(), fixture.token_count, initial.data());
    const float initial_error = maximum_error(
        initial.data(), fixture.expected_initial.data(), d.hidden);
    float logits_error = 0.0f, hidden_error = 0.0f;
    float log_probability_error = 0.0f, entropy_error = 0.0f;
    std::uint64_t greedy_mismatches = 0, counter_mismatches = 0;
    for (std::uint32_t row = 0; row < fixture.events; ++row) {
      actor.step_normalized(
          fixture.hidden.data() + std::size_t(row) * d.hidden,
          fixture.resources.data() + std::size_t(row) * d.resources,
          fixture.cells[row], fixture.stages[row], fixture.previous[row],
          fixture.legal_masks[row], logits.data(), hidden.data());
      logits_error = std::max(logits_error, maximum_error(
          logits.data(), fixture.logits.data() + std::size_t(row) * d.classes,
          d.classes));
      hidden_error = std::max(hidden_error, maximum_error(
          hidden.data(), fixture.next_hidden.data() + std::size_t(row) * d.hidden,
          d.hidden));
      const auto greedy = actor.greedy(logits.data(), fixture.legal_masks[row]);
      const auto evaluated = actor.evaluate(
          logits.data(), fixture.legal_masks[row], fixture.actions[row]);
      const auto sampled = actor.sample(logits.data(), fixture.legal_masks[row],
                                        fixture.rng_seed, row);
      greedy_mismatches += greedy.action != static_cast<int>(fixture.greedy[row]);
      counter_mismatches +=
          sampled.action != static_cast<int>(fixture.counter_actions[row]);
      log_probability_error = std::max(
          log_probability_error,
          std::abs(evaluated.log_probability - fixture.log_probability[row]));
      entropy_error = std::max(
          entropy_error, std::abs(evaluated.entropy - fixture.entropy[row]));
    }

    volatile float sink = 0.0f;
    for (int warmup = 0; warmup < 32; ++warmup) {
      const auto row = std::uint32_t(warmup) % fixture.events;
      actor.step_normalized(
          fixture.hidden.data() + std::size_t(row) * d.hidden,
          fixture.resources.data() + std::size_t(row) * d.resources,
          fixture.cells[row], fixture.stages[row], fixture.previous[row],
          fixture.legal_masks[row], logits.data(), hidden.data());
      sink += hidden[0];
    }
    const auto step_started = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < iterations; ++index) {
      const auto row = std::uint32_t(index % fixture.events);
      actor.step_normalized(
          fixture.hidden.data() + std::size_t(row) * d.hidden,
          fixture.resources.data() + std::size_t(row) * d.resources,
          fixture.cells[row], fixture.stages[row], fixture.previous[row],
          fixture.legal_masks[row], logits.data(), hidden.data());
      sink += hidden[0];
    }
    const double step_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - step_started).count();
    const std::uint64_t initial_iterations = std::max<std::uint64_t>(1, iterations / 10);
    const auto initial_started = std::chrono::steady_clock::now();
    for (std::uint64_t index = 0; index < initial_iterations; ++index) {
      actor.initial_hidden_normalized(
          fixture.context.data(), fixture.observation.data(),
          fixture.observation_length, fixture.token_values.data(),
          fixture.token_categories.data(), fixture.token_count, initial.data());
      sink += initial[0];
    }
    const double initial_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initial_started).count();
    const float tolerance = 5.0e-5f;
    const bool pass = initial_error <= tolerance && logits_error <= tolerance &&
                      hidden_error <= tolerance &&
                      log_probability_error <= 2.0e-5f &&
                      entropy_error <= 2.0e-5f && !greedy_mismatches &&
                      !counter_mismatches && std::isfinite(sink);
    std::cout << std::setprecision(9) << "{\"status\":\""
              << (pass ? "PASS" : "FAIL") << "\",\"events\":"
              << fixture.events << ",\"initial_hidden_max_abs\":"
              << initial_error << ",\"logits_max_abs\":" << logits_error
              << ",\"next_hidden_max_abs\":" << hidden_error
              << ",\"logprob_max_abs\":" << log_probability_error
              << ",\"entropy_max_abs\":" << entropy_error
              << ",\"greedy_mismatches\":" << greedy_mismatches
              << ",\"counter_rng_mismatches\":" << counter_mismatches
              << ",\"iterations\":" << iterations
              << ",\"microseconds_per_event\":"
              << step_seconds * 1e6 / iterations
              << ",\"events_per_second\":" << iterations / step_seconds
              << ",\"microseconds_per_initial\":"
              << initial_seconds * 1e6 / initial_iterations << "}\n";
    return pass ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "native v3 parity failed: " << error.what() << '\n';
    return 1;
  }
}
