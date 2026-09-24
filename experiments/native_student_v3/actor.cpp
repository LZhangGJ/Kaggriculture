#include "actor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#if defined(__aarch64__) && defined(STUDENT_NEON_LINEAR) && !defined(STUDENT_SCALAR_LINEAR)
#include <arm_neon.h>
#endif

namespace student_v3 {
namespace {

constexpr std::size_t kHeaderBytes = 236;
constexpr std::array<unsigned char, 32> kContractSha = {
    0xb1, 0x7a, 0x24, 0x83, 0x4c, 0x02, 0xa6, 0x1b,
    0x12, 0xb3, 0x15, 0x74, 0xcd, 0x62, 0x41, 0xc7,
    0x1e, 0xaf, 0x94, 0xbc, 0xac, 0x27, 0x7e, 0x88,
    0x7d, 0x54, 0x66, 0x7e, 0x20, 0x80, 0x28, 0xb2};
constexpr std::array<unsigned char, 32> kShopContractSha = {
    0xea, 0xc5, 0xeb, 0x1b, 0x81, 0xda, 0x09, 0x3a,
    0x82, 0xb5, 0x46, 0x9c, 0x95, 0x31, 0x29, 0xe2,
    0xd2, 0x5c, 0x6b, 0x92, 0x1e, 0x1c, 0xe6, 0x74,
    0x8e, 0x60, 0xaa, 0x51, 0x62, 0xdb, 0x5a, 0xb2};
constexpr std::array<std::uint32_t, 7> kCategorySizes = {7, 32, 32, 32,
                                                         64, 64, 4};

std::uint32_t u32(const unsigned char* p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}

std::uint64_t u64(const unsigned char* p) {
  return std::uint64_t(u32(p)) | (std::uint64_t(u32(p + 4)) << 32);
}

std::uint32_t rotate_right(std::uint32_t value, unsigned bits) {
  return (value >> bits) | (value << (32 - bits));
}

std::array<unsigned char, 32> sha256(const unsigned char* data,
                                     std::size_t size) {
  static constexpr std::uint32_t constants[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b,
      0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
      0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7,
      0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
      0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152,
      0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
      0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
      0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
      0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
      0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::array<std::uint32_t, 8> state = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const std::size_t padded = ((size + 9 + 63) / 64) * 64;
  std::vector<unsigned char> message(padded, 0);
  std::memcpy(message.data(), data, size);
  message[size] = 0x80;
  const std::uint64_t bits = std::uint64_t(size) * 8;
  for (int i = 0; i < 8; ++i)
    message[padded - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
  for (std::size_t block = 0; block < padded; block += 64) {
    std::uint32_t words[64];
    for (int i = 0; i < 16; ++i) {
      const auto* p = message.data() + block + 4 * i;
      words[i] = (std::uint32_t(p[0]) << 24) |
                 (std::uint32_t(p[1]) << 16) |
                 (std::uint32_t(p[2]) << 8) | p[3];
    }
    for (int i = 16; i < 64; ++i) {
      const auto a = rotate_right(words[i - 15], 7) ^
                     rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
      const auto b = rotate_right(words[i - 2], 17) ^
                     rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
      words[i] = words[i - 16] + a + words[i - 7] + b;
    }
    auto a = state[0], b = state[1], c = state[2], d = state[3];
    auto e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const auto s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                      rotate_right(e, 25);
      const auto choice = (e & f) ^ (~e & g);
      const auto t1 = h + s1 + choice + constants[i] + words[i];
      const auto s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                      rotate_right(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto t2 = s0 + majority;
      h = g; g = f; f = e; e = d + t1;
      d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
  }
  std::array<unsigned char, 32> digest{};
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 4; ++j)
      digest[4 * i + j] = static_cast<unsigned char>(state[i] >> (24 - 8 * j));
  return digest;
}

void linear(const float* weight, const float* bias, const float* values,
            std::size_t inputs, std::size_t outputs,
            float* result) {
  for (std::size_t output = 0; output < outputs; ++output) {
    const float* row = weight + output * inputs;
#if defined(__aarch64__) && defined(STUDENT_NEON_LINEAR) && !defined(STUDENT_SCALAR_LINEAR)
    float32x4_t sum = vdupq_n_f32(0.0f);
    std::size_t input = 0;
    for (; input + 4 <= inputs; input += 4)
      sum = vfmaq_f32(sum, vld1q_f32(row + input),
                      vld1q_f32(values + input));
    float value = bias[output] + vaddvq_f32(sum);
    for (; input < inputs; ++input)
      value += row[input] * values[input];
#else
    float value = bias[output];
    for (std::size_t input = 0; input < inputs; ++input)
      value += row[input] * values[input];
#endif
    result[output] = value;
  }
}

float sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
  const float exponential = std::exp(value);
  return exponential / (1.0f + exponential);
}

std::uint64_t counter_word(std::uint64_t seed, std::uint64_t counter) {
  std::uint64_t value = seed + counter * 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

float counter_uniform(std::uint64_t seed, std::uint64_t counter) {
  const std::uint32_t top = static_cast<std::uint32_t>(
      counter_word(seed, counter) >> 40);
  return (static_cast<float>(top) + 0.5f) * (1.0f / 16777216.0f);
}

float policy_uniform(std::uint64_t seed, std::uint64_t event_index) {
  std::uint64_t value = seed +
      (event_index + 1) * 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  value ^= value >> 31;
  return static_cast<float>(
      (static_cast<double>(value >> 11) + 0.5) /
      static_cast<double>(std::uint64_t{1} << 53));
}

Distribution categorical(const Dimensions& dimensions, const float* logits,
                         std::uint32_t legal_mask, float temperature,
                         bool choose_greedy, float target) {
  if (!std::isfinite(temperature) || temperature <= 0.0f || !legal_mask ||
      (legal_mask >> dimensions.classes))
    throw std::invalid_argument("invalid categorical distribution");
  std::vector<float> scaled(dimensions.classes,
                            -std::numeric_limits<float>::infinity());
  float maximum = -std::numeric_limits<float>::infinity();
  int action = -1;
  for (std::uint32_t i = 0; i < dimensions.classes; ++i) {
    if (!(legal_mask & (1u << i))) continue;
    if (!std::isfinite(logits[i]))
      throw std::invalid_argument("non-finite legal logit");
    scaled[i] = logits[i] / temperature;
    if (action < 0 || scaled[i] > maximum) {
      maximum = scaled[i];
      action = static_cast<int>(i);
    }
  }
  std::vector<float> probabilities(dimensions.classes, 0.0f);
  float total = 0.0f;
  for (std::uint32_t i = 0; i < dimensions.classes; ++i) {
    if (!(legal_mask & (1u << i))) continue;
    probabilities[i] = std::exp(scaled[i] - maximum);
    total += probabilities[i];
  }
  if (!std::isfinite(total) || total <= 0.0f)
    throw std::invalid_argument("invalid categorical mass");
  for (float& probability : probabilities) probability /= total;
  if (!choose_greedy) {
    float cumulative = 0.0f;
    int last = action;
    for (std::uint32_t i = 0; i < dimensions.classes; ++i) {
      if (!(legal_mask & (1u << i))) continue;
      last = static_cast<int>(i);
      cumulative += probabilities[i];
      if (target < cumulative) { action = static_cast<int>(i); break; }
    }
    if (target >= cumulative) action = last;
  }
  const float log_partition = maximum + std::log(total);
  float entropy = 0.0f;
  for (std::uint32_t i = 0; i < dimensions.classes; ++i)
    if (legal_mask & (1u << i))
      entropy += probabilities[i] * (log_partition - scaled[i]);
  return {action, scaled[action] - log_partition, entropy};
}

}  // namespace

Actor Actor::load(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open actor weights: " + path);
  const auto length = stream.tellg();
  if (length < static_cast<std::streamoff>(kHeaderBytes))
    throw std::runtime_error("actor weight file is truncated");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(length));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), length);
  if (!stream) throw std::runtime_error("cannot read actor weights");
  const auto* header = bytes.data();
  const auto version = u32(header + 8);
  const bool shop_action_head = version == 2;
  if (std::memcmp(header, "KAGSV3A\0", 8) ||
      (version != 1 && version != 2) ||
      u32(header + 12) != kHeaderBytes || u32(header + 16) != 0x01020304 ||
      u32(header + 20) != 1 || u32(header + 24) != 8 ||
      u32(header + 28) != (shop_action_head ? 30u : 28u) ||
      (shop_action_head
        ? !std::equal(kShopContractSha.begin(), kShopContractSha.end(), header + 172)
        : !std::equal(kContractSha.begin(), kContractSha.end(), header + 172)))
    throw std::runtime_error("actor weight contract mismatch");
  const auto payload_bytes = u64(header + 100);
  if (payload_bytes % sizeof(float) ||
      payload_bytes != bytes.size() - kHeaderBytes)
    throw std::runtime_error("actor weight payload length mismatch");
  const auto payload_digest = sha256(bytes.data() + kHeaderBytes, payload_bytes);
  if (!std::equal(payload_digest.begin(), payload_digest.end(), header + 204))
    throw std::runtime_error("actor weight payload digest mismatch");

  Actor actor;
  actor.shop_action_head_ = shop_action_head;
  auto& d = actor.dimensions_;
  d.classes = u32(header + 32); d.token_capacity = u32(header + 36);
  d.token_continuous = u32(header + 40); d.categories = u32(header + 44);
  d.context = u32(header + 48); d.observation = u32(header + 52);
  d.resources = u32(header + 56); d.projection = u32(header + 64);
  d.scalar = u32(header + 68); d.embedding = u32(header + 72);
  d.hidden = u32(header + 76); d.resource_hidden = u32(header + 80);
  d.event_embedding = u32(header + 84); d.cells = u32(header + 88);
  d.stages = u32(header + 92); d.previous = u32(header + 96);
  const auto scale = u32(header + 60);
  if (d.classes != 11 || d.token_capacity != 320 ||
      d.token_continuous != 24 || d.categories != 7 || scale < 1 || scale > 4 ||
      d.projection != 16 * scale || d.scalar != 4 * scale ||
      d.embedding != 4 * scale || d.hidden != 64 * scale ||
      d.resource_hidden != 64 * scale || d.event_embedding != 8 * scale ||
      d.cells != 100 || d.stages != 2 || d.previous != d.classes + 1)
    throw std::runtime_error("unsupported actor dimensions");

  actor.payload_.resize(payload_bytes / sizeof(float));
  std::memcpy(actor.payload_.data(), bytes.data() + kHeaderBytes, payload_bytes);
  std::size_t offset = 0;
  auto take = [&](std::size_t size) {
    if (size > actor.payload_.size() - offset)
      throw std::runtime_error("actor tensor exceeds payload");
    Tensor result{actor.payload_.data() + offset, size};
    offset += size;
    return result;
  };
  actor.context_mean_ = take(d.context); actor.context_std_ = take(d.context);
  actor.observation_mean_ = take(d.observation);
  actor.observation_std_ = take(d.observation);
  actor.observation_length_mean_ = take(1);
  actor.observation_length_std_ = take(1);
  actor.resource_mean_ = take(d.resources); actor.resource_std_ = take(d.resources);
  actor.context_weight_ = take(std::size_t(d.projection) * d.context);
  actor.context_bias_ = take(d.projection);
  actor.observation_weight_ = take(std::size_t(d.projection) * d.observation);
  actor.observation_bias_ = take(d.projection);
  actor.observation_length_weight_ = take(d.scalar);
  actor.observation_length_bias_ = take(d.scalar);
  for (auto size : kCategorySizes)
    actor.token_embeddings_.push_back(take(std::size_t(size) * d.embedding));
  actor.token_weight_ = take(std::size_t(d.projection) *
                             (d.token_continuous + d.categories * d.embedding));
  actor.token_bias_ = take(d.projection);
  actor.begin_weight_ = take(std::size_t(d.hidden) *
                             (3 * d.projection + d.scalar));
  actor.begin_bias_ = take(d.hidden);
  actor.resource_weight_ = take(std::size_t(d.resource_hidden) * d.resources);
  actor.resource_bias_ = take(d.resource_hidden);
  actor.cell_embedding_ = take(std::size_t(d.cells) * d.event_embedding);
  actor.stage_embedding_ = take(std::size_t(d.stages) * d.event_embedding);
  actor.previous_embedding_ = take(std::size_t(d.previous) * d.event_embedding);
  const auto gru_input = d.resource_hidden + 3 * d.event_embedding;
  actor.gru_weight_ih_ = take(std::size_t(3 * d.hidden) * gru_input);
  actor.gru_weight_hh_ = take(std::size_t(3 * d.hidden) * d.hidden);
  actor.gru_bias_ih_ = take(3 * d.hidden); actor.gru_bias_hh_ = take(3 * d.hidden);
  actor.head_weight_ = take(std::size_t(d.classes) * d.hidden);
  actor.head_bias_ = take(d.classes);
  if (shop_action_head) {
    if (d.resources != 383)
      throw std::runtime_error("shop action head requires 383 resources");
    actor.shop_gate_weight_ = take(std::size_t(8) * d.hidden);
    actor.shop_gate_bias_ = take(8);
  }
  if (offset != actor.payload_.size())
    throw std::runtime_error("actor payload has trailing floats");
  for (float value : actor.payload_)
    if (!std::isfinite(value)) throw std::runtime_error("non-finite actor weight");
  for (const Tensor* standard_deviation :
       {&actor.context_std_, &actor.observation_std_,
        &actor.observation_length_std_, &actor.resource_std_})
    for (std::size_t i = 0; i < standard_deviation->size; ++i)
      if (standard_deviation->data[i] <= 0.0f)
        throw std::runtime_error("non-positive normalization std");
  return actor;
}

void Actor::initial_hidden_normalized(
    const float* context, const float* observation, float observation_length,
    const float* token_continuous, const std::uint32_t* categories,
    std::uint32_t token_count, float* hidden) const {
  const auto& d = dimensions_;
  if (token_count > d.token_capacity) throw std::invalid_argument("token_count");
  std::vector<float> context_state(d.projection), observation_state(d.projection);
  std::vector<float> length_state(d.scalar), token_state(
      d.token_continuous + d.categories * d.embedding, 0.0f);
  std::vector<float> token_projection(d.projection);
  linear(context_weight_.data, context_bias_.data, context, d.context, d.projection,
         context_state.data());
  linear(observation_weight_.data, observation_bias_.data, observation, d.observation,
         d.projection, observation_state.data());
  linear(observation_length_weight_.data, observation_length_bias_.data,
         &observation_length, 1, d.scalar, length_state.data());
  const float denominator = static_cast<float>(std::max(1u, token_count));
  for (std::uint32_t token = 0; token < token_count; ++token)
    for (std::uint32_t column = 0; column < d.token_continuous; ++column)
      token_state[column] +=
          token_continuous[std::size_t(token) * d.token_continuous + column] /
          denominator;
  for (std::uint32_t category = 0; category < d.categories; ++category) {
    const auto table_size = kCategorySizes[category];
    for (std::uint32_t token = 0; token < token_count; ++token) {
      const auto value = categories[std::size_t(category) * d.token_capacity + token];
      if (value >= table_size) throw std::invalid_argument("token category");
      const float* embedding = token_embeddings_[category].data +
                               std::size_t(value) * d.embedding;
      float* output = token_state.data() + d.token_continuous +
                      std::size_t(category) * d.embedding;
      for (std::uint32_t column = 0; column < d.embedding; ++column)
        output[column] += embedding[column] / denominator;
    }
  }
  linear(token_weight_.data, token_bias_.data, token_state.data(), token_state.size(),
         d.projection, token_projection.data());
  for (auto* vector : {&context_state, &observation_state, &length_state,
                       &token_projection})
    for (float& value : *vector) value = std::max(0.0f, value);
  std::vector<float> state;
  state.reserve(3 * d.projection + d.scalar);
  state.insert(state.end(), context_state.begin(), context_state.end());
  state.insert(state.end(), observation_state.begin(), observation_state.end());
  state.insert(state.end(), length_state.begin(), length_state.end());
  state.insert(state.end(), token_projection.begin(), token_projection.end());
  linear(begin_weight_.data, begin_bias_.data, state.data(), state.size(), d.hidden,
         hidden);
  for (std::uint32_t i = 0; i < d.hidden; ++i) hidden[i] = std::tanh(hidden[i]);
}

void Actor::initial_hidden(const float* context, const float* observation,
                           float observation_length,
                           const float* token_continuous,
                           const std::uint32_t* categories,
                           std::uint32_t token_count, float* hidden) const {
  std::vector<float> normalized_context(dimensions_.context);
  std::vector<float> normalized_observation(dimensions_.observation);
  for (std::uint32_t i = 0; i < dimensions_.context; ++i)
    normalized_context[i] = (context[i] - context_mean_.data[i]) / context_std_.data[i];
  for (std::uint32_t i = 0; i < dimensions_.observation; ++i)
    normalized_observation[i] =
        (observation[i] - observation_mean_.data[i]) / observation_std_.data[i];
  const float normalized_length =
      (observation_length - observation_length_mean_.data[0]) /
      observation_length_std_.data[0];
  initial_hidden_normalized(normalized_context.data(), normalized_observation.data(),
                            normalized_length, token_continuous, categories,
                            token_count, hidden);
}

void Actor::normalize_context(const float* input, float* output) const {
  for (std::uint32_t i = 0; i < dimensions_.context; ++i)
    output[i] = (input[i] - context_mean_.data[i]) / context_std_.data[i];
}

void Actor::normalize_observation(const float* input, float* output) const {
  for (std::uint32_t i = 0; i < dimensions_.observation; ++i)
    output[i] =
        (input[i] - observation_mean_.data[i]) / observation_std_.data[i];
}

float Actor::normalize_observation_length(float input) const {
  return (input - observation_length_mean_.data[0]) /
         observation_length_std_.data[0];
}

void Actor::normalize_resources(const float* input, float* output) const {
  for (std::uint32_t i = 0; i < dimensions_.resources; ++i)
    output[i] = (input[i] - resource_mean_.data[i]) / resource_std_.data[i];
}

void Actor::step_normalized(const float* hidden, const float* resources,
                            std::uint32_t cell, std::uint32_t stage,
                            std::uint32_t previous, std::uint32_t legal_mask,
                            float* logits, float* next_hidden) const {
  const auto& d = dimensions_;
  if (cell >= d.cells || stage >= d.stages || previous >= d.previous ||
      !legal_mask || (legal_mask >> d.classes))
    throw std::invalid_argument("invalid event metadata");
  std::vector<float> input(d.resource_hidden + 3 * d.event_embedding);
  linear(resource_weight_.data, resource_bias_.data, resources, d.resources,
         d.resource_hidden, input.data());
  for (std::uint32_t i = 0; i < d.resource_hidden; ++i)
    input[i] = std::max(0.0f, input[i]);
  float* embedded = input.data() + d.resource_hidden;
  std::memcpy(embedded, cell_embedding_.data + std::size_t(cell) * d.event_embedding,
              d.event_embedding * sizeof(float));
  std::memcpy(embedded + d.event_embedding,
              stage_embedding_.data + std::size_t(stage) * d.event_embedding,
              d.event_embedding * sizeof(float));
  std::memcpy(embedded + 2 * d.event_embedding,
              previous_embedding_.data + std::size_t(previous) * d.event_embedding,
              d.event_embedding * sizeof(float));
  std::vector<float> input_gates(3 * d.hidden), hidden_gates(3 * d.hidden);
  linear(gru_weight_ih_.data, gru_bias_ih_.data, input.data(), input.size(),
         3 * d.hidden,
         input_gates.data());
  linear(gru_weight_hh_.data, gru_bias_hh_.data, hidden, d.hidden, 3 * d.hidden,
         hidden_gates.data());
  for (std::uint32_t i = 0; i < d.hidden; ++i) {
    const float reset = sigmoid(input_gates[i] + hidden_gates[i]);
    const float update = sigmoid(input_gates[d.hidden + i] +
                                 hidden_gates[d.hidden + i]);
    const float candidate = std::tanh(input_gates[2 * d.hidden + i] +
                                      reset * hidden_gates[2 * d.hidden + i]);
    next_hidden[i] = (1.0f - update) * candidate + update * hidden[i];
  }
  linear(head_weight_.data, head_bias_.data, next_hidden, d.hidden, d.classes,
         logits);
  if (shop_action_head_) {
    float gate[8];
    linear(shop_gate_weight_.data, shop_gate_bias_.data, next_hidden,
           d.hidden, 8, gate);
    for (std::uint32_t item = 0; item < 8; ++item)
      logits[item + 3] += gate[item] * resources[374 + item] * (1.0f / 32.0f);
  }
  for (std::uint32_t i = 0; i < d.classes; ++i)
    if (!(legal_mask & (1u << i))) logits[i] = -1.0e9f;
}

void Actor::step(const float* hidden, const float* resources, std::uint32_t cell,
                 std::uint32_t stage, std::uint32_t previous,
                 std::uint32_t legal_mask, float* logits,
                 float* next_hidden) const {
  std::vector<float> normalized(dimensions_.resources);
  for (std::uint32_t i = 0; i < dimensions_.resources; ++i)
    normalized[i] = (resources[i] - resource_mean_.data[i]) / resource_std_.data[i];
  step_normalized(hidden, normalized.data(), cell, stage, previous, legal_mask,
                  logits, next_hidden);
}

Distribution Actor::greedy(const float* logits, std::uint32_t legal_mask,
                           float temperature) const {
  return sample(logits, legal_mask, 0, 0, -temperature);
}

Distribution Actor::evaluate(const float* logits, std::uint32_t legal_mask,
                             std::uint32_t action, float temperature) const {
  if (action >= dimensions_.classes || !(legal_mask & (1u << action)) ||
      !std::isfinite(temperature) || temperature <= 0.0f)
    throw std::invalid_argument("invalid evaluated action");
  Distribution result = greedy(logits, legal_mask, temperature);
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::uint32_t i = 0; i < dimensions_.classes; ++i)
    if (legal_mask & (1u << i))
      maximum = std::max(maximum, logits[i] / temperature);
  float total = 0.0f;
  for (std::uint32_t i = 0; i < dimensions_.classes; ++i)
    if (legal_mask & (1u << i))
      total += std::exp(logits[i] / temperature - maximum);
  result.action = static_cast<int>(action);
  result.log_probability =
      logits[action] / temperature - maximum - std::log(total);
  return result;
}

Distribution Actor::sample(const float* logits, std::uint32_t legal_mask,
                           std::uint64_t seed, std::uint64_t counter,
                           float temperature) const {
  const bool choose_greedy = temperature < 0.0f;
  temperature = std::abs(temperature);
  return categorical(dimensions_, logits, legal_mask, temperature,
                     choose_greedy, counter_uniform(seed, counter));
}

Distribution Actor::sample_policy(const float* logits,
                                  std::uint32_t legal_mask,
                                  std::uint64_t seed,
                                  std::uint64_t event_index,
                                  float temperature) const {
  const bool choose_greedy = temperature < 0.0f;
  temperature = std::abs(temperature);
  return categorical(dimensions_, logits, legal_mask, temperature,
                     choose_greedy, policy_uniform(seed, event_index));
}

}  // namespace student_v3
