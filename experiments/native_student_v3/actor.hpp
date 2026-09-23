#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace student_v3 {

struct Dimensions {
  std::uint32_t classes = 0;
  std::uint32_t token_capacity = 0;
  std::uint32_t token_continuous = 0;
  std::uint32_t categories = 0;
  std::uint32_t context = 0;
  std::uint32_t observation = 0;
  std::uint32_t resources = 0;
  std::uint32_t projection = 0;
  std::uint32_t scalar = 0;
  std::uint32_t embedding = 0;
  std::uint32_t hidden = 0;
  std::uint32_t resource_hidden = 0;
  std::uint32_t event_embedding = 0;
  std::uint32_t cells = 0;
  std::uint32_t stages = 0;
  std::uint32_t previous = 0;
};

struct Distribution {
  int action = -1;
  float log_probability = 0.0f;
  float entropy = 0.0f;
};

class Actor {
 public:
  Actor() = default;
  Actor(const Actor&) = delete;
  Actor& operator=(const Actor&) = delete;
  Actor(Actor&&) = default;
  Actor& operator=(Actor&&) = default;
  static Actor load(const std::string& path);

  const Dimensions& dimensions() const { return dimensions_; }

  // Categories are category-major: [7][token_capacity].
  void initial_hidden_normalized(const float* context,
                                 const float* observation,
                                 float observation_length,
                                 const float* token_continuous,
                                 const std::uint32_t* categories,
                                 std::uint32_t token_count,
                                 float* hidden) const;
  void initial_hidden(const float* context, const float* observation,
                      float observation_length,
                      const float* token_continuous,
                      const std::uint32_t* categories,
                      std::uint32_t token_count, float* hidden) const;

  // PPO rollout capture uses the exact tensors seen by the Python model.
  // Keep normalization next to the frozen binary weights so native rollouts
  // cannot drift from checkpoint statistics.
  void normalize_context(const float* input, float* output) const;
  void normalize_observation(const float* input, float* output) const;
  float normalize_observation_length(float input) const;
  void normalize_resources(const float* input, float* output) const;

  void step_normalized(const float* hidden, const float* resources,
                       std::uint32_t cell, std::uint32_t stage,
                       std::uint32_t previous, std::uint32_t legal_mask,
                       float* logits, float* next_hidden) const;
  void step(const float* hidden, const float* resources, std::uint32_t cell,
            std::uint32_t stage, std::uint32_t previous,
            std::uint32_t legal_mask, float* logits,
            float* next_hidden) const;

  Distribution greedy(const float* logits, std::uint32_t legal_mask,
                      float temperature = 1.0f) const;
  Distribution evaluate(const float* logits, std::uint32_t legal_mask,
                        std::uint32_t action,
                        float temperature = 1.0f) const;
  Distribution sample(const float* logits, std::uint32_t legal_mask,
                      std::uint64_t seed, std::uint64_t counter,
                      float temperature = 1.0f) const;
  // Rollout RNG contract: SplitMix64(policy_seed, global_event_index), with
  // the same 53-bit uniform followed by float32 rounding as the Python/NPU
  // coordinator. `sample` above remains the frozen kernel-fixture contract.
  Distribution sample_policy(const float* logits, std::uint32_t legal_mask,
                             std::uint64_t seed, std::uint64_t event_index,
                             float temperature = 1.0f) const;

 private:
  struct Tensor {
    const float* data = nullptr;
    std::size_t size = 0;
  };

  Dimensions dimensions_;
  std::vector<float> payload_;
  Tensor context_mean_, context_std_, observation_mean_, observation_std_;
  Tensor observation_length_mean_, observation_length_std_;
  Tensor resource_mean_, resource_std_;
  Tensor context_weight_, context_bias_, observation_weight_, observation_bias_;
  Tensor observation_length_weight_, observation_length_bias_;
  std::vector<Tensor> token_embeddings_;
  Tensor token_weight_, token_bias_, begin_weight_, begin_bias_;
  Tensor resource_weight_, resource_bias_, cell_embedding_, stage_embedding_;
  Tensor previous_embedding_, gru_weight_ih_, gru_weight_hh_;
  Tensor gru_bias_ih_, gru_bias_hh_, head_weight_, head_bias_;
};

}  // namespace student_v3
