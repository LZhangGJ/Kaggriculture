#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

at::Tensor tensor(const torch::jit::Module& fixture, const char* name) {
  return fixture.attr(name).toTensor();
}

std::vector<torch::jit::IValue> step_inputs(
    const torch::jit::Module& fixture, int64_t row) {
  std::vector<torch::jit::IValue> values;
  for (const char* name : {"hidden", "resources", "cells", "stages",
                           "previous", "legal"}) {
    values.emplace_back(tensor(fixture, name).narrow(0, row, 1));
  }
  return values;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "usage: " << argv[0]
              << " MODEL.pt FIXTURE.pt [benchmark_iterations]\n";
    return 2;
  }
  const int64_t iterations = argc == 4 ? std::stoll(argv[3]) : 10000;
  if (iterations <= 0) return 2;

  at::set_num_threads(1);
  at::set_num_interop_threads(1);
  torch::InferenceMode inference_guard;
  auto model = torch::jit::load(argv[1], torch::kCPU);
  auto fixture = torch::jit::load(argv[2], torch::kCPU);
  model.eval();

  c10::List<at::Tensor> categories;
  for (int index = 0; index < 7; ++index) {
    categories.push_back(tensor(
        fixture, ("token_category_" + std::to_string(index)).c_str()));
  }
  std::vector<torch::jit::IValue> initial_inputs{
      tensor(fixture, "context"), tensor(fixture, "observation"),
      tensor(fixture, "observation_length"),
      tensor(fixture, "token_continuous"), categories,
      tensor(fixture, "token_count")};
  const auto initial = model.get_method("initial_hidden")(initial_inputs).toTensor();
  double initial_error = (initial - tensor(fixture, "expected_initial_hidden"))
                             .abs().max().item<double>();

  const auto expected_logits = tensor(fixture, "expected_logits");
  const auto expected_next = tensor(fixture, "expected_next_hidden");
  const auto expected_actions = tensor(fixture, "expected_actions");
  const int64_t events = expected_logits.size(0);
  double logits_error = 0.0;
  double hidden_error = 0.0;
  int64_t action_mismatches = 0;
  for (int64_t row = 0; row < events; ++row) {
    const auto output = model.get_method("step")(step_inputs(fixture, row)).toTuple();
    const auto logits = output->elements()[0].toTensor();
    const auto hidden = output->elements()[1].toTensor();
    logits_error = std::max(logits_error,
        (logits - expected_logits.narrow(0, row, 1)).abs().max().item<double>());
    hidden_error = std::max(hidden_error,
        (hidden - expected_next.narrow(0, row, 1)).abs().max().item<double>());
    action_mismatches += logits.argmax(1).item<int64_t>() !=
                         expected_actions[row].item<int64_t>();
  }

  for (int warmup = 0; warmup < 100; ++warmup) {
    (void)model.get_method("step")(step_inputs(fixture, warmup % events));
  }
  auto started = std::chrono::steady_clock::now();
  for (int64_t index = 0; index < iterations; ++index) {
    (void)model.get_method("step")(step_inputs(fixture, index % events));
  }
  auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
  const double microseconds = elapsed * 1e6 / iterations;
  const bool pass = initial_error <= 1e-6 && logits_error <= 1e-6 &&
                    hidden_error <= 1e-6 && action_mismatches == 0;
  std::cout << std::setprecision(10)
            << "{\"status\":\"" << (pass ? "PASS" : "FAIL")
            << "\",\"events\":" << events
            << ",\"initial_hidden_max_abs\":" << initial_error
            << ",\"logits_max_abs\":" << logits_error
            << ",\"next_hidden_max_abs\":" << hidden_error
            << ",\"action_mismatches\":" << action_mismatches
            << ",\"benchmark_iterations\":" << iterations
            << ",\"microseconds_per_event\":" << microseconds
            << ",\"events_per_second\":" << 1e6 / microseconds << "}\n";
  return pass ? 0 : 1;
}
