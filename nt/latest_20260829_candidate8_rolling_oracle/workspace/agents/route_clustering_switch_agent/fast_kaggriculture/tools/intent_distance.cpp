// Licensed under the Apache License, Version 2.0.
// Pairwise distance for intended macro plans.  The binary format is:
// int32 N, int8 planned_layouts[N,5,100], int16 schedules[N,10,12].
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kAnchors = 5;
constexpr int kTiles = 100;
constexpr int kCategories = 10;
constexpr int kPhases = 10;
constexpr int kSchedule = 12;

template <typename T>
void read_exact(std::ifstream& input, T* destination, size_t count) {
  input.read(reinterpret_cast<char*>(destination), sizeof(T) * count);
  if (!input) throw std::runtime_error("truncated intent-distance input");
}

float distance(const int8_t* left_layout, const int8_t* right_layout,
               const int16_t* left_schedule, const int16_t* right_schedule) {
  double composition = 0.0;
  double layout = 0.0;
  for (int anchor = 0; anchor < kAnchors; ++anchor) {
    std::array<int, kCategories> left_counts{}, right_counts{};
    int active = 0, mismatched = 0;
    for (int tile = 0; tile < kTiles; ++tile) {
      const int l = left_layout[anchor * kTiles + tile];
      const int r = right_layout[anchor * kTiles + tile];
      if (l > 0 && l <= kCategories) left_counts[l - 1]++;
      if (r > 0 && r <= kCategories) right_counts[r - 1]++;
      if (l || r) {
        active++;
        mismatched += l != r;
      }
    }
    int left_total = 0, right_total = 0, delta = 0;
    for (int category = 0; category < kCategories; ++category) {
      left_total += left_counts[category];
      right_total += right_counts[category];
      delta += std::abs(left_counts[category] - right_counts[category]);
    }
    composition += double(delta) / std::max(8, std::max(left_total, right_total));
    layout += active ? double(mismatched) / active : 0.0;
  }
  composition /= kAnchors;
  layout /= kAnchors;

  // Compare cumulative phase endpoints.  A macro action drifting over a phase
  // boundary is therefore a small timing difference, not a brand-new plan.
  std::array<int, kSchedule> left_cumulative{}, right_cumulative{};
  double schedule = 0.0;
  for (int phase = 0; phase < kPhases; ++phase) {
    int left_total = 0, right_total = 0, delta = 0;
    for (int category = 0; category < kSchedule; ++category) {
      left_cumulative[category] += left_schedule[phase * kSchedule + category];
      right_cumulative[category] += right_schedule[phase * kSchedule + category];
      left_total += left_cumulative[category];
      right_total += right_cumulative[category];
      delta += std::abs(left_cumulative[category] - right_cumulative[category]);
    }
    schedule += double(delta) / std::max(6, std::max(left_total, right_total));
  }
  schedule /= kPhases;
  return static_cast<float>(0.45 * composition + 0.35 * layout + 0.20 * schedule);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: intent_distance INPUT.bin OUTPUT.f32\n";
    return 2;
  }
  try {
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input");
    int32_t size = 0;
    read_exact(input, &size, 1);
    if (size <= 0) throw std::runtime_error("invalid row count");
    std::vector<int8_t> layouts(size_t(size) * kAnchors * kTiles);
    std::vector<int16_t> schedules(size_t(size) * kPhases * kSchedule);
    read_exact(input, layouts.data(), layouts.size());
    read_exact(input, schedules.data(), schedules.size());
    std::vector<float> distances(size_t(size) * size, 0.0f);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int left = 0; left < size; ++left) {
      for (int right = 0; right < left; ++right) {
        const float value = distance(
            layouts.data() + size_t(left) * kAnchors * kTiles,
            layouts.data() + size_t(right) * kAnchors * kTiles,
            schedules.data() + size_t(left) * kPhases * kSchedule,
            schedules.data() + size_t(right) * kPhases * kSchedule);
        distances[size_t(left) * size + right] = value;
        distances[size_t(right) * size + left] = value;
      }
    }
    std::ofstream output(argv[2], std::ios::binary);
    if (!output) throw std::runtime_error("cannot open output");
    output.write(reinterpret_cast<const char*>(distances.data()),
                 sizeof(float) * distances.size());
    if (!output) throw std::runtime_error("cannot write output");
    std::cout << "computed " << size_t(size) * (size - 1) / 2
              << " intended-plan pairs\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
