#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kProductionSteps = 30;
constexpr int kProductionFields = 12;
constexpr int kLayoutSteps = 5;
constexpr int kLayoutFields = 100;
constexpr int kScheduleSteps = 10;
constexpr int kScheduleFields = 12;

template <typename T>
void Read(std::ifstream& stream, std::vector<T>& values) {
  stream.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!stream) throw std::runtime_error("truncated macro-distance input");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: macro_distance INPUT.bin OUTPUT.f32\n";
    return 2;
  }
  std::ifstream input(argv[1], std::ios::binary);
  if (!input) throw std::runtime_error("cannot open input");
  std::int32_t count = 0;
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  if (!input || count <= 0) throw std::runtime_error("invalid row count");

  std::vector<std::int16_t> production(
      static_cast<std::size_t>(count) * kProductionSteps * kProductionFields);
  std::vector<std::int8_t> layouts(
      static_cast<std::size_t>(count) * kLayoutSteps * kLayoutFields);
  std::vector<std::int16_t> schedule(
      static_cast<std::size_t>(count) * kScheduleSteps * kScheduleFields);
  Read(input, production);
  Read(input, layouts);
  Read(input, schedule);

  std::vector<float> distances(static_cast<std::size_t>(count) * count, 0.0f);
#pragma omp parallel for schedule(dynamic, 1)
  for (std::int32_t left = 0; left < count; ++left) {
    for (std::int32_t right = 0; right < left; ++right) {
      double production_distance = 0.0;
      double land_labor_distance = 0.0;
      for (int step = 0; step < kProductionSteps; ++step) {
        const std::size_t a =
            (static_cast<std::size_t>(left) * kProductionSteps + step) *
            kProductionFields;
        const std::size_t b =
            (static_cast<std::size_t>(right) * kProductionSteps + step) *
            kProductionFields;
        int occupied_left = 0, occupied_right = 0, difference = 0;
        for (int field = 0; field < 10; ++field) {
          occupied_left += production[a + field];
          occupied_right += production[b + field];
          difference += std::abs(static_cast<int>(production[a + field]) -
                                 static_cast<int>(production[b + field]));
        }
        production_distance +=
            static_cast<double>(difference) /
            std::max({8, occupied_left, occupied_right});
        land_labor_distance +=
            std::abs(static_cast<int>(production[a + 10]) -
                     static_cast<int>(production[b + 10])) /
                3.0 +
            std::abs(static_cast<int>(production[a + 11]) -
                     static_cast<int>(production[b + 11])) /
                12.0;
      }
      production_distance /= kProductionSteps;
      land_labor_distance /= (kProductionSteps * 2.0);

      double layout_distance = 0.0;
      for (int step = 0; step < kLayoutSteps; ++step) {
        const std::size_t a =
            (static_cast<std::size_t>(left) * kLayoutSteps + step) * kLayoutFields;
        const std::size_t b =
            (static_cast<std::size_t>(right) * kLayoutSteps + step) * kLayoutFields;
        int active = 0, mismatch = 0;
        for (int field = 0; field < kLayoutFields; ++field) {
          const int x = layouts[a + field], y = layouts[b + field];
          if (x != 0 || y != 0) {
            ++active;
            mismatch += x != y;
          }
        }
        if (active != 0) layout_distance += static_cast<double>(mismatch) / active;
      }
      layout_distance /= kLayoutSteps;

      double schedule_distance = 0.0;
      for (int step = 0; step < kScheduleSteps; ++step) {
        const std::size_t a =
            (static_cast<std::size_t>(left) * kScheduleSteps + step) *
            kScheduleFields;
        const std::size_t b =
            (static_cast<std::size_t>(right) * kScheduleSteps + step) *
            kScheduleFields;
        int total_left = 0, total_right = 0, difference = 0;
        for (int field = 0; field < kScheduleFields; ++field) {
          total_left += schedule[a + field];
          total_right += schedule[b + field];
          difference += std::abs(static_cast<int>(schedule[a + field]) -
                                 static_cast<int>(schedule[b + field]));
        }
        schedule_distance +=
            static_cast<double>(difference) /
            std::max({6, total_left, total_right});
      }
      schedule_distance /= kScheduleSteps;
      const float value = static_cast<float>(
          0.38 * production_distance + 0.38 * layout_distance +
          0.18 * schedule_distance + 0.06 * land_labor_distance);
      distances[static_cast<std::size_t>(left) * count + right] = value;
      distances[static_cast<std::size_t>(right) * count + left] = value;
    }
  }
  std::ofstream output(argv[2], std::ios::binary);
  output.write(reinterpret_cast<const char*>(distances.data()),
               static_cast<std::streamsize>(distances.size() * sizeof(float)));
  if (!output) throw std::runtime_error("cannot write distance matrix");
  std::cout << "rows=" << count << " pairs="
            << (static_cast<std::uint64_t>(count) * (count - 1) / 2) << "\n";
}
