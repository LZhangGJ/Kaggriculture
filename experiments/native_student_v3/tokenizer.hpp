#pragma once

#include <cstdint>
#include <vector>

namespace fastkag { class Simulator; }

namespace student_v3 {

struct EncodedTokens {
  std::vector<float> continuous;          // [capacity, 24]
  std::vector<std::uint32_t> categories;  // [7, capacity]
  std::uint32_t count{};
};

// Exact native counterpart of kaggrl.tokenizer.ObservationTokenizer after
// canonicalizing the acting farm to player zero.
EncodedTokens tokenize(const fastkag::Simulator& environment, int seat,
                       std::uint32_t capacity = 320);

}  // namespace student_v3
