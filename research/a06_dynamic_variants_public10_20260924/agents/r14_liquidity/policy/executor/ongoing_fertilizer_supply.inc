// R3, included inside Controller. Bridge an already selected, due, ongoing
// crop service to preparation. This does NOT select additional maintenance,
// change its value, reserve future manure, or buy for unrelated finite crops.
#ifndef TRI_A06_R12_R3_FERT_SUPPLY
#define TRI_A06_R12_R3_FERT_SUPPLY 1
#endif
static_assert(TRI_A06_R12_R3_FERT_SUPPLY == 0 || TRI_A06_R12_R3_FERT_SUPPLY == 1);
// Internal read-only preparation comparison, not a runtime configuration.
bool suppress_ongoing_fertilizer_supply=false;
int approved_ongoing_fertilizer_gap(const View& o, const std::vector<Job>& js) const {
 if (suppress_ongoing_fertilizer_supply || !p.procure_service_inputs || day != o.day || day < 15 || day >= 29 ||
     crop_service_day != day) return 0;
 std::array<int,100> wanted{}, held{};
 for (const auto& j : js) {
  if (j.pos < 0 || j.pos >= 100 || j.needs[F] <= 0) continue;
  const auto& t = o.own.tiles[j.pos];
  if (!plant(t) || !ongoing(int(t.crop)) || !fertilize_due(t) ||
      crop_birth[j.pos] != t.planted_day || crop_kind[j.pos] != int(t.crop) ||
      crop_fertilize[j.pos] <= 0) continue;
  bool fert = false, wet = t.watered_today, replacement = false;
  for (auto a : j.actions) {
   fert |= a.op == Op::FERTILIZE;
   wet |= a.op == Op::WATER;
   replacement |= a.op == Op::DIG || a.op == Op::PLANT || a.op == Op::PLACE;
  }
  if (fert && wet && !replacement) wanted[j.pos] += j.needs[F];
 }
 if (std::accumulate(wanted.begin(),wanted.end(),0) == 0) return 0;
 // Use the same reservation algorithm as execution, including stable task
 // priority, finite-crop reservations and resource degradation. A copy keeps
 // this read-only; counters and live execution state must not be altered.
 auto probe = *this;
 for (const auto& j : probe.reserve(o,js))
  if (j.pos >= 0 && j.pos < 100) held[j.pos] += j.needs[F];
 int gap = 0;
 for (int pos = 0; pos < 100; ++pos) gap += std::max(0,wanted[pos]-held[pos]);
 return gap;
}
