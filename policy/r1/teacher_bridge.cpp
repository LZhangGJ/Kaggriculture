// Offline-only teacher ABI.  This translation unit intentionally includes the
// normal bridge so a diagnostic .so can reuse the exact Handle/history without
// adding any symbol or branch to the submitted binary.
#include "bridge.cpp"
#include <chrono>

namespace {

constexpr int kTeacherAbiVersion = 2;
constexpr int kTeacherScenarios = 4;
constexpr int kTeacherRowWidth = 7 + 3 * kTeacherScenarios;
constexpr int kExpectedCandidateFeatureWidth = 356;
constexpr int kExpectedContextWidth = 2233;
constexpr int kStudentSlotAbiVersion = 3;
constexpr int kStudentSlotMetaWidth = 4;
constexpr int kStudentSlotResourceWidth = 5 + 12 + 30 * 11;
constexpr const char* kExpectedCandidateFeatureHash =
    "9af5b62e20c54c34b785134f08399d69e32603dfad166918b0813f1b7b970e38";
// Frozen after comparing the names emitted by causal_context() below.  The
// Python writer recomputes SHA-256 and refuses a stale constant.
constexpr const char* kExpectedContextHash =
    "6386bae567618f5c8409075b420dc5dbff3bb616b236bb714aacde151beff3a8";
thread_local double last_short_prepare_seconds = 0;
thread_local double last_scenario_seconds = 0;

struct TeacherObservation {
  int step, day, hour, seat;
  fastkag::Farm a, b;
  fastkag::PrivateState priv;
  fastkag::Market market;
  std::vector<int8_t> shops;
  dp7::View view() const {
    return {step, day, hour, seat == 0 ? a : b, seat == 0 ? b : a,
            priv, market, shops};
  }
};

TeacherObservation decode_teacher_observation(const double* input,
                                               size_t count) {
  Cursor r{input, count};
  TeacherObservation o;
  o.step = r.integer();
  o.day = r.integer();
  o.hour = r.integer();
  o.seat = r.integer();
  o.a = r.farm();
  o.b = r.farm();
  o.priv = r.priv();
  for (auto& x : o.market.inventory) x = r.integer();
  for (auto& x : o.market.prices) x = r.integer();
  const int shops = r.count();
  for (int i = 0; i < shops; ++i) o.shops.push_back(r.integer());
  if (r.i != count || o.day != o.step / 24 || o.hour != o.step % 24 ||
      o.seat < 0 || o.seat > 1)
    throw std::runtime_error("teacher observation");
  return o;
}

void require_first_handoff(const Handle& h, const dp7::View& v) {
  if (v.step != 288 || v.hour != 0 || h.policy.searches != 0 ||
      h.policy.restore_day != -1)
    throw std::runtime_error("teacher v2 requires the first handoff at step 288");
  const int expected_ledger_step = h.policy.begun_step == v.step
      ? v.step : v.step - 1;
  if (h.policy.ledger.observation_step != expected_ledger_step)
    throw std::runtime_error("teacher history is not consecutive at first handoff");
}

const std::vector<std::string>& candidate_feature_names(const Handle& h) {
  if (h.prepared.empty()) throw std::runtime_error("no prepared candidates");
  const auto& names = h.prepared.front().features.names;
  if (names.size() != h.prepared.front().features.x.size() ||
      names.size() != kExpectedCandidateFeatureWidth)
    throw std::runtime_error("candidate feature schema width changed");
  for (const auto& candidate : h.prepared)
    if (candidate.features.x.size() != names.size() ||
        candidate.features.names != names)
      throw std::runtime_error("candidate feature schema mismatch");
  return names;
}

std::string newline_names(const std::vector<std::string>& names) {
  std::ostringstream out;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i) out << '\n';
    out << names[i];
  }
  return out.str();
}

#if R2_STUDENT_SLOT_AUDIT
const std::vector<std::string>& student_slot_resource_names() {
  static const std::vector<std::string> names = [] {
    std::vector<std::string> out = {
        "remaining_budget", "selected_animals", "owned_quadrants",
        "slot_index", "slot_count_at_decision"};
    for (int kind = 0; kind < 12; ++kind)
      out.push_back("stock_left_" + std::to_string(kind));
    for (int day = 0; day < 30; ++day) {
      for (int item = 0; item < 9; ++item)
        out.push_back("prefix_flow_day" + std::to_string(day) + "_item" +
                      std::to_string(item));
      out.push_back("prefix_labor_day" + std::to_string(day));
      out.push_back("prefix_fixed_day" + std::to_string(day));
    }
    return out;
  }();
  return names;
}

int student_slot_kind_bit(int kind) {
  if (kind == -1) return 0;
  if (kind >= 0 && kind < 5) return kind + 1;
  if (kind >= 9 && kind < 12) return kind - 3;
  return -1;
}

int write_student_slot_meta(const std::vector<triad::StudentSlotAudit>& slots,
                            int32_t* output, size_t capacity) {
  if (!output || capacity < slots.size() * kStudentSlotMetaWidth) return -2;
  for (size_t i = 0; i < slots.size(); ++i) {
    const auto& slot = slots[i];
    const int bit = student_slot_kind_bit(slot.label);
    if (bit < 0 || !(slot.legal_mask & (1 << bit))) return -3;
    const size_t at = i * kStudentSlotMetaWidth;
    output[at] = slot.pos;
    output[at + 1] = slot.label;
    output[at + 2] = slot.legal_mask;
    output[at + 3] = slot.terminal;
  }
  return int(slots.size());
}

int write_student_slot_resource(const triad::StudentSlotAudit& slot,
                                double* output, size_t capacity) {
  if (!output || capacity < kStudentSlotResourceWidth) return -2;
  size_t at = 0;
  output[at++] = slot.budget;
  output[at++] = slot.animals;
  output[at++] = slot.owned;
  output[at++] = slot.slot_index;
  output[at++] = slot.slot_count;
  for (int kind = 0; kind < 12; ++kind)
    output[at++] = slot.stock_left[kind];
  for (int day = 0; day < 30; ++day) {
    for (int item = 0; item < 9; ++item)
      output[at++] = slot.prefix.f[day][item];
    output[at++] = slot.prefix.labor[day];
    output[at++] = slot.prefix.fixed[day];
  }
  if (at != kStudentSlotResourceWidth ||
      student_slot_resource_names().size() != kStudentSlotResourceWidth)
    return -3;
  return 1;
}

int write_student_slot_resources(
    const std::vector<triad::StudentSlotAudit>& slots, double* output,
    size_t capacity) {
  if (!output || capacity < slots.size() * kStudentSlotResourceWidth) return -2;
  for (size_t i = 0; i < slots.size(); ++i)
    if (write_student_slot_resource(
            slots[i], output + i * kStudentSlotResourceWidth,
            capacity - i * kStudentSlotResourceWidth) != 1)
      return -3;
  return int(slots.size());
}
#endif

triad::Features causal_context(const Handle& h) {
  triad::Features f;
  f.naming = true;
  auto add = [&](std::string name, double value) {
    f.add(std::move(name), value);
  };
  const auto& policy = h.policy;
  const auto& live = policy.live;
  const auto& ledger = policy.ledger;
  add("state_step", policy.begun_step);
  add("remaining_ticks", 719 - policy.begun_step);
  add("first_handoff", policy.searches == 0);
  add("live_previous_step", live.previous_step);
  add("live_core_day", live.core.day);
  add("controller_restore_day", policy.restore_day);

  add("ledger_observation_step", ledger.observation_step);
  add("ledger_source_step", ledger.source_step);
  add("ledger_accepted", ledger.accepted);
  add("ledger_skipped_floor", ledger.skipped_floor);
  add("ledger_skipped_boundary", ledger.skipped_boundary);
  add("ledger_invalid_market", ledger.invalid_market);
  auto ledger_array = [&](const char* name, const auto& values) {
    for (int item = 0; item < 9; ++item)
      add(std::string("ledger_") + name + "_" + std::to_string(item),
          values[item]);
  };
  ledger_array("lower", ledger.lower);
  ledger_array("upper", ledger.upper);
  ledger_array("added", ledger.added);
  ledger_array("certain_added", ledger.certain_added);
  ledger_array("rival_net", ledger.rival_net);
  ledger_array("valid", ledger.valid);
  ledger_array("own_sales", ledger.own_sales);
  ledger_array("own_valid", ledger.own_valid);

  auto sale_clock = [&](const char* side, const auto& products) {
    for (int item = 0; item < 9; ++item) {
      const auto& product = products[item];
      for (int slot = 0; slot < 3; ++slot) {
        const std::string prefix = std::string("sale_") + side + "_" +
            std::to_string(item) + "_slot" + std::to_string(slot);
        add(prefix + "_day", product.day[slot]);
        for (int hour = 0; hour < 24; ++hour)
          add(prefix + "_hour" + std::to_string(hour),
              product.sales[slot][hour]);
      }
    }
  };
  sale_clock("own", policy.sale_clock.own);
  sale_clock("rival", policy.sale_clock.rival);

  const auto& crop = policy.crop_clock;
  add("crop_previous_step", crop.previous_step);
  for (int kind = 0; kind < 5; ++kind) {
    add("crop_count_" + std::to_string(kind), crop.count[kind]);
    add("crop_cursor_" + std::to_string(kind), crop.cursor[kind]);
    for (int slot = 0; slot < 32; ++slot)
      add("crop_history_" + std::to_string(kind) + "_" +
          std::to_string(slot), crop.history[kind][slot]);
  }

  for (int pos = 0; pos < 100; ++pos) {
    const auto& book = live.book[pos];
    const std::string prefix = "book_" + std::to_string(pos) + "_";
    add(prefix + "kind", book.kind);
    add(prefix + "birth", book.birth);
    add(prefix + "chosen_day", book.chosen_day);
    add(prefix + "length", book.length);
    add(prefix + "successor", book.successor);
    add(prefix + "funded", book.funded);
  }

  const auto& joint = live.joint;
  add("joint_count", joint.count);
  add("joint_created_day", joint.created_day);
  add("joint_animal_limit", joint.animal_limit);
  add("joint_variant", joint.variant);
  add("joint_last_observation", joint.last_observation);
  add("joint_active", joint.active);
  add("joint_source_started", joint.source_started);
  add("joint_source_harvested", joint.source_harvested);
  add("joint_successor_started", joint.successor_started);
  add("joint_cancelled", joint.cancelled);
  add("joint_completed", joint.completed);
  add("joint_last_cancel_reason", joint.last_cancel_reason);
  for (int slot = 0; slot < 2; ++slot) {
    const auto& state = joint.slots[slot];
    const std::string prefix = "joint_slot_" + std::to_string(slot) + "_";
    add(prefix + "pos", state.pos);
    add(prefix + "source_birth", state.source_birth);
    add(prefix + "source_age", state.source_age);
    add(prefix + "next_kind", state.next_kind);
    add(prefix + "stage", state.stage);
    add(prefix + "harvest_step", state.harvest_step);
    add(prefix + "issued_harvest", state.issued_harvest);
    add(prefix + "deadline", state.deadline);
  }
  return f;
}

int prepare_feature_candidates(void* p, const double* input, size_t count) {
  auto& h = *static_cast<Handle*>(p);
  const TeacherObservation decoded = decode_teacher_observation(input, count);
  const dp7::View v = decoded.view();
  require_first_handoff(h, v);
  if (h.policy.begun_step != v.step && td_activate_external(p, input, count))
    return -1;
  h.policy.begin_observation(v);
  h.prepared = h.policy.prepare(v, true);
  h.prepared_scores.clear();
  h.prepared_horizons.clear();
  h.prepared_audits.clear();
  h.prepared_names.assign(h.prepared.size(), "");
  h.prepared_key_unique.assign(h.prepared.size(), true);
  h.prepared_day = v.day;
  candidate_feature_names(h);
  return int(h.prepared.size());
}

// Treat the public asset forecast as the day's available rival output.  This
// diagnostic family changes only sale timing.  Every unit is sold exactly
// once, and the delayed arm carries at most the shared 100-unit shed capacity.
std::array<competitive::Flow, kTeacherScenarios> teacher_flows(
    const competitive::Flow& base, const triad::PublicTradeLedger& ledger,
    int day, double scale) {
  if (!(scale > 0) || !std::isfinite(scale))
    throw std::runtime_error("invalid rival supply scale");
  std::array<competitive::Flow, kTeacherScenarios> out{};
  out[0] = base;
  out[1] = base;
  out[2] = base;
  std::array<double, 9> stock{};
  double lower_total = 0, slack_total = 0;
  for (int item = 0; item < 9; ++item) {
    if (ledger.lower[item] < -1e-9 ||
        ledger.lower[item] > ledger.upper[item] + 1e-9)
      throw std::runtime_error("invalid public stock interval");
    lower_total += ledger.lower[item];
    slack_total += ledger.upper[item] - ledger.lower[item];
  }
  if (lower_total > 100 + 1e-9)
    throw std::runtime_error("public stock lower bound exceeds shared shed");
  const double alpha = slack_total > 0
      ? std::min(1., (100. - lower_total) / slack_total) : 0.;
  for (int item = 0; item < 9; ++item) {
    stock[item] = ledger.lower[item] +
                  alpha * (ledger.upper[item] - ledger.lower[item]);
    out[1][day][item] += ledger.lower[item] / scale;
    out[2][day][item] += stock[item] / scale;
  }

  std::array<double, 9> carried{};
  for (int item = 0; item < 9; ++item)
    carried[item] = stock[item] / scale;
  for (int d = day; d < 30; ++d) {
    std::array<double, 9> available{};
    double total = 0;
    for (int item = 0; item < 9; ++item) {
      available[item] = carried[item] + std::max(0., base[d][item]);
      total += available[item];
      // Negative forecast flow is an input consumed by the rival asset, not a
      // sale.  Preserve its day; only positive sellable output is rescheduled.
      out[3][d][item] = std::min(0., base[d][item]);
    }
    const double keep = d == 29 ? 0. : std::min(100. / scale, total);
    const double fraction = total > 0 ? keep / total : 0.;
    for (int item = 0; item < 9; ++item) {
      carried[item] = available[item] * fraction;
      out[3][d][item] += available[item] - carried[item];
    }
  }
  for (int item = 0; item < 9; ++item) {
    double base_total = 0, delayed_total = 0;
    for (int d = day; d < 30; ++d) {
      base_total += base[d][item];
      delayed_total += out[3][d][item];
    }
    const double expected_physical = scale * base_total + stock[item];
    if (std::abs(scale * delayed_total - expected_physical) >
            1e-7 * (1 + std::abs(expected_physical)) ||
        std::abs(carried[item]) > 1e-9)
      throw std::runtime_error("teacher sale-flow conservation");
  }
  return out;
}

enum CandidateFamily { kNormal = 0, kOuter = 1, kDiagnostic = 2 };

CandidateFamily candidate_family(const std::string& name) {
  if (name.empty()) return kNormal;
  if (name.rfind("outer", 0) == 0) return kOuter;
  return kDiagnostic;
}

std::vector<int> top_indices(const Handle& h, CandidateFamily family,
                             int limit) {
  std::vector<int> indices;
  for (int i = 0; i < int(h.prepared.size()); ++i) {
    if (candidate_family(h.prepared_names[i]) == family) indices.push_back(i);
  }
  std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
    return h.prepared_scores[a] > h.prepared_scores[b];
  });
  if (int(indices.size()) > limit) indices.resize(limit);
  return indices;
}

}  // namespace

extern "C" int td_teacher_abi_version() { return kTeacherAbiVersion; }

extern "C" size_t td_teacher_row_width() { return kTeacherRowWidth; }

extern "C" const char* td_teacher_contract_json() {
  static const std::string contract = std::string(
      "{\"abi_version\":2,\"state_scope\":\"first_handoff_step288_only\","
      "\"candidate_feature_schema\":\"r1_portfolio_features_v1\","
      "\"candidate_feature_width\":356,"
      "\"candidate_feature_names_sha256\":\"") +
      kExpectedCandidateFeatureHash +
      "\",\"causal_context_schema\":\"r1_first_handoff_causal_v1\","
      "\"causal_context_width\":2233,"
      "\"causal_context_names_sha256\":\"" + kExpectedContextHash +
      "\",\"proposal_key_encoding\":\"little_endian_signed_int32\","
      "\"proposal_key_hash\":\"sha256_prefix_128\","
      "\"row_fields\":[\"prepared_index\",\"candidate_id\","
      "\"candidate_family\",\"is_outer\",\"short_score\","
      "\"canonical_anchor_prepared_index\","
      "\"selection_reference_prepared_index\","
      "\"scenario_q_0\",\"scenario_q_1\",\"scenario_q_2\","
      "\"scenario_q_3\",\"paired_advantage_0\","
      "\"paired_advantage_1\",\"paired_advantage_2\","
      "\"paired_advantage_3\",\"selection_paired_advantage_0\","
      "\"selection_paired_advantage_1\","
      "\"selection_paired_advantage_2\","
      "\"selection_paired_advantage_3\"]}";
  return contract.c_str();
}

extern "C" const char* td_teacher_scenarios_json() {
  return "[\"base_point_forecast\",\"flush_certified_lower\","
         "\"flush_feasible_high_stock_particle\","
         "\"capacity_limited_carry_feasible_high_stock_particle\"]";
}

extern "C" int td_teacher_prepare_features_observation(
    void* p, const double* input, size_t count) {
  auto& h = *static_cast<Handle*>(p);
  try {
    return prepare_feature_candidates(p, input, count);
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -1;
  }
}

extern "C" const char* td_teacher_candidate_feature_names(void* p) {
  auto& h = *static_cast<Handle*>(p);
  try {
    h.text = newline_names(candidate_feature_names(h));
    return h.text.c_str();
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return nullptr;
  }
}

extern "C" const char* td_teacher_context_names(void* p) {
  auto& h = *static_cast<Handle*>(p);
  try {
    const auto context = causal_context(h);
    if (context.names.size() != kExpectedContextWidth ||
        context.x.size() != context.names.size())
      throw std::runtime_error("causal context schema width changed");
    h.text = newline_names(context.names);
    return h.text.c_str();
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return nullptr;
  }
}

extern "C" int td_teacher_context(void* p, double* output, size_t capacity) {
  auto& h = *static_cast<Handle*>(p);
  try {
    if (h.policy.begun_step != 288 || h.policy.searches != 0)
      throw std::runtime_error("causal context is only valid at first handoff");
    const auto context = causal_context(h);
    if (context.x.size() != context.names.size() ||
        context.x.size() != kExpectedContextWidth)
      throw std::runtime_error("causal context schema width changed");
    if (capacity < context.x.size()) return -2;
    std::copy(context.x.begin(), context.x.end(), output);
    return int(context.x.size());
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -1;
  }
}

extern "C" int td_teacher_candidate_key(void* p, int index, int32_t* output,
                                          size_t capacity) {
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  const auto key = triad::proposal_key(h.prepared[index].policy);
  if (!output && capacity == 0) return int(key.size());
  if (!output || capacity < key.size()) return -2;
  std::copy(key.begin(), key.end(), output);
  return int(key.size());
}

extern "C" int td_teacher_last_timings(double* output, size_t capacity) {
  if (!output || capacity < 2) return -2;
  output[0] = last_short_prepare_seconds;
  output[1] = last_scenario_seconds;
  return 2;
}

extern "C" int td_teacher_candidate_targets(void* p, int index, int32_t* output,
                                             size_t capacity) {
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  const auto& target = h.prepared[index].policy.core.target;
  if (capacity < target.size() * 2) return -2;
  for (size_t i = 0; i < target.size(); ++i) {
    output[2 * i] = target[i].first;
    output[2 * i + 1] = target[i].second;
  }
  return int(target.size());
}

extern "C" int td_student_candidate_settings(void* p, int index,
                                               double* output,
                                               size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < -1 || index >= int(h.prepared.size())) return -1;
  if (!output || capacity < triad::SETTINGS_COUNT) return -2;
  const auto& settings = index < 0 ? h.policy.live.s : h.prepared[index].policy.s;
  static_assert(sizeof(settings) == triad::SETTINGS_COUNT * sizeof(double));
  std::memcpy(output, &settings, sizeof(settings));
  return triad::SETTINGS_COUNT;
#else
  (void)p; (void)index; (void)output; (void)capacity;
  return -4;
#endif
}

// Exact training-only trace from the production greedy loop.  This is kept in
// the diagnostic translation unit and requires R2_STUDENT_SLOT_AUDIT; the
// submitted bridge has no added state or branch.
extern "C" int td_student_slot_abi_version() {
  return R2_STUDENT_SLOT_AUDIT ? kStudentSlotAbiVersion : 0;
}

extern "C" const char* td_student_slot_contract_json() {
#if R2_STUDENT_SLOT_AUDIT
  return "{\"abi_version\":3,\"meta_width\":4,\"meta_fields\":[\"cell\","
         "\"proposed_kind\",\"legal_mask_9bit\",\"terminal\"],"
         "\"class_kinds\":[-1,0,1,2,3,4,9,10,11],"
         "\"resource_width\":347,"
         "\"label_semantics\":\"actor_callback_action_before_preview\","
         "\"actor_scope\":\"all_executed_plant_place_no_hidden_commitments\","
         "\"prefix_semantics\":\"before_current_slot\"}";
#else
  return "{\"abi_version\":0,\"available\":false}";
#endif
}

extern "C" const char* td_student_slot_resource_names() {
#if R2_STUDENT_SLOT_AUDIT
  static const std::string names = newline_names(student_slot_resource_names());
  return names.c_str();
#else
  return nullptr;
#endif
}

extern "C" int td_student_candidate_slot_count(void* p, int index) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  return int(h.prepared[index].policy.student_slot_audit.size());
#else
  (void)p; (void)index;
  return -4;
#endif
}

extern "C" int td_student_candidate_slot_meta(void* p, int index,
                                                int32_t* output,
                                                size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  return write_student_slot_meta(
      h.prepared[index].policy.student_slot_audit, output, capacity);
#else
  (void)p; (void)index; (void)output; (void)capacity;
  return -4;
#endif
}

extern "C" int td_student_candidate_slot_resources(void* p, int index,
                                                     double* output,
                                                     size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  return write_student_slot_resources(
      h.prepared[index].policy.student_slot_audit, output, capacity);
#else
  (void)p; (void)index; (void)output; (void)capacity;
  return -4;
#endif
}

// Training-only live export.  Python first activates a warmed external handle,
// snapshots this pre-choice context, then calls the ordinary Agent.__call__.
// The second begin_observation() inside act() is intentionally idempotent.
extern "C" int td_student_pre_context_observation(
    void* p, const double* input, size_t count, double* output,
    size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  auto& h = *static_cast<Handle*>(p);
  try {
    const TeacherObservation decoded = decode_teacher_observation(input, count);
    const dp7::View v = decoded.view();
    if (v.step < 288 || v.hour != 0)
      throw std::runtime_error("student live context requires a post-288 day boundary");
    h.policy.begin_observation(v);
    const auto context = causal_context(h);
    if (!output || capacity < context.x.size()) return -2;
    std::copy(context.x.begin(), context.x.end(), output);
    return int(context.x.size());
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -1;
  }
#else
  (void)p; (void)input; (void)count; (void)output; (void)capacity;
  return -4;
#endif
}

extern "C" int td_student_live_meta(void* p, int32_t* output,
                                      size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (!output || capacity < 6) return -2;
  output[0] = h.policy.begun_step;
  output[1] = h.policy.searches;
  output[2] = h.policy.live.core.day;
  output[3] = h.policy.restore_day;
  output[4] = h.policy.live.joint.active;
  output[5] = h.policy.live.student_slot_audit.size();
  return 6;
#else
  (void)p; (void)output; (void)capacity;
  return -4;
#endif
}

extern "C" int td_student_live_slot_count(void* p) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  return int(h.policy.live.student_slot_audit.size());
#else
  (void)p;
  return -4;
#endif
}

extern "C" int td_student_live_slot_meta(void* p, int32_t* output,
                                          size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  return write_student_slot_meta(
      h.policy.live.student_slot_audit, output, capacity);
#else
  (void)p; (void)output; (void)capacity;
  return -4;
#endif
}

extern "C" int td_student_live_slot_resources(void* p, double* output,
                                               size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  return write_student_slot_resources(
      h.policy.live.student_slot_audit, output, capacity);
#else
  (void)p; (void)output; (void)capacity;
  return -4;
#endif
}

int write_student_plan_targets(const triad::Controller& policy,
                               int32_t* output, size_t capacity) {
  if (!output || capacity < 100) return -2;
  std::fill(output, output + 100, int32_t(-2));
  for (const auto& [cell, kind] : policy.core.target) {
    if (cell < 0 || cell >= 100 || kind < -1 || kind >= 12) return -3;
    output[cell] = kind;
  }
  return 100;
}

int write_student_plan_release(const triad::Controller& policy,
                               int32_t* output, size_t capacity) {
  if (!output || capacity < 100) return -2;
  std::copy(policy.release.begin(), policy.release.end(), output);
  return 100;
}

int write_student_plan_meta(const triad::Controller& policy,
                            int32_t* output, size_t capacity) {
  if (!output || capacity < 5) return -2;
  output[0] = policy.core.planned_land;
  output[1] = policy.greedy_preview_checked;
  output[2] = policy.greedy_preview_proposed;
  output[3] = policy.greedy_preview_started;
  output[4] = policy.greedy_preview_removed;
  return 5;
}

extern "C" int td_student_live_targets(void* p, int32_t* output,
                                         size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  return write_student_plan_targets(
      static_cast<Handle*>(p)->policy.live, output, capacity);
#else
  (void)p; (void)output; (void)capacity; return -4;
#endif
}

extern "C" int td_student_live_release(void* p, int32_t* output,
                                         size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  return write_student_plan_release(
      static_cast<Handle*>(p)->policy.live, output, capacity);
#else
  (void)p; (void)output; (void)capacity; return -4;
#endif
}

extern "C" int td_student_live_plan_meta(void* p, int32_t* output,
                                          size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  return write_student_plan_meta(
      static_cast<Handle*>(p)->policy.live, output, capacity);
#else
  (void)p; (void)output; (void)capacity; return -4;
#endif
}

extern "C" int td_student_candidate_plan_audit(void* p, int index,
                                                 int32_t* targets,
                                                 int32_t* release,
                                                 int32_t* meta) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  if (write_student_plan_targets(h.prepared[index].policy, targets, 100) != 100 ||
      write_student_plan_release(h.prepared[index].policy, release, 100) != 100 ||
      write_student_plan_meta(h.prepared[index].policy, meta, 5) != 5)
    return -2;
  return 0;
#else
  (void)p; (void)index; (void)targets; (void)release; (void)meta; return -4;
#endif
}

// Build one complete executable plan while constraining only the already
// sampled autoregressive prefix.  Replanning from the same pre-choice live
// controller makes the next slot's mask and resource calendar conditional on
// that exact prefix.  This intentionally lives only in the offline bridge.
extern "C" int td_student_prepare_prefix_observation(
    void* p, const double* input, size_t count, const int32_t* prefix_cells,
    const int32_t* prefix_kinds, size_t prefix_count,
    const double* scaffold_settings, size_t scaffold_count) {
#if R2_STUDENT_SLOT_AUDIT && R2_OUTER_NEIGHBOR_AUDIT
  auto& h = *static_cast<Handle*>(p);
  try {
    if (prefix_count > 100 ||
        (prefix_count && (!prefix_cells || !prefix_kinds)))
      throw std::runtime_error("invalid student prefix buffers");
    const TeacherObservation decoded = decode_teacher_observation(input, count);
    const dp7::View v = decoded.view();
    if (v.step < 288 || v.hour != 0)
      throw std::runtime_error("student prefix plan requires a post-288 day boundary");
    h.policy.begin_observation(v);

    triad::Settings settings = h.policy.base;
    if (scaffold_count) {
      if (!scaffold_settings || scaffold_count != triad::SETTINGS_COUNT)
        throw std::runtime_error("student scaffold settings width");
      for (size_t i = 0; i < scaffold_count; ++i)
        if (!std::isfinite(scaffold_settings[i]))
          throw std::runtime_error("non-finite student scaffold setting");
      std::memcpy(&settings, scaffold_settings, sizeof(settings));
    }

    triad::Controller planned = h.policy.live;
    planned.configure(settings);
    planned.forced_kind.fill(-2);
    std::array<bool, 100> seen{};
    for (size_t i = 0; i < prefix_count; ++i) {
      const int cell = prefix_cells[i], kind = prefix_kinds[i];
      const bool valid_kind = kind == -1 || (kind >= 0 && kind < 5) ||
                              (kind >= 9 && kind < 12);
      if (cell < 0 || cell >= 100 || seen[cell] || !valid_kind ||
          (kind == -1 && i + 1 != prefix_count))
        throw std::runtime_error("invalid student prefix decision");
      seen[cell] = true;
      planned.forced_kind[cell] = int8_t(kind);
    }
    planned.plan(v);
    const auto& slots = planned.student_slot_audit;
    if (slots.size() < prefix_count)
      throw std::runtime_error("student prefix ended before requested decisions");
    for (size_t i = 0; i < prefix_count; ++i) {
      const auto& slot = slots[i];
      if (slot.pos != prefix_cells[i] || slot.label != prefix_kinds[i])
        throw std::runtime_error("student prefix does not reconstruct plan order");
      const int bit = student_slot_kind_bit(slot.label);
      if (bit < 0 || !(slot.legal_mask & (1 << bit)))
        throw std::runtime_error("student prefix contains an illegal decision");
    }
    if (prefix_count && prefix_kinds[prefix_count - 1] == -1 &&
        slots.size() != prefix_count)
      throw std::runtime_error("student terminal prefix did not stop planning");

    h.prepared.clear();
    h.prepared.push_back({0, std::move(planned), {}});
    h.prepared_scores.clear();
    h.prepared_horizons.clear();
    h.prepared_audits.clear();
    h.prepared_names.assign(1, "student_prefix");
    h.prepared_key_unique.assign(1, true);
    h.prepared_day = v.day;
    return int(h.prepared.front().policy.student_slot_audit.size());
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -1;
  }
#else
  (void)p; (void)input; (void)count; (void)prefix_cells;
  (void)prefix_kinds; (void)prefix_count; (void)scaffold_settings;
  (void)scaffold_count;
  return -4;
#endif
}

extern "C" int td_student_install_prepared(void* p) {
#if R2_STUDENT_SLOT_AUDIT && R2_OUTER_NEIGHBOR_AUDIT
  auto& h = *static_cast<Handle*>(p);
  if (h.prepared.size() != 1 || h.prepared_day < 0) return -1;
  ++h.policy.searches;
  return td_install(p, 0);
#else
  (void)p;
  return -4;
#endif
}

using StudentPlanCallback = int (*)(void*, int32_t, int32_t, int32_t,
                                    const double*, size_t);
using StudentReleaseCallback = int (*)(void*, int32_t, int32_t, int32_t,
                                       const double*, size_t);
using StudentV3SlotCallback = int (*)(void*, int32_t, int32_t, int32_t,
                                      const double*, size_t);

#if R2_STUDENT_SLOT_AUDIT
void make_student_actor_owned(triad::Controller& planned,
                              triad::Settings& settings) {
  settings.keep_commitments = 0;
  settings.portfolio_swaps = 0;
  planned.joint = triad::JointBundleState{};
  for (auto& commitment : planned.book) commitment.successor = -1;
}

void require_no_hidden_student_placement(const triad::Controller& planned,
                                         const dp7::View& view) {
  std::array<int, 100> selected{};
  selected.fill(-1);
  static constexpr std::array<int, 8> kinds{0, 1, 2, 3, 4, 9, 10, 11};
  for (const auto& event : planned.student_v3_slot_audit) {
    if (event.label_class >= 3) selected[event.slot.pos] = kinds[event.label_class - 3];
  }
  for (const auto& job : planned.core.jobs(view)) {
    for (const auto& action : job.actions) {
      if ((action.op == fastkag::Op::PLANT || action.op == fastkag::Op::PLACE) &&
          (job.pos < 0 || job.pos >= 100 || selected[job.pos] != int(action.item)))
        throw std::runtime_error("student scaffold emitted hidden placement");
    }
  }
}
#endif

// One native plan, one callback per exact autoregressive slot.  Unlike the
// prefix reconstruction diagnostic above, this advances budget, inventory and
// the 30-day resource calendar in place and is therefore the fast interaction
// seam used by rollout/RL experiments.
extern "C" int td_student_plan_callback_observation(
    void* p, const double* input, size_t count, StudentPlanCallback callback,
    void* user, const double* scaffold_settings, size_t scaffold_count) {
#if R2_STUDENT_SLOT_AUDIT
  auto& h = *static_cast<Handle*>(p);
  try {
    if (!callback) throw std::runtime_error("student plan callback is null");
    const TeacherObservation decoded = decode_teacher_observation(input, count);
    const dp7::View v = decoded.view();
    if (v.step < 288 || v.hour != 0)
      throw std::runtime_error("student callback plan requires a post-288 day boundary");
    h.policy.begin_observation(v);
    triad::Settings settings = h.policy.base;
    if (scaffold_count) {
      if (!scaffold_settings || scaffold_count != triad::SETTINGS_COUNT)
        throw std::runtime_error("student callback scaffold settings width");
      for (size_t i = 0; i < scaffold_count; ++i)
        if (!std::isfinite(scaffold_settings[i]))
          throw std::runtime_error("non-finite student callback scaffold setting");
      std::memcpy(&settings, scaffold_settings, sizeof(settings));
    }
    triad::Controller planned = h.policy.live;
    make_student_actor_owned(planned, settings);
    planned.configure(settings);
    planned.student_selector = [&](const triad::StudentSlotAudit& slot) {
      std::array<double, kStudentSlotResourceWidth> resources{};
      if (write_student_slot_resource(slot, resources.data(), resources.size()) != 1)
        throw std::runtime_error("student callback resource export failed");
      return callback(user, slot.pos, slot.label, slot.legal_mask,
                      resources.data(), resources.size());
    };
    planned.plan(v);
    planned.student_selector = {};
    h.prepared.clear();
    h.prepared.push_back({0, std::move(planned), {}});
    h.prepared_scores.clear();
    h.prepared_horizons.clear();
    h.prepared_audits.clear();
    h.prepared_names.assign(1, "student_callback");
    h.prepared_key_unique.assign(1, true);
    h.prepared_day = v.day;
    return int(h.prepared.front().policy.student_slot_audit.size());
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -1;
  }
#else
  (void)p; (void)input; (void)count; (void)callback; (void)user;
  (void)scaffold_settings; (void)scaffold_count;
  return -4;
#endif
}

int student_v3_class_from_kind(int kind) {
  if (kind < 0) return 0;
  if (kind >= 0 && kind < 5) return kind + 3;
  if (kind >= 9 && kind < 12) return kind - 1;
  return -1;
}

extern "C" int td_student_plan_v3_callback_observation(
    void* p, const double* input, size_t count,
    StudentReleaseCallback release_callback, StudentV3SlotCallback slot_callback,
    void* user, const double* scaffold_settings, size_t scaffold_count) {
#if R2_STUDENT_SLOT_AUDIT
  auto& h = *static_cast<Handle*>(p);
  try {
    if (!release_callback || !slot_callback)
      throw std::runtime_error("student v3 callback is null");
    const TeacherObservation decoded = decode_teacher_observation(input, count);
    const dp7::View v = decoded.view();
    if (v.step < 288 || v.hour != 0)
      throw std::runtime_error("student v3 requires a post-288 day boundary");
    triad::Settings settings = h.policy.base;
    if (scaffold_count) {
      if (!scaffold_settings || scaffold_count != triad::SETTINGS_COUNT)
        throw std::runtime_error("student v3 scaffold settings width");
      for (size_t i = 0; i < scaffold_count; ++i)
        if (!std::isfinite(scaffold_settings[i]))
          throw std::runtime_error("non-finite student v3 scaffold setting");
      std::memcpy(&settings, scaffold_settings, sizeof(settings));
    }
    const int owned_land = std::popcount(unsigned(v.own.unlocked_mask));
    if (owned_land < 2 || owned_land > 3 || int(settings.max_land) != 3)
      throw std::runtime_error("student v3 land invariant");
    h.policy.begin_observation(v);
    triad::Controller planned = h.policy.live;
    make_student_actor_owned(planned, settings);
    planned.configure(settings);
    planned.student_release_selector = [&](const triad::StudentReleaseAudit& event) {
      std::array<double, kStudentSlotResourceWidth> resources{};
      if (write_student_slot_resource(event.state, resources.data(),
                                      resources.size()) != 1)
        throw std::runtime_error("student v3 release resource export failed");
      return release_callback(user, event.pos, event.label, event.legal_mask,
                              resources.data(), resources.size());
    };
    planned.student_v3_selector = [&](const triad::StudentSlotAudit& slot, int mask) {
      std::array<double, kStudentSlotResourceWidth> resources{};
      if (write_student_slot_resource(slot, resources.data(), resources.size()) != 1)
        throw std::runtime_error("student v3 resource export failed");
      const int suggested = student_v3_class_from_kind(slot.label);
      if (suggested < 0) throw std::runtime_error("student v3 suggested kind");
      return slot_callback(user, slot.pos, suggested, mask,
                           resources.data(), resources.size());
    };
    planned.plan(v);
    planned.core.p.student_preserve_weed_plant = true;
    planned.student_release_selector = {};
    planned.student_v3_selector = {};
    require_no_hidden_student_placement(planned, v);
    h.prepared.clear();
    h.prepared.push_back({0, std::move(planned), {}});
    h.prepared_scores.clear();
    h.prepared_horizons.clear();
    h.prepared_audits.clear();
    h.prepared_names.assign(1, "student_v3_callback");
    h.prepared_key_unique.assign(1, true);
    h.prepared_day = v.day;
    const auto& policy = h.prepared.front().policy;
    return int(policy.student_release_audit.size() +
               policy.student_v3_slot_audit.size());
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -1;
  }
#else
  (void)p; (void)input; (void)count; (void)release_callback;
  (void)slot_callback; (void)user; (void)scaffold_settings;
  (void)scaffold_count; return -4;
#endif
}

extern "C" int td_student_candidate_v3_release_meta(
    void* p, int index, int32_t* output, size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  const auto& events = h.prepared[index].policy.student_release_audit;
  if (!output || capacity < events.size() * 3) return -2;
  for (size_t i = 0; i < events.size(); ++i) {
    output[3 * i] = events[i].pos;
    output[3 * i + 1] = events[i].label;
    output[3 * i + 2] = events[i].legal_mask;
  }
  return int(events.size());
#else
  (void)p; (void)index; (void)output; (void)capacity; return -4;
#endif
}

extern "C" int td_student_candidate_v3_release_resources(
    void* p, int index, double* output, size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  const auto& events = h.prepared[index].policy.student_release_audit;
  if (!output || capacity < events.size() * kStudentSlotResourceWidth) return -2;
  for (size_t i = 0; i < events.size(); ++i)
    if (write_student_slot_resource(events[i].state,
            output + i * kStudentSlotResourceWidth,
            capacity - i * kStudentSlotResourceWidth) != 1)
      return -3;
  return int(events.size());
#else
  (void)p; (void)index; (void)output; (void)capacity; return -4;
#endif
}

extern "C" int td_student_candidate_v3_slot_meta(
    void* p, int index, int32_t* output, size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  const auto& events = h.prepared[index].policy.student_v3_slot_audit;
  if (!output || capacity < events.size() * 4) return -2;
  for (size_t i = 0; i < events.size(); ++i) {
    output[4 * i] = events[i].slot.pos;
    output[4 * i + 1] = events[i].label_class;
    output[4 * i + 2] = events[i].legal_mask;
    output[4 * i + 3] = events[i].label_class == 0;
  }
  return int(events.size());
#else
  (void)p; (void)index; (void)output; (void)capacity; return -4;
#endif
}

extern "C" int td_student_candidate_v3_slot_resources(
    void* p, int index, double* output, size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  const auto& h = *static_cast<Handle*>(p);
  if (index < 0 || index >= int(h.prepared.size())) return -1;
  const auto& events = h.prepared[index].policy.student_v3_slot_audit;
  if (!output || capacity < events.size() * kStudentSlotResourceWidth) return -2;
  for (size_t i = 0; i < events.size(); ++i)
    if (write_student_slot_resource(events[i].slot,
            output + i * kStudentSlotResourceWidth,
            capacity - i * kStudentSlotResourceWidth) != 1)
      return -3;
  return int(events.size());
#else
  (void)p; (void)index; (void)output; (void)capacity; return -4;
#endif
}

// Final per-cell executable job summary after preview/repair.  Each row is
// [op bitset, PLANT item, PLACE item, contributing job count].  This is the
// teacher truth for release/placement projection; core.target and release[]
// are planning metadata and may be stale after preview removes a project.
extern "C" int td_student_candidate_job_actions_observation(
    void* p, int index, const double* input, size_t count,
    int32_t* output, size_t capacity) {
#if R2_STUDENT_SLOT_AUDIT
  auto& h = *static_cast<Handle*>(p);
  try {
    if (index < -1 || index >= int(h.prepared.size())) return -1;
    if (!output || capacity < 400) return -2;
    const TeacherObservation decoded = decode_teacher_observation(input, count);
    const dp7::View v = decoded.view();
    std::fill(output, output + 400, int32_t(0));
    for (int cell = 0; cell < 100; ++cell) {
      output[4 * cell + 1] = -1;
      output[4 * cell + 2] = -1;
    }
    const auto& policy = index < 0 ? h.policy.live : h.prepared[index].policy;
    for (const auto& job : policy.core.jobs(v)) {
      if (job.pos < 0 || job.pos >= 100) return -3;
      ++output[4 * job.pos + 3];
      for (const auto& action : job.actions) {
        const int op = int(action.op);
        if (op < 0 || op >= 31) return -4;
        output[4 * job.pos] |= int32_t(uint32_t(1) << op);
        if (action.op == fastkag::Op::PLANT)
          output[4 * job.pos + 1] = int(action.item);
        if (action.op == fastkag::Op::PLACE)
          output[4 * job.pos + 2] = int(action.item);
      }
    }
    return 100;
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -5;
  }
#else
  (void)p; (void)index; (void)input; (void)count;
  (void)output; (void)capacity; return -6;
#endif
}

// Raw Q is the training target.  paired_advantage is anchored to canonical
// normal id0 and therefore does not drift with top_k.  The old top-k/base-Q
// winner is retained separately as selection_paired_advantage for diagnostics.
extern "C" int td_teacher_batch_observation(
    void* p, const double* input, size_t count, int top_k, double* output,
    size_t capacity) {
#if !R2_SCENARIO_CARRY_RIVAL
  (void)p; (void)input; (void)count; (void)top_k; (void)output; (void)capacity;
  return -4;
#else
  auto& h = *static_cast<Handle*>(p);
  try {
    if (top_k < 1 || top_k > 64) throw std::runtime_error("teacher top_k");
    const TeacherObservation decoded = decode_teacher_observation(input, count);
    const dp7::View v = decoded.view();
    require_first_handoff(h, v);
    // td_prepare_observation calls the shared idempotent begin_observation()
    // prefix used by act(), then stops before choose()/action execution.
    const auto prepare_started = std::chrono::steady_clock::now();
    const int prepared = td_prepare_observation(p, input, count);
    last_short_prepare_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - prepare_started).count();
    if (prepared < 0) return prepared;
    const auto& feature_names = candidate_feature_names(h);
    if (int(feature_names.size()) != kExpectedCandidateFeatureWidth)
      throw std::runtime_error("candidate feature width changed");
    auto ranked_normal = top_indices(h, kNormal, top_k);
    auto outer = top_indices(h, kOuter, top_k);
    if (ranked_normal.empty())
      throw std::runtime_error("teacher has no normal candidate");
    int canonical_anchor = -1;
    for (int i = 0; i < int(h.prepared.size()); ++i)
      if (candidate_family(h.prepared_names[i]) == kNormal &&
          h.prepared[i].id == 0) {
        if (canonical_anchor >= 0)
          throw std::runtime_error("multiple canonical id0 candidates");
        canonical_anchor = i;
      }
    if (canonical_anchor < 0)
      throw std::runtime_error("canonical id0 candidate missing");
    auto normal = ranked_normal;
    if (std::find(normal.begin(), normal.end(), canonical_anchor) == normal.end())
      normal.push_back(canonical_anchor);
    std::vector<int> selected = normal;
    selected.insert(selected.end(), outer.begin(), outer.end());
    if (capacity < selected.size() * kTeacherRowWidth) return -2;

    const auto scenario_started = std::chrono::steady_clock::now();
    std::vector<std::array<double, kTeacherScenarios>> scores(selected.size());
    for (size_t row = 0; row < selected.size(); ++row) {
      const auto flows = teacher_flows(h.prepared[selected[row]].policy.model.rival,
                                       h.policy.ledger, v.day,
                                       h.policy.base.supply);
      for (int scenario = 0; scenario < kTeacherScenarios; ++scenario) {
        auto proposal = h.prepared[selected[row]];
        proposal.policy.model.rival = flows[scenario];
        proposal.policy.model.use_rival_forecast = true;
        proposal.policy.model.rival_forecast = flows[scenario];
        scores[row][scenario] = h.policy.score(
            v, proposal, std::max(1, 30 - v.day));
      }
    }
    last_scenario_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - scenario_started).count();
    size_t selection_reference = 0;
    double best = -1e100;
    for (size_t row = 0; row < ranked_normal.size(); ++row) {
      // These are uncalibrated support points, not a probability distribution.
      if (scores[row][0] > best) {
        best = scores[row][0];
        selection_reference = row;
      }
    }
    const auto canonical_at = std::find(
        selected.begin(), selected.end(), canonical_anchor);
    if (canonical_at == selected.end())
      throw std::runtime_error("canonical anchor not selected");
    const size_t canonical_row = canonical_at - selected.begin();
    for (size_t row = 0; row < selected.size(); ++row) {
      const size_t at = row * kTeacherRowWidth;
      const auto family = candidate_family(h.prepared_names[selected[row]]);
      output[at] = selected[row];
      output[at + 1] = h.prepared[selected[row]].id;
      output[at + 2] = family;
      output[at + 3] = family == kOuter;
      output[at + 4] = h.prepared_scores[selected[row]];
      output[at + 5] = canonical_anchor;
      output[at + 6] = selected[selection_reference];
      for (int scenario = 0; scenario < kTeacherScenarios; ++scenario) {
        output[at + 7 + scenario] = scores[row][scenario];
        output[at + 7 + kTeacherScenarios + scenario] =
            scores[row][scenario] - scores[canonical_row][scenario];
        output[at + 7 + 2 * kTeacherScenarios + scenario] =
            scores[row][scenario] - scores[selection_reference][scenario];
      }
    }
    return int(selected.size());
  } catch (const std::exception& error) {
    h.text = std::string("ERROR: ") + error.what();
    return -1;
  }
#endif
}

// One process currently owns one warm history.  A two-handle warm check crashed
// even without OpenMP, so fail closed instead of exposing unsafe state batching.
// Candidate/scenario work still crosses the ABI once; independent states use
// the repository's existing process-level scheduler.
extern "C" int td_teacher_batch_many(
    void* const* handles, int batch, const double* packed,
    const size_t* offsets, int top_k, double* output, size_t capacity,
    int32_t* row_counts) {
  if (!handles || !packed || !offsets || !output || !row_counts || batch != 1 ||
      top_k < 1 || top_k > 64)
    return -1;
  const size_t stride = size_t((2 * top_k + 1) * kTeacherRowWidth);
  if (capacity < size_t(batch) * stride) return -2;
  row_counts[0] = td_teacher_batch_observation(
      handles[0], packed + offsets[0], offsets[1] - offsets[0], top_k,
      output, stride);
  return row_counts[0] < 0 ? row_counts[0] : 1;
}
