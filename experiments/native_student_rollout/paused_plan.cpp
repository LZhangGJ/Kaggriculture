#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "prefix_batch.hpp"

#include <dlfcn.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int kContextWidth = 2233;
constexpr int kResourceWidth = 347;
constexpr int kClassCount = 11;
constexpr std::size_t kDefaultStackBytes = 2u << 20;
constexpr std::size_t kMaxSessions = 256;

using Callback = int (*)(void*, int32_t, int32_t, int32_t,
                         const double*, std::size_t);

class Runtime {
 public:
  explicit Runtime(const std::string& path) {
    library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error("dlopen: " + dl_error());
    try {
      activate = symbol<Activate>("td_activate_external");
      pre_context = symbol<PreContext>("td_student_pre_context_observation");
      plan = symbol<Plan>("td_student_plan_v3_callback_observation");
      install = symbol<Install>("td_student_install_prepared");
      debug = symbol<Debug>("td_debug");
      abi_version = symbol<AbiVersion>("td_student_slot_abi_version");
      if (abi_version() != 3)
        throw std::runtime_error("student slot ABI v3 is required");
    } catch (...) {
      dlclose(library_);
      library_ = nullptr;
      throw;
    }
  }

  ~Runtime() {
    if (library_) dlclose(library_);
  }

  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  using Activate = int (*)(void*, const double*, std::size_t);
  using PreContext = int (*)(void*, const double*, std::size_t, double*,
                             std::size_t);
  using Plan = int (*)(void*, const double*, std::size_t, Callback, Callback,
                       void*, const double*, std::size_t);
  using Install = int (*)(void*);
  using Debug = const char* (*)(void*);
  using AbiVersion = int (*)();

  Activate activate{};
  PreContext pre_context{};
  Plan plan{};
  Install install{};
  Debug debug{};
  AbiVersion abi_version{};

 private:
  static std::string dl_error() {
    const char* message = dlerror();
    return message ? message : "unknown dynamic-loader error";
  }

  template <class T>
  T symbol(const char* name) {
    dlerror();
    void* value = dlsym(library_, name);
    if (const char* error = dlerror())
      throw std::runtime_error(std::string("dlsym ") + name + ": " + error);
    return reinterpret_cast<T>(value);
  }

  void* library_{};
};

struct Event {
  int stage{};
  int cell{};
  int suggested{};
  int legal_mask{};
  std::uint64_t sequence{};
  std::vector<double> resources;
};

enum class State { kCreated, kRunning, kWaiting, kDone, kFailed, kCancelled };

class Batch;

class Session {
 public:
  Session(Batch* owner, std::shared_ptr<Runtime> runtime, void* handle,
          std::vector<double> packed, bool activate_external)
      : owner_(owner), runtime_(std::move(runtime)), handle_(handle),
        packed_(std::move(packed)), activate_external_(activate_external) {
    if (!handle_) throw std::invalid_argument("null R1 handle");
    if (packed_.empty()) throw std::invalid_argument("empty packed observation");
  }

  ~Session() { cancel_and_join(); }

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  void start(std::size_t stack_bytes) {
    std::lock_guard lock(mutex_);
    if (state_ != State::kCreated) throw std::logic_error("session already started");
    state_ = State::kRunning;
    pthread_attr_t attr;
    if (pthread_attr_init(&attr)) throw std::runtime_error("pthread_attr_init failed");
    const std::size_t minimum = std::max<std::size_t>(PTHREAD_STACK_MIN, 256u << 10);
    stack_bytes = std::max(stack_bytes, minimum);
    int error = pthread_attr_setstacksize(&attr, stack_bytes);
    if (!error) error = pthread_create(&thread_, &attr, &Session::thread_entry, this);
    pthread_attr_destroy(&attr);
    if (error) {
      state_ = State::kFailed;
      error_ = "pthread_create/setstacksize failed: " + std::to_string(error);
      throw std::runtime_error(error_);
    }
    started_ = true;
  }

  void wait_prepared() const {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return context_ready_ || terminal_unlocked(); });
    if (!context_ready_)
      throw std::runtime_error(error_.empty() ? "session prepare failed" : error_);
  }

  const std::vector<double>& context() const { return context_; }

  State inspect(Event& output) const {
    std::lock_guard lock(mutex_);
    if (state_ == State::kWaiting) output = event_;
    return state_;
  }

  State state() const {
    std::lock_guard lock(mutex_);
    return state_;
  }

  bool terminal() const {
    const State value = state();
    return value == State::kDone || value == State::kFailed ||
           value == State::kCancelled;
  }

  void validate_action(int action) const {
    std::lock_guard lock(mutex_);
    if (state_ != State::kWaiting) throw std::logic_error("session is not waiting");
    if (action < 0 || action >= kClassCount ||
        !(event_.legal_mask & (1 << action)))
      throw std::invalid_argument("action is outside the published legal mask");
  }

  void apply(int action) {
    std::lock_guard lock(mutex_);
    if (state_ != State::kWaiting) throw std::logic_error("session is not waiting");
    response_ = action;
    response_ready_ = true;
    state_ = State::kRunning;
    cv_.notify_one();
  }

  int result_count() const {
    std::lock_guard lock(mutex_);
    return result_count_;
  }

  std::string error() const {
    std::lock_guard lock(mutex_);
    return error_;
  }

  void join_terminal() {
    {
      std::lock_guard lock(mutex_);
      if (state_ == State::kWaiting || state_ == State::kRunning)
        throw std::logic_error("cannot join a non-terminal session");
    }
    join();
  }

  void cancel_and_join() noexcept {
    {
      std::lock_guard lock(mutex_);
      if (!terminal_unlocked()) {
        cancel_ = true;
        response_ready_ = true;
        response_ = -1000;
        cv_.notify_one();
      }
    }
    join();
  }

 private:
  static void* thread_entry(void* self) noexcept {
    static_cast<Session*>(self)->run();
    return nullptr;
  }

  static int release_callback(void* user, int32_t cell, int32_t suggested,
                              int32_t mask, const double* resources,
                              std::size_t width) {
    return static_cast<Session*>(user)->publish(
        0, cell, suggested, mask, resources, width);
  }

  static int slot_callback(void* user, int32_t cell, int32_t suggested,
                           int32_t mask, const double* resources,
                           std::size_t width) {
    return static_cast<Session*>(user)->publish(
        1, cell, suggested, mask, resources, width);
  }

  int publish(int stage, int cell, int suggested, int mask,
              const double* resources, std::size_t width);

  void run() noexcept;

  void prepare() {
    if (activate_external_ &&
        runtime_->activate(handle_, packed_.data(), packed_.size()))
      throw std::runtime_error(debug_text("activate external failed"));
    context_.resize(kContextWidth);
    const int width = runtime_->pre_context(
        handle_, packed_.data(), packed_.size(), context_.data(), context_.size());
    if (width != kContextWidth)
      throw std::runtime_error(debug_text("pre-context failed"));
  }

  std::string debug_text(const char* prefix) const {
    const char* detail = runtime_->debug(handle_);
    return std::string(prefix) + (detail ? std::string(": ") + detail : "");
  }

  bool terminal_unlocked() const {
    return state_ == State::kDone || state_ == State::kFailed ||
           state_ == State::kCancelled;
  }

  void join() noexcept {
    if (started_ && !joined_) {
      pthread_join(thread_, nullptr);
      joined_ = true;
    }
  }

  Batch* owner_{};
  std::shared_ptr<Runtime> runtime_;
  void* handle_{};
  std::vector<double> packed_;
  std::vector<double> context_;
  bool activate_external_{};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  State state_{State::kCreated};
  Event event_;
  std::uint64_t next_sequence_{};
  bool response_ready_{};
  bool context_ready_{};
  int response_{};
  bool cancel_{};
  int result_count_{-1};
  std::string error_;
  pthread_t thread_{};
  bool started_{};
  bool joined_{};
};

class Batch {
 public:
  Batch(const std::string& library_path, const std::vector<std::uintptr_t>& handles,
        py::sequence packed, bool activate_external,
        std::size_t stack_bytes = kDefaultStackBytes)
      : runtime_(std::make_shared<Runtime>(library_path)),
        stack_bytes_(stack_bytes) {
    if (handles.empty() || handles.size() > kMaxSessions)
      throw std::invalid_argument("session count must be in [1,256]");
    if (py::len(packed) != static_cast<py::ssize_t>(handles.size()))
      throw std::invalid_argument("one packed observation is required per handle");
    sessions_.reserve(handles.size());
    for (std::size_t i = 0; i < handles.size(); ++i) {
      auto array = py::array_t<double, py::array::c_style | py::array::forcecast>(
          packed[py::int_(i)]);
      if (array.ndim() != 1) throw std::invalid_argument("packed observation must be 1-D");
      const double* begin = array.data();
      std::vector<double> values(begin, begin + array.size());
      sessions_.push_back(std::make_unique<Session>(
          this, runtime_, reinterpret_cast<void*>(handles[i]), std::move(values),
          activate_external));
    }
    try {
      for (auto& session : sessions_) session->start(stack_bytes_);
      {
        py::gil_scoped_release release;
        for (const auto& session : sessions_) session->wait_prepared();
      }
    } catch (...) {
      close();
      throw;
    }
  }

  ~Batch() { close(); }

  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;

  py::array_t<double> contexts() const {
    py::array_t<double> output({static_cast<py::ssize_t>(sessions_.size()),
                                static_cast<py::ssize_t>(kContextWidth)});
    auto view = output.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(sessions_.size()); ++i)
      for (py::ssize_t j = 0; j < kContextWidth; ++j)
        view(i, j) = sessions_[i]->context()[j];
    return output;
  }

  py::dict collect_ready(int timeout_ms, bool barrier) {
    if (timeout_ms < -1) throw std::invalid_argument("timeout_ms must be -1 or nonnegative");
    std::vector<std::pair<int, Event>> ready;
    const auto deadline = timeout_ms < 0
        ? std::chrono::steady_clock::time_point::max()
        : std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
      std::uint64_t seen;
      {
        std::lock_guard lock(notify_mutex_);
        seen = notify_epoch_;
      }
      ready.clear();
      bool all_terminal = true, all_quiescent = true;
      for (std::size_t i = 0; i < sessions_.size(); ++i) {
        Event event;
        const State state = sessions_[i]->inspect(event);
        if (state == State::kWaiting)
          ready.emplace_back(int(i), std::move(event));
        const bool terminal = state == State::kDone || state == State::kFailed ||
                              state == State::kCancelled;
        if (!terminal) all_terminal = false;
        if (!terminal && state != State::kWaiting) all_quiescent = false;
      }
      if ((!ready.empty() && (!barrier || all_quiescent)) || all_terminal ||
          timeout_ms == 0)
        break;
      {
        py::gil_scoped_release release;
        std::unique_lock lock(notify_mutex_);
        if (timeout_ms < 0) {
          notify_cv_.wait(lock, [&] { return notify_epoch_ != seen; });
        } else if (!notify_cv_.wait_until(
                       lock, deadline, [&] { return notify_epoch_ != seen; })) {
          break;
        }
      }
    }
    return ready_dict(ready);
  }

  void apply(py::array_t<int32_t, py::array::c_style | py::array::forcecast> indices,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> actions) {
    if (indices.ndim() != 1 || actions.ndim() != 1 || indices.size() != actions.size())
      throw std::invalid_argument("indices/actions must be equal-length 1-D arrays");
    auto ii = indices.unchecked<1>();
    auto aa = actions.unchecked<1>();
    std::unordered_set<int> unique;
    for (py::ssize_t row = 0; row < indices.size(); ++row) {
      const int index = ii(row);
      if (index < 0 || index >= static_cast<int>(sessions_.size()) ||
          !unique.insert(index).second)
        throw std::invalid_argument("invalid or duplicate session index");
      sessions_[index]->validate_action(aa(row));
    }
    for (py::ssize_t row = 0; row < indices.size(); ++row)
      sessions_[ii(row)]->apply(aa(row));
  }

  py::dict summary() const {
    py::list states, errors, counts;
    for (const auto& session : sessions_) {
      states.append(state_name(session->state()));
      errors.append(session->error());
      counts.append(session->result_count());
    }
    py::dict result;
    result["states"] = std::move(states);
    result["errors"] = std::move(errors);
    result["event_counts"] = std::move(counts);
    result["stack_bytes"] = py::int_(stack_bytes_);
    return result;
  }

  void join() {
    for (const auto& session : sessions_) session->join_terminal();
  }

  void close() noexcept {
    if (closed_) return;
    for (auto& session : sessions_) session->cancel_and_join();
    closed_ = true;
  }

  void notify() {
    {
      std::lock_guard lock(notify_mutex_);
      ++notify_epoch_;
    }
    notify_cv_.notify_all();
  }

 private:
  static const char* state_name(State state) {
    switch (state) {
      case State::kCreated: return "created";
      case State::kRunning: return "running";
      case State::kWaiting: return "waiting";
      case State::kDone: return "done";
      case State::kFailed: return "failed";
      case State::kCancelled: return "cancelled";
    }
    return "unknown";
  }

  py::dict ready_dict(const std::vector<std::pair<int, Event>>& ready) const {
    const py::ssize_t count = ready.size();
    py::array_t<int32_t> indices(count), stages(count), cells(count),
        suggested(count), masks(count);
    py::array_t<std::uint64_t> sequences(count);
    py::array_t<double> resources({count, static_cast<py::ssize_t>(kResourceWidth)});
    auto oi = indices.mutable_unchecked<1>();
    auto os = stages.mutable_unchecked<1>();
    auto oc = cells.mutable_unchecked<1>();
    auto og = suggested.mutable_unchecked<1>();
    auto om = masks.mutable_unchecked<1>();
    auto oq = sequences.mutable_unchecked<1>();
    auto ores = resources.mutable_unchecked<2>();
    for (py::ssize_t row = 0; row < count; ++row) {
      const auto& [index, event] = ready[row];
      oi(row) = index;
      os(row) = event.stage;
      oc(row) = event.cell;
      og(row) = event.suggested;
      om(row) = event.legal_mask;
      oq(row) = event.sequence;
      for (py::ssize_t column = 0; column < kResourceWidth; ++column)
        ores(row, column) = event.resources[column];
    }
    py::array_t<std::uint8_t> terminal(sessions_.size());
    auto ot = terminal.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(sessions_.size()); ++i)
      ot(i) = sessions_[i]->terminal();
    py::dict result;
    result["session_indices"] = std::move(indices);
    result["stages"] = std::move(stages);
    result["cells"] = std::move(cells);
    result["suggested"] = std::move(suggested);
    result["legal_masks"] = std::move(masks);
    result["sequences"] = std::move(sequences);
    result["resources"] = std::move(resources);
    result["terminal"] = std::move(terminal);
    return result;
  }

  std::shared_ptr<Runtime> runtime_;
  std::vector<std::unique_ptr<Session>> sessions_;
  std::size_t stack_bytes_{};
  mutable std::mutex notify_mutex_;
  std::condition_variable notify_cv_;
  std::uint64_t notify_epoch_{};
  bool closed_{};
};

int Session::publish(int stage, int cell, int suggested, int mask,
                     const double* resources, std::size_t width) {
  std::unique_lock lock(mutex_);
  if (cancel_) return -1000;
  const bool valid_stage_mask =
      (stage == 0 && (mask & ~0b110) == 0 && (mask & 0b110)) ||
      (stage == 1 && !(mask & (1 << 2)) && (mask & (1 << 1)));
  if (cell < 0 || cell >= 100 || suggested < 0 || suggested >= kClassCount ||
      !mask || (mask >> kClassCount) || !(mask & (1 << suggested)) ||
      width != kResourceWidth || !resources || !valid_stage_mask) {
    error_ = "invalid callback event";
    return -1000;
  }
  event_ = Event{stage, cell, suggested, mask, next_sequence_++,
                 std::vector<double>(resources, resources + width)};
  response_ready_ = false;
  state_ = State::kWaiting;
  owner_->notify();
  cv_.wait(lock, [&] { return response_ready_ || cancel_; });
  if (cancel_) return -1000;
  const int result = response_;
  response_ready_ = false;
  return result;
}

void Session::run() noexcept {
  try {
    prepare();
    bool cancelled = false;
    {
      std::lock_guard lock(mutex_);
      context_ready_ = true;
      cv_.notify_all();
      if (cancel_) {
        state_ = State::kCancelled;
        cancelled = true;
      }
    }
    owner_->notify();
    if (cancelled) return;
  } catch (const std::exception& error) {
    {
      std::lock_guard lock(mutex_);
      error_ = error.what();
      state_ = cancel_ ? State::kCancelled : State::kFailed;
      cv_.notify_all();
    }
    owner_->notify();
    return;
  } catch (...) {
    {
      std::lock_guard lock(mutex_);
      error_ = "unknown session prepare error";
      state_ = cancel_ ? State::kCancelled : State::kFailed;
      cv_.notify_all();
    }
    owner_->notify();
    return;
  }
  const int count = runtime_->plan(
      handle_, packed_.data(), packed_.size(), &Session::release_callback,
      &Session::slot_callback, this, nullptr, 0);
  bool cancelled;
  {
    std::lock_guard lock(mutex_);
    cancelled = cancel_;
  }
  int installed = -1;
  if (!cancelled && count >= 0) installed = runtime_->install(handle_);
  {
    std::lock_guard lock(mutex_);
    result_count_ = count;
    if (cancelled) {
      state_ = State::kCancelled;
    } else if (count < 0 || installed) {
      state_ = State::kFailed;
      if (error_.empty()) error_ = debug_text(count < 0 ? "plan failed" : "install failed");
    } else {
      state_ = State::kDone;
    }
  }
  owner_->notify();
}

}  // namespace

PYBIND11_MODULE(_paused_plan, module) {
  module.doc() = "Batched paused-callback seam for the native v3 student planner";
  py::class_<Batch>(module, "PlanBatch")
      .def(py::init<const std::string&, const std::vector<std::uintptr_t>&,
                    py::sequence, bool, std::size_t>(),
           py::arg("library_path"), py::arg("handles"), py::arg("packed"),
           py::arg("activate_external") = true,
           py::arg("stack_bytes") = kDefaultStackBytes)
      .def_property_readonly("contexts", &Batch::contexts)
      .def("collect_ready", &Batch::collect_ready,
           py::arg("timeout_ms") = 0, py::arg("barrier") = true)
      .def("apply", &Batch::apply, py::arg("session_indices"),
           py::arg("actions"))
      .def("summary", &Batch::summary)
      .def("join", &Batch::join)
      .def("close", &Batch::close);
  module.attr("CONTEXT_WIDTH") = kContextWidth;
  module.attr("RESOURCE_WIDTH") = kResourceWidth;
  module.attr("SHOP_RESOURCE_SEMANTICS") = 1;
  module.attr("CLASS_COUNT") = kClassCount;
  module.attr("MAX_SESSIONS") = kMaxSessions;
#ifdef ECONOMIC_FEATURES_MATURE_STORED
  module.attr("ECONOMIC_FEATURES_SEMANTICS") = 2;
#else
  module.attr("ECONOMIC_FEATURES_SEMANTICS") = 1;
#endif
#ifdef STUDENT_EXPLICIT_SHOP_TOKENS
#ifndef SHOP_TOKEN_GAIN
#define SHOP_TOKEN_GAIN 1
#endif
  module.attr("SHOP_TOKEN_SEMANTICS") = 2;
  module.attr("SHOP_TOKEN_GAIN") = float(SHOP_TOKEN_GAIN);
#else
  module.attr("SHOP_TOKEN_SEMANTICS") = 1;
  module.attr("SHOP_TOKEN_GAIN") = 1.0f;
#endif
  bind_prefix_batch(module);
}
