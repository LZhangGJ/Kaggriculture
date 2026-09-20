#include "joint_fixed_move_oracle.hpp"
#include "nt_trace_bank.hpp"
#include "reactive_production.hpp"
#include "route_loader.hpp"
#include "weed_absorption.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace jfmo = joint_fixed_move_oracle;
namespace prod = joint_fixed_move_oracle::production;
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using jfmo::Tape;

namespace {

enum class Mode { Legacy, RawFixed, LocalV3, ReactiveCursor };

struct Options {
  std::uint64_t seed_begin{991001};
  int seeds{128};
  int threads{static_cast<int>(std::max(
      1u, std::thread::hardware_concurrency()))};
  std::string output{"artifacts/weed-reactive-v1/report.json"};
  std::string tapes{JFMO_TAPES};
  std::string library{JFMO_LIBRARY};
  std::string nt_root{JFMO_NT_ROOT};
};

struct Opponent {
  std::string name;
  const Tape* tape{};
  const public_ports::NtTraceBankAgent* native{};
};
struct OpponentState { public_ports::NtTraceBankState native; };
struct WeedTx { bool active{}; int start{-1}; Action intended{}; };
struct LegacyState { std::vector<WeedTx> actors; };
struct Event { int step{}; int actor{}; fastkag::Position position{}; Action intended{}; };
struct Trace {
  std::vector<std::vector<fastkag::Position>> positions;
  std::vector<Event> weeds;
};

struct Metrics {
  double own{};
  double opponent{};
  int failures{};
  int actual_weed_episodes{};
  int legacy_triggers{};
  int legacy_drop_move_risks{};
  int absolute_move_mismatches{};
  int ordered_move_day_failures{};
  int missing_or_extra_moves{};
  prod::RouteCursorAudit reactive{};
  int local_patches{};
  Mode mode{Mode::Legacy};
  int score() const { return own > opponent ? 2 : own == opponent ? 1 : 0; }
  double margin() const { return own - opponent; }
};

struct Row {
  std::string opponent;
  std::uint64_t seed{};
  int seat{};
  std::array<Metrics, 5> arms{};  // legacy, raw, local, reactive, best-four
};

struct Job { int opponent{}; std::uint64_t seed{}; int seat{}; };

const char* mode_name(int arm) {
  constexpr std::array<const char*, 5> names{
      "legacy", "raw-fixed", "local-v3", "full-reactive", "best-four"};
  return names[static_cast<std::size_t>(arm)];
}

std::string escape(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (const char c : value) { if (c == '"' || c == '\\') out << '\\'; out << c; }
  out << '"';
  return out.str();
}

const PlayerAction& frame(const Tape& tape, int step) {
  if (tape.empty()) throw std::runtime_error("empty tape");
  return tape[std::min<std::size_t>(static_cast<std::size_t>(step), tape.size()-1)];
}

std::vector<fastkag::Position> positions(const fastkag::Farm& farm) {
  std::vector<fastkag::Position> out{farm.farmer};
  out.insert(out.end(), farm.hands.begin(), farm.hands.end());
  return out;
}

const fastkag::Tile* tile_at(const fastkag::Simulator& sim, int player,
                             fastkag::Position position) {
  if (position.x < 0 || position.y < 0 ||
      position.x >= sim.config().board_size ||
      position.y >= sim.config().board_size) return nullptr;
  return &sim.farms()[player].tiles[static_cast<std::size_t>(
      position.y * sim.config().board_size + position.x)];
}

PlayerAction apply_legacy(PlayerAction out, const Tape& tape,
                          const fastkag::Simulator& sim, int player,
                          LegacyState& state, Metrics& metrics) {
  const int step = sim.step_count();
  const auto pos = positions(sim.farms()[player]);
  state.actors.resize(out.units.size());
  int hires = 0;
  for (int ahead=0; ahead<3; ++ahead)
    hires += std::count_if(frame(tape, std::min(step+ahead,
        static_cast<int>(tape.size())-1)).market.begin(),
        frame(tape, std::min(step+ahead, static_cast<int>(tape.size())-1)).market.end(),
        [](const Action& value){ return value.op == Op::HIRE; });
  const bool farmer_barrier = hires >= 5;
  for (std::size_t actor=0; actor<state.actors.size(); ++actor) {
    auto& tx = state.actors[actor];
    if (farmer_barrier && actor==0) { tx.active=false; continue; }
    if (!tx.active) continue;
    const int age=step-tx.start;
    if (age==1) out.units[actor]=tx.intended;
    else if (age>=2 && age<=9) {
      const auto& previous=frame(tape,std::max(0,step-1));
      out.units[actor]=actor<previous.units.size()?previous.units[actor]:Action{};
    } else tx.active=false;
  }
  for (std::size_t actor=0; actor<out.units.size() && actor<pos.size(); ++actor) {
    auto& tx=state.actors[actor];
    if (tx.active || (farmer_barrier && actor==0)) continue;
    const auto op=out.units[actor].op;
    if (op!=Op::PLANT && op!=Op::BUILD_PASTURE) continue;
    const auto* tile=tile_at(sim,player,pos[actor]);
    if (!tile || tile->kind!=fastkag::TileKind::WEED) continue;
    tx={true,step,out.units[actor]}; out.units[actor]={Op::DIG};
    ++metrics.legacy_triggers;
    const auto& dropped=frame(tape,std::min(step+9,static_cast<int>(tape.size())-1));
    if (actor<dropped.units.size() && jfmo::is_move(dropped.units[actor].op))
      ++metrics.legacy_drop_move_risks;
  }
  return out;
}

PlayerAction opponent_action(const Opponent& opponent, OpponentState& state,
                             const fastkag::Simulator& sim, int seat) {
  if (opponent.tape) return frame(*opponent.tape,sim.step_count());
  return opponent.native->action(sim,seat,state.native);
}

void certify_route(const Tape& baseline, const Tape& emitted,
                   const std::vector<std::size_t>& active_actors,
                   Metrics& value, int turns_per_day) {
  for (std::size_t step=0; step<emitted.size() && step<baseline.size(); ++step) {
    const std::size_t actors=step<active_actors.size()?active_actors[step]:0;
    for (std::size_t actor=0; actor<actors; ++actor) {
      const Action expected=actor<baseline[step].units.size()?baseline[step].units[actor]:Action{};
      const Action actual=actor<emitted[step].units.size()?emitted[step].units[actor]:Action{};
      if ((jfmo::is_move(expected.op)||jfmo::is_move(actual.op)) &&
          !jfmo::action_equal(expected,actual)) ++value.absolute_move_mismatches;
    }
  }
  const int days=(static_cast<int>(emitted.size())+turns_per_day-1)/turns_per_day;
  for (int day=0; day<days; ++day) {
    const int begin=day*turns_per_day;
    const int end=std::min(begin+turns_per_day,static_cast<int>(emitted.size()));
    std::size_t actors=0;
    for (int step=begin; step<end && step<static_cast<int>(active_actors.size()); ++step)
      actors=std::max(actors,active_actors[step]);
    for (std::size_t actor=0; actor<actors; ++actor) {
      std::vector<Op> expected,actual;
      int first_active=end;
      for(int step=begin;step<end && step<static_cast<int>(active_actors.size());++step)
        if(actor<active_actors[step]){first_active=step;break;}
      for (int step=first_active; step<end; ++step) {
        if(step>=static_cast<int>(active_actors.size())||actor>=active_actors[step])continue;
        if (actor<baseline[step].units.size() && jfmo::is_move(baseline[step].units[actor].op))
          expected.push_back(baseline[step].units[actor].op);
        if (actor<emitted[step].units.size() && jfmo::is_move(emitted[step].units[actor].op))
          actual.push_back(emitted[step].units[actor].op);
      }
      if (expected!=actual) {
        ++value.ordered_move_day_failures;
        value.missing_or_extra_moves += std::abs(static_cast<int>(expected.size())-
                                                 static_cast<int>(actual.size()));
      }
    }
  }
}

Metrics run_game(const Tape& baseline, const Tape& candidate, Mode mode,
                 const Opponent& opponent, std::uint64_t seed, int seat,
                 Trace* trace=nullptr, int local_patches=0,
                 const prod::ReactiveOptions& reactive_options={}) {
  if (mode != Mode::ReactiveCursor) jfmo::require_absolute_moves(baseline,candidate);
  fastkag::Simulator sim({},seed); OpponentState opponent_state; LegacyState legacy;
  prod::RouteCursorState cursor; Metrics result; result.mode=mode;
  result.local_patches=local_patches; Tape emitted;
  std::vector<std::size_t> active_actors;
  std::vector<bool> was_weed;
  if (trace) trace->positions.resize(static_cast<std::size_t>(sim.config().episode_steps));
  while (!sim.done()) {
    const int step=sim.step_count(); const auto& raw=frame(candidate,step);
    const auto& own_tiles=sim.farms()[seat].tiles;
    if (was_weed.size()!=own_tiles.size()) was_weed.assign(own_tiles.size(),false);
    for(std::size_t tile=0;tile<own_tiles.size();++tile) {
      const bool weed=own_tiles[tile].kind==fastkag::TileKind::WEED;
      if(weed&&!was_weed[tile]) ++result.actual_weed_episodes;
      was_weed[tile]=weed;
    }
    auto own = mode==Mode::ReactiveCursor
        ? prod::apply_reactive_route_cursor(candidate,sim,seat,cursor,
                                             result.reactive,reactive_options)
        : raw;
    if (mode==Mode::Legacy) own=apply_legacy(std::move(own),candidate,sim,seat,legacy,result);
    const auto pos=positions(sim.farms()[seat]);
    active_actors.push_back(pos.size());
    if (trace) {
      trace->positions[static_cast<std::size_t>(step)]=pos;
      for (std::size_t actor=0; actor<raw.units.size() && actor<pos.size(); ++actor) {
        const auto* tile=tile_at(sim,seat,pos[actor]);
        if (!tile || tile->kind!=fastkag::TileKind::WEED) continue;
        const auto& action=raw.units[actor];
        if (action.op==Op::PLANT || action.op==Op::BUILD_PASTURE)
          trace->weeds.push_back({step,static_cast<int>(actor),pos[actor],action});
      }
    }
    for (std::size_t actor=0; actor<own.units.size() && actor<pos.size(); ++actor)
      if (own.units[actor].op!=Op::PASS &&
          !g001::weed_absorption::projected_unit_action_succeeds(
              sim,seat,own,static_cast<int>(actor))) ++result.failures;
    emitted.push_back(own);
    std::array<PlayerAction,2> actions;
    actions[seat]=std::move(own);
    actions[1-seat]=opponent_action(opponent,opponent_state,sim,1-seat);
    sim.step(actions);
  }
  result.own=sim.farms()[seat].money; result.opponent=sim.farms()[1-seat].money;
  certify_route(baseline,emitted,active_actors,result,sim.config().turns_per_day);
  return result;
}

bool better(const Metrics& left,const Metrics& right) {
  return std::tuple(left.score(),left.margin(),left.own,-left.failures) >
         std::tuple(right.score(),right.margin(),right.own,-right.failures);
}

std::vector<Tape> local_candidates(const Tape& baseline,const Trace& trace) {
  std::vector<Tape> out;
  for (const auto& event:trace.weeds) {
    auto patches=jfmo::enumerate_absolute_weed_patches(
        baseline,baseline,trace.positions,
        {event.step,event.actor,event.position,event.intended},24,24);
    for (auto& patch:patches) out.push_back(std::move(patch.tape));
  }
  return out;
}

std::array<Metrics,5> evaluate(const Tape& baseline,const Opponent& opponent,
                               std::uint64_t seed,int seat) {
  std::array<Metrics,5> out;
  out[0]=run_game(baseline,baseline,Mode::Legacy,opponent,seed,seat);
  Trace raw_trace;
  out[1]=run_game(baseline,baseline,Mode::RawFixed,opponent,seed,seat,&raw_trace);
  // Conservative local-v3 comparison is route-valid: raw fixed-MOVE plus
  // absolute-tick DIG/PLANT/WATER patches, never legacy fallback.
  out[2]=out[1];
  for (const auto& tape:local_candidates(baseline,raw_trace)) {
    auto value=run_game(baseline,tape,Mode::LocalV3,opponent,seed,seat,nullptr,1);
    if (better(value,out[2])) out[2]=std::move(value);
  }
  // Full reactive is a finite, terminal-selected portfolio over causal event
  // subsets and activation horizons.  Raw fixed-MOVE is its no-op member, so
  // the achievable lower bound cannot be damaged by a bad repair event.
  out[3]=out[1];
  auto try_reactive=[&](const prod::ReactiveOptions& options) {
    auto value=run_game(baseline,baseline,Mode::ReactiveCursor,opponent,seed,
                        seat,nullptr,0,options);
    const int candidates=value.reactive.production.candidate_weed_events;
    if (better(value,out[3])) out[3]=std::move(value);
    return candidates;
  };
  for (const bool old_crop : {false,true})
    for (const int cutoff : {199,399,718}) {
      prod::ReactiveOptions options;
      options.recover_old_crop_weeds=old_crop;
      options.latest_new_debt_step=cutoff;
      (void)try_reactive(options);
    }
  for (const int cutoff : {199,299,399,519,718}) {
    prod::ReactiveOptions options;
    options.recover_old_crop_weeds=true;
    options.manage_all_known_crops=true;
    options.latest_new_debt_step=cutoff;
    (void)try_reactive(options);
  }
  prod::ReactiveOptions monitor;
  monitor.allowed_event_ordinal=1'000'000;
  const int candidate_events=try_reactive(monitor);
  for(int event=0;event<std::min(128,candidate_events);++event) {
    prod::ReactiveOptions options;
    options.allowed_event_ordinal=event;
    (void)try_reactive(options);
  }
  for(const int gap : {1,2,4,8,16,32})
    for(int event=0;event+gap<std::min(64,candidate_events);++event) {
      prod::ReactiveOptions options;
      options.allowed_event_ordinal=event;
      options.allowed_event_ordinal_second=event+gap;
      (void)try_reactive(options);
    }
  // A two-event-only oracle can understate recovery when DIG/PLANT/WATER debt
  // on several different fields must be repaired together.  Probe a bounded
  // family of evenly spaced triples; this remains a finite, reproducible
  // achievable set rather than an unconstrained theoretical optimum.
  for(const int gap : {1,2,4,8,16})
    for(int event=0;event+2*gap<std::min(96,candidate_events);++event) {
      prod::ReactiveOptions options;
      options.allowed_event_ordinal=event;
      options.allowed_event_ordinal_second=event+gap;
      options.allowed_event_ordinal_third=event+2*gap;
      (void)try_reactive(options);
    }
  // Preserve the raw-trajectory eligible-event count even when the terminal
  // selector chooses the no-op raw member or a sparse event subset.
  out[3].reactive.production.candidate_weed_events=candidate_events;
  out[4]=out[0];
  for (int arm=1;arm<4;++arm) if (better(out[arm],out[4])) out[4]=out[arm];
  return out;
}

struct Aggregate {
  int games{}, losses{}, affected_losses{}, repair_opportunity_losses{}, legacy_affected_losses{};
  int loss_to_win{}, affected_loss_to_win{}, repair_opportunity_loss_to_win{}, legacy_affected_loss_to_win{};
  int baseline_wins{}, wins_retained{}, ordered_failures{}, absolute_mismatches{};
  int legacy_triggers{}, actual_weed_episodes{}, weed_events{}, candidate_weed_events{};
  double own_delta{},margin_delta{},score{};
};

Aggregate aggregate(const std::vector<Row>& rows,std::string_view opponent,int arm) {
  Aggregate out;
  for (const auto& row:rows) {
    if (!opponent.empty() && row.opponent!=opponent) continue;
    const auto& base=row.arms[0]; const auto& value=row.arms[arm]; ++out.games;
    const bool loss=base.score()==0;
    const bool affected=loss&&base.actual_weed_episodes>0;
    const bool repair_opportunity=loss&&row.arms[3].reactive.production.candidate_weed_events>0;
    const bool legacy_affected=loss&&base.legacy_triggers>0;
    out.losses+=loss; out.affected_losses+=affected;
    out.repair_opportunity_losses+=repair_opportunity;
    out.legacy_affected_losses+=legacy_affected;
    out.loss_to_win+=loss&&value.score()==2;
    out.affected_loss_to_win+=affected&&value.score()==2;
    out.repair_opportunity_loss_to_win+=repair_opportunity&&value.score()==2;
    out.legacy_affected_loss_to_win+=legacy_affected&&value.score()==2;
    out.baseline_wins+=base.score()==2;
    out.wins_retained+=base.score()==2&&value.score()==2;
    out.ordered_failures+=value.ordered_move_day_failures;
    out.absolute_mismatches+=value.absolute_move_mismatches;
    out.legacy_triggers+=value.legacy_triggers;
    out.actual_weed_episodes+=value.actual_weed_episodes;
    out.weed_events+=value.reactive.production.weed_events;
    out.candidate_weed_events+=value.reactive.production.candidate_weed_events;
    out.own_delta+=value.own-base.own; out.margin_delta+=value.margin()-base.margin();
    out.score+=value.score()/2.0;
  }
  return out;
}

Options parse(int argc,char** argv) {
  Options out;
  for (int i=1;i<argc;++i) {
    const std::string arg=argv[i]; auto next=[&]{if(++i>=argc)throw std::invalid_argument("missing "+arg);return std::string(argv[i]);};
    if(arg=="--seed-begin")out.seed_begin=std::stoull(next());
    else if(arg=="--seeds")out.seeds=std::stoi(next());
    else if(arg=="--threads")out.threads=std::stoi(next());
    else if(arg=="--output")out.output=next();
    else if(arg=="--tapes")out.tapes=next();
    else if(arg=="--library")out.library=next();
    else if(arg=="--nt-root")out.nt_root=next();
    else throw std::invalid_argument("unknown option "+arg);
  }
  if(out.seeds<=0||out.threads<=0)throw std::invalid_argument("positive seeds/threads required");
  for(int i=0;i<out.seeds;++i)if(out.seed_begin+static_cast<std::uint64_t>(i)==976)throw std::invalid_argument("seed 976 forbidden");
  return out;
}

std::string bank(std::string_view name) {
  return name=="hasegawa"?"hasegawa_current_trace_bank_v1.npz":"rank04_arman_trace_bank_v1.npz";
}

}  // namespace

int main(int argc,char** argv) try {
  const auto options=parse(argc,argv); const auto started=std::chrono::steady_clock::now();
  const auto baseline=g001::repair::load_route(options.tapes,options.library,"G001");
  const auto g096=g001::repair::load_route(options.tapes,options.library,"G096");
  const public_ports::NtTraceBankAgent hasegawa(
      (std::filesystem::path(options.nt_root)/bank("hasegawa")).string(),
      public_ports::router_for_slug("hasegawa_current"));
  const public_ports::NtTraceBankAgent rank04(
      (std::filesystem::path(options.nt_root)/bank("rank04")).string(),
      public_ports::router_for_slug("rank04_arman"));
  const std::array<Opponent,4> opponents{{
      {"G001",&baseline,nullptr},{"G096",&g096,nullptr},
      {"hasegawa",nullptr,&hasegawa},{"rank04",nullptr,&rank04}}};
  std::vector<Job> jobs; for(int o=0;o<4;++o)for(int s=0;s<options.seeds;++s)for(int seat=0;seat<2;++seat)
    jobs.push_back({o,options.seed_begin+static_cast<std::uint64_t>(s),seat});
  std::vector<Row> rows(jobs.size()); std::atomic<std::size_t> cursor{}; std::mutex mutex;
  std::exception_ptr failure; std::vector<std::thread> workers;
  for(int worker=0;worker<std::min<int>(options.threads,jobs.size());++worker)workers.emplace_back([&]{try{while(true){const auto i=cursor.fetch_add(1);if(i>=jobs.size())break;const auto& j=jobs[i];rows[i]={opponents[j.opponent].name,j.seed,j.seat,evaluate(baseline,opponents[j.opponent],j.seed,j.seat)};}}catch(...){std::lock_guard lock(mutex);if(!failure)failure=std::current_exception();cursor.store(jobs.size());}});
  for(auto& worker:workers) worker.join();
  if(failure) std::rethrow_exception(failure);
  const auto parent=std::filesystem::path(options.output).parent_path();if(!parent.empty())std::filesystem::create_directories(parent);
  std::ofstream report(options.output); report<<std::fixed<<std::setprecision(6);
  report<<"{\n  \"schema\": \"weed-reactive-route-cursor-v1\",\n  \"bound\": \"finite-oracle-achievable-lower-bound\",\n  \"route_contract\": \"per-actor per-day ordered MOVE subsequence; production may time-dilate within day; all daily MOVE must close before midnight\",\n  \"seed_begin\": "<<options.seed_begin<<",\n  \"seeds_per_opponent\": "<<options.seeds<<",\n  \"arms\": {\n";
  for(int arm=0;arm<5;++arm) {
    report<<"    "<<escape(mode_name(arm))<<": {\n";
    for(std::size_t o=0;o<opponents.size();++o) {
      const auto a=aggregate(rows,opponents[o].name,arm);
      const double n=std::max(1,a.games);
      report<<"      "<<escape(opponents[o].name)<<": {\"games\": "<<a.games
            <<", \"baseline_losses\": "<<a.losses
            <<", \"weed_affected_baseline_losses\": "<<a.affected_losses
            <<", \"repair_opportunity_baseline_losses\": "<<a.repair_opportunity_losses
            <<", \"legacy_trigger_baseline_losses\": "<<a.legacy_affected_losses
            <<", \"loss_to_win\": "<<a.loss_to_win
            <<", \"affected_loss_to_win\": "<<a.affected_loss_to_win
            <<", \"affected_flip_rate\": "
            <<double(a.affected_loss_to_win)/std::max(1,a.affected_losses)
            <<", \"repair_opportunity_loss_to_win\": "
            <<a.repair_opportunity_loss_to_win
            <<", \"legacy_affected_loss_to_win\": "
            <<a.legacy_affected_loss_to_win
            <<", \"wins_retained\": "<<a.wins_retained
            <<", \"baseline_wins\": "<<a.baseline_wins
            <<", \"own_delta_mean\": "<<a.own_delta/n
            <<", \"margin_delta_mean\": "<<a.margin_delta/n
            <<", \"score_rate\": "<<a.score/n
            <<", \"ordered_move_day_failures\": "<<a.ordered_failures
            <<", \"absolute_move_mismatches\": "<<a.absolute_mismatches
            <<", \"legacy_triggers\": "<<a.legacy_triggers
            <<", \"actual_weed_episodes\": "<<a.actual_weed_episodes
            <<", \"reactive_candidate_weed_events\": "<<a.candidate_weed_events
            <<", \"reactive_weed_events\": "<<a.weed_events<<"}"
            <<(o+1==opponents.size()?"\n":",\n");
    }
    report<<"    }"<<(arm==4?"\n":",\n");
  }
  const auto all=aggregate(rows,"",4);
  const auto full=aggregate(rows,"",3);
  const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  report<<"  },\n  \"best_four_total\": {\"games\": "<<all.games
        <<", \"baseline_losses\": "<<all.losses
        <<", \"weed_affected_baseline_losses\": "<<all.affected_losses
        <<", \"legacy_trigger_baseline_losses\": "<<all.legacy_affected_losses
        <<", \"loss_to_win\": "<<all.loss_to_win
        <<", \"affected_loss_to_win\": "<<all.affected_loss_to_win
        <<", \"affected_flip_rate\": "
        <<double(all.affected_loss_to_win)/std::max(1,all.affected_losses)
        <<", \"legacy_affected_loss_to_win\": "
        <<all.legacy_affected_loss_to_win
        <<", \"ordered_move_day_failures\": "<<all.ordered_failures
        <<"},\n  \"full_reactive_total\": {\"games\": "<<full.games
        <<", \"baseline_losses\": "<<full.losses
        <<", \"weed_affected_baseline_losses\": "<<full.affected_losses
        <<", \"repair_opportunity_baseline_losses\": "<<full.repair_opportunity_losses
        <<", \"legacy_trigger_baseline_losses\": "<<full.legacy_affected_losses
        <<", \"loss_to_win\": "<<full.loss_to_win
        <<", \"affected_loss_to_win\": "<<full.affected_loss_to_win
        <<", \"affected_flip_rate\": "
        <<double(full.affected_loss_to_win)/std::max(1,full.affected_losses)
        <<", \"repair_opportunity_loss_to_win\": "
        <<full.repair_opportunity_loss_to_win
        <<", \"legacy_affected_loss_to_win\": "
        <<full.legacy_affected_loss_to_win
        <<", \"wins_retained\": "<<full.wins_retained
        <<", \"baseline_wins\": "<<full.baseline_wins
        <<", \"ordered_move_day_failures\": "<<full.ordered_failures
        <<"},\n  \"wall_seconds\": "<<elapsed<<"\n}\n";
  std::ofstream detail(options.output+".rows.jsonl");for(const auto& row:rows){detail<<"{\"opponent\":"<<escape(row.opponent)<<",\"seed\":"<<row.seed<<",\"seat\":"<<row.seat<<",\"arms\":[";for(int arm=0;arm<5;++arm){const auto& v=row.arms[arm];detail<<"{\"name\":"<<escape(mode_name(arm))<<",\"selected_mode\":"<<static_cast<int>(v.mode)<<",\"own\":"<<v.own<<",\"opponent_money\":"<<v.opponent<<",\"margin\":"<<v.margin()<<",\"score\":"<<v.score()/2.0<<",\"actual_weed_episodes\":"<<v.actual_weed_episodes<<",\"legacy_triggers\":"<<v.legacy_triggers<<",\"legacy_drop_move_risks\":"<<v.legacy_drop_move_risks<<",\"local_patches\":"<<v.local_patches<<",\"reactive_candidate_weed_events\":"<<v.reactive.production.candidate_weed_events<<",\"reactive_weed_events\":"<<v.reactive.production.weed_events<<",\"reactive_digs\":"<<v.reactive.production.digs<<",\"reactive_plants\":"<<v.reactive.production.plants<<",\"reactive_waters\":"<<v.reactive.production.waters<<",\"reactive_harvests\":"<<v.reactive.production.harvests<<",\"inserted_before_move\":"<<v.reactive.inserted_before_move<<",\"skipped_nonmoves\":"<<v.reactive.skipped_nonmoves<<",\"ordered_move_day_failures\":"<<v.ordered_move_day_failures<<",\"absolute_move_mismatches\":"<<v.absolute_move_mismatches<<"}"<<(arm==4?"":",");}detail<<"]}\n";}
  std::cout<<"weed reactive games="<<rows.size()<<" report="<<options.output<<" seconds="<<elapsed<<'\n';
  return 0;
} catch(const std::exception& error){std::cerr<<"weed_reactive_eval: "<<error.what()<<'\n';return 2;}
