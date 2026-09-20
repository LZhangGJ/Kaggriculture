#include "native_teammate.hpp"
#include "route_loader.hpp"
#include "scheduler.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

namespace {
using fastkag::Action;
using fastkag::Item;
using fastkag::Op;
using fastkag::PlayerAction;
using fastkag::Simulator;

struct Options {
  std::string tapes, library, refs, output;
  int threads{static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))};
  int seeds{128};
  std::uint64_t block{4000000};
};

template<class T> T number(std::string_view s) {
  T x{}; const auto p = std::from_chars(s.data(), s.data() + s.size(), x);
  if (p.ec != std::errc{} || p.ptr != s.data() + s.size())
    throw std::invalid_argument("invalid number: " + std::string(s));
  return x;
}

Options parse(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto next = [&] { if (++i >= argc) throw std::invalid_argument("missing value"); return std::string_view(argv[i]); };
    if (a == "--tapes") o.tapes = next();
    else if (a == "--library") o.library = next();
    else if (a == "--refs") o.refs = next();
    else if (a == "--output") o.output = next();
    else if (a == "--threads") o.threads = number<int>(next());
    else if (a == "--seeds") o.seeds = number<int>(next());
    else if (a == "--block") o.block = number<std::uint64_t>(next());
    else throw std::invalid_argument("unknown argument: " + std::string(a));
  }
  if (o.tapes.empty() || o.library.empty() || o.refs.empty() || o.output.empty())
    throw std::invalid_argument("--tapes --library --refs --output required");
  if (o.threads <= 0 || o.seeds <= 0) throw std::invalid_argument("positive sizes required");
  if (o.block <= 976 && o.block + static_cast<std::uint64_t>(o.seeds) > 976)
    throw std::invalid_argument("reserved seed 976 is forbidden");
  return o;
}

bool action_equal(const Action& a, const Action& b) {
  return a.op == b.op && a.item == b.item && a.quantity == b.quantity;
}
bool action_list_equal(const std::vector<Action>& a, const std::vector<Action>& b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), action_equal);
}
bool player_action_equal(const PlayerAction& a, const PlayerAction& b) {
  return action_list_equal(a.units, b.units) && action_list_equal(a.market, b.market);
}
bool pair_action_equal(const std::array<PlayerAction,2>& a,
                       const std::array<PlayerAction,2>& b) {
  return player_action_equal(a[0], b[0]) && player_action_equal(a[1], b[1]);
}
bool market_actions_equal(const std::array<PlayerAction,2>& a,
                          const std::array<PlayerAction,2>& b) {
  return action_list_equal(a[0].market, b[0].market) &&
         action_list_equal(a[1].market, b[1].market);
}
bool public_market_equal(const Simulator& a, const Simulator& b) {
  return a.step_count() == b.step_count() &&
         a.market().inventory == b.market().inventory &&
         a.market().prices == b.market().prices && a.shops() == b.shops() &&
         a.farms()[0].money == b.farms()[0].money &&
         a.farms()[1].money == b.farms()[1].money;
}
bool tile_equal(const fastkag::Tile& a, const fastkag::Tile& b) {
  return a.kind==b.kind && a.crop==b.crop && a.animal==b.animal &&
    a.planted_day==b.planted_day && a.placed_day==b.placed_day &&
    a.yield_units==b.yield_units && a.consecutive_unwatered==b.consecutive_unwatered &&
    a.consecutive_unfed==b.consecutive_unfed &&
    a.fertilized_until_day==b.fertilized_until_day &&
    a.pending_care_bonus==b.pending_care_bonus &&
    a.max_lifespan_step==b.max_lifespan_step &&
    a.watered_today==b.watered_today && a.fed_today==b.fed_today &&
    a.cared_today==b.cared_today &&
    a.fertilizer_available==b.fertilizer_available;
}
bool exact_env_equal(const Simulator& a, const Simulator& b) {
  if (!public_market_equal(a,b) || a.day()!=b.day() || a.hour()!=b.hour()) return false;
  for (int p=0;p<2;++p) {
    const auto& x=a.farms()[p]; const auto& y=b.farms()[p];
    if (x.farmer.x!=y.farmer.x || x.farmer.y!=y.farmer.y ||
        x.hands.size()!=y.hands.size() || x.tiles.size()!=y.tiles.size() ||
        x.unlocked_mask!=y.unlocked_mask || x.hires_today!=y.hires_today) return false;
    for (std::size_t i=0;i<x.hands.size();++i)
      if (x.hands[i].x!=y.hands[i].x || x.hands[i].y!=y.hands[i].y) return false;
    for (std::size_t i=0;i<x.tiles.size();++i) if (!tile_equal(x.tiles[i],y.tiles[i])) return false;
    const auto& q=a.privates()[p]; const auto& r=b.privates()[p];
    if (q.shed!=r.shed || q.seeds!=r.seeds || q.inventories!=r.inventories ||
        q.inventory_order!=r.inventory_order) return false;
  }
  return true;
}

bool production_op(Op op) {
  return op==Op::PLANT || op==Op::WATER || op==Op::HARVEST ||
    op==Op::FERTILIZE || op==Op::DIG || op==Op::BUILD_COOP ||
    op==Op::BUILD_PASTURE || op==Op::FEED ||
    op==Op::COLLECT_FERTILIZER || op==Op::CARE || op==Op::PLACE;
}
std::array<int,fastkag::N_ITEMS> total_goods(const Simulator& env, int player) {
  std::array<int,fastkag::N_ITEMS> out=env.privates()[player].shed;
  for (const auto& inv:env.privates()[player].inventories)
    for (int i=0;i<fastkag::N_ITEMS;++i) out[i]+=inv[i];
  return out;
}

struct Metrics {
  double own{}, opponent{};
  int production_attempts{}, unit_failures{}, market_failures{}, overflow{};
  std::array<int,fastkag::N_ITEMS> production_output{};
};
void observe_step(Metrics& m, const Simulator& env, int seat,
                  const std::array<PlayerAction,2>& actions) {
  for (const auto& a:actions[seat].units) m.production_attempts += production_op(a.op);
  m.unit_failures += fastkag::native_macro_unit_failures(env,seat,actions[seat]);
  const auto before=total_goods(env,seat);
  const auto preview=env.preview_unit_phase(actions);
  const auto after=total_goods(preview,seat);
  for (int i=0;i<fastkag::N_ITEMS;++i)
    m.production_output[i] += std::max(0,after[i]-before[i]);
}
void observe_after(Metrics& m, const Simulator& env, int seat,
                   const PlayerAction& action) {
  m.market_failures += fastkag::native_macro_market_failures(env,seat,action);
  m.overflow += env.last_end_of_day_overflow()[seat];
}

fastkag::NativeTapeLibrary load_library(const Options& o,
                                        std::vector<PlayerAction>& g001) {
  fastkag::NativeTapeLibrary out;
  g001=g001::repair::load_route(o.tapes,o.library,"G001");
  out.routes.push_back(g001);
  out.r5_reference=g001::repair::load_route(o.refs,o.library,"R5");
  out.md_reference=g001::repair::load_route(o.refs,o.library,"MD");
  constexpr std::array<std::string_view,5> labels{
    "10C4S_3Q","8C6S_3Q","6C8S_3Q","6C12S_4Q_FIRST_YARN","6C12S_4Q_SECOND_YARN"};
  for (std::size_t i=0;i<labels.size();++i) {
    out.moon[i]=g001::repair::load_route(o.refs,o.library,"MOON_"+std::string(labels[i]));
    out.moon_legacy[i]=g001::repair::load_route(o.refs,o.library,"MOON_LEGACY_"+std::string(labels[i]));
  }
  return out;
}

struct ForkResult {
  std::uint64_t seed{}; int seat{},actor{},start{},skip{},realign{};
  bool used_pass{}, source9_move{}, initial_actions_equal{}, post_trigger_env_equal{};
  int first_market_action_div{-1}, first_market_state_div{-1};
  int intended_op{}, intended_item{}, x{}, y{}, own_goods{}, own_seeds{};
  int market_inventory{}, market_price{}, shop_count{}, own_hands{}, opponent_hands{};
  double own_money{}, opponent_money{};
  Metrics base,cand;
};

std::optional<weed_audit::Choice> safe_pass_before_move(
    const std::vector<PlayerAction>& tape, int step, int actor) {
  const int last=std::min({static_cast<int>(tape.size())-1,
      (step/24+1)*24-1,step+16});
  for(int source=step+1;source<=last;++source) {
    const auto op=weed_audit::tape_unit(tape,source,actor).op;
    if(g001::repair::is_movement(op)) break;
    if(op==Op::PASS) return weed_audit::Choice{source,op,false,source==last,true};
  }
  return std::nullopt;
}

ForkResult run_fork(const fastkag::NativeTeammateExecutor& executor,
                    const std::vector<PlayerAction>& tape,
                    const Simulator& snapshot,
                    const std::array<fastkag::NativeAgentState,2>& state_snapshot,
                    int seat,int actor,const weed_audit::Choice& safe_choice) {
  Simulator base_env=snapshot, cand_env=snapshot;
  auto base_state=state_snapshot, cand_state=state_snapshot;
  ForkResult out; out.seed=snapshot.seed(); out.seat=seat; out.actor=actor;
  out.start=snapshot.step_count();
  const std::optional<weed_audit::Choice> choice=safe_choice;
  out.skip=choice->skip_step;
  out.realign=choice->skip_step+1;
  out.used_pass=true;
  out.source9_move=g001::repair::is_movement(
      weed_audit::tape_unit(tape,out.start+9,actor).op);
  const auto intended=weed_audit::tape_unit(tape,out.start,actor);
  out.intended_op=static_cast<int>(intended.op);
  out.intended_item=static_cast<int>(intended.item);
  const auto& farm=snapshot.farms()[seat];
  const auto& pos=actor==0 ? farm.farmer : farm.hands.at(static_cast<std::size_t>(actor-1));
  out.x=pos.x;out.y=pos.y;out.own_money=farm.money;
  out.opponent_money=snapshot.farms()[1-seat].money;
  out.own_hands=static_cast<int>(farm.hands.size());
  out.opponent_hands=static_cast<int>(snapshot.farms()[1-seat].hands.size());
  if(out.intended_item>=0&&out.intended_item<fastkag::N_ITEMS) {
    out.own_goods=total_goods(snapshot,seat)[out.intended_item];
    if(out.intended_item<fastkag::N_CROPS)
      out.own_seeds=snapshot.privates()[seat].seeds[out.intended_item];
    if(out.intended_item<fastkag::N_PRODUCTS) {
      out.market_inventory=snapshot.market().inventory[out.intended_item];
      out.market_price=snapshot.market().prices[out.intended_item];
      out.shop_count=static_cast<int>(std::count(snapshot.shops().begin(),
          snapshot.shops().end(),out.intended_item));
    }
  }

  bool first=true;
  while (!base_env.done()) {
    if (base_env.step_count()!=cand_env.step_count())
      throw std::runtime_error("fork clocks diverged");
    const int step=base_env.step_count();
    // action_external uses the deployed legacy option set.  Its final overlay
    // executes an injected transaction, but the caller must retire that
    // offline-only transaction after its skip (the whole-policy audit does the
    // same).  Retiring it is what restores ordinary legacy weed handling.
    if (choice && actor < static_cast<int>(cand_state[seat].experimental_realign.size())) {
      auto& transaction=cand_state[seat].experimental_realign[actor];
      if (transaction.active && step>transaction.skipped_source_step)
        transaction.active=false;
    }
    if (step>=out.realign && out.first_market_state_div<0 &&
        !public_market_equal(base_env,cand_env)) out.first_market_state_div=step;
    std::array<PlayerAction,2> ba{
      executor.action_external(base_env,0,0,base_state[0]),
      executor.action_external(base_env,1,0,base_state[1])};
    std::array<PlayerAction,2> ca{
      executor.action_external(cand_env,0,0,cand_state[0]),
      executor.action_external(cand_env,1,0,cand_state[1])};
    if (first) {
      out.initial_actions_equal=pair_action_equal(ba,ca);
      if (actor>=static_cast<int>(cand_state[seat].weed.size()) ||
          !cand_state[seat].weed[actor].active ||
          cand_state[seat].weed[actor].start!=step)
        throw std::runtime_error("snapshot did not reproduce weed trigger");
      if (choice) {
        cand_state[seat].experimental_realign.resize(ca[seat].units.size());
        cand_state[seat].experimental_realign[actor]=
          {true,step,choice->skip_step,Action{Op::DIG}};
        cand_state[seat].weed[actor]={};
      }
    }
    if (step>=out.realign && out.first_market_action_div<0 &&
        !market_actions_equal(ba,ca)) out.first_market_action_div=step;
    observe_step(out.base,base_env,seat,ba);
    observe_step(out.cand,cand_env,seat,ca);
    base_env.step(ba); cand_env.step(ca);
    observe_after(out.base,base_env,seat,ba[seat]);
    observe_after(out.cand,cand_env,seat,ca[seat]);
    if (first) { out.post_trigger_env_equal=exact_env_equal(base_env,cand_env); first=false; }
  }
  out.base.own=base_env.farms()[seat].money;
  out.base.opponent=base_env.farms()[1-seat].money;
  out.cand.own=cand_env.farms()[seat].money;
  out.cand.opponent=cand_env.farms()[1-seat].money;
  return out;
}

std::vector<ForkResult> run_case(const fastkag::NativeTeammateExecutor& executor,
                                 const std::vector<PlayerAction>& tape,
                                 std::uint64_t seed,int seat) {
  Simulator env({},seed);
  std::array<fastkag::NativeAgentState,2> states;
  std::vector<ForkResult> out;
  while (!env.done()) {
    const Simulator snapshot=env;
    const auto state_snapshot=states;
    std::array<PlayerAction,2> actions{
      executor.action_external(env,0,0,states[0]),
      executor.action_external(env,1,0,states[1])};
    for (int actor=0;actor<static_cast<int>(states[seat].weed.size());++actor) {
      const auto& w=states[seat].weed[actor];
      if (!w.active || w.start!=env.step_count()) continue;
      const auto choice=safe_pass_before_move(tape,env.step_count(),actor);
      if(choice && g001::repair::is_movement(
          weed_audit::tape_unit(tape,env.step_count()+9,actor).op))
        out.push_back(run_fork(executor,tape,snapshot,state_snapshot,seat,actor,*choice));
    }
    env.step(actions);
  }
  return out;
}

double lower_cvar(std::vector<double> x,double alpha) {
  if (x.empty()) return 0;
  std::sort(x.begin(),x.end());
  const std::size_t n=std::max<std::size_t>(1,static_cast<std::size_t>(std::ceil(alpha*x.size())));
  return std::accumulate(x.begin(),x.begin()+static_cast<std::ptrdiff_t>(n),0.0)/n;
}
double mean(const std::vector<double>& x) {
  return x.empty()?0:std::accumulate(x.begin(),x.end(),0.0)/x.size();
}
struct Summary {
  int n{},own_negative{},margin_negative{},strict_local_better{},nonnegative_both{},zero_effect{};
  int gained_wins{},lost_wins{},initial_mismatch{},post_mismatch{};
  int production_attempt_delta{},unit_failure_delta{},market_failure_delta{},overflow_delta{};
  int market_action_div{},market_state_div{};
  std::array<int,fastkag::N_ITEMS> output_delta{};
  std::vector<double> own,opponent,margin,score,market_action_lag,market_state_lag;
};
double game_score(double own,double opponent) {
  return own>opponent ? 1.0 : own<opponent ? 0.0 : 0.5;
}
Summary summarize(const std::vector<ForkResult>& rows,
                  const std::function<bool(const ForkResult&)>& keep) {
  Summary s;
  for (const auto& r:rows) if (keep(r)) {
    ++s.n; const double own=r.cand.own-r.base.own;
    const double opp=r.cand.opponent-r.base.opponent;
    const double margin=own-opp;
    s.own.push_back(own);s.opponent.push_back(opp);s.margin.push_back(margin);
    s.score.push_back(game_score(r.cand.own,r.cand.opponent)-
                      game_score(r.base.own,r.base.opponent));
    s.own_negative+=own<0;s.margin_negative+=margin<0;
    s.strict_local_better+=own>0 && margin>0;
    s.nonnegative_both+=own>=0 && margin>=0;
    s.zero_effect+=own==0 && margin==0;
    s.gained_wins+=r.base.own<=r.base.opponent && r.cand.own>r.cand.opponent;
    s.lost_wins+=r.base.own>r.base.opponent && r.cand.own<=r.cand.opponent;
    s.initial_mismatch+=!r.initial_actions_equal;
    s.post_mismatch+=!r.post_trigger_env_equal;
    s.production_attempt_delta+=r.cand.production_attempts-r.base.production_attempts;
    s.unit_failure_delta+=r.cand.unit_failures-r.base.unit_failures;
    s.market_failure_delta+=r.cand.market_failures-r.base.market_failures;
    s.overflow_delta+=r.cand.overflow-r.base.overflow;
    s.market_action_div+=r.first_market_action_div>=0;
    s.market_state_div+=r.first_market_state_div>=0;
    if(r.first_market_action_div>=0)
      s.market_action_lag.push_back(r.first_market_action_div-r.realign);
    if(r.first_market_state_div>=0)
      s.market_state_lag.push_back(r.first_market_state_div-r.realign);
    for(int i=0;i<fastkag::N_ITEMS;++i)
      s.output_delta[i]+=r.cand.production_output[i]-r.base.production_output[i];
  }
  return s;
}
void write_summary_row(std::ostream& os,std::string_view name,const Summary& s) {
  const auto worst=[](const std::vector<double>& x){return x.empty()?0:*std::min_element(x.begin(),x.end());};
  const auto median=[](std::vector<double> x){
    if(x.empty()) return 0.0;
    std::sort(x.begin(),x.end());
    const auto n=x.size();
    return n%2?x[n/2]:(x[n/2-1]+x[n/2])/2.0;};
  os<<"{\"stratum\":\""<<name<<"\",\"n\":"<<s.n
    <<",\"mean_own_delta\":"<<mean(s.own)
    <<",\"mean_opponent_delta\":"<<mean(s.opponent)
    <<",\"mean_margin_delta\":"<<mean(s.margin)
    <<",\"median_own_delta\":"<<median(s.own)
    <<",\"median_margin_delta\":"<<median(s.margin)
    <<",\"mean_score_delta\":"<<mean(s.score)
    <<",\"worst_own_delta\":"<<worst(s.own)<<",\"worst_margin_delta\":"<<worst(s.margin)
    <<",\"lower_cvar10_own\":"<<lower_cvar(s.own,.10)
    <<",\"lower_cvar10_margin\":"<<lower_cvar(s.margin,.10)
    <<",\"lower_cvar25_own\":"<<lower_cvar(s.own,.25)
    <<",\"lower_cvar25_margin\":"<<lower_cvar(s.margin,.25)
    <<",\"own_negative\":"<<s.own_negative<<",\"margin_negative\":"<<s.margin_negative
    <<",\"strict_own_and_margin_better\":"<<s.strict_local_better
    <<",\"nonnegative_own_and_margin\":"<<s.nonnegative_both
    <<",\"exact_zero_effect\":"<<s.zero_effect
    <<",\"gained_wins\":"<<s.gained_wins<<",\"lost_wins\":"<<s.lost_wins
    <<",\"initial_action_mismatch\":"<<s.initial_mismatch
    <<",\"post_trigger_env_mismatch\":"<<s.post_mismatch
    <<",\"production_attempt_delta\":"<<s.production_attempt_delta
    <<",\"unit_failure_delta\":"<<s.unit_failure_delta
    <<",\"market_failure_delta\":"<<s.market_failure_delta
    <<",\"overflow_delta\":"<<s.overflow_delta
    <<",\"first_market_action_divergence_count\":"<<s.market_action_div
    <<",\"first_market_state_divergence_count\":"<<s.market_state_div
    <<",\"mean_first_market_action_divergence_lag\":"<<mean(s.market_action_lag)
    <<",\"mean_first_market_state_divergence_lag\":"<<mean(s.market_state_lag)
    <<",\"production_output_delta\":[";
  for(int i=0;i<fastkag::N_ITEMS;++i){if(i)os<<',';os<<s.output_delta[i];}
  os<<"]}";
}
void write_metrics(std::ostream& os,const Metrics& m) {
  os<<"{\"own\":"<<m.own<<",\"opponent\":"<<m.opponent
    <<",\"production_attempts\":"<<m.production_attempts
    <<",\"unit_failures\":"<<m.unit_failures
    <<",\"market_failures\":"<<m.market_failures<<",\"overflow\":"<<m.overflow
    <<",\"production_output\":[";
  for(int i=0;i<fastkag::N_ITEMS;++i){if(i)os<<',';os<<m.production_output[i];}
  os<<"]}";
}
} // namespace

int main(int argc,char** argv) try {
  const auto o=parse(argc,argv);
  std::vector<PlayerAction> tape;
  const fastkag::NativeTeammateExecutor executor(load_library(o,tape));
  std::vector<std::pair<std::uint64_t,int>> tasks;
  for(int i=0;i<o.seeds;++i)for(int seat=0;seat<2;++seat)tasks.emplace_back(o.block+i,seat);
  std::vector<std::vector<ForkResult>> result(tasks.size());
  std::atomic<std::size_t> cursor{};
  std::vector<std::thread> workers;
  for(int t=0;t<std::min<int>(o.threads,tasks.size());++t) workers.emplace_back([&]{
    for(;;){const auto i=cursor.fetch_add(1);if(i>=tasks.size())break;
      result[i]=run_case(executor,tape,tasks[i].first,tasks[i].second);}
  });
  for(auto& t:workers)t.join();
  std::vector<ForkResult> rows;
  for(auto& v:result)rows.insert(rows.end(),v.begin(),v.end());
  std::sort(rows.begin(),rows.end(),[](const auto& a,const auto& b){
    return std::tie(a.seed,a.seat,a.start,a.actor)<std::tie(b.seed,b.seat,b.start,b.actor);});
  std::filesystem::create_directories(o.output);
  std::ofstream detail(std::filesystem::path(o.output)/"forks.jsonl");
  detail<<std::fixed<<std::setprecision(6);
  for(const auto& r:rows){
    detail<<"{\"seed\":"<<r.seed<<",\"seat\":"<<r.seat<<",\"actor\":"<<r.actor
      <<",\"start\":"<<r.start<<",\"skip\":"<<r.skip<<",\"realign\":"<<r.realign
      <<",\"sink\":\""<<(r.skip<0?"NONE":r.used_pass?"PASS":"NONPASS")
      <<"\",\"legacy_source9\":\""<<(r.source9_move?"MOVE":"NONMOVE")
      <<"\",\"initial_actions_equal\":"<<(r.initial_actions_equal?"true":"false")
      <<",\"post_trigger_env_equal\":"<<(r.post_trigger_env_equal?"true":"false")
      <<",\"first_market_action_divergence\":"<<r.first_market_action_div
      <<",\"first_market_state_divergence\":"<<r.first_market_state_div
      <<",\"features\":{\"step\":"<<r.start<<",\"day\":"<<(r.start/24)
      <<",\"hour\":"<<(r.start%24)<<",\"actor\":"<<r.actor
      <<",\"intended_op\":"<<r.intended_op<<",\"intended_item\":"<<r.intended_item
      <<",\"x\":"<<r.x<<",\"y\":"<<r.y<<",\"own_money\":"<<r.own_money
      <<",\"opponent_money\":"<<r.opponent_money
      <<",\"money_gap\":"<<(r.own_money-r.opponent_money)
      <<",\"own_goods\":"<<r.own_goods<<",\"own_seeds\":"<<r.own_seeds
      <<",\"market_inventory\":"<<r.market_inventory
      <<",\"market_price\":"<<r.market_price<<",\"shop_count\":"<<r.shop_count
      <<",\"own_hands\":"<<r.own_hands<<",\"opponent_hands\":"<<r.opponent_hands
      <<"},\"baseline\":";
    write_metrics(detail,r.base);detail<<",\"candidate\":";write_metrics(detail,r.cand);
    const double own=r.cand.own-r.base.own,opp=r.cand.opponent-r.base.opponent;
    detail<<",\"own_delta\":"<<own<<",\"opponent_delta\":"<<opp
      <<",\"margin_delta\":"<<(own-opp)<<"}\n";
  }
  std::ofstream summary(std::filesystem::path(o.output)/"summary.json");
  summary<<std::fixed<<std::setprecision(6)
    <<"{\n  \"schema\":\"weed_single_trigger_fork_v2\",\n  \"causal_design\":\"copy Simulator and both NativeAgentState values immediately before each eligible trigger; absorb only a PASS before the next MOVE; all later triggers use legacy\",\n"
    <<"  \"block\":"<<o.block<<",\n  \"seeds\":"<<o.seeds
    <<",\n  \"dual_seat_cases\":"<<tasks.size()<<",\n  \"trigger_forks\":"<<rows.size()
    <<",\n  \"strata\":[\n    ";
  struct Named {const char* name;std::function<bool(const ForkResult&)> keep;};
  const std::vector<Named> strata{
    {"all",[](const auto&){return true;}},
    {"sink_PASS",[](const auto&r){return r.skip>=0&&r.used_pass;}},
    {"sink_NONPASS",[](const auto&r){return r.skip>=0&&!r.used_pass;}},
    {"sink_NONE",[](const auto&r){return r.skip<0;}},
    {"source9_MOVE",[](const auto&r){return r.source9_move;}},
    {"source9_NONMOVE",[](const auto&r){return !r.source9_move;}},
    {"PASS_source9_MOVE",[](const auto&r){return r.used_pass&&r.source9_move;}},
    {"PASS_source9_NONMOVE",[](const auto&r){return r.used_pass&&!r.source9_move;}},
    {"NONPASS_source9_MOVE",[](const auto&r){return r.skip>=0&&!r.used_pass&&r.source9_move;}},
    {"NONPASS_source9_NONMOVE",[](const auto&r){return r.skip>=0&&!r.used_pass&&!r.source9_move;}}
  };
  for(std::size_t i=0;i<strata.size();++i){if(i)summary<<",\n    ";write_summary_row(summary,strata[i].name,summarize(rows,strata[i].keep));}
  summary<<"\n  ]\n}\n";
  std::cout<<"trigger_forks="<<rows.size()<<" output="<<o.output<<'\n';
  return 0;
} catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<'\n'; return 2; }
