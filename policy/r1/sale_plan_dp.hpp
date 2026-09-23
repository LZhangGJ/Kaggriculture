#pragma once
// Optimal selling plan for one product over the remaining game, given a KNOWN demand path and a
// KNOWN rival-supply path.
//
// Both inputs are genuinely known, not forecast: demand follows from the shop list and the step
// (`PublicTradeLedger::consumption`), and the rival's ready supply is countable off its public
// tiles (`public_rival_quantity`). That is what makes this a solvable problem rather than a
// prediction.
//
// Why it is small: a product's price depends only on that product's market inventory, so the nine
// products decouple into nine independent 1-D problems. And because our sales only move goods
// between the market and our shed,
//
//     I' = I + q + R - D          (market inventory)
//     s' = s + P - q              (our stock)
//     => (I + s)' = (I + s) + P + R - D
//
// the SUM (I + s) is exogenous in q. So `total = I + s` is tracked exogenously and the DP state is
// OUR STOCK s carried INTO a period (before that period's production arrives); the market inventory
// is recovered as I = total - s.
//
// TIMING: a period's own production reaches the shed before that period's market closes, so it is
// sellable in the SAME period, matching planner.hpp::value()'s `trade(i, inv[i], a.f[d][i])`.
// Available in period k is s + production[k]; the stock carried out is (s + production[k]) - q.
//
// SATURATION: ConditionalMarket::execute stops raising the inventory once the price reaches its
// floor (price>1 gate in the engine's _commit_unit). Selling into that region pays exactly 1 per
// unit; execute() caps the inflow itself and pays 1/unit for the overflow, so the plan is NOT
// restricted there. Clamping q to (saturation - I) made the plan sell nothing whenever the market
// sat at its floor, which for most products is most of the game (-24k a game).
//
// The state axis is quantised to ISTEP units and the sell grid is QN points. Callers plan at day
// granularity, matching planner.hpp's own `dem`/`rival` arrays.
#include <array>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <vector>
#include "conditional_market.hpp"

namespace competitive {
namespace sale_plan {

constexpr int QN = 17;                   // candidate sell amounts per period, 0..whole stock
constexpr int ISTEP = 10;
constexpr int MAX_INV = 20000;           // matches ConditionalMarket::prefix' table

// One period, mirroring planner.hpp value(): demand takes half, the rival sells half, we sell, the
// rival sells the rest, then demand takes the second half. `inv_after` is the quantised next state.
inline void day_step_parts(int item, double inv, double q, double rival_total, double demand,
                           double sat, double* ours, double* enemy, int* inv_after) {
  double half = demand * .5, r = rival_total * .5;
  double stock = inv - half;
  // Pass the full quantities: execute() caps what enters the market at saturation and pays
  // 1/unit for the overflow. Clamping q to (sat - stock) here made the plan sell NOTHING
  // whenever the market sat at its floor -- which, since most products saturate within ~60
  // units of I0, is most of the game. That burst the shed and destroyed goods (-24k/game).
  double e = ConditionalMarket::execute(item, stock, r);
  double o = ConditionalMarket::execute(item, stock, q);
  e += ConditionalMarket::execute(item, stock, r);
  stock -= half;
  if (ours) *ours = o;
  if (enemy) *enemy = e;
  if (inv_after) *inv_after = int(std::lround(stock / ISTEP)) * ISTEP;
}
inline double day_step(int item, double inv, double q, double rival_total, double demand,
                       double competition, double sat, int* inv_after = nullptr) {
  double o = 0., e = 0.;
  day_step_parts(item, inv, q, rival_total, demand, sat, &o, &e, inv_after);
  return o - competition * e;
}

struct Input {
  std::vector<double> demand, rival, production;   // per period, same length
  double competition = 0.;
  // Hard cap on how much we may be holding at any time. Kept for the single-product offline tests;
  // the live planner no longer uses it (see hold_penalty).
  double hold_cap = 1e9;
  // SHADOW PRICE of shed space, per unit of stock carried out of a period. The shed is 100 units
  // SHARED by all nine products and the future arrivals are known, so the exact problem is
  //     max sum_i V_i(q_i)   s.t.  sum_i s_i[k] <= 100 for every k,
  // which is a separable concave allocation: pricing shed space at lambda and solving each product
  // independently is exact for the right lambda, and the right lambda is found by bisection.
  // A fixed per-product cap is only a crude stand-in for this and has to be guessed.
  double hold_penalty = 0.;
  // Per-period discount, matching planner.hpp::value(). WITHOUT it the DP treats a unit of cash on
  // the last day as worth the same as one on the first, so it defers sales for free -- while in the
  // real game early cash hires workers, buys seed and buys animals, and those compound. That is the
  // structural reason a "more optimal" plan earned LESS: measured, the DP arm's cash ran -1,073
  // behind over days 12-20 and never recovered, with the herd and the final stock both normal.
  double discount = 0.;
  int count() const { return int(demand.size()); }
};

struct Plan {
  std::vector<double> sell;              // per period
  double value = 0.;                     // our revenue - competition * rival's
  double revenue = 0.;                   // OUR revenue alone (value() needs this for `cash`)
  std::vector<double> inventory;         // market inventory at the START of each period
  std::vector<double> stock;             // OUR carried stock at the start of each period
};

// Roll one sell schedule forward. Stock grows by production, shrinks by what we sell; because a
// sale only moves goods between the shed and the market, the market inventory is total[k] - stock.
// This single function is both the model and the objective, so a plan can never be "optimal
// according to the DP" while scoring differently when replayed.
inline double rollout(int item, double inv0, double stock0, const std::vector<double>& q,
                      const Input& in, std::vector<double>* inv_path = nullptr,
                      double* revenue = nullptr) {
  const int N = in.count();
  const double sat = ConditionalMarket::saturation(item) - 1.;
  std::vector<double> total(N + 1);
  total[0] = inv0 + stock0;
  for (int k = 0; k < N; k++)
    total[k + 1] = total[k] + in.production[k] + in.rival[k] - in.demand[k];
  double s = stock0, v = 0., rev = 0.;
  for (int k = 0; k < N; k++) {
    double I = total[k] - s;
    if (inv_path) inv_path->push_back(I);
    // Period k's own production reaches the shed BEFORE the market closes, so it is sellable in
    // period k -- `value()` credits exactly that (`cash += trade(i, inv[i], a.f[d][i])`). Adding
    // production AFTER the sale makes the plan sell one day late, which in a real game means
    // carrying a whole day of output (800+/day here) through a 100-unit shed: the engine discards
    // it. Measured against the live opponent that cost -4425/-4112/-4294 margin.
    double avail = s + in.production[k];
    double sell = std::min(std::max(0., q[k]), std::max(0., avail));
    double o = 0., e = 0.;
    day_step_parts(item, I, sell, in.rival[k], in.demand[k], sat, &o, &e, nullptr);
    rev += o;
    v += (o - in.competition * e) / std::pow(1. + in.discount, double(k));
    s = avail - sell;
    v -= in.hold_penalty * s / std::pow(1. + in.discount, double(k));
  }
  if (revenue) *revenue = rev;
  return v;
}

// Backward DP, state = OUR STOCK on a grid anchored at 0 (so s=0 is always representable), with
// the exact transition s' = s - q + production. The market inventory is derived, never a state:
//   * an inventory-anchored grid lost the single feasible state at k=0 (stock exactly 0) whenever
//     inv0 was not a grid multiple, and the whole DP collapsed to "sell nothing";
//   * a coordinate-descent version was correct but too weak a neighbourhood -- random schedules
//     beat it in 104/400 trials, so it is not the answer either.
// The returned value is measured by rollout(), the same function the search optimises, so a plan
// can never score one way here and another way when replayed.
// `polish` is the expensive part (O(N^3) rollouts); the DP table itself is ~1/7 of the cost. The
// shadow-price search below only needs the TABLE, so it runs with polish=false and the chosen
// lambda is then solved once for real.
inline Plan solve(int item, double inv0, double stock0, const Input& in, bool polish = true) {
  const int N = in.count();
  Plan out;
  out.sell.assign(N, 0.);
  if (N == 0) return out;
  const double sat = ConditionalMarket::saturation(item) - 1.;

  std::vector<double> total(N + 1), maxs(N + 1);
  total[0] = inv0 + stock0;
  maxs[0] = stock0;
  for (int k = 0; k < N; k++) {
    total[k + 1] = total[k] + in.production[k] + in.rival[k] - in.demand[k];
    maxs[k + 1] = maxs[k] + std::max(0., in.production[k]);
  }
  const int W = int(maxs[N] / ISTEP) + 2;
  auto sg = [](double s) { return int(std::lround(s / ISTEP)); };
  // V is only tabulated at multiples of ISTEP, and neither the opening stock nor a period's
  // production is a multiple of it, so the state leaves the grid immediately. Reading the nearest
  // grid value therefore replays a DIFFERENT policy than the one that was optimised -- which is
  // what made the plan lose to random schedules. Interpolate instead.
  auto lookup = [&](const std::vector<double>& V, double s) {
    double x = s / ISTEP;
    if (x <= 0) return V[0];
    if (x >= W - 1) return V[W - 1];
    int a = int(std::floor(x));
    double f = x - a;
    return V[a] * (1. - f) + V[a + 1] * f;
  };

  std::vector<double> next(W, 0.), cur(W);
  std::vector<std::vector<int>> pol(N, std::vector<int>(W, 0));
  for (int k = N - 1; k >= 0; k--) {
    std::fill(cur.begin(), cur.end(), -1e18);
    for (int w = 0; w < W; w++) {
      double s = double(w) * ISTEP;
      // Reachability: we cannot be holding more than we have produced by period k. Without this
      // the DP evaluates unreachable states, where I = total[k] - s goes negative and the price
      // curve returns fantasy values that the search then steers toward (278/400 worse than
      // random schedules before this guard was restored).
      if (s > maxs[k] + ISTEP * 0.5) continue;
      double I = total[k] - s;
      double avail = s + in.production[k];
      double best = -1e18;
      int bq = 0;
      for (int j = 0; j < QN; j++) {
        double q = std::floor(avail * double(j) / double(QN - 1));
        double s2 = avail - q;
        // The engine DISCARDS end-of-day shed overflow, so carrying more than the shed holds
        // destroys goods. hold_cap was plumbed into Input but never read until now.
        if (s2 > in.hold_cap + 1e-9) continue;
        double o = 0., e = 0.;
        day_step_parts(item, I, q, in.rival[k], in.demand[k], sat, &o, &e, nullptr);
        double cont = (k + 1 < N) ? lookup(next, s2) : 0.;
        double v = (o - in.competition * e) / std::pow(1. + in.discount, double(k)) + cont - in.hold_penalty * s2;
        if (v > best) { best = v; bq = j; }   // store the FRACTION, see below
      }
      cur[w] = best;
      pol[k][w] = bq;
    }
    next.swap(cur);
  }

  // The policy is stored as a fraction of the stock, not an absolute quantity: the DP only ever
  // tabulates at multiples of ISTEP, and real stocks are not, so an absolute policy would be read
  // off the wrong state and replay a different plan than the one optimised.
  double s = stock0;
  for (int k = 0; k < N; k++) {
    int w = sg(s);
    int j = (w >= 0 && w < W) ? pol[k][w] : (QN - 1);
    double avail = s + in.production[k];
    double q = std::min(std::floor(avail * double(j) / double(QN - 1)), std::max(0., avail));
    out.sell[k] = q;
    s = avail - q;
  }
  if (!polish) {
   out.value = rollout(item, inv0, stock0, out.sell, in);
   { double st = stock0; out.stock.clear();
     for (int k = 0; k < N; k++) { out.stock.push_back(st); double a = st + in.production[k];
       st = a - std::min(std::max(0., out.sell[k]), std::max(0., a)); } }
   return out;
  }
  // The DP is a WARM START. Its grid (stock quantised to ISTEP, sells as QN fractions of the
  // stock available in a period) can leave it a fraction of a percent off the exact optimum on
  // short horizons, which is enough to lose to a plain random schedule. So the returned plan is
  // the best of several starting points after an exact-objective local search:
  //
  //   * every schedule is kept engine-legal: integer sales, and the stock carried out of a period
  //     never exceeds hold_cap (the engine discards the overflow, so carrying more destroys goods);
  //   * `repair` enforces both, and is applied to every candidate before it is scored;
  //   * starts are the DP plan, sell-everything, an even spread, and random fractions of the
  //     available stock (the same family the DP quantises);
  //   * starts are polished in descending order of raw value and the loop STOPS as soon as a
  //     start's raw value no longer beats the best polished result -- so the usual case is one
  //     polish, and extra polish only happens when a start genuinely beats what we already have.
  {
    auto repair = [&](std::vector<double>& t) {
      t.resize(N);
      double s = stock0;
      for (int k = 0; k < N; k++) {
        double a = s + in.production[k];
        double lim = std::max(0., std::floor(a));
        t[k] = std::min(std::max(0., std::floor(t[k])), lim);
        double excess = (a - t[k]) - in.hold_cap;
        if (excess > 0) t[k] = std::min(lim, t[k] + std::ceil(excess));
        s = a - t[k];
      }
    };
    auto polish = [&](std::vector<double> cur, double curv) {
      auto apply = [&](std::vector<double> t) {
        repair(t);
        double v = rollout(item, inv0, stock0, t, in);
        if (v > curv + 1e-9) { curv = v; cur = std::move(t); return true; }
        return false;
      };
      for (int sweep = 0; sweep < (N <= 40 ? 6 : 2); sweep++) {
        bool moved = false;
        // Nudge one period up or down.
        for (int k = 0; k < N; k++) {
          static const std::array<double, 8> deltas{{1., 2., 4., 8., 16., 32., 1e9, -1e9}};
          for (double d : deltas) {
            std::vector<double> t = cur;
            t[k] = (d > 0) ? std::floor(cur[k] + (d >= 1e8 ? 1e8 : d))
                           : std::max(0., std::floor(cur[k] + d));
            if (apply(std::move(t))) moved = true;
          }
        }
        // Move quantity BETWEEN two periods. O(N^2) rollouts per sweep, so it is switched off at
        // the 108-period (4-step) horizon the DP is meant to run at; single-period moves carry it.
        for (int a = 0; a < N && N <= 40; a++)
          for (int b = 0; b < N; b++) {
            if (a == b) continue;
            for (double d : {1., 4., 16., 64.}) {
              double take = std::min(std::floor(cur[a]), d);
              if (take <= 0) continue;
              std::vector<double> t = cur;
              t[a] = std::floor(cur[a]) - take;
              t[b] = std::floor(cur[b]) + take;
              if (apply(std::move(t))) moved = true;
            }
          }
        if (!moved) break;
      }
      return std::pair<std::vector<double>, double>(std::move(cur), curv);
    };

    std::vector<std::vector<double>> starts;
    starts.push_back(out.sell);                       // the DP's own plan
    { std::vector<double> sa(N, 0.);                  // sell everything on hand
      double s = stock0;
      for (int k = 0; k < N; k++) { double a = s + in.production[k]; sa[k] = std::floor(a); s = a - sa[k]; }
      starts.push_back(std::move(sa)); }
    { std::vector<double> sp(N, 0.);                  // spread evenly over the horizon
      double s = stock0;
      for (int k = 0; k < N; k++) { double a = s + in.production[k]; sp[k] = std::floor(a / double(N - k)); s = a - sp[k]; }
      starts.push_back(std::move(sp)); }
    { uint64_t h = 0x9e3779b97f4a7c15ull ^ uint64_t(item * 131 + N * 977 + int(stock0) * 31);
      for (int r = 0; r < (N <= 40 ? 64 : 16); r++) {
        std::vector<double> q(N, 0.); double s = stock0;
        for (int k = 0; k < N; k++) {
          h ^= h << 13; h ^= h >> 7; h ^= h << 17;
          double a = s + in.production[k];
          q[k] = std::floor(a * double(h % 101) / 100.);
          s = a - q[k];
        }
        starts.push_back(std::move(q));
      } }
    std::vector<std::pair<double, int>> ranked;
    for (size_t r = 0; r < starts.size(); r++) {
      repair(starts[r]);
      ranked.emplace_back(rollout(item, inv0, stock0, starts[r], in), int(r));
    }
    std::sort(ranked.begin(), ranked.end(), [](auto& a, auto& b) { return a.first > b.first; });
    double bestv = -1e18;
    for (auto& [rawv, idx] : ranked) {
      if (rawv <= bestv) break;            // nothing left that could beat what we already have
      auto [plan, val] = polish(std::move(starts[idx]), rawv);
      if (val > bestv) { bestv = val; out.sell = std::move(plan); }
    }
  }
  out.value = rollout(item, inv0, stock0, out.sell, in, &out.inventory, &out.revenue);
  { double st = stock0; out.stock.clear();
    for (int k = 0; k < N; k++) { out.stock.push_back(st); double a = st + in.production[k];
      st = a - std::min(std::max(0., out.sell[k]), std::max(0., a)); } }
  return out;
}

// Objective of an arbitrary sell schedule -- delegates to the same rollout used to optimise.
inline double evaluate(int item, double inv0, const std::vector<double>& sell, const Input& in,
                       double stock0 = 0.) {
  return rollout(item, inv0, stock0, sell, in);
}

}  // namespace sale_plan
}  // namespace competitive
