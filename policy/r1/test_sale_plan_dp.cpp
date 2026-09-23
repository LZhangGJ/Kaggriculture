// Offline optimality check for sale_plan_dp.hpp -- no agent, no env needed.
//   solve() must be self-consistent, and must beat every reference schedule evaluated under the
//   SAME objective (rollout). Every reference uses the same recurrence the model does: a period's
//   production is available to sell IN that period, so avail = s + production[k] and
//   s' = avail - q. `ref_sell_all` used to add production AFTER the sale, i.e. it was one period
//   behind the model -- which is exactly the defect the test was supposed to catch, so it could
//   never catch it. The DP carried the same lag until it was fixed.
#include <cstdio>
#include <random>
#include <vector>
#include "sale_plan_dp.hpp"
namespace sp = competitive::sale_plan;

// sell-everything on hand, in true stock terms
static std::vector<double> ref_sell_all(const sp::Input& in, double stock0) {
  std::vector<double> q(in.count(), 0.); double s = stock0;
  for (int k = 0; k < in.count(); k++) { double avail = s + in.production[k]; q[k] = std::floor(avail); s = avail - q[k]; }
  return q;
}
// A schedule is only physically realisable if it never carries more than the shed holds -- the
// engine discards the overflow. Sell more to shed the excess, exactly as the live agent does.
static void make_feasible(const sp::Input& in, double stock0, std::vector<double>& q) {
  double s = stock0;
  for (int k = 0; k < in.count(); k++) {
    double avail = s + in.production[k];
    double lim = std::max(0., std::floor(avail));
    // The engine sells whole units. A fractional reference is not a schedule the agent could ever
    // execute, and letting one through here is what made solve() look 0.3% suboptimal on 20 cases.
    q[k] = std::min(std::max(0., std::floor(q[k])), lim);
    double excess = (avail - q[k]) - in.hold_cap;
    if (excess > 0) q[k] = std::min(lim, q[k] + std::ceil(excess));
    s = avail - q[k];
  }
}
// spread evenly over the remaining periods
static std::vector<double> ref_spread(const sp::Input& in, double stock0) {
  int N = in.count(); std::vector<double> q(N, 0.); double s = stock0;
  for (int k = 0; k < N; k++) {
    s += in.production[k];
    q[k] = std::floor(s / double(N - k));
    s -= q[k];
  }
  return q;
}

int main() {
  std::mt19937 rng(20260922);
  std::uniform_real_distribution<double> dq(0., 30.), rq(0., 30.), pq(0., 60.), hp(10., 600.);
  int bad_self = 0, bad_ref = 0, n = 0; double worst = 0.;
  for (int t = 0; t < 400; t++) {
    int item = int(rng() % 9), N = 3 + int(rng() % 8);
    sp::Input in; in.competition = 0.; in.hold_cap = hp(rng);
    for (int k = 0; k < N; k++) {
      in.demand.push_back(dq(rng)); in.rival.push_back(rq(rng)); in.production.push_back(pq(rng));
    }
    double inv0 = 9500. + double(rng() % 1500), stock0 = double(rng() % 40);
    auto plan = sp::solve(item, inv0, stock0, in);
    for (auto& x : plan.sell) x = std::floor(x);   // engine-legal, same as the references
    double replay = sp::rollout(item, inv0, stock0, plan.sell, in);
    if (std::abs(replay - plan.value) > 1e-9 * std::max(1., std::abs(plan.value))) {
      if (++bad_self <= 3) printf("  SELF item=%d N=%d dp=%.4f replay=%.4f\n", item, N, plan.value, replay);
    }
    auto ref = [&](std::vector<double> q) { make_feasible(in, stock0, q);
      return sp::rollout(item, inv0, stock0, q, in); };
    double best_ref = ref(ref_sell_all(in, stock0));
    best_ref = std::max(best_ref, ref(ref_spread(in, stock0)));
    for (int r = 0; r < 8; r++) {              // random schedules too
      std::vector<double> q(N, 0.); double s = stock0;
      for (int k = 0; k < N; k++) { s += in.production[k]; q[k] = std::floor(s * (double(rng() % 101) / 100.)); s -= q[k]; }
      best_ref = std::max(best_ref, ref(q));
    }
    double gap = plan.value - best_ref;
    double rel = gap / std::max(1., std::abs(best_ref));
    if (rel < -1e-3) {                    // materially worse than a reference (>0.1%)
      bad_ref++;
      if (bad_ref <= 5) printf("  BEATEN item=%d N=%d dp=%.3f ref=%.3f gap=%.3f (%.3f%%)\n",
                               item, N, plan.value, best_ref, gap, 100.*rel);
    }
    if (rel < worst) worst = rel;
    n++;
  }
  printf("cases=%d  self-inconsistent=%d  beaten(>0.1%%) by reference=%d  worst relative gap=%.5f\n", n, bad_self, bad_ref, worst);
  printf("%s\n", (bad_self == 0 && bad_ref == 0) ? "PASS" : "FAIL");
  return (bad_self == 0 && bad_ref == 0) ? 0 : 1;
}
