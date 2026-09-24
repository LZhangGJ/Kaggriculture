// Standalone audit of the opt-in sale DP's bucket-to-day scoring interface.
// g++ -std=c++20 -O2 -Ipolicy/r1 experiments/audit_sale_dp_contract.cpp -o /tmp/audit_sale_dp_contract
#include "planner.hpp"
#include <cassert>
#include <cstdio>

int main() {
  using namespace competitive;
  fastkag::Farm own, rival;
  own.money = 10000;
  fastkag::PrivateState priv;
  fastkag::Market market;
  market.inventory.fill(10000);
  market.inventory[dp7::M] = 10060; // observed at step 288, episode 112731551
  std::vector<int8_t> shops;
  dp7::View view{288, 12, 0, own, rival, priv, market, shops};
  competitive::Config cfg;
  cfg.sale_dp = 1;
  cfg.discount = 0;
  cfg.competition = 0;
  cfg.risk = 0;
  Planner planner(cfg);
  planner.day = 12;
  planner.demand(view);
  for (double daily : {6., 12., 18.}) {
    Asset asset;
    for (int d = 12; d < 30; ++d) asset.f[d][dp7::M] = daily;
    double scored = planner.value(view, asset, nullptr, dp7::M);
    double by_day = 0, current_units = 0, actual_units = 0;
    double inv = market.inventory[dp7::M];
    for (int d = 12; d < 30; ++d) {
      double q = 0;
      for (int b = 6 * (d - 12); b < 6 * (d - 11); ++b) q += planner.live_q[dp7::M][b];
      by_day += Planner::trade(dp7::M, inv, q);
      inv -= planner.dem[d][dp7::M];
      current_units += planner.live_q[dp7::M][d];
      actual_units += q;
    }
    std::printf("daily=%.0f current_score=%.0f six_bucket_score=%.0f current_units=%.0f planned_units=%.0f\n",
                daily, scored, by_day, current_units, actual_units);
    assert(actual_units > current_units); // this is a bug witness, not a correctness claim
  }
  // A $1 sale leaves market stock unchanged, but rollout reconstructs it from I+s.
  sale_plan::Input floor;
  floor.production = {20., 10.};
  floor.rival = {0., 0.};
  floor.demand = {0., 30.};
  const double sat = ConditionalMarket::saturation(dp7::WO);
  std::vector<double> inferred;
  double model = sale_plan::rollout(dp7::WO, sat, 0., {20., 10.}, floor, &inferred);
  double exact_inv = sat, exact = 0;
  for (int k = 0; k < 2; ++k) {
    exact_inv -= floor.demand[k] / 2;
    exact += ConditionalMarket::execute(dp7::WO, exact_inv, k ? 10. : 20.);
    exact_inv -= floor.demand[k] / 2;
  }
  std::printf("floor model=%.0f exact=%.0f next_inventory_model=%.0f exact=%.0f\n",
              model, exact, inferred[1], sat);
  assert(inferred[1] != sat);

  // The same 60 existing goods are sellable now, not delivered ten per bucket.
  sale_plan::Input arriving;
  arriving.demand.assign(6, 0.);
  arriving.rival.assign(6, 0.);
  arriving.rival[1] = 100.;
  arriving.production.assign(6, 10.);
  auto exact_revenue = [&](const std::vector<double>& sales) {
    double inv = 10060., revenue = 0.;
    for (int b = 0; b < 6; ++b) {
      ConditionalMarket::execute(dp7::M, inv, arriving.rival[b] / 2);
      revenue += ConditionalMarket::execute(dp7::M, inv, sales[b]);
      ConditionalMarket::execute(dp7::M, inv, arriving.rival[b] / 2);
    }
    return revenue;
  };
  double spread_rev = exact_revenue({10., 10., 10., 10., 10., 10.});
  double now_rev = exact_revenue({60., 0., 0., 0., 0., 0.});
  std::printf("stock_clock spread_revenue=%.0f available_now_revenue=%.0f difference=%.0f\n",
              spread_rev, now_rev, now_rev - spread_rev);
  assert(now_rev > spread_rev);

  // WHEAT/FERTILIZER portfolio flow may be negative (feed/fertilizer consumption).
  sale_plan::Input consumption;
  consumption.production = {-1., 2.};
  consumption.rival = {0., 0.};
  consumption.demand = {0., 0.};
  auto impossible = sale_plan::solve(dp7::W, 10000., 0., consumption, false);
  std::printf("negative_arrival first_sale=%.0f next_stock=%.0f\n",
              impossible.sell[0], impossible.stock[1]);
  assert(impossible.stock[1] < 0.);
}
