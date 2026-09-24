#pragma once
// Included inside dp7 after Job/Order. No forecast cash or crop revenue is
// created here: this only makes an existing wage reservation executable.
namespace a06 {
inline bool ongoing_survival_water(const View& o, const std::vector<Job>& jobs) {
 for (const auto& job : jobs) {
  if (job.pos < 0 || job.pos >= 100) continue;
  const auto& tile = o.own.tiles[job.pos];
  if (!plant(tile) || !ongoing(int(tile.crop)) || tile.watered_today ||
      tile.consecutive_unwatered < 1) continue;
  // A replacement's water is not maintenance of the incumbent plant.
  bool water = false, replace = false;
  for (auto a : job.actions) {
   water |= a.op == Op::WATER;
   replace |= a.op == Op::DIG || a.op == Op::PLANT;
  }
  if (!water || replace) continue;
  const int kind = int(tile.crop);
  for (int wave = 0; wave < 4; ++wave) {
   const int production_day = tile.planted_day + first[kind] + wave * interval[kind];
   if (production_day > o.day && production_day <= 29) return true;
  }
 }
 return false;
}
inline bool reserve_ongoing_wages(const View& o, const std::vector<Job>& jobs,
                                const Acts& sales, double admitted_cost,
                                int hires, int wages) {
 // A fully cash-funded plan retains the parent ordering. Unconfirmed sale
 // receipts must not fund capital before wages already reserved by admit().
 return hires > 0 && !sales.empty() && o.own.money < admitted_cost + wages &&
        ongoing_survival_water(o, jobs);
}
inline Acts preparation_with_wage_reserve(const Acts& sales,
                                         const std::vector<Order>& purchases,
                                         int hires, bool protect) {
 Acts result = sales;
 if (!protect) {
  for (const auto& purchase : purchases) result.push_back(purchase.a);
  for (int i = 0; i < hires; ++i) result.push_back(action(Op::HIRE));
  return result;
 }
 auto current_feed = [](const Order& x) {
  return x.priority == 0 && x.a.op == Op::BUY_PRODUCT && int(x.a.item) == W;
 };
 // Do not starve existing animals to finance the maintenance workforce.
 // Future feed buffers and fertilizer retain their original later priority.
 for (const auto& purchase : purchases) if (current_feed(purchase)) result.push_back(purchase.a);
 for (int i = 0; i < hires; ++i) result.push_back(action(Op::HIRE));
 for (const auto& purchase : purchases) if (!current_feed(purchase)) result.push_back(purchase.a);
 return result;
}
} // namespace a06
