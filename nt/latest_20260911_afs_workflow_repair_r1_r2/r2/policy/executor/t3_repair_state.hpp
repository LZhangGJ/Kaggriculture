#pragma once
#ifndef T3_OBLIGATION_REPAIR
#define T3_OBLIGATION_REPAIR 1
#endif
#ifndef T3_CAPACITY_REPAIR
#define T3_CAPACITY_REPAIR 1
#endif
#ifndef T3_RECEIPT_REPAIR
#define T3_RECEIPT_REPAIR 1
#endif
namespace dp7 {
struct T3RepairState {
 // Value-owned snapshot of the CURRENT P16 sale policy. A copied day probe
 // advances its own deadlines; it cannot mutate the live controller's memory.
 std::function<void(const View&,const fastkag::PrivateState&,
   const std::vector<fastkag::Action>&,std::array<int,12>&)> project_sales;
 bool suspended=false;
 int transport_admissions=0;
 int obligation_checks=0,water_insertions=0,water_unresolved=0;
 int capacity_checks=0,capacity_rollouts=0,delivery_insertions=0;
 int projected_overflow_saved=0,receipt_checks=0,receipt_seeds=0,receipt_hires=0;
 int receipt_expected_hands=0,receipt_day=-1,receipt_attempt_day=-1;
 int obligation_last_step=-1,capacity_last_step=-1;
 std::array<int8_t,100> missing_water{};
};
}
