#pragma once
#ifndef A08_PAID_CONTINUATION
#define A08_PAID_CONTINUATION 1
#endif
namespace triad {
struct PaidTarget { int pos=-1,kind=-1,birth=-1; };
struct PaidContinuation {
 bool probe=false;
 int checked_day=-1,checks_today=0,last_cash=-1,last_hands=-1;
 int pending_day=-1,pending_step=-1,pending_workers=0;
 std::vector<PaidTarget> pending;
 int hire_day=-1,request_step=-1,request_count=0,request_workers=0,hire_deficit=0;
 int checks=0,alternatives=0,selected=0,hire_requested=0,observed_new_workers=0;
 int routes_committed=0,water_committed=0,unfilled=0,deadline_rejected=0;
 std::string last_json="{}";
};
struct Controller;
void paid_observe(Controller&,const dp7::View&);
void paid_record(Controller&,const dp7::View&,const fastkag::PlayerAction&);
bool paid_choose(Controller&,const dp7::View&,fastkag::PlayerAction&);
}
