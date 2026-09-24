#pragma once
#include "planner.hpp"
namespace competitive {
// Conditional extra harvest at a fixed deadline, net of fertilizer and work.
// Bounded maintenance heuristic; not globally optimal crop/route planning.
inline bool finite_fertilizer_choice(int k,int age,int finish_age,int quantity,
 bool will_water,bool already_watered,int fertile_remaining,
 double harvest_price,double fertilizer_price,double action_cost){
 if(!will_water||already_watered||fertile_remaining>0||quantity<0)return false;
 int last=k==W?4:k==C?3:12,cap=k==C?4:6;
 if(age<(last+1)/2||age>last||age>finish_age||quantity>=cap)return false;
 int remaining=std::min(last,finish_age)-age+1;
 int natural=std::min(cap,quantity+remaining);
 int boosted=std::min(cap,quantity+remaining+std::min(3,remaining));
 return (boosted-natural)*harvest_price>fertilizer_price+action_cost+1e-9;
}
}
