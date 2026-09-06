#pragma once
#include "local_reference.hpp"
#include "executor/investment_candidates.hpp"
namespace competitive {
// Canonical selection deliberately retains candidate deduplication semantics.
// A faster direct semantic shortcut failed equivalence and was removed.
inline dp7::Controller reference_pick(const dp7::Controller&incoming,const dp7::View&o){
 auto menu=dp7branch::generate(incoming,o);int chosen=0;
 if(o.day<29){auto&r=switch_plans[0][o.day];for(int n=0;n<int(menu.size());n++)if(menu[n].family==r.family&&menu[n].kind==r.kind&&menu[n].amount==r.amount){chosen=n;break;}}
 return std::move(menu[chosen].controller);
}
}
