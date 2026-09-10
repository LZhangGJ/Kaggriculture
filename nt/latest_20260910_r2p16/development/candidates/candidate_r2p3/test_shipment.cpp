#include "policy/planner.hpp"
#include "policy/shipment_value.hpp"
#include <cassert>
#include <cmath>
int main(){
 competitive::Asset a;
 a.f[4][dp7::W]=-7;a.f[4][dp7::MI]=18;a.f[5][dp7::MI]=3;
 a.f[29][dp7::WO]=9;a.fixed[4]=-100;a.labor[4]=14;a.first_cost=100;
 competitive::shipment_value(a);
 constexpr double lag=R2_PRODUCT_HANDOFF_DELAY;
 assert(a.f[4][dp7::W]==-7);
 assert(a.f[4][dp7::MI]==18*(1-lag));
 assert(a.f[5][dp7::MI]==3*(1-lag)+18*lag);
 assert(a.f[6][dp7::MI]==3*lag);
 assert(a.f[29][dp7::WO]==9);
 assert(a.fixed[4]==-100&&a.labor[4]==14&&a.first_cost==100);
 double total=0;for(auto& day:a.f)total+=day[dp7::MI];assert(total==21);
}
