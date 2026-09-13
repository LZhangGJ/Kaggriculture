#include "policy/animal_service_dp.hpp"
#include <cstdio>
using namespace competitive;
int main() {
 Flow prices{};for(auto &d:prices){d.fill(30.);d[W]=25.;d[F]=2.;}
 AnimalServiceDP poor;poor.solve(9,21,28,prices,4.);
 for(auto &d:prices)d[E]=100.;
 AnimalServiceDP rich;rich.solve(9,21,28,prices,4.);
 auto q=rich.choices[28][1][0];
 std::printf("poor.feed=%d poor.value=%.17g rich.feed=%d rich.care=%d rich.choice_value=%.17g rich.value=%.17g terminal=%.17g\n",poor.choices[28][1][0].feed,poor.value[28][1][0],q.feed,q.care,q.value,rich.value[28][1][0],rich.value[29][0][0]);
 return !(poor.choices[28][1][0].feed==0 && poor.value[28][1][0]==0 && q.feed==1 && q.care==0 && q.value==65 && rich.value[28][1][0]==65 && rich.value[29][0][0]==0);
}
