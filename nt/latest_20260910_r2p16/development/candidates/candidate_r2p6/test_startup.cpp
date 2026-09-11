#include "policy/search.hpp"
#include <cassert>
#include <iostream>
int main(){
 triad::Settings s;s.animal_bias=1.25;s.competition=2;s.reserve=120;
 triad::SearchController c(s);
 double factor=R2_STARTUP_MODE==1?.6:R2_STARTUP_MODE==2?1.6:1.;
 assert(c.base.animal_bias==1.25*factor);assert(c.live.s.animal_bias==1.25*factor);
 c.restore_startup(0);assert(c.base.animal_bias==1.25*factor);
 c.restore_startup(1);assert(c.base.animal_bias==s.animal_bias);assert(c.live.s.animal_bias==s.animal_bias);
 assert(c.base.competition==s.competition&&c.base.reserve==s.reserve);
 c.live.s.animal_bias=9.;c.restore_startup(2);assert(c.live.s.animal_bias==9.);
 triad::SearchController mid(s);mid.restore_startup(17);assert(mid.base.animal_bias==s.animal_bias);
 std::cout<<"PASS startup mode "<<R2_STARTUP_MODE<<"\n";
}
