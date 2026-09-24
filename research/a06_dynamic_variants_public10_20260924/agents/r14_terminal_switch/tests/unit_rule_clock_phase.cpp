#include "../policy/search.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
static int checks=0;
static void ck(bool ok,const char*name){++checks;if(!ok){std::cerr<<name<<"\n";std::exit(1);}}
int main(){
 triad::SaleClock clock;
 int total=0;
 for(int h=0;h<24;h++){int q=clock.quantity(4,100,h);ck(q==((h==1||h==17)?50:0),"unobserved clock fallback");total+=q;}
 ck(total==100,"fallback conservation");
 clock.rival[4].add(0,5,8);
 ck(clock.quantity(4,100,1)==50,"one observed day still falls back");
 clock.rival[4].add(1,10,14);
 int expected[24]={};expected[1]=1;expected[5]=2;expected[10]=2;expected[17]=1;
 total=0;
 for(int h=0;h<24;h++){int q=clock.quantity(4,6,h);ck(q==expected[h],"two days use regularized public arrival density");total+=q;}
 ck(total==6,"observed clock conservation");
 ck(!triad::learned::available,"trained model disabled");
 triad::Settings invalid;invalid.scenario=-1;bool rejected=false;
 try{triad::SearchController c(invalid);}catch(const std::invalid_argument&){rejected=true;}
 ck(rejected,"learned scenario cannot be enabled");
 for(int mode:{0,3})for(int day:{0,13,14,29}){
  fastkag::Simulator sim(fastkag::Config{},0);
  dp7::View v{day*24,day,0,sim.farms()[0],sim.farms()[1],sim.privates()[0],sim.market(),sim.shops()};
  triad::Settings setting;setting.scenario=0;setting.layout=mode;setting.competition=2;
  triad::SearchController c(setting);c.act(v);
  ck(c.base.competition==((mode==3&&day>=14)?1:2),"day-14 competition phase boundary");
  ck(c.nominal.competition==2,"nominal settings retained");
 }
 std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"scope\":\"public sale-clock conservation, no-model rejection, phase-14 boundary\"}\n";
}
