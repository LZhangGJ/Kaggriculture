#include "policy/planner.hpp"
#include "policy/sale_clock.hpp"
#include <cassert>
#include <iostream>
int main(){
 triad::SaleClock c;
 for(double q:{-31.,-3.6,-1.,0.,.3,.8,1.,3.6,30.,51.2})for(int i=0;i<9;i++){
  int sum=0;for(int h=0;h<24;h++){int x=c.quantity(i,q,h);assert(x==(h==1?int(std::floor(q*.5)):h==17?int(std::round(q))-int(std::floor(q*.5)):0));sum+=x;}assert(sum==int(std::round(q)));
 }
 c.rival[7].add(5,10,2);assert(c.rival[7].count()==1);assert(c.quantity(7,10,1)==5);
 c.rival[7].add(7,10,8);assert(c.rival[7].count()==2);
 auto p=c.rival[7].density();assert(std::abs(p[10]-2./3)<1e-12);assert(std::abs(std::accumulate(p.begin(),p.end(),0.)-1)<1e-12);
 for(double q:{.1,.8,1.,3.6,30.2,91.9}){int sum=0;for(int h=0;h<24;h++){int x=c.quantity(7,q,h);assert(x>=0);sum+=x;}assert(sum==int(std::round(q)));}
 c.own[7]=c.rival[7];assert(std::abs(c.early_shares()[7]-.5)<1e-12);
 c.rival[7].add(9,20,5);c.rival[7].add(11,21,7);assert(c.rival[7].day[0]==7&&c.rival[7].day[2]==11);
 assert(c.rival[6].count()==0);
 triad::PublicTradeLedger l;l.source_step=49;l.valid[6]=1;l.rival_net[6]=12;l.own_valid[6]=1;l.own_sales[6]=3;c.observe(l);
 assert(c.rival[6].day[2]==2&&c.rival[6].sales[2][1]==12&&c.own[6].sales[2][1]==3);
 triad::SaleClock fresh;for(int i=0;i<9;i++)assert(fresh.rival[i].count()==0);
 std::cout<<"PASS clock fallback, past-day buffer, independent products, fractional quantity conservation, public-only update and reset\n";
}
