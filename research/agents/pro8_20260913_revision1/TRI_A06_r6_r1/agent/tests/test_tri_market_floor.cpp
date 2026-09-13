// One production fix: no fictitious supply from a sale at the $1 floor.
#include "policy/planner.hpp"
#include <iostream>
#include <cassert>
#include <iomanip>
#include <cstring>
using competitive::Planner;using competitive::ConditionalMarket;
static double legacy(int i,double& stock,double q){double p=Planner::quote(i,stock,q);if(q<0||p>1.)stock+=q;return q*p;}
static bool bits(double a,double b){return std::memcmp(&a,&b,sizeof(double))==0;}
int main(int argc,char**argv){
 int positive=0,negative=0,fractional=0,detected=0,recovery=0;
 bool export_rows=argc>1&&std::string(argv[1])=="--rows";
 for(int i=0;i<9;i++){
  double sat=ConditionalMarket::saturation(i);
  assert(dp7::price(i,sat)==1&&dp7::price(i,sat-1)>1);
  for(int d=-12;d<=12;d++)for(int q:{1,2,3,5,8,13,21,34,55,89,100}){
   double s=sat-d,a=s,b=s,c=s;double pa=Planner::trade(i,a,q),pb=legacy(i,b,q),pc=ConditionalMarket::execute(i,c,q);
   if(s+q>sat){positive++;assert(pa==pc&&a==c);if(pa!=pb||a!=b)detected++;
    if(export_rows)std::cout<<std::setprecision(17)<<i<<' '<<s<<' '<<q<<' '<<pa<<' '<<a<<'\n';
   }else{negative++;assert(bits(pa,pb)&&bits(a,b));}
  }
  // Purchases, zero flows and all ordinary non-crossing estimates are unchanged.
  for(double s:{9700.,10000.,sat-15,sat,sat+15})for(double q:{-100.,-31.,-10.,-3.,-2.,-1.,0.}){
   double a=s,b=s;double pa=Planner::trade(i,a,q),pb=legacy(i,b,q);negative++;assert(bits(pa,pb)&&bits(a,b));
  }
  for(double d:{-3.75,-.25,0.,.25,.75,1.5,3.5})for(double q:{.25,.75,1.5,3.75,15.5}){
   double s=sat-d;if(s+q<=sat)continue;
   double a=s,b=s;double pa=Planner::trade(i,a,q),pb=ConditionalMarket::execute(i,b,q);fractional++;assert(pa==pb&&a==b);
  }
  // A subsequent known demand pulse starts from true counted inventory, not
  // discarded floor units. These are conditional primitive trajectories only.
  double a=sat-2,old=a;Planner::trade(i,a,15);legacy(i,old,15);
  assert(a==sat);a-=3;double exact=a;auto cash=Planner::trade(i,a,10),truth=ConditionalMarket::execute(i,exact,10);
  assert(cash==truth&&a==exact);recovery++;
 }
 if(!export_rows)std::cout<<"{\"positive_cases\":"<<positive<<",\"legacy_failures\":"<<detected<<",\"negative_bit_identical\":"<<negative<<",\"fractional_cases\":"<<fractional<<",\"demand_recovery_cases\":"<<recovery<<"}\n";
 return 0;
}
