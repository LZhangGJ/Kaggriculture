#include "policy/planner.hpp"
#include <iomanip>
#include <iostream>
int main(){
 int i;double stock,q;
 while(std::cin>>i>>stock>>q){double actual=stock,old=stock;
  double v=competitive::ConditionalMarket::execute(i,actual,q);
  double p=competitive::Planner::quote(i,old,q);if(q<0||p>1)old+=q;
  std::cout<<std::setprecision(17)<<i<<' '<<stock<<' '<<q<<' '<<v<<' '<<actual<<' '<<p*q<<' '<<old<<' '<<competitive::ConditionalMarket::saturation(i)<<'\n';
 }
}
