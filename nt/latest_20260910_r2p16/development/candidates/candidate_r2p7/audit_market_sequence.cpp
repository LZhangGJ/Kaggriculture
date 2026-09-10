#include "policy/planner.hpp"
#include <iostream>
#include <iomanip>
int main(){
 int day;double supply,competition;
 while(std::cin>>day>>supply>>competition){
  std::array<double,9>old{},fixed{};competitive::Flow dem{},own{},rival{};
  for(auto&x:old)std::cin>>x;fixed=old;
  for(auto*p:{&dem,&own,&rival})for(auto&row:*p)for(auto&x:row)std::cin>>x;
  int crosses=0,first=-1;std::array<int,9>by{};double before=0,after=0;
  for(int d=day;d<30;d++)for(int i=0;i<9;i++){
   old[i]-=dem[d][i]*.5;fixed[i]-=dem[d][i]*.5;
   double r=rival[d][i]*supply*.5;
   for(int part=0;part<3;part++){
    double q=part==1?own[d][i]:r,weight=part==1?1:-competition;
    double price=competitive::Planner::quote(i,old[i],q),clipped=old[i];
    competitive::ConditionalMarket::update(i,clipped,q);
    if(q<0||price>1)old[i]+=q;
    if(std::abs(old[i]-clipped)>1e-7){crosses++;by[i]++;if(first<0)first=d;}
    before+=weight*q*price;
    after+=weight*competitive::ConditionalMarket::execute(i,fixed[i],q);
   }
   old[i]-=dem[d][i]*.5;fixed[i]-=dem[d][i]*.5;
  }
  std::cout<<crosses<<' '<<first<<' '<<std::setprecision(17)<<after-before;
  for(auto x:by)std::cout<<' '<<x;
  std::cout<<std::endl;
 }
}
