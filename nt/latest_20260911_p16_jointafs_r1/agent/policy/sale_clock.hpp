#pragma once
#include "public_trade_ledger.hpp"
#ifndef R2_SALE_CLOCK_MODE
#define R2_SALE_CLOCK_MODE 0
#endif
namespace triad {
struct SaleClock {
 struct Product {
  std::array<int,3> day{{-1,-1,-1}};
  std::array<std::array<double,24>,3> sales{};
  void add(int d,int h,int q){
   if(q<=0)return;
   if(day[2]!=d){day[0]=day[1];day[1]=day[2];day[2]=d;sales[0]=sales[1];sales[1]=sales[2];sales[2]={};}
   sales[2][h]+=q;
  }
  int count()const{int n=0;for(int d:day)n+=d>=0;return n;}
  std::array<double,24> density()const{
   std::array<double,24> p{};p[1]=p[17]=.5;
   int n=count();if(n<2)return p;
   for(int k=0;k<3;k++)if(day[k]>=0){double total=std::accumulate(sales[k].begin(),sales[k].end(),0.);for(int h=0;h<24;h++)p[h]+=sales[k][h]/total;}
   for(double&x:p)x/=1+n;return p;
  }
 };
 std::array<Product,9> own{},rival{};
 void observe(const PublicTradeLedger&ledger){
  int s=ledger.source_step;if(s<0)return;
  for(int i=1;i<=7;i++){
   if(ledger.valid[i])rival[i].add(s/24,s%24,ledger.rival_net[i]);
   if(ledger.own_valid[i])own[i].add(s/24,s%24,ledger.own_sales[i]);
  }
 }
 std::array<double,9> early_shares()const{
  std::array<double,9>a{};a.fill(.5);
  for(int i=1;i<=7;i++)if(own[i].count()>=2 && rival[i].count()>=2){
   auto ours=own[i].density(),theirs=rival[i].density();double cdf=0,p=0;
   for(int h=0;h<24;h++){p+=ours[h]*(cdf+.5*theirs[h]);cdf+=theirs[h];}a[i]=p;
  }return a;
 }
 int quantity(int item,double total,int hour)const{
  if(item<1||item>7||total<=0||rival[item].count()<2){
   return hour==1?int(std::floor(total*.5)):hour==17?int(std::round(total))-int(std::floor(total*.5)):0;
  }
  auto p=rival[item].density();double before=0;for(int h=0;h<hour;h++)before+=p[h];double after=before+p[hour];
  auto cumulative=[&](double c){return c>=1-1e-12?int(std::round(total)):int(std::floor(total*c+1e-12));};
  return cumulative(after)-cumulative(before);
 }
};
}
