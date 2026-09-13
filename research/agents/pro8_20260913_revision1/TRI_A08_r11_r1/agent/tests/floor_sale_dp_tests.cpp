#define A08_SALE_SCHEDULE_DP 1
#include "../policy/sale_schedule_dp.hpp"
#include <iostream>
#include <chrono>
using namespace triad;
using namespace sale_dp;
static long assertions=0,problems=0,optima=0,primitives=0;
static void require(bool x,const char* message){++assertions;if(!x)throw std::runtime_error(message);}
static Cash independent(const Problem&p,const Schedule&s,bool traffic,Ordering order){
 int stock=p.stock;Cash total;
 for(int t=0;t<p.steps;t++){
  auto c=reference_trade(p.item,stock,s[t],traffic?p.rival[t]:0,order);
  total.own+=c.own;total.rival+=c.rival;stock-=p.demand[t];
 }return total;
}
static void check(const Problem&p,bool exhaustive){
 require(supported(p),"unexpected unsupported fixture");Quotes q(p);auto selected=select(p);
 int total=0;for(int t=0;t<p.steps;t++){require(selected.chosen[t]>=0,"negative sale");total+=selected.chosen[t];}
 require(total==p.quantity,"physical quantity not conserved");require(selected.chosen[0]>=p.minimum_now,"delays parent's immediate sale");
 require(selected.worst[selected.index]>=selected.worst[0]-1e-7,"scenario-set rank regressed");
 for(int scenario=0;scenario<4;scenario++){
  const bool traffic=scenario>0;const Ordering order=Ordering(std::max(0,scenario-1));
  auto s=solve(p,q,traffic,order);auto c=independent(p,s,traffic,order);auto fast=evaluate(p,q,s,traffic,order);
  require(c.own==fast.own&&c.rival==fast.rival,"optimized quotes vs independent full path");
  if(exhaustive){
   double best=-1e100;Schedule brute{};
   std::function<void(int,int)> walk=[&](int t,int remaining){
    if(t==p.steps-1){brute[t]=remaining;auto b=independent(p,brute,traffic,order);best=std::max(best,b.own-p.competition*b.rival);return;}
    for(int amount=t==0?p.minimum_now:0;amount<=remaining;amount++){brute[t]=amount;walk(t+1,remaining-amount);}
   };walk(0,p.quantity);
   require(std::abs(c.own-p.competition*c.rival-best)<1e-7,"saturating Bellman vs independent exhaustive search");++optima;
  }
  for(auto a:selected.alternatives){
   auto expected=independent(p,a,traffic,order);auto actual=evaluate(p,q,a,traffic,order);
   require(expected.own==actual.own&&expected.rival==actual.rival,"candidate scenario cash not exact");
  }
 }
 ++problems;
}
int main(){auto start=std::chrono::steady_clock::now();try{
 // Off-floor egg curves must not overflow an int while finding a remote floor.
 require(saturation_stock(dp7::E)==20001,"egg out-of-domain floor sentinel");
 for(int item=1;item<=7;item++){
  int floor=saturation_stock(item);
  for(int offset:{-15,-7,-2,-1,0,1})for(int own:{0,1,2,3,7,24,100})for(int rival:{0,1,2,3,7,24,100})for(int o=0;o<3;o++){
   int stock=(floor>20000?10000:floor)+offset;int expected_stock=stock;
   auto expected=reference_trade(item,expected_stock,own,rival,Ordering(o));
   require(next_stock(item,stock,own,rival,Ordering(o))==expected_stock,"saturation/paired overshoot stock transition");
   if(own>0){Problem p;p.item=item;p.stock=stock;p.quantity=own;p.steps=1;p.rival[0]=rival;
    if(supported(p)){Quotes q(p);auto actual=q.trade(stock,own,rival,Ordering(o));require(actual.own==expected.own&&actual.rival==expected.rival,"cached quotes across price floor");}}
   ++primitives;
  }
  if(floor>20000)continue;
  require(dp7::price(item,floor)==1&&dp7::price(item,floor-1)>1,"floor boundary");
  require(next_stock(item,floor-1,1,1,LOCKSTEP)==floor+1,"last paired nonfloor quote must overshoot by one");
  for(int offset:{-15,-7,-2,-1,0,1})for(int Q:{1,3,7})for(int T:{2,3,5})for(int shape=0;shape<2;shape++){
   Problem p;p.item=item;p.stock=floor+offset;p.quantity=Q;p.steps=T;p.minimum_now=shape?Q/3:0;p.competition=shape?2:1;
   p.demand[T-2]=1+item%4;
   if(shape&&T>2)p.demand[0]++;
   for(int t=0;t<T;t++)p.rival[t]=(t+shape)%3==0?3+item%5:0;
   check(p,true);
  }
  // Practical worst quantity, broader finite traffic, non-exhaustive stress.
  for(int T:{2,3,5})for(int offset:{-90,-1,1}){
   Problem p;p.item=item;p.stock=floor+offset;p.quantity=100;p.steps=T;p.minimum_now=0;p.competition=2;
   p.demand[T-2]=16;for(int t=0;t<T;t++)p.rival[t]=t%2?35:12;
   check(p,false);
  }
 }
 Problem p;p.item=dp7::WO;p.stock=saturation_stock(p.item)-12;p.quantity=24;p.steps=4;p.minimum_now=0;p.demand[2]=4;p.rival[1]=18;p.rival[2]=12;p.competition=2;
 auto good=select(p);require(good.chosen[0]>0,"congestion fixture should advance real stock");
 Problem bad=p;bad.stock=20001;require(!supported(bad),"out-of-domain stock guard");bad=p;bad.quantity=101;require(!supported(bad),"inventory capacity guard");bad=p;bad.demand[0]=-1;require(!supported(bad),"negative demand guard");bad=p;bad.item=-1;require(!supported(bad),"invalid product guard");
 const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 std::cout<<"{\"scope\":\"synthetic exact floor DP tests; not games\",\"passed\":true,\"problems\":"<<problems<<",\"independent_exhaustive_optima\":"<<optima<<",\"primitive_cases\":"<<primitives<<",\"assertions\":"<<assertions<<",\"seconds\":"<<seconds<<",\"congestion_example_first_sale\":"<<good.chosen[0]<<"}\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"FAIL after "<<assertions<<" assertions / "<<problems<<" problems: "<<e.what()<<"\n";return 1;}}
