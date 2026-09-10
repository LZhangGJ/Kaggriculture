#define R2_LOCAL_SALE_TIMING 2
#include "policy/triad.hpp"
#include <iostream>
#include <cassert>
int main(){using namespace triad;using namespace dp7;
 for(int s=0;s<719;s++){int d=LocalSaleTiming::due(s);assert(d>s&&d-s<=4&&(d-1)%4==0);}
 Farm a,b;a.money=1000;PrivateState p;p.inventories.resize(1);p.inventory_order.resize(1);p.shed[S]=8;
 Market m;for(int i=0;i<9;i++){m.inventory[i]=10000;m.prices[i]=price(i,10000);}
 std::vector<int8_t>shops{3};View v{96,4,0,a,b,p,m,shops};SaleClock history;Counts sell{};sell[S]=8;
 LocalSaleTiming timer;timer.filter(v,p,{},sell,history,120,2);assert(timer.holds==1&&timer.deadline[S]==97&&sell[S]<8);
 View next{97,4,1,a,b,p,m,shops};sell[S]=8;timer.filter(next,p,{},sell,history,120,2);assert(sell[S]==8&&timer.deadline[S]<0);
 for(int guard=0;guard<4;guard++){auto priv=p;auto farm=a;Acts buys;int step=96;
  if(guard==0)farm.money=119;if(guard==1)buys.push_back(action(Op::BUY_SEED,S,1));if(guard==2)priv.shed[W]=90;if(guard==3)step=695;
  View x{step,step/24,step%24,farm,b,priv,m,shops};LocalSaleTiming t;Counts s{};s[S]=8;t.filter(x,priv,buys,s,history,120,2);assert(s[S]==8&&t.holds==0);
 }
 // A fixed existing deadline is never rolled to the next consumption window.
 LocalSaleTiming fixed;fixed.deadline[S]=101;View before{99,4,3,a,b,p,m,shops};sell[S]=8;fixed.filter(before,p,{},sell,history,120,2);assert(fixed.deadline[S]==101&&sell[S]==0);
 int item,inv,q,n,dem,rival;double c;
 std::cout.precision(17);
 while(std::cin>>item>>inv>>q>>n>>dem>>rival>>c){double gain;int selected=LocalSaleTiming::amount_now(item,inv,q,dem,rival,c,&gain);std::cout<<LocalSaleTiming::payoff(item,inv,q,n,dem,rival,c)<<" "<<selected<<" "<<gain<<"\n";}
}
