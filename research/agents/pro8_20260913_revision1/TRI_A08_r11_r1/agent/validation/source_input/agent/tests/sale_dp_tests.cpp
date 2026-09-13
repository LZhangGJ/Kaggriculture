#define A08_SALE_SCHEDULE_DP 1
#include "../policy/sale_schedule_dp.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>
using namespace triad;using namespace sale_dp;
int checks=0;
void require(bool x,const char*name){checks++;if(!x)throw std::runtime_error(name);}
void distribution(const Schedule&s,int T,int Q,int minimum){int n=0;for(int t=0;t<T;t++){require(s[t]>=0,"negative shipment");n+=s[t];}require(n==Q,"quantity conservation");require(s[0]>=minimum,"delayed original cash");}
int main(){try{
 int cases=0,optimal=0,primitive=0,guard=0;std::cout<<"{\"scope\":\"synthetic unit/exhaustive tests; no games\",\"rows\":[";
 for(int item=1;item<=7;item++)for(int inv:{9900,9950,9980,9990,10000})for(int Q:{1,3,6,10})for(int T:{2,3,5}){
  Problem p;p.item=item;p.stock=inv;p.quantity=Q;p.steps=T;p.minimum_now=Q/3;p.competition=2;
  p.demand[T-2]=1+item%3;for(int t=0;t<T;t++)p.rival[t]=(t+item)%3==0?2+item%4:0;
  if(!supported(p))continue;Quotes quote(p);auto selected=select(p);distribution(selected.chosen,T,Q,p.minimum_now);
  require(selected.worst[selected.index]+1e-6>=selected.worst[0],"cross-evaluation degraded base");
  for(int scenario=0;scenario<4;scenario++){
   bool traffic=scenario>0;Ordering ord=Ordering(std::max(0,scenario-1));auto s=solve(p,quote,traffic,ord);auto cs=evaluate(p,quote,s,traffic,ord);double actual=cs.own-p.competition*cs.rival;
   double best=-1e100;Schedule brute{};
   std::function<void(int,int)> dfs=[&](int t,int r){
    if(t==T-1){brute[t]=r;auto c=evaluate(p,quote,brute,traffic,ord);best=std::max(best,c.own-p.competition*c.rival);return;}
    for(int a=(t==0?p.minimum_now:0);a<=r;a++){brute[t]=a;dfs(t+1,r-a);}
   };dfs(0,Q);require(std::abs(actual-best)<1e-7,"Bellman vs exhaustive");optimal++;
   for(auto profile:selected.alternatives){int stock=inv;Cash total;
    for(int t=0;t<T;t++){int r=traffic?p.rival[t]:0;auto expected=quote.trade(stock,profile[t],r,ord);auto c=reference_trade(item,stock,profile[t],r,ord);require(c.own==expected.own&&c.rival==expected.rival,"cached lockstep quotes");total.own+=c.own;total.rival+=c.rival;stock-=p.demand[t];primitive++;}
    auto predicted=evaluate(p,quote,profile,traffic,ord);require(total.own==predicted.own&&total.rival==predicted.rival,"complete cash path");
   }
  }
  if(cases++)std::cout<<",";
  std::cout<<"{\"item\":"<<item<<",\"stock\":"<<inv<<",\"quantity\":"<<Q<<",\"ticks\":"<<T<<",\"selected\":"<<selected.index<<",\"minimum_gain\":"<<selected.gain<<",\"exhaustive_scenarios\":4}";
 }
 for(int item=1;item<=7;item++){Problem p;p.item=item;p.stock=20001;p.quantity=5;p.steps=3;require(!supported(p),"out-of-domain fallback");guard++;}
 fastkag::Farm own,rival;own.money=500;fastkag::PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);priv.shed[dp7::MI]=6;
 fastkag::Market market;market.inventory.fill(9981);market.prices.fill(100);
 std::vector<int8_t>shops{6};dp7::View v{266,11,2,own,rival,priv,market,shops};SaleClock clock;
 for(int d=8;d<=10;d++){clock.rival[dp7::MI].add(d,4,2);clock.rival[dp7::MI].add(d,5,5);}
 dp7::Counts available{};available[dp7::MI]=6;std::array<int,9>dead;dead.fill(-1);dead[dp7::MI]=269;
 auto exercise=[&](int mode){SaleScheduleDP dp;auto left=available,sell=available;sell[dp7::MI]=0;auto dl=dead;auto h=clock;dp7::Acts buys;
  if(mode==1)own.money=119;else own.money=500;
  if(mode==2)buys.push_back(dp7::action(fastkag::Op::HIRE));
  if(mode==3)h=SaleClock{};
  if(mode==4)priv.shed[dp7::F]=84;else priv.shed[dp7::F]=0;
  dp.filter(v,priv,buys,left,sell,dl,h,120,2);
  require(sell[dp7::MI]>=0&&sell[dp7::MI]<=6,"actual stock bound");
  if(mode){require(sell[dp7::MI]==0,"guard must preserve baseline");require(dp.changed==0,"guard no new action");}else require(sell[dp7::MI]==6,"congested fixture advances actual batch");
  guard++;
 };for(int mode=0;mode<=4;mode++)exercise(mode);
 std::cout<<"],\"synthetic_problems\":"<<cases<<",\"exhaustive_scenario_optima\":"<<optimal<<",\"exact_path_transitions\":"<<primitive<<",\"guard_cases\":"<<guard<<",\"assertions\":"<<checks<<",\"passed\":true}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
