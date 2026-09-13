#define A08_SALE_SCHEDULE_DP 1
#include "../policy/sale_schedule_dp.hpp"
#include <iostream>
using namespace triad;
int main(){try{
 int tests=0;auto require=[&](bool x,const char*why){tests++;if(!x)throw std::runtime_error(why);};
 fastkag::Farm own,rival;own.money=500;fastkag::PrivateState priv;
 priv.inventories.resize(1);priv.inventory_order.resize(1);priv.shed[dp7::MI]=9;
 fastkag::Market market;market.inventory.fill(10000);market.inventory[dp7::MI]=10063;market.prices.fill(100);
 std::vector<int8_t>shops{6};dp7::View v{434,18,2,own,rival,priv,market,shops};SaleClock clock;
 for(int d=15;d<=17;d++)clock.rival[dp7::MI].add(d,5,9);
 dp7::Counts available{};available[dp7::MI]=9;
 auto run=[&](int anchor,int stock,int quantity,bool buying,double weight=2.){
  market.inventory[dp7::MI]=stock;available[dp7::MI]=quantity;
  auto sell=available;sell[dp7::MI]=0;std::array<int,9>dead;dead.fill(-1);dead[dp7::MI]=437;
  SaleScheduleDP dp;dp.floor_observation_step=anchor;dp7::Acts buys;
  if(buying)buys.push_back(dp7::action(fastkag::Op::HIRE));
  dp.filter(v,priv,buys,available,sell,dead,clock,120,weight);
  require(sell[dp7::MI]>=0&&sell[dp7::MI]<=quantity,"cannot sell beyond executable surplus");
  return std::pair(dp,sell[dp7::MI]);
 };
 auto actual=run(434,10063,9,false);require(actual.first.floor_eligible==1&&actual.second==9,"actual saturated observation should advance batch");
 for(int anchor:{-1,433,435,430}){auto future=run(anchor,10063,9,false);require(future.first.floor_eligible==0&&future.second==0,"synthetic/future observation cannot authorize new floor branch");}
 for(double weight:{.5,1.,2.,3.}){auto x=run(434,10063,9,false,weight);require(x.second==actual.second,"macro preference leaked into new cash-margin floor action");require(x.first.last_json.find("\"objective_opponent_cash_weight\":1")!=std::string::npos,"floor objective must be actual cash difference");}
 auto reserve=run(434,10063,5,false);require(reserve.second<=5,"reserved stock sold");
 auto no_stock=run(434,10063,0,false);require(no_stock.second==0,"phantom future stock sold");
 auto buying=run(434,10063,9,true);require(buying.first.floor_eligible==0&&buying.second==0,"current purchase order must be preserved");
 auto ordinary=run(-1,9981,9,false);require(ordinary.first.eligible==1,"non-floor parent DP must remain enabled in forecasts");
 require(priv.shed[dp7::MI]==9,"prediction mutated actual inventory");
 std::cout<<"{\"scope\":\"real-observation authorization and physical-surplus guards; no games\",\"assertions\":"<<tests<<",\"passed\":true}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
