#define A08_SALE_SCHEDULE_DP 1
#include "../policy/sale_schedule_dp.hpp"
#include <iostream>
using namespace triad::sale_dp;
int main(){try{
 Problem p;p.item=3;p.stock=10055;p.quantity=24;p.steps=2;p.rival[1]=1;p.demand[0]=3;p.competition=3;
 auto weighted=select(p);p.competition=1;auto actual=select(p);Quotes q(p);
 auto cash_margin=[&](const Schedule&s){double worst=1e100;for(int scene=0;scene<4;scene++){auto c=evaluate(p,q,s,scene>0,Ordering(std::max(0,scene-1)));worst=std::min(worst,c.own-c.rival);}return worst;};
 double old=cash_margin(weighted.chosen),base=cash_margin(actual.original),now=cash_margin(actual.chosen);
 if(!(old<base&&now>=base&&now>old))throw std::runtime_error("cash-margin regression was not repaired");
 std::cout<<"{\"scope\":\"synthetic low-traffic floor batch; not a match\",\"macro_weight3_selected_now\":"<<weighted.chosen[0]<<",\"cash_margin_selected_now\":"<<actual.chosen[0]<<",\"old_weighted_min_actual_margin\":"<<old<<",\"baseline_min_actual_margin\":"<<base<<",\"new_min_actual_margin\":"<<now<<",\"passed\":true}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
