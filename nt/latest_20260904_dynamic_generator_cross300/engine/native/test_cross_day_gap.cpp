#include "policy.hpp"
#include <iostream>
using namespace dp7;
int main(){
 Simulator sim(Config{},9182);while(sim.day()<12)sim.step({});
 Farm f=sim.farms()[0];f.tiles.assign(100,Tile{});f.money=5000;f.farmer={4,4};
 PrivateState p=sim.privates()[0];p.shed={};p.seeds={};p.inventories.assign(1,{});p.inventory_order.assign(1,{});
 Controller c;c.day=12;c.target={{44,W}};c.p.max_land=1;c.p.operating_reserve=100;c.p.economic_land=true;
 View v{288,12,0,f,sim.farms()[1],p,sim.market(),sim.shops()};
 auto empty=dayvalue::residual(c,v);auto abandoned=c;abandoned.target={{44,-1}};auto stopped=dayvalue::residual(abandoned,v);
 auto cal=portfolio::crop_delivery(c,v,W,Tile{});double future=0;for(int d=12;d<30;d++)future+=cal.cash.quantity[d][W];
 c.new_day(v);int proposed=0,crop_count=0;for(auto[pos,k]:c.target){proposed+=k>=0;crop_count+=k>=0&&k<5;}
 bool purchase=false;for(auto a:c.queue)purchase|=a.op==Op::BUY_SEED;
 if(!empty.known||empty.calendars!=0||empty.score!=5000||stopped.score!=empty.score||future<=0||proposed<=0||!purchase)return 2;
 // This establishes missing continuation coverage, NOT optimal actions or cash.
 std::cout<<"{\"status\":\"PASS_REPRODUCED_OMITTED_OPTION\",\"empty_value\":"<<empty.score<<",\"stopped_value\":"<<stopped.score<<",\"empty_works\":"<<empty.calendars<<",\"conditional_wheat_future_units\":"<<future<<",\"next_day_proposed\":"<<proposed<<",\"next_day_crops\":"<<crop_count<<",\"seed_purchase_proposed\":"<<(purchase?"true":"false")<<"}\n";
}
