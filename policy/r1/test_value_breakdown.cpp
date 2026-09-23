#include "planner.hpp"
#include <cassert>
#include <cmath>

int main(){
 competitive::Planner p;p.day=12;p.cfg.competition=2;p.cfg.discount=.015;p.cfg.risk=.25;
 fastkag::Farm own,rival;own.money=100;own.tiles.resize(100);rival.tiles.resize(100);
 fastkag::PrivateState priv;fastkag::Market market;market.inventory.fill(10000);std::vector<int8_t>shops;
 dp7::View view{288,12,0,own,rival,priv,market,shops};
 competitive::Asset a;a.fixed[12]=-30;a.f[12][0]=4;a.labor[12]=8;
 p.rival[12][0]=3;
 competitive::ValueBreakdown b;double value=p.value(view,a,nullptr,-1,&b);
 double sum=b.fixed+b.own_trade+b.wages+b.actions+b.rival_penalty+b.liquidity_penalty;
 assert(std::abs(value-b.total)<1e-9);
 assert(std::abs(value-sum)<1e-9);
 competitive::ValueBreakdown frozen;competitive::ValueBasis basis{11,100};
 double shifted=p.value(view,a,nullptr,-1,&frozen,&basis);
 assert(std::abs(shifted-frozen.total)<1e-9);
 assert(std::abs(shifted-value/(1+p.cfg.discount*(1-100./20000)))<1e-9);

 // Town demand raises the later wool price, so a cash-maximising rival may profitably carry its
 // day-12 output.  The diagnostic delta must therefore expose an adverse c=2 timing correction.
 p.dem={};p.rival={};a={};market.inventory[dp7::WO]=10050;
 for(int d=12;d<=14;d++)p.dem[d][dp7::WO]=20;
 p.rival[12][dp7::WO]=20;
 double stock_delta=p.rival_stock_response_delta(view,a,dp7::WO);
 assert(std::isfinite(stock_delta)&&stock_delta<0);
}
