#define R2_FLOW_AUDIT 1
#include "public_flow_scenario.hpp"
#include <cassert>
#include <cmath>

static dp7::View view_at(int step,fastkag::Farm&own,fastkag::Farm&rival,
                        fastkag::PrivateState&priv,fastkag::Market&market,
                        std::vector<int8_t>&shops){
 return {step,step/24,step%24,own,rival,priv,market,shops};
}

int main(){
 fastkag::Farm own,rival;own.tiles.resize(100);rival.tiles.resize(100);
 fastkag::PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);
 fastkag::Market market;for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=dp7::price(i,10000);}
 std::vector<int8_t>shops;competitive::Flow flow{};

 // A requested purchase larger than cash/capacity must be reported as a
 // short fill, with the exact cash, stock and residual-shed conservation.
 flow[1][dp7::W]=-600;
 fastkag::PublicFlowScenario buy(view_at(25,own,rival,priv,market,shops),flow,1.);
 buy.advance({});auto a=buy.audit();auto shed=buy.rival_shed();
 assert(a.requested[dp7::W]==-300&&a.rounded[dp7::W]==-300);
 assert(a.filled[dp7::W]<0&&a.filled[dp7::W]>a.rounded[dp7::W]);
 assert(a.market_delta[dp7::W]==a.filled[dp7::W]);
 assert(a.cash_delta[dp7::W]<0&&shed[dp7::W]==-a.filled[dp7::W]);

 // At the $1 floor every unit still fills and pays, but only the prefix before
 // saturation advances market inventory.
 flow={};flow[1][dp7::M]=6;market.inventory[dp7::M]=competitive::ConditionalMarket::saturation(dp7::M)-1;
 market.prices[dp7::M]=dp7::price(dp7::M,market.inventory[dp7::M]);
 fastkag::PublicFlowScenario sell(view_at(25,own,rival,priv,market,shops),flow,1.);
 sell.advance({});a=sell.audit();shed=sell.rival_shed();
 assert(a.requested[dp7::M]==3&&a.rounded[dp7::M]==3&&a.filled[dp7::M]==3);
 assert(a.market_delta[dp7::M]==1&&a.cash_delta[dp7::M]>=3&&shed[dp7::M]==0);
}
