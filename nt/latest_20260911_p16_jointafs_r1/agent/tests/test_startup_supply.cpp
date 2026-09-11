#define R2_STARTUP_SUPPLY_MODE 2
#define R2_FINITE_FERTILIZER 1
#define R2_LOCAL_SALE_TIMING 1
#define R2_CROP_CLOCK_MODE 1
#include "policy/search.hpp"
#include <cassert>
#include <iostream>
using namespace fastkag;
struct Fixture{
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t>shops;
 Fixture(){own.tiles.resize(100);rival.tiles.resize(100);own.money=rival.money=3000;own.unlocked_mask=rival.unlocked_mask=1;priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(10000);for(int k=0;k<9;k++)market.prices[k]=dp7::price(k,10000);}
 dp7::View at(int t)const{return{t,t/24,t%24,own,rival,priv,market,shops};}
};
double total(const competitive::Flow&f){double v=0;for(auto&d:f)for(double x:d)v+=std::abs(x);return v;}
int main(){Fixture f;triad::Settings s;s.scenario=0;s.rotation=0;s.repeat=0;s.replant=0;s.max_land=3;s.labor_hours=10;s.work_price=4;s.feed_cover=1;s.competition=2;s.service_reconcile=2;s.batch_delivery=1;
 auto original=triad::startup_supply_prior(f.at(0),s);assert(total(original)>0);
 f.priv.shed.fill(9999);f.priv.seeds.fill(9999);f.priv.inventories[0].fill(9999);
 auto poisoned=triad::startup_supply_prior(f.at(0),s);assert(original==poisoned); // own private irrelevant to rival proxy
 for(int t:{1,24,240,718})assert(total(triad::startup_supply_prior(f.at(t),s))==0);
 for(auto&d:original)for(int k=0;k<9;k++){assert(std::isfinite(d[k]));if(k>0&&k<8)assert(d[k]>=0);}
 f.rival.tiles[0].kind=TileKind::PLANT;f.rival.tiles[0].crop=Item::WHEAT;
 assert(total(triad::startup_supply_prior(f.at(0),s))==0);
 Fixture g;triad::SearchController agent(s);agent.act(g.at(0));assert(total(agent.live.model.extra_rival)>0);
 // The MPC copies and reconfigures controllers. Verify that both public
 // signals survive this path and actually change valuation, not just debug.
 agent.live.model.rival_harvest_weights[0][3]=1.;
 auto proposals=agent.prepare(g.at(0));assert(!proposals.empty());
 bool value_changed=false;
 for(const auto&p:proposals){
  assert(p.policy.model.extra_rival==agent.live.model.extra_rival);
  assert(p.policy.model.rival_harvest_weights==agent.live.model.rival_harvest_weights);
  assert(total(p.policy.model.rival)>0);
  auto without=p.policy;without.model.extra_rival={};without.plan(g.at(0));
  value_changed|=std::abs(without.predicted-p.policy.predicted)>1e-6;
 }
 assert(value_changed);
 // With the real P16 public trade ledger enabled, skipping from t=0 to
 // t=24 violates its consecutive-observation contract. Advance actual
 // conditional engine states for every step instead of weakening that guard.
 Simulator env(Config{},314159);triad::SearchController continuous(s);
 for(int step=0;step<=24;step++){
  dp7::View v{env.step_count(),env.day(),env.hour(),env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  auto action=continuous.act(v);
  if(step==0)assert(total(continuous.live.model.extra_rival)>0);
  if(step<24)env.step({action,PlayerAction{}});
 }
 assert(total(continuous.live.model.extra_rival)==0);
 std::cout<<"PASS: initial-only, private isolation, finite legal net flows, existing-crop guard, MPC proposal propagation and valuation effect, real next-day clear\n";
}
