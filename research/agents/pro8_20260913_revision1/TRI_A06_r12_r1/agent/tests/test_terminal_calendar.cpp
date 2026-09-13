// TRI_A06_r12_r1: deterministic source-level boundary/negative controls.
// No replay future, seed lookup, opponent identity, or simulated win claim.
#include "policy/triad.hpp"
#include <cassert>
#include <iostream>
#include <string>
using namespace triad; using namespace dp7;
struct TerminalFixture {
 Farm own,other; PrivateState priv; Market market; std::vector<int8_t>shops{7};
 TerminalFixture(){
  own.tiles.resize(100);other.tiles.resize(100);own.money=1200;other.money=1500;
  own.farmer=other.farmer={4,4};own.unlocked_mask=other.unlocked_mask=1;
  priv.inventories.resize(1);priv.inventory_order.resize(1);
  for(int i=0;i<9;i++){market.inventory[i]=9900;market.prices[i]=price(i,9900);}
  for(int pos=0;pos<100;pos++)if(quad(pos)!=0)own.tiles[pos].kind=other.tiles[pos].kind=TileKind::LOCKED;
 }
 View view(int day,int hour=0){return {day*24+hour,day,hour,own,other,priv,market,shops};}
};
Settings settings(bool enabled){Settings s;s.a06_r12_calendar=enabled;s.repeat=0;s.rotation=0;s.scenario=0;s.competition=2;return s;}
bool has(const std::vector<Job>&js,Op op,int item=-2){for(auto&j:js)for(auto a:j.actions)if(a.op==op&&(item==-2||int(a.item)==item))return true;return false;}
int main(int argc,char**argv){
 const bool legacy=argc>1&&std::string(argv[1])=="--expect-legacy";
 TerminalFixture f;
 int checked=0;
 for(auto [day,kind]:{std::pair(26,W),{27,W},{27,C}}){
  auto o=f.view(day);triad::Controller c(settings(true));c.model.day=c.core.day=day;
  c.model.demand(o);c.model.public_rival(o);Asset base;c.model.value(o,base,&c.prices);
  auto dp=c.rotations_dp(44,c.prices,day);auto q=c.choose_crop(kind,day,44,c.prices,dp,&o,&base,&c.model);
  assert(q.kind==kind&&day+q.length<=29&&q.length>=first[kind]&&q.a.f[day+q.length][kind]>0);
  c.core.target={{44,kind}};c.core.triad_crop_age[44]=q.length;
  auto jobs=c.core.jobs(o);bool planted=has(jobs,Op::PLANT,kind);
  std::cout<<"day="<<day<<" crop="<<kind<<" selected_finish="<<day+q.length<<" positive_output="<<q.a.f[day+q.length][kind]<<" compiled_plant="<<planted<<"\n";
  assert(planted==!legacy);checked++;
 }
 if(legacy){std::cout<<"REPRODUCED: 3 legal positive-output calendar paths rejected by original compiler\n";return 0;}
 // Negative controls: OFF/recurrence, maturity beyond day 29, all crop types.
 for(bool on:{false,true})for(int day=0;day<30;day++)for(int k=0;k<5;k++){
  auto o=f.view(day);triad::Controller c(settings(on));c.core.day=day;c.core.target={{44,k}};
  bool expected=day<29&&day+(on&&!ongoing(k)?first[k]:harvest_age[k])<=29;
  assert(has(c.core.jobs(o),Op::PLANT,k)==expected);checked++;
 }
 {auto s=settings(true);s.repeat=1;triad::Controller c(s);c.core.day=27;c.core.target={{44,C}};assert(!has(c.core.jobs(f.view(27)),Op::PLANT));checked++;}
 // Incumbent age is NOT the successor's age. Preserve harvest-before-plant.
 {auto o=f.view(27);auto&t=f.own.tiles[44];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=23;t.yield_units=4;t.watered_today=true;
  triad::Controller c(settings(true));c.core.day=27;c.core.target={{44,C}};c.core.triad_crop_age[44]=4;
  auto js=c.core.jobs(o);assert(has(js,Op::HARVEST)&&has(js,Op::PLANT,C));assert(c.core.triad_crop_age[44]==4);
  assert(js.size()==1&&js[0].seeds[C]==1&&js[0].seeds[W]==0);
  int h=-1,p=-1;for(int i=0;i<int(js[0].actions.size());i++){if(js[0].actions[i].op==Op::HARVEST)h=i;if(js[0].actions[i].op==Op::PLANT)p=i;}assert(h>=0&&p>h);
  c.core.plant_not_before[44]=30;auto deferred=c.core.jobs(o);assert(has(deferred,Op::HARVEST)&&!has(deferred,Op::PLANT));f.own.tiles[44]=Tile{};checked+=2;
 }
 auto setup=[&](){triad::Controller c(settings(true));c.core.day=27;c.core.phase=3;c.core.plans.resize(1);c.core.admission_blend=1;c.core.admission_scope=2;
  c.core.admission_values=[](const auto&,const auto&,const auto&){std::array<double,12>v;v.fill(-1e90);v[W]=100;return v;};return c;};
 PlayerAction action;action.units.resize(1);
 {auto o=f.view(27,4);auto c=setup();assert(c.core.crop_project(W).out==0); // Legacy proxy intentionally unchanged.
  auto q=intraday::propose(c.core,o,action);assert(q.valid&&q.kind==W&&q.expense==seed_price[W]&&q.job.seeds[W]==1);checked++;
  // No unexecuted purchase is treated as a receipt.
  c.core.pending_admission={true,27,o.step,q.unit,q.pos,q.kind};intraday::complete_pending(c.core,o);assert(c.core.intraday_activated==0&&c.core.intraday_unfilled==1);checked++;
  c=setup();f.priv.seeds[W]=1;auto paid=intraday::propose(c.core,o,action);assert(paid.valid&&paid.buy.empty()&&paid.expense==0);
  c.core.pending_admission={true,27,o.step,paid.unit,paid.pos,paid.kind};intraday::complete_pending(c.core,o);assert(c.core.intraday_activated==1&&c.core.plans[0].a.size()>=2);f.priv.seeds[W]=0;checked++;
 }
 {auto o=f.view(27,4);auto c=setup();f.own.money=0;assert(!intraday::propose(c.core,o,action).valid);f.own.money=1200;checked++;}
 {auto o=f.view(27,4);auto c=setup();c.core.admission_values=[](const auto&,const auto&,const auto&){std::array<double,12>v;v.fill(-1);return v;};assert(!intraday::propose(c.core,o,action).valid);checked++;}
 {auto o=f.view(27,4);auto c=setup();c.core.admission_values={};assert(!intraday::propose(c.core,o,action).valid);checked++;}
 {auto o=f.view(27,4);auto c=setup();c.core.plans[0].a.push_back(dp7::action(Op::WATER));c.core.plans[0].target.push_back(44);assert(!intraday::propose(c.core,o,action).valid);checked++;}
 {auto o=f.view(27,23);auto c=setup();assert(!intraday::propose(c.core,o,action).valid);checked++;}
 {auto o=f.view(28,4);auto c=setup();c.core.day=28;assert(!intraday::propose(c.core,o,action).valid);checked++;}
 {auto o=f.view(27,4);auto c=setup();c.configure(settings(false));assert(!intraday::propose(c.core,o,action).valid);checked++;}
 {auto o=f.view(27,4);auto c=setup();for(int i=0;i<10;i++)action.market.push_back(dp7::action(Op::BUY_PRODUCT,W));assert(!intraday::propose(c.core,o,action).valid);action.market.clear();checked++;}
 std::cout<<"PASS terminal calendar admission: "<<checked<<" positive/negative checks; actual cash/receipt, idle labor, order cap, end time, paid successor and OFF preserved\n";
}
