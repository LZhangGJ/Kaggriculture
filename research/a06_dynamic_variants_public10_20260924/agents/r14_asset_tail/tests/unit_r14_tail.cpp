#include "../policy/search.hpp"
#include <cstdlib>
#include <iostream>
using namespace dp7;
static int checks=0;
static void ck(bool ok,const char* m){++checks;if(!ok){std::cerr<<"FAIL "<<m<<"\n";std::exit(1);}}
struct Fixture{
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t> shops;
 Fixture(){
  own.tiles.resize(100);rival.tiles.resize(100);own.unlocked_mask=rival.unlocked_mask=1;
  own.farmer=rival.farmer={4,4};own.money=5000;rival.money=3000;
  for(int p=0;p<100;p++){if(quad(p)!=0){own.tiles[p].kind=TileKind::LOCKED;rival.tiles[p].kind=TileKind::LOCKED;}}
  priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(10000);
  for(int k=0;k<9;k++)market.prices[k]=price(k,10000);
 }
 View view(int day){return {day*24,day,0,own,rival,priv,market,shops};}
};
static triad::Settings settings(){triad::Settings s;s.rotation=0;s.repeat=0;s.max_land=4;s.max_hands=14;s.a06_r11_recovery=0;s.scenario=0;s.preview=1;s.keep_commitments=1;s.reserve=120;s.a06_r12_calendar=1;return s;}
static int investments(const triad::Controller& c){int n=0;for(auto a:c.core.queue)if(a.op==Op::BUY_SEED||a.op==Op::BUY_ANIMAL||a.op==Op::BUY_LAND)n++;return n;}
int main(){
 for(int day:{0,5,12,20,26}){
  Fixture f;auto s=settings();triad::Controller c(s);c.hold_only_valuation=true;auto v=f.view(day);
  const auto before_cash=f.own.money;const auto before_seed=f.priv.seeds;const auto before_market=f.market.inventory;
  c.plan(v);
  ck(investments(c)==0,"held critic cannot buy unfunded seed/animal/land");
  ck(c.core.planned_land==1,"held critic cannot add unfunded land");
  ck(std::all_of(c.core.target.begin(),c.core.target.end(),[](auto p){return p.second<0;}),"empty unfunded estate remains empty in held valuation");
  ck(f.own.money==before_cash&&f.priv.seeds==before_seed&&f.market.inventory==before_market,"held valuation cannot mutate authoritative cash/input/market");
  ck(std::isfinite(c.predicted),"held estate value is finite");
 }
 for(int k:{W,C}){
  Fixture f;auto v=f.view(5);f.priv.seeds[k]=1;triad::Controller held(settings());held.hold_only_valuation=true;
  held.book[44]={k,-1,5,4,-1,true};held.plan(v);
  bool retained=false;for(auto [p,kind]:held.core.target)retained|=p==44&&kind==k;
  ck(retained,"already-paid unfinished crop remains represented");
  ck(held.kept==1,"paid commitment counted once");
  ck(held.paths[44].first_cost==0,"paid seed not charged twice");
  ck(f.priv.seeds[k]==1&&f.own.money==5000,"valuation does not consume actual paid seed");
  ck(investments(held)==0,"paid crop need not buy another seed");
 }
 {
  Fixture f;auto&t=f.own.tiles[44];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=2;t.yield_units=4;t.consecutive_unwatered=0;t.max_lifespan_step=7*24;
  auto v=f.view(5);triad::Controller held(settings());held.hold_only_valuation=true;held.plan(v);
  double flow=0;for(auto&d:held.portfolio.f)flow+=d[W];
  ck(flow>0,"standing harvest retained in held asset value");ck(investments(held)==0,"standing harvest does not imply unfunded renewal");
  ck(t.yield_units==4&&t.kind==TileKind::PLANT,"valuation cannot consume standing crop");
 }
 {
  Fixture f;auto s=settings();s.scenario=1;s.portfolio_passes=2;s.candidate_extra=0;s.r14_tail_weight=.25;
  triad::SearchController search(s);auto v=f.view(0);search.choose(v);
  ck(!search.live.hold_only_valuation,"valuation-only flag cannot leak to installed live controller");
  ck(search.last_search.find("expanded_tail_value")!=std::string::npos,"trace records expanded valuation");
  ck(search.last_search.find("held_tail_value")!=std::string::npos,"trace records held valuation");
  ck(search.last_search.find("r14_tail_weight")!=std::string::npos,"trace records interpolation weight");
  ck(f.own.money==5000&&f.priv.seeds[W]==0,"candidate rollouts cannot mutate authoritative facts");
 }
 std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"scope\":\"R14 held/funded asset valuation, no unfunded expansion, sunk-input accounting, authoritative-state isolation, installed-policy flag isolation, critic diagnostics\"}\n";
}
