#include "policy.hpp"
#include "observed_day_scenario.hpp"
#include <iostream>
#include <sstream>
using namespace dp7;
static int checks=0,windows=0,transitions=0;
void check(bool yes,const char*why){checks++;if(!yes)throw std::runtime_error(why);}
View view(const Simulator&e,int seat=0){return {e.step_count(),e.day(),e.hour(),e.farms()[seat],e.farms()[1-seat],e.privates()[seat],e.market(),e.shops()};}
Farm& farm(Simulator&e,int p=0){return const_cast<Farm&>(e.farms()[p]);}
PrivateState& inv(Simulator&e,int p=0){return const_cast<PrivateState&>(e.privates()[p]);}
auto tile_key(const Tile&t){return std::tuple(int(t.kind),int(t.crop),int(t.animal),t.planted_day,t.placed_day,t.yield_units,t.consecutive_unwatered,t.consecutive_unfed,t.fertilized_until_day,t.pending_care_bonus,t.max_lifespan_step,t.watered_today,t.fed_today,t.cared_today,t.fertilizer_available);}
bool same_position(Position a,Position b){return a.x==b.x&&a.y==b.y;}
bool same_private(const PrivateState&a,const PrivateState&b){return a.shed==b.shed&&a.seeds==b.seeds&&a.inventories==b.inventories&&a.inventory_order==b.inventory_order;}
void compare(const ObservedDayScenario&s,const Simulator&e,int seat,bool boundary){
 const auto&a=s.own();const auto&b=e.farms()[seat];
 check(a.money==b.money,"money mismatch");check(a.unlocked_mask==b.unlocked_mask&&a.hires_today==b.hires_today,"capacity mismatch");
 check(same_position(a.farmer,b.farmer)&&a.hands.size()==b.hands.size(),"unit reset mismatch");
 for(size_t u=0;u<a.hands.size();u++)check(same_position(a.hands[u],b.hands[u]),"hand position mismatch");
 check(same_private(s.inventory(),e.privates()[seat]),"private state/order mismatch");
 for(int k=0;k<100;k++){
  // Only a NEW random weed on an otherwise empty plot is outside the model.
  // Death/decay is tested explicitly below and must still create a weed.
  if(boundary&&a.tiles[k].kind==TileKind::EMPTY&&b.tiles[k].kind==TileKind::WEED)continue;
  check(tile_key(a.tiles[k])==tile_key(b.tiles[k]),"productive plot mismatch");
 }
 check(s.market().inventory==e.market().inventory&&s.market().prices==e.market().prices,"market/town mismatch");
 check(s.fills()==e.last_market_fills()[seat],"ordered fills mismatch");
 check(s.overflow()==e.last_end_of_day_overflow()[seat],"automatic overflow mismatch");
 check(s.step_count()==e.step_count(),"clock mismatch");
}
Simulator at(int step,uint64_t seed=741){Simulator e(Config{},seed);while(e.step_count()<step)e.step({});return e;}
void units(Simulator&e,int n){farm(e).hands.resize(n-1,{4,4});inv(e).inventories.resize(n);inv(e).inventory_order.resize(n);}
void bag(Simulator&e,int u,std::initializer_list<std::pair<int,int>>items){inv(e).inventories[u]={};inv(e).inventory_order[u].clear();for(auto[i,q]:items){inv(e).inventories[u][i]=q;inv(e).inventory_order[u].push_back(i);}}
void once(Simulator&e,const PlayerAction&a){ObservedDayScenario s(view(e));s.advance(a);e.step({a,{}});compare(s,e,0,s.finished());}
template<class Fn>void rejects(Fn&&f,const char*why){bool yes=false;try{f();}catch(const std::exception&){yes=true;}check(yes,why);}
int main(){try{
 // Unit phase happens before finance; exact same-turn inventory can fund
 // ordered SELL -> BUY, never retroactively fund an earlier BUY.
 auto e=at(4);farm(e).money=0;bag(e,0,{{MI,3}});
 PlayerAction sellbuy{{action(Op::PLACE,MI,3)},{action(Op::SELL,MI,3),action(Op::BUY_ANIMAL,CO,1)}};
 ObservedDayScenario good(view(e));good.advance(sellbuy);auto reversed=sellbuy;std::reverse(reversed.market.begin(),reversed.market.end());ObservedDayScenario bad(view(e));bad.advance(reversed);
 check(good.inventory().shed[CO]==1&&bad.inventory().shed[CO]==0,"financing order ignored");once(e,sellbuy);
 // Same work/movement and identical endpoint stock can earn different cash:
 // waiting across CURRENTLY KNOWN demand is sometimes better, not always
 // earlier completion. This is not a fabricated per-tick reward.
 e=at(4);bag(e,0,{{MI,3}});const_cast<std::vector<int8_t>&>(e.shops())={3,3,3,3};
 ObservedDayScenario early(view(e)),later(view(e));
 PlayerAction delivery{{action(Op::DROP)},{action(Op::SELL,MI,3)}};
 early.advance(delivery);later.advance({});early.advance({});later.advance(delivery);
 check(later.own().money>early.own().money,"known-demand timing consequence not detected");
 check(later.inventory().shed==early.inventory().shed&&later.market().inventory==early.market().inventory,"timing example changed endpoint assets");
 // A same-turn purchase cannot supply the earlier two unit PICKUPs.
 e=at(5);units(e,2);inv(e).shed[W]=1;
 PlayerAction pickup{{action(Op::PICKUP,W,1),action(Op::PICKUP,W,1)},{action(Op::BUY_PRODUCT,W,1)}};
 once(e,pickup);check(inv(e).inventories[0][W]==1&&inv(e).inventories[1][W]==0&&inv(e).shed[W]==1,"shared warehouse timing");
 // Aggregate seed demand is checked before any unit PLANT.
 e=at(5);units(e,2);farm(e).farmer={2,2};farm(e).hands[0]={3,2};inv(e).seeds[S]=1;
 once(e,{{action(Op::PLANT,S),action(Op::PLANT,S)}, {}});check(inv(e).seeds[S]==1&&farm(e).tiles[22].kind==TileKind::EMPTY&&farm(e).tiles[23].kind==TileKind::EMPTY,"aggregate PLANT guard");
 // Two workers on one crop: FERT -> WATER differs from WATER -> FERT.
 e=at(48);units(e,2);auto&t=farm(e).tiles[44];t=Tile{};t.kind=TileKind::PLANT;t.crop=Item(W);t.planted_day=0;t.yield_units=1;bag(e,0,{{F,1}});bag(e,1,{{F,1}});
 ObservedDayScenario fw(view(e)),wf(view(e));fw.advance({{action(Op::FERTILIZE),action(Op::WATER)}, {}});wf.advance({{action(Op::WATER),action(Op::FERTILIZE)}, {}});
 check(fw.own().tiles[44].yield_units==3&&wf.own().tiles[44].yield_units==2,"cross-worker order lost");once(e,{{action(Op::FERTILIZE),action(Op::WATER)}, {}});
 // Bag insertion order governs a capacity-limited DROP, with discarded rest.
 e=at(22);inv(e).shed[W]=99;bag(e,0,{{WO,2},{MI,2}});once(e,{{action(Op::DROP)}, {}});
 check(inv(e).shed[WO]==1&&inv(e).shed[MI]==0&&sum(inv(e).inventories[0])==0,"DROP order/capacity");
 // Selective placement keeps maintenance material and unfilled product.
 e=at(5);inv(e).shed[W]=99;bag(e,0,{{W,2},{MI,4}});once(e,{{action(Op::PLACE,MI,4)}, {}});
 check(inv(e).shed[MI]==1&&inv(e).inventories[0][MI]==3&&inv(e).inventories[0][W]==2,"selective deposit retention");
 // Day-end auto-deposit is after market: not sellable retroactively.
 e=at(23);units(e,2);inv(e).shed[W]=98;bag(e,0,{{WO,2},{MI,1}});bag(e,1,{{MI,3}});
 double cash=farm(e).money;once(e,{{},{action(Op::SELL,WO,2)}});
 check(farm(e).money==cash&&inv(e).shed[WO]==2&&e.last_end_of_day_overflow()[0]==4,"auto-deposit/market order");
 // Maintenance failures are real consequences, not suppressed random weeds.
 e=at(23);auto&crop=farm(e).tiles[33];crop=Tile{};crop.kind=TileKind::PLANT;crop.crop=Item(S);crop.consecutive_unwatered=1;
 auto&cow=farm(e).tiles[34];cow=Tile{};cow.kind=TileKind::ANIMAL;cow.animal=Item(CO);cow.consecutive_unfed=1;
 once(e,{});check(farm(e).tiles[33].kind==TileKind::WEED&&farm(e).tiles[34].kind==TileKind::PASTURE,"maintenance loss omitted");
 // Expiry is evaluated at its actual tick after units, so HARVEST can save it.
 e=at(72);auto&exp=farm(e).tiles[44];exp=Tile{};exp.kind=TileKind::PLANT;exp.crop=Item(C);exp.planted_day=0;exp.yield_units=1;exp.max_lifespan_step=72;
 ObservedDayScenario collect(view(e)),wait(view(e));collect.advance({{action(Op::HARVEST)}, {}});wait.advance({});
 check(collect.inventory().inventories[0][C]==1&&wait.own().tiles[44].kind==TileKind::WEED,"expiry timing");
 // Known town demand occurs on absolute step, after this turn's transactions.
 e=at(4);const_cast<std::vector<int8_t>&>(e.shops())={7};auto inv0=e.market().inventory;once(e,{});
 check(e.market().inventory[WO]==inv0[WO]-2,"known yarn demand missing");
 // At minimum price a sale need not increase market stock.
 e=at(7);auto&market=const_cast<Market&>(e.market());market.inventory[MI]=1000000;market.prices[MI]=price(MI,1000000);inv(e).shed[MI]=3;
 once(e,{{},{action(Op::SELL,MI,3)}});check(e.market().inventory[MI]==1000000,"floor sale inventory mismatch");
 // Fixed horizon, final 719th transition, and no synthetic next-day policy.
 e=at(71);ObservedDayScenario horizon(view(e));horizon.advance({});check(horizon.finished()&&horizon.ticks()==1,"day horizon");check(horizon.shops()==e.shops(),"unseen shop leaked");rejects([&]{horizon.advance({});},"day advance allowed");rejects([&]{horizon.view();},"next day view allowed");
 e=at(718);ObservedDayScenario terminal(view(e));terminal.advance({});check(terminal.finished()&&terminal.step_count()==719,"terminal off by one");rejects([&]{terminal.advance({});},"after terminal allowed");
 // The constructor has no seed or rival-private argument. Two real engines
 // differing only in those hidden fields produce identical public scenarios.
 e=at(36);auto hidden=e;hidden.reseed_future(999123);inv(hidden,1).shed[MI]=500;inv(hidden,1).seeds[S]=1000;bag(hidden,0,{}); // Own bag already empty.
 ObservedDayScenario a(view(e)),b(view(hidden));while(!a.finished()){a.advance({});b.advance({});check(a.own().money==b.own().money&&same_private(a.inventory(),b.inventory())&&a.market().inventory==b.market().inventory,"hidden state influenced scenario");}
 // Reject ambiguous own inventory ordering instead of inventing a dict order.
 e=at(6);bag(e,0,{{W,1}});inv(e).inventory_order[0].clear();rejects([&]{ObservedDayScenario s(view(e));},"missing own order accepted");
 auto v=view(e);v.hour++;rejects([&]{ObservedDayScenario s(v);},"inconsistent clock accepted");
 // Live-policy snapshots, independent complete own continuations in a stated
 // PASS-opponent scenario. Source game remains responsive self-vs-self and
 // is never changed by any hypothetical window.
 Params p;p.fix_logistics=p.capacity_hauling=p.shared_task_atoms_v2=p.stepwise_recoordination=p.preparation_pipeline_v2=true;
 p.intraday_admission=p.intraday_procurement=p.shared_service_insertions=true;
 for(int seed=20263001;seed<20263009;seed++){
  Simulator live(Config{},seed);Controller own(p),other(p);
  while(!live.done()){
   if(live.hour()%6==0)for(int seat:{0,1}){
    auto reference=live;auto sc=seat?other:own,rc=sc;ObservedDayScenario preview(view(live,seat));int start=live.step_count();auto old_private=live.privates()[seat];double oldcash=live.farms()[seat].money;windows++;
    while(!preview.finished()){
     auto pa=sc.act(preview.view()),ra=rc.act(view(reference,seat));
     check(pa.units.size()==ra.units.size()&&pa.market.size()==ra.market.size(),"policy action shape mismatch");
     for(size_t i=0;i<pa.units.size();i++)check(Controller::same_action(pa.units[i],ra.units[i]),"public unit decision mismatch");
     for(size_t i=0;i<pa.market.size();i++)check(Controller::same_action(pa.market[i],ra.market[i]),"public market decision mismatch");
     preview.advance(pa);std::array<PlayerAction,2>joint{};joint[seat]=ra;reference.step(joint);transitions++;
     compare(preview,reference,seat,preview.finished());
    }
    check(live.step_count()==start&&live.farms()[seat].money==oldcash&&same_private(live.privates()[seat],old_private),"source state mutated");
   }
   auto pa=own.act(view(live)),pb=other.act(view(live,1));live.step({pa,pb});
  }
 }
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<",\"public_windows\":"<<windows<<",\"scenario_transitions\":"<<transitions<<"}\n";
}catch(const std::exception&e){std::cerr<<"check "<<checks<<": "<<e.what()<<"\n";return 1;}}
