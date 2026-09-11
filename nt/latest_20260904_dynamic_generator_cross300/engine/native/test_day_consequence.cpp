#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;
void check(bool yes,const char*why){checks++;if(!yes)throw std::runtime_error(why);}
View view(const Simulator&e){return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
Simulator at(int day){Simulator e(Config{},8821);while(e.day()<day)e.step({});return e;}
Farm& farm(Simulator&e){return const_cast<Farm&>(e.farms()[0]);}
PrivateState& pr(Simulator&e){return const_cast<PrivateState&>(e.privates()[0]);}
double outputs(const portfolio::DeliveryCalendar&cal,int item){double n=0;for(int d=0;d<30;d++)n+=cal.cash.quantity[d][item];return n;}
int main(){try{
 auto e=at(12);Controller c;c.day=12;c.p.max_hands=14;
 auto&t=farm(e).tiles[44];t=Tile{};t.kind=TileKind::ANIMAL;t.animal=Item(CO);t.placed_day=4;
 auto lean=dayvalue::animal_calendar(c,view(e),t);t.yield_units=5;t.pending_care_bonus=2;
 auto loaded=dayvalue::animal_calendar(c,view(e),t);
 check(outputs(loaded,MI)>outputs(lean,MI),"stored milk and care bonus ignored");
 auto cared=t;cared.pending_care_bonus=0;t.yield_units=0;
 auto bonus=dayvalue::animal_calendar(c,view(e),t);t.pending_care_bonus=0;
 auto no_bonus=dayvalue::animal_calendar(c,view(e),t);
 check(outputs(bonus,MI)>=outputs(no_bonus,MI),"bonus loses produce");
 t.fed_today=true;auto fed=dayvalue::animal_calendar(c,view(e),t);
 check(fed.feed[12]==0&&no_bonus.feed[12]==1,"current feed charged twice");
 t.fertilizer_available=true;auto fertilizer=dayvalue::animal_calendar(c,view(e),t);
 check(outputs(fertilizer,F)==outputs(fed,F)+1,"current manure omitted/doubled");
 // Independent official-kernel rollout for several ages/types and states.
 int animal_cases=0;
 for(int kind:{G,CO,SH})for(int start:{0,8,18,28})for(int stored:{0,2}){
  auto actual=at(start);auto&f=farm(actual);f.tiles.assign(100,Tile{});f.farmer={4,4};
  auto&a=f.tiles[44];a.kind=TileKind::ANIMAL;a.animal=Item(kind);a.placed_day=std::max(0,start-7);a.yield_units=stored;a.pending_care_bonus=1;a.fertilizer_available=true;
  c.day=start;auto cal=dayvalue::animal_calendar(c,view(actual),a);Counts harvested{};int feed=0;
  while(!actual.done()){
   int d=actual.day();Controller one(c.p);one.day=d;one.target={{44,kind}};one.p.shared_task_atoms_v2=one.p.split_service_jobs=false;
   pr(actual).shed={};pr(actual).shed[W]=1;pr(actual).inventories.assign(1,{});pr(actual).inventory_order.assign(1,{});
   auto jobs=one.jobs(view(actual));Acts todo;for(auto&j:jobs)if(j.needs[W])todo.push_back(action(Op::PICKUP,W,j.needs[W]));for(auto&j:jobs)todo.insert(todo.end(),j.actions.begin(),j.actions.end());size_t k=0;
   while(!actual.done()&&actual.day()==d){
    Action op=k<todo.size()?todo[k++]:action(Op::PASS);int hour=actual.hour();
    auto before=actual.project_unit_phase(0,{op});
    if(op.op==Op::FEED)feed+=pr(actual).inventories[0][W]-before.privates()[0].inventories[0][W];
    for(int i:{product[kind-9],F})if(d<29||hour<22)harvested[i]+=std::max(0,before.privates()[0].inventories[0][i]-pr(actual).inventories[0][i]);
    actual.step({PlayerAction{{op},{}},{}});
   }
  }
  check(outputs(cal,product[kind-9])==harvested[product[kind-9]],"animal product projection differs from engine");
  check(outputs(cal,F)==harvested[F],"animal manure projection differs from engine");
  check(-outputs(cal,W)==feed,"animal feed obligation differs from engine");animal_cases++;
 }
 // Empty holdings retain cash; feed stock is sold only when no future use.
 e=at(25);farm(e).tiles.assign(100,Tile{});farm(e).money=10000;c.day=25;
 auto zero=dayvalue::residual(c,view(e));check(zero.known&&zero.score==10000&&zero.wages==0,"empty estate value");
 pr(e).shed[MI]=3;auto stocked=dayvalue::residual(c,view(e));check(stocked.score>zero.score,"owned products ignored");
 pr(e).shed={};pr(e).seeds[S]=20;auto unused=dayvalue::residual(c,view(e));check(unused.score==zero.score,"unused seeds credited as sellable cash");
 auto&crop=farm(e).tiles[44];crop=Tile{};crop.kind=TileKind::PLANT;crop.crop=Item(W);crop.planted_day=21;crop.yield_units=3;
 pr(e).seeds={};auto buyseed=dayvalue::residual(c,view(e));pr(e).seeds[W]=5;auto ownseed=dayvalue::residual(c,view(e));
 check(buyseed.known&&ownseed.known&&ownseed.seeds<buyseed.seeds&&ownseed.score>buyseed.score,"owned seed was double charged");
 Controller::Calendar flow{};flow[26][W]=-3;flow[27][W]=5;flow[28][W]=-4;
 check(dayvalue::reserve_after(flow,25,W)==3,"future feed production timing ignored");
 flow[26][W]=5;flow[27][W]=-3;check(dayvalue::reserve_after(flow,25,W)==2,"feed reserve too large");
 std::vector<dayvalue::Work>compact,spread;portfolio::DeliveryCalendar cal;cal.water[26]=1;
 for(int pos:{44,43,42,41,40,30,31,32,33,34,24,23,22,21,20})compact.push_back({pos,W,cal});
 for(int pos:{0,9,90,99,10,19,80,89,20,29,70,79,30,39,60})spread.push_back({pos,W,cal});
 check(dayvalue::wages_on(c,compact,26)<dayvalue::wages_on(c,spread,26),"joint workforce ignores geography");
 e=at(27);farm(e).tiles.assign(100,Tile{});farm(e).money=10000;c.day=27;
 auto&an=farm(e).tiles[44];an=Tile{};an.kind=TileKind::ANIMAL;an.animal=Item(CO);an.placed_day=0;an.yield_units=5;
 auto pass=dayvalue::residual(c,view(e));auto&opp=const_cast<Farm&>(e.farms()[1]);opp.tiles.assign(100,Tile{});for(int i=0;i<12;i++)opp.tiles[i]=an;
 c.p.day_value_public_supply=true;auto pressure=dayvalue::residual(c,view(e));check(pressure.score<pass.score,"public competing supply has no effect");
 // Pure fallback when the maintained-project model cannot fund itself.
 e=at(25);farm(e).tiles.assign(100,Tile{});farm(e).money=0;c.day=25;
 auto&famine=farm(e).tiles[0];famine.kind=TileKind::ANIMAL;famine.animal=Item(CO);famine.placed_day=25;
 auto gap=dayvalue::residual(c,view(e));check(!gap.known&&gap.funding_gap>0,"unfunded continuation treated as exact");
 // Disabled selector does not run, and per-game mutable counters reset.
 Params p;p.stepwise_recoordination=p.shared_task_atoms_v2=p.shared_service_insertions=true;
 Controller off(p),fresh(p),dirty(p);dirty.day_value_checks=12;dirty.day_value_rejected=3;dirty.day_value_unknown=2;dirty.last_step=500;dirty.day=20;
 Simulator reset(Config{},328);dirty.act(view(reset));fresh.act(view(reset));off.act(view(reset));
 check(dirty.day_value_checks==0&&dirty.day_value_rejected==0&&dirty.day_value_unknown==0,"day-value state leaked across games");
 check(off.day_value_checks==0,"disabled valuation invoked");
 // A controlled late-day legal omission must not beat preserving a cow whose
 // known pending care produces milk at the imminent daily settlement.
 e=at(27);while(e.hour()<23)e.step({});farm(e).tiles.assign(100,Tile{});farm(e).money=10000;
 auto&valuable=farm(e).tiles[44];valuable.kind=TileKind::ANIMAL;valuable.animal=Item(CO);valuable.placed_day=20;valuable.consecutive_unfed=1;valuable.pending_care_bonus=4;
 pr(e).inventories[0][W]=1;pr(e).inventory_order[0]={W};Controller decision;decision.day=27;decision.phase=3;decision.target={{44,CO}};
 Plan keep;keep.a={action(Op::FEED)};keep.target={44};
 auto maintained=dayvalue::schedule_value(decision,view(e),{keep});auto abandoned=dayvalue::schedule_value(decision,view(e),{Plan{}});
 check(maintained.known&&abandoned.known&&maintained.score>abandoned.score,"pending production not valued over idle cash");
 check(!dayvalue::accept(decision,view(e),{keep},{Plan{}})&&decision.day_value_rejected==1,"known inferior schedule not rejected");
 // Active selector runs on actual multi-unit continuations and is bounded.
 p.day_consequence_compare=true;p.intraday_admission=p.intraday_procurement=true;
 Controller active(p);Simulator live(Config{},732);int evaluated=0;
 while(!live.done()){auto own=active.act(view(live));live.step({own,{}});}
 evaluated=active.day_value_checks;check(evaluated>0,"selector is a dead flag");
 check(active.day_value_rejected<=evaluated,"invalid decision counters");
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<",\"animal_cases\":"<<animal_cases<<",\"live_checks\":"<<evaluated<<",\"live_rejections\":"<<active.day_value_rejected<<",\"live_unknown\":"<<active.day_value_unknown<<"}\n";
}catch(const std::exception&e){std::cerr<<"check "<<checks<<": "<<e.what()<<"\n";return 1;}}
