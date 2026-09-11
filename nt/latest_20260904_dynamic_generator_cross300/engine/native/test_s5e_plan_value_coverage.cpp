// Read-only diagnostic of what the existing residual estimator represents.
// This does not add a policy or assign a hindsight value to any decision.
#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;
void check(bool ok,const char*why){++checks;if(!ok)throw std::runtime_error(why);}
struct Fixture{
 Farm f,r;PrivateState pr;Market m;std::vector<int8_t>shops;Controller c;int day=14;
 Fixture(){
  f.tiles.resize(100);r.tiles.resize(100);f.unlocked_mask=7;f.money=100000;
  f.farmer=r.farmer={4,4};pr.inventories.resize(1);pr.inventory_order.resize(1);
  pr.seeds.fill(10);pr.shed[W]=20;c.day=day;c.p.fix_finite_projection=true;c.p.fix_calendar=true;
  m.inventory.fill(10000);for(int i=0;i<9;i++)m.prices[i]=price(i,10000);
 }
 View view(){return {24*day,day,0,f,r,pr,m,shops};}
};
int count_plant(const Controller&c,const View&o,int kind){
 int n=0;for(const auto&j:c.jobs(o))for(auto a:j.actions)n+=a.op==Op::PLANT&&int(a.item)==kind;return n;
}
int main(){try{
 int invisible_empty=0,invisible_wait=0,invisible_replacement=0;
 for(int kind=0;kind<5;kind++)for(int delay:{1,3})for(int pos:{43,62}){
  Fixture f;auto planned=f.c;planned.target={{pos,kind}};planned.plant_not_before[pos]=f.day+delay;
  auto a=dayvalue::residual(f.c,f.view()),b=dayvalue::residual(planned,f.view());
  check(a.known&&b.known,"unexpected unknown for a funded empty farm");
  check(a.score==b.score&&b.calendars==0,"empty-project value is no longer invisible; review this diagnosis");
  auto future=planned;future.day=f.day+delay;View v{24*future.day,future.day,0,f.f,f.r,f.pr,f.m,f.shops};
  check(count_plant(planned,f.view(),kind)==0,"deferred planting was executable early");
  check(count_plant(future,v,kind)==1,"deferred project does not generate a later planting intent");
  ++invisible_empty;
 }
 for(int kind:{W,C,M})for(int delay:{1,3}){
  Fixture f;auto&t=f.f.tiles[43];t.kind=TileKind::PLANT;t.crop=Item(kind);
  t.planted_day=f.day-f.c.p.crop_harvest_age[kind];t.yield_units=3;t.max_lifespan_step=24*(t.planted_day+(kind==W?4:kind==C?3:12)+1);
  f.c.target={{43,kind}};auto wait=f.c;wait.plant_not_before[43]=f.day+delay;
  check(count_plant(f.c,f.view(),kind)==1&&count_plant(wait,f.view(),kind)==0,"wait does not alter executable semantics");
  auto a=dayvalue::residual(f.c,f.view()),b=dayvalue::residual(wait,f.view());
  check(a.known==b.known&&a.score==b.score&&a.wages==b.wages&&a.seeds==b.seeds,"wait affects residual; revise diagnosis");++invisible_wait;
  for(int next=0;next<5;next++)if(next!=kind){
   auto changed=f.c;changed.target={{43,next}};
   check(count_plant(changed,f.view(),next)==1,"replacement intent absent");
   auto d=dayvalue::residual(changed,f.view());
   check(a.known==d.known&&a.score==d.score&&a.wages==d.wages&&a.seeds==d.seeds,"target affects residual; revise diagnosis");++invisible_replacement;
  }
 }
 std::cout<<"{\"status\":\"PASS_DIAGNOSTIC_NOT_REPAIR\",\"checks\":"<<checks
  <<",\"empty_deferred_projects_not_valued\":"<<invisible_empty
  <<",\"standing_crop_wait_intents_not_valued\":"<<invisible_wait
  <<",\"standing_crop_replacement_intents_not_valued\":"<<invisible_replacement<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
