#include "hybrid.hpp"
#include <cassert>
#include <iostream>
using namespace fastkag;using namespace dp7;
static bool harvests(const Controller&c,const View&v){for(auto&j:c.jobs(v))for(auto&a:j.actions)if(a.op==Op::HARVEST)return true;return false;}
int main(){
 Simulator env;Farm own=env.farms()[0],opp=env.farms()[1];auto pr=env.privates()[0];
 competitive::Config off;off.new_limit=1;off.hold=0;off.expansion=2;off.reserve=150;off.competition=1.5;off.service_dp=1;off.service_cost=0;off.execution_variant=4;
 auto on=off;on.harvest_threshold=1;
 competitive::Hybrid legacy(off),candidate(on);
 assert(!legacy.core.p.frequent_harvest&&candidate.core.p.frequent_harvest&&candidate.core.p.frequent_threshold==1);
 for(auto*c:{&legacy.core,&candidate.core}){c->day=10;c->target={{44,CO}};}
 Tile t;t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.placed_day=0;t.yield_units=1;t.fertilizer_available=true;t.fed_today=true;own.tiles[44]=t;
 View v{240,10,0,own,opp,pr,env.market(),env.shops()};
 assert(!harvests(legacy.core,v));assert(harvests(candidate.core,v));
 std::cout<<"PASS small harvest becomes a live executable task without future overflow\n";
 own.tiles[44].yield_units=0;assert(!harvests(candidate.core,v));
 std::cout<<"PASS no harvest is generated from zero product\n";
 own.tiles[44].yield_units=held[1];assert(harvests(candidate.core,v)&&harvests(legacy.core,v));
 std::cout<<"PASS mandatory full-capacity harvest retained\n";
 for(auto*c:{&legacy.core,&candidate.core})c->day=29;
 own.tiles[44].yield_units=1;View terminal{696,29,0,own,opp,pr,env.market(),env.shops()};
 assert(harvests(candidate.core,terminal)&&harvests(legacy.core,terminal));
 std::cout<<"PASS terminal liquidation retained\n";
}
