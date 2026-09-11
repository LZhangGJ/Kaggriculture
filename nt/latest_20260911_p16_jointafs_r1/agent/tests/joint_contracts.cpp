#include "policy/search.hpp"
#include <fstream>
#include <cstring>
#include <iostream>
using namespace triad;
int checks=0;
void require(bool ok,const char*message){checks++;if(!ok)throw std::runtime_error(message);}
View view(const fastkag::Simulator&e){return{e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
int main(int argc,char**argv){try{
 require(argc==2,"settings path");Settings s;std::ifstream in(argv[1]);std::vector<double>x;double y;while(in>>y)x.push_back(y);require(x.size()==SETTINGS_COUNT,"38-parameter ABI unchanged");std::memcpy(&s,x.data(),sizeof(s));
 fastkag::Simulator env(fastkag::Config{},101);auto o=view(env);triad::Controller live(s);live.model.extra_rival=startup_supply_prior(o,s);
 auto parents=generate_proposals(live,s,o);require(!parents.empty(),"KEEP exists");auto keep=parents[0].policy;
 auto bundles=joint_candidates(live,keep,o);require(bundles.size()>=2,"at least two distinct structural candidates");
 for(const auto&b:bundles){
  require(b.policy.joint.active,"live state owned by candidate");require(!live.joint.active,"candidate construction did not mutate original");
  require(b.feeders==b.policy.joint.count&&b.feeders>=1&&b.feeders<=2,"bounded batch count");
  require(b.removed>=1&&b.removed<=2,"only bounded new-animal exchange");
  int planned_animals=0;for(auto[p,k]:b.policy.core.target)planned_animals+=k>=9;
  require(planned_animals<=b.policy.joint.animal_limit,"initial animal cap honored");
  for(int i=0;i<b.feeders;i++){
   auto slot=b.policy.joint.slots[i];require(!animal(o.own.tiles[slot.pos])&&!plant(o.own.tiles[slot.pos]),"only vacant owned plots changed");
   require(b.policy.length[slot.pos]>=2&&b.policy.length[slot.pos]<=4,"finite wheat duration");
   require(slot.next_kind>=0&&slot.next_kind<5,"one valid successor crop");
   auto path=b.policy.paths[slot.pos];require(path.fixed[0]<0||o.priv.seeds[W]>0,"current seed expense real");
   double future_setup=0;for(int d=1;d<30;d++)future_setup+=path.fixed[d];require(future_setup==0,"successor not counted as already planted");
  }
 }
 // Confirm facts, not elapsed clock, advance the option state.
 Farm own=o.own;auto pr=o.priv;JointBundleState j;j.active=true;j.count=1;j.created_day=0;j.animal_limit=4;j.slots[0].pos=0;j.slots[0].next_kind=S;j.slots[0].source_age=3;
 auto at=[&](int step){return View{step,step/24,step%24,own,o.opponent,pr,o.market,o.shops};};
 j.observe(at(1));require(j.slots[0].stage==0,"unexecuted PLANT not assumed");
 auto&t=own.tiles[0];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=0;t.yield_units=2;
 j.observe(at(2));require(j.source_started==1&&j.slots[0].stage==1,"actual planted source detected");
 j.observe(at(2));require(j.source_started==1,"same observation idempotent");
 // A disappearance without issued harvest is drought/invalid, not financing.
 auto dead=j;own.tiles[0]=Tile{};dead.observe(at(3));require(!dead.active&&dead.last_cancel_reason==2&&dead.source_harvested==0,"disappearance never fabricated as harvest");
 own.tiles[0]=t;own.tiles[0].kind=TileKind::PLANT;own.tiles[0].crop=Item::WHEAT;own.tiles[0].planted_day=0;own.tiles[0].yield_units=2;
 own.farmer={0,0};PlayerAction action;action.units={dp7::action(Op::HARVEST)};
 j.record(at(74),action);own.tiles[0]=Tile{};j.observe(at(75));
 require(j.source_harvested==1&&j.slots[0].stage==2,"issued harvest plus actual disappearance confirmed");
 require(j.force_kind(0,3)==-1,"same-day expected receipts do not start successor");
 require(j.force_kind(0,4)==S,"following-day successor becomes eligible");
 require(j.seed_reserve(at(75))>0,"successor cash earmark survives harvest");
 auto lost=j;lost.observe(at(168));require(!lost.active&&lost.last_cancel_reason==4,"unaffordable successor expires explicitly");
 own.tiles[0].kind=TileKind::PLANT;own.tiles[0].crop=Item::STRAWBERRY;own.tiles[0].planted_day=4;j.observe(at(98));
 require(j.successor_started==1&&j.completed==1&&!j.active,"actual successor closes option");
 // State is value-owned; a speculative branch cannot consume the real promise.
 require(live.joint.completed==0,"no hypothetical completion in real policy");
 // Test cap scope and cash: 'less today' must not ban tomorrow's investments.
 auto c=bundles[0].policy;Farm empty=o.own;auto zero=o.priv;empty.money=0;
 auto&slot=c.joint.slots[0];c.joint.count=1;slot.stage=2;slot.harvest_step=74;slot.deadline=6;slot.next_kind=M;c.joint.last_observation=-1;
 View no_cash{96,4,0,empty,o.opponent,zero,o.market,o.shops};c.plan(no_cash);
 require(c.core.p.max_animals==int(s.max_animals),"animal cap only affects original admission day");
 bool ordered=false;for(auto a:c.core.queue)ordered|=a.op==Op::BUY_SEED&&a.item==Item::MELON;
 require(!ordered,"releasing an earmark cannot create cash");
 for(auto[p,k]:c.core.target)if(p==slot.pos)require(k!=M,"unfunded forced successor not valued as started");
 // Common event horizon; only simulation facts admit a bundle.
 SearchController eval(s);bool valid=false;int starts=0;auto&b=bundles[0];
 auto before=live.joint;double score=eval.event_value(o,b.policy,keep.model.rival,b.age+2,b.feeders,valid,starts);
 require(std::isfinite(score),"finite common-horizon value");require(valid==(starts>=b.feeders),"candidate admission tied to executed successor");
 require(live.joint.completed==before.completed&&live.joint.active==before.active,"rollout side effects isolated");
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<",\"bundles\":"<<bundles.size()<<"}\n";
 }catch(const std::exception&e){std::cerr<<"FAIL after "<<checks<<": "<<e.what()<<"\n";return 1;}}
