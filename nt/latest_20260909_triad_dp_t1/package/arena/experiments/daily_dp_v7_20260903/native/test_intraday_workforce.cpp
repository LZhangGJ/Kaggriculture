#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;
void check(bool yes,const char*msg){++checks;if(!yes)throw std::runtime_error(msg);}
int main(){
 Simulator env(Config{},92413);auto&f=const_cast<Farm&>(env.farms()[0]);auto&pr=const_cast<PrivateState&>(env.privates()[0]);
 f.unlocked_mask=15;f.money=10000;f.tiles.assign(100,Tile{});f.farmer={4,4};pr.inventories.resize(1);pr.inventory_order.resize(1);
 Controller c;c.day=5;c.phase=3;c.p.efficient_water=true;c.p.portfolio_crop_calendar=true;c.plans.resize(1);
 View o{120,5,0,f,env.farms()[1],pr,env.market(),env.shops()};
 // Actual plot geometry and the existing workload change marginal wages.
 auto empty=portfolio::future_wages(c,o,{});check(empty==0,"empty farm costs wages");
 auto near=portfolio::future_wages(c,o,{{44,W}}),far=portfolio::future_wages(c,o,{{0,W}});
 check(std::isfinite(near)&&std::isfinite(far),"workforce forecast invalid");
 auto compact=portfolio::future_wages(c,o,{{44,W},{43,W}}),spread=portfolio::future_wages(c,o,{{0,W},{99,W}});
 check(spread>compact,"geography ignored by joint workforce");
 std::vector<std::pair<int,int>> commitments;bool found_increase=false;double old=empty,max_marginal=0;
 for(int pos:{44,43,42,41,40,30,31,32,33,34,24,23,22,21,20,10,11,12,13,14}){
  commitments.emplace_back(pos,SH);double now=portfolio::future_wages(c,o,commitments);
  check(std::isfinite(now),"ordinary joint work declared impossible");
  max_marginal=std::max(max_marginal,now-old);found_increase|=now>old;old=now;
 }
 check(found_increase&&max_marginal>0,"workload never changes staffing");
 // A future project is counted once despite duplicate planned metadata.
 PlayerAction a;a.units={action(Op::PLANT,W)};Plan pending;
 pending.a={action(Op::PLANT,W),action(Op::PLANT,C),action(Op::PLANT,M)};
 pending.target={44,43,42};c.plans={pending};
 auto declared=intraday::committed_new_targets(c,o,a);
 check(declared==std::vector<std::pair<int,int>>{{44,W},{43,C},{42,M}},"duplicate current/planned commitment");
 f.tiles[43].kind=TileKind::PLANT;f.tiles[43].crop=Item(C);f.tiles[43].planted_day=4;
 declared=intraday::committed_new_targets(c,o,a);check(declared.size()==2,"existing crop counted twice");
 View late{143,5,23,f,env.farms()[1],pr,env.market(),env.shops()};
 check(intraday::committed_new_targets(c,late,a)==std::vector<std::pair<int,int>>{{44,W}},"after-day plans credited");
 // No hidden inventory or opponent action is consulted for staffing.
 auto opponent=env.farms()[1];opponent.money=987654;opponent.tiles.assign(100,Tile{});
 View other{120,5,0,f,opponent,pr,env.market(),env.shops()};
 check(portfolio::future_wages(c,other,{})==portfolio::future_wages(c,o,{}),"opponent identity/cash affected staffing");
 // Correction replaces the old proxy rather than stacking penalties.
 intraday::Proposal q;q.valid=true;q.kind=W;q.value=123.;q.pos=44;q.unit=0;
 double proxy=c.p.action_shadow*c.crop_project(W).actions*c.p.bias[W];
 check(std::abs(intraday::workforce_score(c,q,30,50)-(q.value+proxy-20*c.p.bias[W]))<1e-9,"double-charged labour");
 check(intraday::workforce_score(c,q,INFINITY,50)==q.value,"unknown forecast hard rejected");
 check(intraday::workforce_score(c,q,50,30)==q.value+proxy,"negative heuristic cost credit");
 std::array<intraday::Proposal,12> reps{};reps[W]=q;
 check(intraday::compare_workforce(c,o,a,reps,q).value==q.value&&c.intraday_workforce_checks==0,"disabled not literal no-op");
 c.p.intraday_future_workforce=true;c.plans={{}};a.units={{}};f.tiles.assign(100,Tile{});
 q.job=intraday::project_job(c,o,44,W);q.value=-1;reps[W]=q;
 auto chosen=intraday::compare_workforce(c,o,a,reps,{});
 check(chosen.valid&&chosen.kind==W&&chosen.value>0,"spare existing capacity never affects choice");
 check(c.intraday_workforce_switches==1&&c.intraday_workforce_candidates==1,"effect counters absent");
 // End of season cannot create future wage obligations.
 c.day=29;View terminal{696,29,0,f,env.farms()[1],pr,env.market(),env.shops()};
 check(portfolio::future_wages(c,terminal,{{44,W}})==0,"terminal wages beyond game");
 Params p;p.intraday_admission=true;p.intraday_procurement=true;p.intraday_future_workforce=true;
 Controller dirty(p),fresh(p);dirty.day=20;dirty.last_step=600;dirty.intraday_workforce_checks=99;dirty.intraday_workforce_unknown=88;
 Simulator reset(Config{},8382);View initial{0,0,0,reset.farms()[0],reset.farms()[1],reset.privates()[0],reset.market(),reset.shops()};
 auto aa=dirty.act(initial),bb=fresh.act(initial);
 check(dirty.intraday_workforce_checks==fresh.intraday_workforce_checks&&dirty.intraday_workforce_unknown==fresh.intraday_workforce_unknown,"cross-game counters leaked");
 check(aa.units.size()==bb.units.size()&&aa.market.size()==bb.market.size(),"reset policy changed");
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<",\"empty_wages\":"<<empty<<",\"one_near_wheat\":"<<near<<",\"one_far_wheat\":"<<far<<",\"max_extra_sheep_wage\":"<<max_marginal<<"}\n";
}
