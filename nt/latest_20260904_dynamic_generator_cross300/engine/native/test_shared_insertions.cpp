#include "policy.hpp"
#include <iostream>
using namespace dp7;
int checks=0;
void check(bool yes,const char*why){++checks;if(!yes)throw std::runtime_error(why);}
View view(const Simulator&e){return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
Plan route(int start,std::initializer_list<std::pair<Action,int>>ops){Plan p;for(auto[a,pos]:ops){Controller::walk(p,start,pos);p.a.push_back(a);p.target.push_back(pos);}return p;}
Job job(int pos,std::initializer_list<Action>actions){Job j;j.pos=pos;j.actions=actions;return j;}
std::set<Controller::TaskKey> keys(const std::vector<Job>&js){std::set<Controller::TaskKey>r;for(auto&j:js)for(auto a:j.actions)r.insert(Controller::task_key(j.pos,a));return r;}
int count(const std::vector<Plan>&ps,Op op,int pos){int n=0;for(auto&p:ps)for(size_t k=0;k<p.a.size();k++)n+=p.a[k].op==op&&p.target[k]==pos;return n;}
bool material_same(const Plan&a,const Plan&b,const std::set<Controller::TaskKey>&ks){
 auto x=Controller::plan_semantics(a),y=Controller::plan_semantics(b);
 for(auto*v:{&x,&y})v->erase(std::remove_if(v->begin(),v->end(),[&](auto q){return ks.contains(Controller::task_key(q.second,q.first));}),v->end());
 if(x.size()!=y.size())return false;for(size_t k=0;k<x.size();k++)if(!Controller::same_action(x[k].first,y[k].first)||x[k].second!=y[k].second)return false;return true;
}
int main(){
 Simulator env(Config{},981724);auto&farm=const_cast<Farm&>(env.farms()[0]);auto&pr=const_cast<PrivateState&>(env.privates()[0]);
 farm.farmer={4,4};farm.hands={{3,3}};farm.unlocked_mask=15;pr.inventories.resize(2);pr.inventory_order.resize(2);
 Controller c;c.day=0;c.phase=3;
 // Insertion route reconstruction and delta are exact for all slots, including
 // same-cell operations and nonzero remaining-plan indices.
 auto original=route(44,{{action(Op::PICKUP,W,2),44},{action(Op::FEED),33},{action(Op::DROP),44}});
 for(int pos:{0,33,34,44,45,99})for(int at=0;at<=3;at++){
  auto j=job(pos,{action(Op::CARE)});auto inserted=c.insert_service(view(env),0,original,j,at);
  auto sem=c.plan_semantics(original);int prev=at?sem[at-1].second:44;
  int delta=dist(prev,pos)+1;if(at<int(sem.size()))delta+=dist(pos,sem[at].second)-dist(prev,sem[at].second);
  check(int(inserted.a.size())==int(original.a.size())+delta,"incorrect insertion movement cost");
  check(material_same(original,inserted,keys({j})),"material order/quantity changed");
 }
 // Two workers share care of the same animal as well as distinct plots.
 std::vector<Plan>base{route(44,{{action(Op::FEED),44},{action(Op::CARE),33},{action(Op::CARE),34}}),{}};
 std::vector<Job>js{job(33,{action(Op::CARE)}),job(34,{action(Op::CARE)})};int assigned=0;
 auto planned=c.insert_shared_services(view(env),base,js,keys(js),24,assigned);
 check(assigned==2,"shared jobs not fully assigned");
 check(count(planned,Op::CARE,33)==1&&count(planned,Op::CARE,34)==1,"duplicate/missing shared work");
 check(material_same(base[0],planned[0],keys(js)),"FEED ownership/order changed");
 check(c.plan_load(planned)<c.plan_load(base),"shared insertion not better than original route");
 for(int pos:{33,34,44}){auto&t=farm.tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item(SH);t.fed_today=pos!=44;}
 pr.inventories[0][W]=1;pr.inventory_order[0]={W};
 auto live=env;for(int k=0;k<c.plan_load(planned).first;k++){PlayerAction own;own.units.resize(2);for(int u=0;u<2;u++)if(k<int(planned[u].a.size()))own.units[u]=planned[u].a[k];live.step({own,PlayerAction{}});}
 check(live.farms()[0].tiles[33].cared_today&&live.farms()[0].tiles[34].cared_today&&live.farms()[0].tiles[44].fed_today,"shared simulated effects incomplete");
 // A promised delivery cannot disappear when HARVEST changes worker. Cheapest
 // unguarded insertion would choose the idle worker with no DROP.
 farm.hands[0]={3,3};base={route(44,{{action(Op::HARVEST),33},{action(Op::DROP),44}}),{}};
 js={job(33,{action(Op::HARVEST)})};planned=c.insert_shared_services(view(env),base,js,keys(js),24,assigned);
 check(assigned==1,"delivery-safe insertion missing");
 check(count(planned,Op::HARVEST,33)==1,"harvest duplicated");
 bool delivered=false;for(auto&p:planned){bool collected=false;for(auto a:p.a){if(a.op==Op::HARVEST)collected=true;if(a.op==Op::DROP&&collected)delivered=true;}}
 check(delivered,"harvest moved after/beyond its delivery");
 auto&t=farm.tiles[33];t.yield_units=3;pr.inventories[0]={};pr.inventory_order[0].clear();pr.shed={};
 live=env;for(int k=0;k<c.plan_load(planned).first;k++){PlayerAction own;own.units.resize(2);for(int u=0;u<2;u++)if(k<int(planned[u].a.size()))own.units[u]=planned[u].a[k];live.step({own,PlayerAction{}});}
 check(live.privates()[0].shed[WO]==3,"harvest never reached warehouse");
 // Cross-worker fertilizer before water must hold, or the alternative must be
 // rejected. Unit-id order in a simultaneous turn is not a planning shortcut.
 base={route(44,{{action(Op::FERTILIZE),33},{action(Op::WATER),33}}),{}};
 js={job(33,{action(Op::WATER)})};planned=c.insert_shared_services(view(env),base,js,keys(js),24,assigned);
 if(assigned==1){int ft=-1,wt=1000;for(auto&p:planned)for(size_t k=0;k<p.a.size();k++){if(p.a[k].op==Op::FERTILIZE)ft=int(k);if(p.a[k].op==Op::WATER)wt=int(k);}check(ft<wt,"cross-worker causal dependency broken");}
 else check(assigned==-1,"unsafe alternative not explicitly rejected");
 // Insufficient day budget cannot be presented as a complete assignment.
 js={job(0,{action(Op::CARE)}),job(99,{action(Op::CARE)})};base.resize(2);base[0]={};base[1]={};
 planned=c.insert_shared_services(view(env),base,js,keys(js),1,assigned);check(assigned<int(js.size()),"incomplete alternative passed full-set gate");
 auto material=job(33,{action(Op::FEED)});material.needs[W]=1;bool rejected=false;
 try{c.insert_shared_services(view(env),base,{material},keys({material}),24,assigned);}catch(const std::logic_error&){rejected=true;}
 check(rejected,"unfunded material task admitted as free service");
 // Disabled is a literal no-op extension of the existing controller.
 Params p;p.stepwise_recoordination=true;p.shared_task_atoms_v2=true;
 Controller legacy(p),off(p);legacy.day=off.day=0;legacy.phase=off.phase=3;legacy.target=off.target={{33,SH},{34,SH},{44,SH}};legacy.plans=off.plans=base;
 legacy.recoordinate(view(env));off.recoordinate(view(env));
 for(size_t u=0;u<off.plans.size();u++)check(c.same_remaining(off.plans[u],legacy.plans[u]),"disabled extension changed schedule");
 check(off.shared_insertion_checks==0&&off.shared_insertion_applied==0,"disabled counter changed");
 Controller dirty(p),fresh(p);dirty.p.shared_service_insertions=fresh.p.shared_service_insertions=true;dirty.day=12;dirty.last_step=500;dirty.shared_insertion_checks=11;dirty.shared_insertion_applied=3;dirty.shared_insertion_peak_saved=17;dirty.shared_insertion_steps_saved=9;
 Simulator reset_env(Config{},812);dirty.act(view(reset_env));fresh.act(view(reset_env));
 check(dirty.shared_insertion_checks==fresh.shared_insertion_checks&&dirty.shared_insertion_applied==fresh.shared_insertion_applied&&dirty.shared_insertion_peak_saved==fresh.shared_insertion_peak_saved&&dirty.shared_insertion_steps_saved==fresh.shared_insertion_steps_saved,"cross-game insertion state leaked");
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";
}
