#pragma once
// A08 r4: repair a terminal rolling extension as a COMPLETE route, not a
// positive field-yield quote. Existing commitments are never deleted.
#include "observed_day_scenario.hpp"
#include "resource_exchange.hpp"
#include <map>
namespace dp7::completion {
struct Audit {
 bool shape=true,executed=true,known=true;double cash=0.;
 int work=0,waits=0,failed=0,ticks=0;Counts sold{},stranded{},deposited{};
 std::map<Controller::TaskKey,int>effects,harvests;
};
inline bool productive(Op op){return op==Op::HARVEST||op==Op::COLLECT_FERTILIZER;}
inline int start(const View&o,int u){return cell(u?o.own.hands.at(u-1):o.own.farmer);}
inline Plan remaining(const Plan&p){Plan r;for(size_t k=p.index;k<p.a.size();k++){r.a.push_back(p.a[k]);r.target.push_back(p.target[k]);}return r;}
inline bool open_load(const View&o,int u,const Plan&p){
 bool load=sum(o.priv.inventories.at(u))>0;
 for(size_t k=p.index;k<p.a.size();k++){
  if(p.a[k].op==Op::DROP)load=false;
  else if(productive(p.a[k].op)||p.a[k].op==Op::PICKUP)load=true;
 }return load;
}
inline bool close_route(const View&o,int u,Plan&p){
 if(!open_load(o,u,p))return false;
 int pos=Controller::plan_end(o,u,p),dest=depot[0];
 for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;
 Controller::walk(p,pos,dest);p.a.push_back(action(Op::DROP));p.target.push_back(dest);return true;
}
// Same action has to produce its effect, not just pass a shape/type test.
inline bool effect(const View&before,const Simulator&after,int u,Action a,int pos){
 const auto&t=before.own.tiles[pos];const auto&nt=after.farms()[0].tiles[pos];
 const auto&old=before.priv.inventories[u];const auto&now=after.privates()[0].inventories[u];int i=int(a.item);
 switch(a.op){
 case Op::PASS:return true;
 case Op::NORTH:case Op::SOUTH:case Op::EAST:case Op::WEST:{
  int wanted=pos+(a.op==Op::NORTH?-10:a.op==Op::SOUTH?10:a.op==Op::EAST?1:-1);
  return start({before.step,before.day,before.hour,after.farms()[0],after.farms()[1],after.privates()[0],after.market(),after.shops()},u)==wanted;
 }
 case Op::DROP:return sum(now)==0&&sum(after.privates()[0].shed)-sum(before.priv.shed)==sum(old);
 case Op::PICKUP:return i>=0&&i<12&&now[i]-old[i]==a.quantity;
 case Op::HARVEST:return t.yield_units>0&&sum(now)>sum(old);
 case Op::COLLECT_FERTILIZER:return animal(t)&&t.fertilizer_available&&!nt.fertilizer_available&&now[F]==old[F]+1;
 case Op::WATER:return plant(t)&&!t.watered_today&&nt.watered_today;
 case Op::FERTILIZE:return plant(t)&&old[F]>0&&now[F]==old[F]-1&&nt.fertilized_until_day>=before.day;
 case Op::FEED:return animal(t)&&!t.fed_today&&nt.fed_today&&now[W]==old[W]-1;
 case Op::CARE:return animal(t)&&!t.cared_today&&nt.cared_today;
 case Op::DIG:return t.kind!=TileKind::EMPTY&&!animal(t)&&nt.kind==TileKind::EMPTY;
 case Op::BUILD_COOP:return t.kind==TileKind::EMPTY&&nt.kind==TileKind::COOP;
 case Op::BUILD_PASTURE:return t.kind==TileKind::EMPTY&&nt.kind==TileKind::PASTURE;
 case Op::PLACE:return i>=9&&i<12&&!animal(t)&&int(nt.animal)==i&&now[i]==old[i]-a.quantity;
 case Op::PLANT:return i>=0&&i<5&&t.kind==TileKind::EMPTY&&plant(nt)&&int(nt.crop)==i&&after.privates()[0].seeds[i]==before.priv.seeds[i]-1;
 default:return false;
 }
}
// Terminal horizon only: no night, no unknown shop/weed draw, no residual asset
// value. Current public rival farm is frozen and sends NO market orders.
// Its actual private stock and future script are never accepted. The result is
// conditional cash, NOT a real opponent-game score or a promised fill.
inline Audit audit(const Controller&source,const View&o,const std::vector<Plan>&ps){
 Audit a;if(o.day!=29||ps.size()!=o.priv.inventories.size()){a.known=a.shape=false;return a;}
 const int horizon=719-o.step;
 for(const auto&p:ps)if(p.a.size()!=p.target.size()||p.index>p.a.size()||int(p.a.size()-p.index)>horizon){a.shape=false;}
 if(!a.shape){a.known=false;return a;}
 ObservedDayScenario world(o);std::vector<size_t>idx;for(auto&p:ps)idx.push_back(p.index);
 while(!world.finished()){
  const auto v=world.view();std::vector<Action>acts(ps.size());
  for(size_t u=0;u<ps.size();u++)if(idx[u]<ps[u].a.size())acts[u]=ps[u].a[idx[u]];
  for(size_t u=0;u<ps.size();u++){
   if(idx[u]>=ps[u].a.size())continue;
   auto before=world.project_units(acts,int(u));
   View pv{v.step,v.day,v.hour,before.farms()[0],before.farms()[1],before.privates()[0],before.market(),before.shops()};
   int pos=start(pv,int(u)),target=ps[u].target[idx[u]];auto act=acts[u];
   if(target>=0&&target!=pos){a.executed=false;a.failed++;}
   if(act.op==Op::DROP&&sum(pv.priv.inventories[u])>100-sum(pv.priv.shed)){
    // Match the live full-DROP guard: wait for a real sale to free room;
    // a wait consumes one tick and can make later commitments infeasible.
    acts[u]=action(Op::PASS);a.waits++;a.work++;continue;
   }
   auto after=world.project_units(acts,int(u+1));
   bool ok=effect(pv,after,int(u),act,pos);
   if(!ok){a.executed=false;a.failed++;}
   if(ok&&!Controller::movement(act.op)&&act.op!=Op::PASS&&act.op!=Op::DROP&&act.op!=Op::PICKUP){
    const auto key=Controller::task_key(pos,act);a.effects[key]++;
    if(productive(act.op))a.harvests[key]+=sum(after.privates()[0].inventories[u])-sum(pv.priv.inventories[u]);
   }
   if(act.op==Op::DROP&&ok)for(int i=0;i<12;i++)a.deposited[i]+=pv.priv.inventories[u][i];
   a.work+=act.op!=Op::PASS;idx[u]++;
  }
  auto post=world.project_units(acts,-1);Counts sale{};for(int i=0;i<9;i++)sale[i]=post.privates()[0].shed[i];
  auto orders=source.sales_sorted(v,sale);world.advance({acts,orders});
  for(size_t k=0;k<orders.size();k++)a.sold[int(orders[k].item)]+=world.fills()[k];
  ++a.ticks;
 }
 for(size_t u=0;u<ps.size();u++){if(idx[u]!=ps[u].a.size())a.executed=false;add(a.stranded,world.inventory().inventories[u]);}
 add(a.stranded,world.inventory().shed);a.cash=world.own().money-o.own.money;
 a.known=std::isfinite(a.cash);return a;
}
inline bool preserves(const Audit&base,const Audit&trial){
 if(!trial.known||!trial.shape||!trial.executed)return false;
 for(auto&[k,n]:base.effects){auto it=trial.effects.find(k);if(it==trial.effects.end()||it->second<n)return false;}
 for(auto&[k,n]:base.harvests){auto it=trial.harvests.find(k);if(it==trial.harvests.end()||it->second<n)return false;}
 for(int i=0;i<9;i++)if(trial.stranded[i]>base.stranded[i]||trial.sold[i]<base.sold[i])return false;
 return true;
}
struct Candidate{std::vector<Plan>plans;std::vector<int>new_indices;int unit=-1,at=-1,extra=0;};
inline Candidate inserted(const View&o,const std::vector<Plan>&base,const Job&job,int u,int slot){
 Candidate c;c.plans=base;c.unit=u;c.at=slot;auto sem=Controller::plan_semantics(base[u]);Plan p;int pos=start(o,u);
 for(int k=0;k<=int(sem.size());k++){
  if(k==slot){Controller::walk(p,pos,job.pos);for(auto a:job.actions){c.new_indices.push_back(int(p.a.size()));p.a.push_back(a);p.target.push_back(job.pos);}}
  if(k<int(sem.size())){Controller::walk(p,pos,sem[k].second);p.a.push_back(sem[k].first);p.target.push_back(sem[k].second);}
 }
 close_route(o,u,p);c.extra=int(p.a.size())-int(base[u].a.size());c.plans[u]=std::move(p);return c;
}
inline bool precedence(const std::vector<Plan>&base,const Candidate&c,const Job&whole){
 auto old=c.plans;for(int k:c.new_indices)old[c.unit].target[k]=-1;
 if(exchange::order(old)!=exchange::order(base))return false;
 // Match the full fresh WATER/HARVEST (or other service) sequence across ALL
 // workers, including a predecessor already committed to another worker.
 auto ordered=exchange::order(c.plans);const auto it=ordered.find(whole.pos);if(it==ordered.end())return false;
 size_t k=0;for(auto a:whole.actions){while(k<it->second.size()&&it->second[k]!=exchange::Event{int(a.op),int(a.item),a.quantity})k++;if(k==it->second.size())return false;k++;}
 return true;
}
inline bool within(const View&o,const std::vector<Plan>&plans){
 const int limit=719-o.step;for(const auto&p:plans)if(int(p.a.size()-p.index)>limit)return false;return true;
}
inline bool close(Controller&c,const View&o,std::vector<Plan>&plans){
 if(!A08_COMPLETE_ROLLING_ROUTES||o.day!=29)return false;
 bool any=false;
 // One unreturnable worker must not veto another worker's reachable return.
 // Consider every single tail as well as their joint closure, then re-evaluate
 // the WHOLE fleet after each accepted choice. At most one pass per worker.
 for(size_t pass=0;pass<plans.size();pass++){
  std::vector<std::vector<Plan>>choices;auto all=plans;bool pending=false;
  for(int u=0;u<int(plans.size());u++){
   auto single=plans;if(close_route(o,u,single[u])){pending=true;choices.push_back(std::move(single));close_route(o,u,all[u]);}
  }
  if(!pending)break;choices.push_back(std::move(all));++c.completion_checks;
  auto base=audit(c,o,plans);c.completion_evaluations++;if(!base.known)break;
  int best=-1;double best_net=0,best_cash=0,best_cost=0;
  for(size_t k=0;k<choices.size();k++){
   const auto&trial=choices[k];
   if(!within(o,trial)){c.completion_horizon_rejects++;continue;}
   if(!exchange::resources(o,trial)){c.completion_resource_rejects++;continue;}
   auto end=audit(c,o,trial);c.completion_evaluations++;
   const double cost=c.p.action_shadow*std::max(0,end.work-base.work);
   if(!preserves(base,end)){c.completion_execution_rejects++;continue;}
   const double net=end.cash-base.cash-cost;
   if(net<=1e-6){c.completion_value_rejects++;continue;}
   if(best<0||net>best_net+1e-6){best=int(k);best_net=net;best_cash=end.cash-base.cash;best_cost=cost;}
  }
  if(best<0)break;
  c.completion_last_step=o.step;c.completion_last_unit=-1;c.completion_last_pos=-1;
  c.completion_last_cash_delta=best_cash;c.completion_last_work_cost=best_cost;
  plans=std::move(choices[best]);c.completion_closed++;any=true;
 }
 return any;
}
inline bool insert(Controller&c,const View&o,std::vector<Plan>&plans,const Job&whole,const Job&missing){
 if(!A08_COMPLETE_ROLLING_ROUTES||o.day!=29||missing.actions.empty()||!Controller::resource_free_service(missing))return false;
 ++c.completion_checks;std::vector<Candidate>choices;
 for(int u=0;u<int(plans.size());u++){
  const auto sem=Controller::plan_semantics(plans[u]);
  for(int at=0;at<=int(sem.size());at++){
   auto trial=inserted(o,plans,missing,u,at);++c.completion_candidates;
   if(!within(o,trial.plans)){c.completion_horizon_rejects++;continue;}
   if(!precedence(plans,trial,whole)){c.completion_precedence_rejects++;continue;}
   if(!exchange::resources(o,trial.plans)){c.completion_resource_rejects++;continue;}
   choices.push_back(std::move(trial));
  }
 }
 if(choices.empty())return false;
 std::stable_sort(choices.begin(),choices.end(),[](const Candidate&a,const Candidate&b){return std::tuple(a.extra,a.plans[a.unit].a.size(),a.unit,a.at)<std::tuple(b.extra,b.plans[b.unit].a.size(),b.unit,b.at);});
 // Bounded terminal insertion search, not a claim of global optimality.
 if(choices.size()>64){c.completion_bounded_skips+=choices.size()-64;choices.resize(64);}
 auto base=audit(c,o,plans);c.completion_evaluations++;if(!base.known)return false;
 int best=-1;double best_net=0,best_cash=0,best_cost=0;
 for(int k=0;k<int(choices.size());k++){
  auto end=audit(c,o,choices[k].plans);c.completion_evaluations++;
  if(!preserves(base,end)){c.completion_execution_rejects++;continue;}
  const double cost=c.p.action_shadow*std::max(0,end.work-base.work),delta=end.cash-base.cash,net=delta-cost;
  if(net<=1e-6){c.completion_value_rejects++;continue;}
  if(best<0||net>best_net+1e-6){best=k;best_net=net;best_cash=delta;best_cost=cost;}
 }
 if(best<0)return false;
 c.completion_last_step=o.step;c.completion_last_unit=choices[best].unit;c.completion_last_pos=whole.pos;
 c.completion_last_cash_delta=best_cash;c.completion_last_work_cost=best_cost;
 plans=std::move(choices[best].plans);c.completion_applied++;return true;
}
} // namespace dp7::completion
