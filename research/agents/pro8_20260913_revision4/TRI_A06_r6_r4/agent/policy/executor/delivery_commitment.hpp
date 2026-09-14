#pragma once
// Executable delivery commitments: a bounded own-visible, same-day comparison.
// Rival orders, private inventories, future shops and random events are NOT inputs.
#include "observed_day_scenario.hpp"
#include <map>
namespace dp7::delivery {
#ifdef R4_TESTING
// Offline ablation only: absent from the shipped production build.
inline thread_local bool testing_enabled=true;
inline bool enabled(){return testing_enabled;}
#else
inline bool enabled(){return true;}
#endif
using Sem=std::vector<std::pair<Action,int>>;
using Events=std::map<std::tuple<int,int,int>,int>;
inline bool warehouse(Action a,int pos){return a.op==Op::DROP||a.op==Op::PICKUP||(a.op==Op::PLACE&&int(a.item)>=0&&int(a.item)<9&&at_depot(pos));}
inline bool field(Action a,int pos){return pos>=0&&!Controller::movement(a.op)&&a.op!=Op::PASS&&!warehouse(a,pos);}
inline Plan rebuild(const View&o,int u,const Sem&s){Plan p;int pos=cell(u?o.own.hands.at(u-1):o.own.farmer);for(auto[a,d]:s){Controller::walk(p,pos,d);p.a.push_back(a);p.target.push_back(d);}return p;}
inline auto field_order(const std::vector<Plan>&ps){
 std::map<int,std::vector<std::tuple<int,int,int,int,int>>> timed;
 for(size_t u=0;u<ps.size();u++)for(size_t k=ps[u].index;k<ps[u].a.size();k++){auto a=ps[u].a[k];int pos=ps[u].target[k];if(field(a,pos))timed[pos].push_back({int(k-ps[u].index),int(u),int(a.op),int(a.item),a.quantity});}
 std::map<int,std::vector<std::tuple<int,int,int>>> out;for(auto&[pos,ev]:timed){std::sort(ev.begin(),ev.end());for(auto[t,u,op,i,q]:ev)out[pos].push_back({op,i,q});}return out;
}
inline bool deadline(const View&o,const std::vector<Plan>&ps){
 int limit=std::min(24-o.hour,719-o.step);
 for(const auto&p:ps){if(int(p.a.size()-p.index)>limit)return false;
  for(size_t k=p.index;k<p.a.size();k++)if(p.a[k].op==Op::HARVEST&&p.target[k]>=0){const auto&t=o.own.tiles[p.target[k]];if(plant(t)&&t.max_lifespan_step>=0&&o.step+int(k-p.index)+1>=t.max_lifespan_step)return false;}
 }return true;
}
inline int steps(const std::vector<Plan>&ps){int n=0;for(const auto&p:ps)n+=int(p.a.size()-p.index);return n;}
inline std::vector<Plan> remaining(const Controller&c,const View&o){std::vector<Plan>ps;int stale=0;for(int u=0;u<int(c.plans.size());u++)ps.push_back(c.repair_plan(o,u,c.plans[u],stale));return ps;}
inline int unit_effect(const Simulator&before,const Simulator&after,int u,Action a,int pos){
 const auto&t=before.farms()[0].tiles[pos];const auto&nt=after.farms()[0].tiles[pos];const auto&iv=before.privates()[0].inventories[u];const auto&nv=after.privates()[0].inventories[u];
 switch(a.op){
 case Op::HARVEST:case Op::COLLECT_FERTILIZER:return std::max(0,sum(nv)-sum(iv));
 case Op::WATER:return plant(t)&&!t.watered_today&&nt.watered_today;
 case Op::CARE:return animal(t)&&!t.cared_today&&nt.cared_today;
 case Op::FEED:return std::max(0,iv[W]-nv[W]);
 case Op::FERTILIZE:return std::max(0,iv[F]-nv[F]);
 case Op::PLANT:return !plant(t)&&plant(nt);
 case Op::PLACE:return !animal(t)&&animal(nt);
 case Op::DIG:return t.kind!=TileKind::EMPTY&&nt.kind==TileKind::EMPTY;
 case Op::BUILD_COOP:case Op::BUILD_PASTURE:return t.kind!=nt.kind;
 default:return 0;
 }
}
struct Eval {bool valid=false;double cash=0,quote=0;int overflow=0,deposited=0;Events effects;Counts remaining{};};
// Fixed paid workforce and fixed semantic jobs; no speculative investment/hire.
// Warehouse unit phase precedes market. Only physically deposited stock is sold.
// Current shops persist through this bounded day; no opponent orders are assumed.
inline Eval evaluate(const Controller&source,const View&o,const std::vector<Plan>&ps){
 Eval e;if(ps.size()!=o.priv.inventories.size()||!deadline(o,ps))return e;
 auto c=source;c.plans=ps;c.phase=3;c.day=o.day;c.last_step=o.step-1;c.queue.clear();c.pending_admission={};c.admission_values={};
 c.p.early_deposit=0;c.p.funded_labor=false;c.p.intraday_admission=false;c.p.intraday_procurement=false;
 c.p.recover_service_inputs=false;c.p.procure_service_inputs=false;c.p.finance_service_inputs=false;c.p.idle_task_handoff=false;c.p.continuous_market_execution=false;
 ObservedDayScenario w(o);
 while(!w.finished()){
  auto v=w.view();c.resume_compiled_tick=true;auto a=c.act(v);
  if(!a.market.empty())a.market.clear(); // Only the fixed schedule, no new purchases.
  auto post=w.project_units(a.units,-1);
  for(int u=0;u<int(a.units.size());u++){
   auto x=a.units[u];int pos=cell(u?v.own.hands[u-1]:v.own.farmer);
   if(field(x,pos)){auto pre=w.project_units(a.units,u);auto end=w.project_units(a.units,u+1);int n=unit_effect(pre,end,u,x,pos);if(n)e.effects[{pos,int(x.op),int(x.item)}]+=n;}
   if(x.op==Op::DROP||(x.op==Op::PLACE&&int(x.item)>=0&&int(x.item)<9&&at_depot(pos))){auto pre=w.project_units(a.units,u);auto end=w.project_units(a.units,u+1);e.deposited+=std::max(0,sum(end.privates()[0].shed)-sum(pre.privates()[0].shed));}
  }
  Counts reserve{};if(v.day<29){reserve=intraday::reserved(c).shed;int animals=0,heldfeed=0;for(auto&t:post.farms()[0].tiles)animals+=animal(t);for(auto&iv:post.privates()[0].inventories)heldfeed+=iv[W];if(v.hour<23)reserve[W]=std::max(reserve[W],std::max(0,animals-heldfeed));}
  Counts sell{};for(int i=0;i<9;i++)sell[i]=std::max(0,post.privates()[0].shed[i]-reserve[i]);a.market=c.sales_sorted(v,sell);
  w.advance(a);e.overflow+=w.overflow();
 }
 e.cash=w.own().money;e.quote=w.liquidation_quote();e.remaining=w.inventory().shed;for(const auto&iv:w.inventory().inventories)add(e.remaining,iv);e.valid=true;return e;
}
inline bool preserves(const Eval&base,const Eval&alt){if(!base.valid||!alt.valid||alt.overflow>base.overflow)return false;for(auto[key,n]:base.effects){auto it=alt.effects.find(key);if(it==alt.effects.end()||it->second<n)return false;}return true;}
struct Trial {std::vector<Plan>plans;int extra=0,unit=-1,moved=0,quantity=0;double rank=0;};
inline void keep(std::vector<Trial>&q,Trial t,int cap=6){q.push_back(std::move(t));std::stable_sort(q.begin(),q.end(),[](const Trial&a,const Trial&b){return std::tuple(-a.rank,a.extra,a.unit)<std::tuple(-b.rank,b.extra,b.unit);});if(int(q.size())>cap)q.resize(cap);}
inline bool new_precedence(const View&o,const std::vector<Plan>&ps,const Job&j){
 // Existing same-plot order is checked separately. A newly inserted service
 // must not jump ahead of an already-promised prerequisite on another worker.
 auto order=field_order(ps);auto it=order.find(j.pos);if(it==order.end())return false;
 bool seen_water=false,seen_fert=false,seen_feed=false;bool needs_water=false,needs_fert=false,needs_feed=false;
 for(auto[op,i,q]:it->second){needs_water|=op==int(Op::WATER);needs_fert|=op==int(Op::FERTILIZE);needs_feed|=op==int(Op::FEED);}
 for(auto[op,i,q]:it->second){
  if(op==int(Op::HARVEST)&&plant(o.own.tiles[j.pos])&&!ongoing(int(o.own.tiles[j.pos].crop))&&needs_water&&!seen_water)return false;
  if(op==int(Op::WATER)&&needs_fert&&!seen_fert)return false;
  if(op==int(Op::CARE)&&needs_feed&&!seen_feed)return false;
  seen_water|=op==int(Op::WATER);seen_fert|=op==int(Op::FERTILIZE);seen_feed|=op==int(Op::FEED);
 }return true;
}
inline bool terminal_add(Controller&c,const View&o,std::vector<Plan>&base,const Job&j,int limit){
 if(o.day<29||j.actions.empty()||!Controller::resource_free_service(j))return false;
 bool yield=false;for(auto a:j.actions)yield|=a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER;
 // Terminal free care/water alone cannot produce cash without some harvest;
 // still test them as part of a complete route rather than blanket-disabling.
 c.delivery_terminal_checks++;
 std::vector<Trial>trials;const auto old_order=field_order(base);int old_steps=steps(base);
 for(int u=0;u<int(base.size());u++){
  auto sem=Controller::plan_semantics(base[u]);
  for(int at=0;at<=int(sem.size());at++){
   Sem s=sem;s.insert(s.begin()+at,j.actions.size(),{Action{},j.pos});for(size_t k=0;k<j.actions.size();k++)s[at+k]={j.actions[k],j.pos};
   Plan p=rebuild(o,u,s);
   if(yield&&(s.empty()||s.back().first.op!=Op::DROP)){
    int pos=Controller::plan_end(o,u,p),d=depot[0];for(int x:depot)if(dist(pos,x)<dist(pos,d))d=x;
    Controller::walk(p,pos,d);p.a.push_back(action(Op::DROP));p.target.push_back(d);
   }
   if(int(p.a.size())>limit)continue;
   auto all=base;all[u]=std::move(p);if(!deadline(o,all)||!new_precedence(o,all,j))continue;
   // Remove exactly the inserted actions to verify original cross-worker order.
   auto stripped=all;for(auto a:j.actions){bool removed=false;for(size_t k=0;k<stripped[u].a.size();k++)if(!removed&&stripped[u].target[k]==j.pos&&Controller::same_action(stripped[u].a[k],a)){stripped[u].target[k]=-1;removed=true;}}
   if(field_order(stripped)!=old_order)continue;
   int extra=steps(all)-old_steps;keep(trials,{std::move(all),extra,u,0,0,-double(extra)});
  }
 }
 if(trials.empty())return false;
 auto initial=evaluate(c,o,base);c.delivery_evaluations++;
 double gain=0;int best=-1;
 for(int k=0;k<int(trials.size());k++){auto e=evaluate(c,o,trials[k].plans);c.delivery_evaluations++;if(!preserves(initial,e))continue;
  double value=e.cash-initial.cash-c.p.action_shadow*std::max(0,trials[k].extra);
  if(value>gain+1e-6){gain=value;best=k;}
 }
 if(best<0)return false;base=std::move(trials[best].plans);c.delivery_terminal_added++;c.delivery_conditional_gain+=gain;return true;
}
// Under overflow pressure, unload a currently carried surplus product while
// retaining ALL existing jobs. One resource-free service may change worker;
// material tasks, original field order and resource reservations remain intact.
inline bool overflow_wave(Controller&c,const View&o){
 if(o.day>=29||o.hour>=22||c.phase!=3||c.plans.size()!=o.priv.inventories.size()||c.pending_admission.active)return false;
 int excess=c.expected_auto_deposit(o)-(100-c.p.shed_safety);if(excess<=0)return false;
 c.delivery_wave_checks++;auto base=remaining(c,o);if(!deadline(o,base))return false;
 const auto original=field_order(base);int base_steps=steps(base);std::vector<Trial>trials;
 for(int u=0;u<int(base.size());u++){
  auto sem=Controller::plan_semantics(base[u]);if(sem.empty())continue;
  bool underway=false;Counts need{};for(auto[a,p]:sem){if(a.op==Op::PLACE&&int(a.item)>=0&&int(a.item)<9)underway=true;if(a.op==Op::FEED)need[W]++;if(a.op==Op::FERTILIZE)need[F]++;}
  if(underway)continue;
  int item=-1,qty=0;double best_value=0;
  for(int i=0;i<9;i++){int q=std::max(0,o.priv.inventories[u][i]-need[i]);double val=q?revenue(i,o.market.inventory[i],q):0;if(q>0&&val>best_value){item=i;qty=q;best_value=val;}}
  if(item<0)continue;
  // Prefer a single-product PLACE: unlike DROP, it leaves feed/fertilizer in
  // the carrier and does not discard other products when warehouse is full.
  for(int d:depot){
   auto attempt=[&](const std::vector<Plan>&from,int moved){
    Sem s=Controller::plan_semantics(from[u]);s.insert(s.begin(),{action(Op::PLACE,item,qty),d});auto all=from;all[u]=rebuild(o,u,s);
    int deposit_tick=near(cell(u?o.own.hands[u-1]:o.own.farmer));
    // Actual chosen access tile, not just distance to the nearest one.
    deposit_tick=dist(cell(u?o.own.hands[u-1]:o.own.farmer),d);
    if(o.hour+deposit_tick>22||!deadline(o,all)||field_order(all)!=original)return;
    int extra=steps(all)-base_steps;double charge=c.p.action_shadow*std::max(0,extra);
    double benefit=best_value*std::min(qty,excess)/qty-charge;if(benefit<=0)return;
    keep(trials,{std::move(all),extra,u,moved,qty,benefit/std::max(1,extra)},6);
   };
   attempt(base,0);
   // One-service handoff releases haul slack without deleting paid work.
   // No animal/seed/feed/fertilizer input is transferred or invented.
   for(int k=0;k<int(sem.size());k++){
    auto[a,pos]=sem[k];if(a.op!=Op::WATER&&a.op!=Op::CARE)continue;
    Sem rest=sem;rest.erase(rest.begin()+k);auto moved=base;moved[u]=rebuild(o,u,rest);
    for(int v=0;v<int(base.size());v++)if(v!=u){
     auto other=Controller::plan_semantics(base[v]);
     for(int at=0;at<=int(other.size());at++){auto joined=other;joined.insert(joined.begin()+at,{a,pos});moved[v]=rebuild(o,v,joined);
      if(o.hour+int(moved[v].a.size())<=24)attempt(moved,1);
     }moved[v]=base[v];
    }
   }
  }
 }
 if(trials.empty())return false;
 auto initial=evaluate(c,o,base);c.delivery_evaluations++;if(!initial.valid)return false;
 double gain=0;int best=-1;
 for(int k=0;k<int(trials.size());k++){
  auto e=evaluate(c,o,trials[k].plans);c.delivery_evaluations++;
  if(!preserves(initial,e)||e.deposited<=initial.deposited)continue;
  double value=e.cash+e.quote-initial.cash-initial.quote-c.p.action_shadow*std::max(0,trials[k].extra);
  if(value>gain+1e-6){gain=value;best=k;}
 }
 if(best<0)return false;
 c.plans=std::move(trials[best].plans);c.midroute_delivery_insertions++;c.midroute_delivery_quantity+=trials[best].quantity;
 c.delivery_wave_added++;c.delivery_handoffs+=trials[best].moved;c.delivery_added_steps+=trials[best].extra;c.delivery_conditional_gain+=gain;return true;
}
} // namespace dp7::delivery
