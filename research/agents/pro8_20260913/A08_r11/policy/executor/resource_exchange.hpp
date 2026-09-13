#pragma once
// Online public-state equal-input service exchange. No Simulator or future state.
#include "policy.hpp"
#include <map>
namespace dp7::exchange {
struct Group {int pos=-1;Acts actions;};
using Groups=std::vector<Group>;
inline Groups groups(const Plan&p){
 Groups out;for(size_t k=p.index;k<p.a.size();k++){
  if(Controller::movement(p.a[k].op)||p.target[k]<0)continue;
  if(out.empty()||out.back().pos!=p.target[k])out.push_back({p.target[k],{}});
  out.back().actions.push_back(p.a[k]);
 }return out;
}
inline bool transferable(const Group&g){
 bool input=false;for(auto a:g.actions){
  input|=a.op==Op::FEED||a.op==Op::FERTILIZE;
  if(a.op!=Op::FEED&&a.op!=Op::FERTILIZE&&a.op!=Op::CARE&&a.op!=Op::WATER)return false;
 }return input;
}
inline bool identical(const Group&a,const Group&b){
 if(a.actions.size()!=b.actions.size())return false;
 for(size_t k=0;k<a.actions.size();k++)if(!Controller::same_action(a.actions[k],b.actions[k]))return false;
 return true;
}
inline Plan rebuild(const View&o,int unit,const Groups&gs){
 Plan out;int pos=cell(unit?o.own.hands.at(unit-1):o.own.farmer);
 for(auto&g:gs){int dest=g.pos;
  if(!g.actions.empty()&&(g.actions[0].op==Op::PICKUP||g.actions[0].op==Op::DROP)){
   dest=depot[0];for(int d:depot)if(dist(pos,d)<dist(pos,dest))dest=d;
  }
  Controller::walk(out,pos,dest);for(auto a:g.actions){out.a.push_back(a);out.target.push_back(dest);}
 }return out;
}
// Whole-team field order, not per-worker nearest-task greed. Changing the
// original FERTILIZE/WATER, HARVEST/PLANT, or other same-tile order is rejected.
using Event=std::tuple<int,int,int>; // op, item, quantity
inline std::map<int,std::vector<Event>> order(const std::vector<Plan>&ps){
 std::map<int,std::vector<std::tuple<int,int,Event>>> timed;
 for(size_t u=0;u<ps.size();u++)for(size_t k=0;k<ps[u].a.size();k++){
  auto a=ps[u].a[k];if(ps[u].target[k]<0||Controller::movement(a.op)||a.op==Op::PICKUP||a.op==Op::DROP)continue;
  timed[ps[u].target[k]].push_back({int(k),int(u),{int(a.op),int(a.item),a.quantity}});
 }
 std::map<int,std::vector<Event>> out;for(auto&[p,es]:timed){std::sort(es.begin(),es.end());for(auto&[k,u,e]:es)out[p].push_back(e);}return out;
}
inline bool resources(const View&o,const std::vector<Plan>&ps){
 Counts pickups{},seed_need{};if(ps.size()!=o.priv.inventories.size())return false;
 for(size_t u=0;u<ps.size();u++){
  Counts held=o.priv.inventories[u];bool field_started=false;
  for(size_t k=0;k<ps[u].a.size();k++){
   auto a=ps[u].a[k];if(Controller::movement(a.op))continue;int i=int(a.item);
   if(a.op==Op::PICKUP){
    // Do not finance pickups from a future DROP or a different worker's cargo.
    if(field_started||i<0||i>=12||a.quantity<=0)return false;
    pickups[i]+=a.quantity;held[i]+=a.quantity;
   }else if(a.op==Op::DROP){held={};}
   else {field_started=true;
    int consume=a.op==Op::FEED?W:a.op==Op::FERTILIZE?F:a.op==Op::PLACE?i:-1;
    if(consume>=0){if(consume>=12||held[consume]<=0)return false;held[consume]--;}
    if(a.op==Op::PLANT){if(i<0||i>=5)return false;seed_need[i]++;}
    // Deliberately no credit for uncollected fertilizer or future harvests.
   }
  }
 }
 for(int i=0;i<12;i++)if(pickups[i]>o.priv.shed[i])return false;
 for(int i=0;i<5;i++)if(seed_need[i]>o.priv.seeds[i])return false;
 return true;
}
inline bool deadlines(const View&o,const std::vector<Plan>&ps){
 for(auto&p:ps){if(o.hour+int(p.a.size())>(o.day>=29?23:24))return false;
  for(size_t k=0;k<p.a.size();k++)if(p.target[k]>=0&&p.a[k].op==Op::HARVEST){
   auto&t=o.own.tiles[p.target[k]];
   if(plant(t)&&t.max_lifespan_step>=0&&o.step+int(k)+1>=t.max_lifespan_step)return false;
  }
 }return true;
}
struct Choice {
 std::vector<Plan>base,best;
 int compatible=0,shorter=0,feasible=0,unit_a=-1,unit_b=-1,pos_a=-1,pos_b=-1;
 int total_saved=0,peak_saved=0;bool base_resources=false,base_deadline=false;
};
inline Choice propose(const Controller&ctl,const View&o){
 Choice out;std::vector<Groups>gs;int stale=0;
 for(size_t u=0;u<ctl.plans.size();u++){auto p=ctl.repair_plan(o,int(u),ctl.plans[u],stale);gs.push_back(groups(p));out.base.push_back(rebuild(o,int(u),gs.back()));}
 out.best=out.base;out.base_resources=resources(o,out.base);out.base_deadline=deadlines(o,out.base);
 if(!out.base_resources||!out.base_deadline)return out;
 auto initial_load=Controller::plan_load(out.base),best_load=initial_load;auto causal_order=order(out.base);
 for(size_t u=0;u<gs.size();u++)for(size_t a=0;a<gs[u].size();a++)if(transferable(gs[u][a]))
  for(size_t v=u+1;v<gs.size();v++)for(size_t b=0;b<gs[v].size();b++){
   if(!transferable(gs[v][b])||gs[u][a].pos==gs[v][b].pos||!identical(gs[u][a],gs[v][b]))continue;
   out.compatible++;std::swap(gs[u][a],gs[v][b]);auto trial=out.base;
   trial[u]=rebuild(o,int(u),gs[u]);trial[v]=rebuild(o,int(v),gs[v]);std::swap(gs[u][a],gs[v][b]);
   auto load=Controller::plan_load(trial);
   // Never lengthen a team's total work to improve its peak, or vice versa.
   if(load.first>initial_load.first||load.second>=initial_load.second)continue;out.shorter++;
   if(!deadlines(o,trial)||!resources(o,trial)||order(trial)!=causal_order)continue;out.feasible++;
   if(load<best_load){best_load=load;out.best=std::move(trial);out.unit_a=int(u);out.unit_b=int(v);out.pos_a=gs[u][a].pos;out.pos_b=gs[v][b].pos;}
  }
 out.total_saved=initial_load.second-best_load.second;out.peak_saved=initial_load.first-best_load.first;return out;
}

inline void apply(Controller&ctl,const View&o){
 if(!ctl.p.resource_aware_exchange||ctl.phase!=3||ctl.day!=o.day||ctl.plans.size()!=o.priv.inventories.size())return;
 ctl.resource_exchange_checks++;auto c=propose(ctl,o);
 ctl.resource_exchange_pairs+=c.compatible;
 if(c.total_saved<=0)return;
 ctl.plans=std::move(c.best);ctl.resource_exchange_applied++;
 ctl.resource_exchange_local_steps_saved+=c.total_saved;
}
}
