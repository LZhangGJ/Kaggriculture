#pragma once
#include "observed_day_scenario.hpp"
namespace dp7::handoff {
inline bool transferable(Op op){return op==Op::WATER||op==Op::CARE||op==Op::HARVEST||op==Op::COLLECT_FERTILIZER;}
inline int position(const View&o,int u){return cell(u?o.own.hands[u-1]:o.own.farmer);}
inline int drop_delay(const Plan&pl,size_t after){for(size_t k=after;k<pl.a.size();k++)if(pl.a[k].op==Op::DROP)return int(k-pl.index)+1;return -1;}
inline View own_view(const Simulator&e){return {e.step_count(),e.day(),e.hour(),e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};}
inline bool effect(const View&before,const Simulator&after,int u,Action a,int pos){
 auto&t=before.own.tiles[pos];auto&nt=after.farms()[0].tiles[pos];
 if(a.op==Op::WATER)return plant(t)&&!t.watered_today&&nt.watered_today;
 if(a.op==Op::CARE)return animal(t)&&!t.cared_today&&nt.cared_today;
 return sum(after.privates()[0].inventories[u])>sum(before.priv.inventories[u]);
}
inline void apply(Controller&c,const View&o,PlayerAction&out){
 if(!c.p.idle_task_handoff||c.phase!=3||c.plans.size()!=out.units.size()||out.units.size()!=o.own.hands.size()+1)return;
 bool idle=std::any_of(out.units.begin(),out.units.end(),[](Action a){return a.op==Op::PASS;});if(!idle)return;
 c.idle_handoff_checks++;ObservedDayScenario phase(o);
 for(size_t u=0;u<out.units.size();u++)if(out.units[u].op==Op::PASS){
  int pos=position(o,int(u)),owner=-1;size_t which=0;std::tuple<int,int,int>rank{100000,100000,100000};
  auto prefix=phase.project_units(out.units,int(u));auto pv=own_view(prefix);
  for(size_t v=0;v<c.plans.size();v++)if(v!=u){const auto&pl=c.plans[v];bool prior_same_plot=false;
   for(size_t k=pl.index;k<pl.a.size();k++)if(pl.target[k]==pos&&!Controller::movement(pl.a[k].op)){
    auto a=pl.a[k];bool first=!prior_same_plot;prior_same_plot=true;if(!first||!transferable(a.op))continue;
    bool conflict=false;
    for(size_t w=0;w<out.units.size();w++)if(position(o,int(w))==pos){
     if(Controller::same_action(out.units[w],a))conflict=true;
     if(w>u&&a.op==Op::WATER&&out.units[w].op==Op::FERTILIZE)conflict=true;
    }
    // A finite crop can gain immediate units when watered. Do not pull its
    // harvest ahead of water owned by another unit or scheduled later.
    const auto&t=pv.own.tiles[pos];bool finite_harvest=a.op==Op::HARVEST&&plant(t)&&!ongoing(int(t.crop))&&!t.watered_today;
    for(const auto&other:c.plans)for(size_t j=other.index;j<other.a.size();j++)if(other.target[j]==pos){
     if(a.op==Op::WATER&&other.a[j].op==Op::FERTILIZE)conflict=true;
     if(finite_harvest&&other.a[j].op==Op::WATER)conflict=true;
    }
    if(finite_harvest)for(size_t w=u+1;w<out.units.size();w++)if(position(o,int(w))==pos&&out.units[w].op==Op::WATER)conflict=true;
    if(conflict)continue;
    bool output=a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER;
    if(output){int promised=drop_delay(pl,k+1),receiver=drop_delay(c.plans[u],c.plans[u].index);
     if((promised>=0&&(receiver<0||receiver>promised))||(o.day>=29&&receiver<0))continue;
    }
    Counts pickups{},seeds{};auto valid=c.valid(pv,int(u),a,pos,pickups,seeds);if(!valid||!Controller::same_action(*valid,a))continue;
    auto trial=out.units;trial[u]=a;auto projected=phase.project_units(trial,int(u+1));if(!effect(pv,projected,int(u),a,pos))continue;
    auto order=std::tuple(int(k-pl.index)+1,int(v),int(k));if(order<rank){rank=order;owner=int(v);which=k;}
   }
  }
  if(owner>=0){auto&pl=c.plans[owner];Action a=pl.a[which];Counts pickups{},seeds{};auto initial=c.valid(o,int(u),a,pos,pickups,seeds);
   c.idle_handoff_prefix+=!initial||!Controller::same_action(*initial,a);out.units[u]=a;
   pl.a.erase(pl.a.begin()+which);pl.target.erase(pl.target.begin()+which);c.idle_handoff_applied++;
  }
 }
}
}
