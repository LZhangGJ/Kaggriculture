#pragma once
// A single transport repair: transfer one resource-free, same-plot job group
// so an otherwise deadline-blocked cargo worker can visit the warehouse.
// The job multiset and whole-fleet same-plot order must remain identical.
// No future prices, rival inventory, episode seed, or opponent identity.
#include "policy.hpp"
#include "resource_exchange.hpp"
#ifndef A08_DELIVERY_HANDOFF
#define A08_DELIVERY_HANDOFF 0
#endif
static_assert(A08_DELIVERY_HANDOFF==0||A08_DELIVERY_HANDOFF==1);
namespace dp7::deliveryhandoff {
using exchange::Groups;
struct Choice {
 bool selected=false;
 int donor=-1,receiver=-1,plot=-1,depot=-1,quantity=0,extra_actions=0,arrival=-1;
 int overflow_before=0,overflow_after=0,candidates=0,deadline_rejects=0;
 int order_rejects=0,capacity_rejects=0,economy_rejects=0;
 double conditional_quote=0,score=0;
 std::vector<Plan> base,best;
};
inline Plan remaining(const Plan&p){
 Plan out;
 if(p.index>p.a.size()||p.target.size()!=p.a.size())return out;
 out.a.assign(p.a.begin()+p.index,p.a.end());
 out.target.assign(p.target.begin()+p.index,p.target.end());return out;
}
inline bool free_group(const exchange::Group&g){
 if(g.actions.empty())return false;
 for(auto a:g.actions)if(a.op!=Op::WATER&&a.op!=Op::HARVEST&&
    a.op!=Op::CARE&&a.op!=Op::COLLECT_FERTILIZER)return false;
 return true;
}
inline bool donor_safe(const Plan&p,const Counts&cargo,bool funded_seeds=false){
 for(int i=9;i<12;i++)if(cargo[i])return false;
 for(size_t k=0;k<p.a.size();k++){
  auto a=p.a[k];if(!Controller::movement(a.op)&&p.target[k]<0)return false;
  // Stronger than merely subtracting current inputs. A moved harvest may not
  // be the later source of this worker's wheat/fertilizer/animal commitment.
  if((a.op==Op::PLANT&&!funded_seeds)||a.op==Op::DROP||a.op==Op::PICKUP||a.op==Op::FEED||
     a.op==Op::FERTILIZE||a.op==Op::PLACE)return false;
 }return true;
}
// All currently planned seed consumption must already be funded. Retiming
// a HARVEST/PLANT/WATER group is safe when current seeds cover the WHOLE fleet,
// regardless of the atomic same-tick PLANT order. No future purchase is credited.
inline bool seeds_funded(const View&o,const std::vector<Plan>&ps){
 std::array<int,5>need{};for(const auto&p:ps)for(auto a:p.a)if(a.op==Op::PLANT){
  int i=int(a.item);if(i<0||i>=5)return false;need[i]++;
 }for(int i=0;i<5;i++)if(need[i]>o.priv.seeds[i])return false;return true;
}
inline bool pickup_prefix_preserved(const Plan&before,const Plan&after){
 int last=-1;for(size_t k=0;k<before.a.size();k++)if(before.a[k].op==Op::PICKUP)last=int(k);
 if(last>=int(after.a.size()))return false;
 for(int k=0;k<=last;k++)if(!Controller::same_action(before.a[k],after.a[k])||before.target[k]!=after.target[k])return false;
 return true;
}
// Rebuild only the two changed routes. Material operations keep their original
// destination and relative order, including pickups at the existing depot.
inline Plan route(const View&o,int u,const Groups&gs,int first_depot=-1){
 Plan out;int pos=cell(u?o.own.hands.at(u-1):o.own.farmer);
 if(first_depot>=0){Controller::walk(out,pos,first_depot);out.a.push_back(action(Op::DROP));out.target.push_back(first_depot);}
 for(auto&g:gs){Controller::walk(out,pos,g.pos);for(auto a:g.actions){out.a.push_back(a);out.target.push_back(g.pos);}}
 return out;
}
inline int cargo_upper(const View&o,int u,const Plan&p,int through){
 // No speculative market clearance is credited. Count all current cargo and
 // all visible scheduled harvests up to arrival, even those already consumed.
 int q=sum(o.priv.inventories[u]);std::set<int>harvested,collected;
 for(int k=0;k<int(p.a.size())&&k<=through;k++){
  auto a=p.a[k];int pos=p.target[k];
  if(a.op==Op::PICKUP)q+=std::max(0,a.quantity);
  if(pos<0)continue;const auto&t=o.own.tiles[pos];
  if(a.op==Op::HARVEST&&harvested.insert(pos).second){
   int visible=std::max(0,int(t.yield_units));
   // One finite-crop WATER can add at most two units today. Parent forecasts
   // omit this increment; capacity safety must not rely on that omission.
   if(plant(t)&&!ongoing(int(t.crop))&&!t.watered_today)visible+=2;
   q+=visible;
  }else if(a.op==Op::COLLECT_FERTILIZER&&animal(t)&&t.fertilizer_available&&collected.insert(pos).second)q++;
 }return q;
}
inline bool capacity_safe(const View&o,const std::vector<Plan>&ps,int donor,int arrival,int qty){
 int upper=sum(o.priv.shed)+qty;
 for(size_t u=0;u<ps.size();u++)if(int(u)!=donor){
  // Other units that can possibly reach ANY depot during this prefix may
  // preempt the new DROP. Bound their whole visible load, without assuming a
  // sale order succeeds. Existing runtime DROP capacity protection also stays.
  int start=cell(u?o.own.hands[u-1]:o.own.farmer);
  if(near(start)<=arrival)upper+=cargo_upper(o,int(u),ps[u],arrival);
 }return upper<=100;
}
inline Choice propose(const Controller&ctl,const View&o){
 Choice out;
 if(!A08_DELIVERY_HANDOFF||ctl.delivery_observation_step!=o.step||!ctl.p.midroute_delivery||
    ctl.phase!=3||ctl.day!=o.day||o.day>=29||o.hour>=22||ctl.plans.size()!=o.priv.inventories.size())return out;
 out.overflow_before=ctl.expected_auto_deposit(o)-(100-ctl.p.shed_safety);
 if(out.overflow_before<=0)return out;
 std::vector<Groups>groups;for(auto&p:ctl.plans){out.base.push_back(remaining(p));groups.push_back(exchange::groups(out.base.back()));}
 out.best=out.base;
 if(!exchange::deadlines(o,out.base)||!seeds_funded(o,out.base))return out;
 const auto causal_order=exchange::order(out.base);
 for(size_t u=0;u<groups.size();u++){
  auto&cargo=o.priv.inventories[u];int qty=sum(cargo);
  if(qty<=0||qty>100||groups[u].empty()||!donor_safe(out.base[u],cargo,true))continue;
  int start=cell(u?o.own.hands[u-1]:o.own.farmer);
  for(int depot_pos:dp7::depot){
   const int arrival=dist(start,depot_pos);if(o.hour+arrival+1>22)continue;
   const auto single=route(o,int(u),groups[u],depot_pos);
   // The existing single-worker midroute policy already handles these. This
   // repair addresses its remaining-route deadline rejection, not a new sale
   // timing policy or a broad replacement of the old dispatch priorities.
   if(o.hour+int(single.a.size())+1<=24)continue;
   for(size_t g=0;g<groups[u].size();g++)if(free_group(groups[u][g])){
    auto dg=groups[u];auto transfer=dg[g];dg.erase(dg.begin()+g);
    auto dp=route(o,int(u),dg,depot_pos);
    if(o.hour+int(dp.a.size())>24)continue;
    for(size_t v=0;v<groups.size();v++)if(v!=u){
     // Do not insert before a scheduled input pickup: preserve the recipient's
     // warehouse allocation ordering rather than funding it from new cargo.
     size_t first=0;bool unsupported=false;
     for(size_t j=0;j<groups[v].size();j++)for(auto a:groups[v][j].actions)
      {if(a.op==Op::PICKUP)first=j+1;if(a.op==Op::DROP)unsupported=true;}
     for(size_t k=0;k<out.base[v].a.size();k++)if(!Controller::movement(out.base[v].a[k].op)&&out.base[v].target[k]<0)unsupported=true;
     if(unsupported)continue;
     for(size_t at=first;at<=groups[v].size();at++){
      auto rg=groups[v];rg.insert(rg.begin()+at,transfer);auto rp=route(o,int(v),rg);
      if(o.hour+int(rp.a.size())>24||!pickup_prefix_preserved(out.base[v],rp))continue;
      out.candidates++;auto trial=out.base;trial[u]=dp;trial[v]=std::move(rp);
      if(!exchange::deadlines(o,trial)){out.deadline_rejects++;continue;}
      if(exchange::order(trial)!=causal_order){out.order_rejects++;continue;}
      if(!capacity_safe(o,trial,int(u),arrival,qty)){out.capacity_rejects++;continue;}
      const int extra=std::max(1,int(trial[u].a.size()+trial[v].a.size())-int(out.base[u].a.size()+out.base[v].a.size()));
      // Quote only sale products, never wheat/fertilizer reserved elsewhere.
      // This is a conservative opportunity ranking, NOT recovered terminal
      // cash. The unchanged market DP still decides actual sale quantities.
      double quote=0;for(int i=1;i<=WO;i++)quote+=revenue(i,o.market.inventory[i],cargo[i]);
      quote*=double(std::min(qty,out.overflow_before))/qty;
      if(quote<=ctl.p.action_shadow*extra+1e-6){out.economy_rejects++;continue;}
      double score=(quote-ctl.p.action_shadow*extra)/extra;
      if(!out.selected||score>out.score+1e-6){
       out.selected=true;out.score=score;out.conditional_quote=quote;out.donor=int(u);out.receiver=int(v);
       out.plot=transfer.pos;out.depot=depot_pos;out.quantity=qty;out.extra_actions=extra;out.arrival=arrival;
       out.overflow_after=out.overflow_before-qty;out.best=std::move(trial);
      }
     }
    }
   }
  }
 }
 return out;
}
inline bool apply(Controller&ctl,const View&o){
 if(!A08_DELIVERY_HANDOFF||ctl.delivery_observation_step!=o.step)return false;
 ctl.delivery_handoff_checks++;auto c=propose(ctl,o);if(!c.selected)return false;
 // Only these two plans are replaced. All other workers and the original
 // investment, maintenance, market DP and input-recovery machinery are intact.
 ctl.plans[c.donor]=std::move(c.best[c.donor]);ctl.plans[c.receiver]=std::move(c.best[c.receiver]);
 ctl.delivery_handoff_insertions++;ctl.delivery_handoff_quantity+=c.quantity;
 ctl.delivery_handoff_last_step=o.step;ctl.delivery_handoff_donor=c.donor;ctl.delivery_handoff_receiver=c.receiver;
 return true;
}
}
