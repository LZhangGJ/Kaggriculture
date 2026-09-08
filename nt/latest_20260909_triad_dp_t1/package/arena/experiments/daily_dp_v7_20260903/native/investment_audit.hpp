#pragma once
// Offline lifecycle evidence only. Nothing here is passed to the policy.
#include "policy.hpp"
namespace dp7audit {
struct InvestmentEvent {int step,op,item,quantity,unit,position;};
struct ProposalEvent {
 int step,position,kind,current_kind,first_output_day,project_output,capital;
 double standalone_quoted_value;
};
struct Cohort {
 int kind,position,start_step,first_output_state_step=-1,first_harvest_step=-1;
 int end_state_step=-1,harvested=0,output_seen=0;
};
struct InvestmentAudit {
 std::vector<ProposalEvent>proposals;
 std::vector<InvestmentEvent>events;
 std::vector<Cohort>cohorts;
 std::array<int,100>active;
 InvestmentAudit(){active.fill(-1);}
 static int kind(const fastkag::Tile&t){return dp7::animal(t)?int(t.animal):dp7::plant(t)?int(t.crop):-1;}
 void propose(const dp7::Controller&c,const dp7::View&v){
  // Only new-kind/empty-tile proposals. Same-kind replant targets are not
  // counted as new proposals; actual replants are still recorded as cohorts.
  auto values=c.values(v,c.existing(v),{});
  for(auto[pos,k]:c.target){if(k<0||kind(v.own.tiles[pos])==k)continue;
   for(auto[value,pr]:values)if(pr.kind==k){
    int delay=k>=9?dp7::afirst[k-9]:dp7::ongoing(k)?dp7::first[k]:c.h_age(k);
    proposals.push_back({v.step,pos,k,kind(v.own.tiles[pos]),v.day+delay,pr.out,pr.capital,value});break;
   }
  }
 }
 void unit(const fastkag::Simulator&before,const fastkag::Simulator&after,int seat,int u,const fastkag::Action&a){
  using namespace fastkag;int pos=dp7::cell(u?before.farms()[seat].hands[u-1]:before.farms()[seat].farmer);
  auto&t=before.farms()[seat].tiles[pos];auto&nt=after.farms()[seat].tiles[pos];int item=int(a.item),n=0;
  if(a.op==Op::PLANT&&item>=0&&item<5)n=before.privates()[seat].seeds[item]-after.privates()[seat].seeds[item];
  if(a.op==Op::PLACE&&item>=9&&item<12&&kind(nt)==item&&kind(t)!=item)n=1;
  if(n>0){
   if(active[pos]>=0)cohorts[active[pos]].end_state_step=before.step_count();
   active[pos]=int(cohorts.size());cohorts.push_back({item,pos,before.step_count()});
   events.push_back({before.step_count(),int(a.op),item,n,u,pos});
  }
  if(a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER){
   item=a.op==Op::COLLECT_FERTILIZER?dp7::F:dp7::animal(t)?dp7::product[int(t.animal)-9]:int(t.crop);
   if(item>=0&&item<9){n=after.privates()[seat].inventories[u][item]-before.privates()[seat].inventories[u][item];
    if(n>0){events.push_back({before.step_count(),int(a.op),item,n,u,pos});
     if(a.op==Op::HARVEST&&active[pos]>=0){auto&r=cohorts[active[pos]];if(r.first_harvest_step<0)r.first_harvest_step=before.step_count();r.harvested+=n;}
    }
   }
  }
  observe(pos,nt,before.step_count());
 }
 void observe(int pos,const fastkag::Tile&t,int state_step){
  int id=active[pos];if(id<0)return;auto&r=cohorts[id];
  if(kind(t)!=r.kind){r.end_state_step=state_step;active[pos]=-1;return;}
  if(t.yield_units>0&&r.first_output_state_step<0)r.first_output_state_step=state_step;
  r.output_seen=std::max(r.output_seen,int(t.yield_units));
 }
 void finish(const fastkag::Simulator&before,const fastkag::Simulator&after,int seat,const fastkag::PlayerAction&a){
  for(int pos=0;pos<100;pos++)observe(pos,after.farms()[seat].tiles[pos],after.step_count());
  const auto&fills=after.last_market_fills()[seat];
  for(size_t i=0;i<fills.size();i++)if(fills[i]){auto&m=a.market[i];events.push_back({before.step_count(),int(m.op),int(m.item),fills[i],-1,-1});}
 }
};
}
