#pragma once
// A bounded, public-state project option. Not an opening tape or a private
// replica of inventory. Only two feeder plots and one successor per plot.
#ifndef P16_JOINT_BUNDLES
#define P16_JOINT_BUNDLES 0
#endif
namespace triad {
struct JointSlot {
 int pos=-1,source_birth=-1,source_age=3,next_kind=-1;
 int stage=0; // 0 await PLANT, 1 feeder live, 2 confirmed HARVEST, 3 successor live
 int harvest_step=-1,issued_harvest=-1,deadline=-1;
};
struct JointBundleState {
 std::array<JointSlot,2> slots{};
 int count=0,created_day=-1,animal_limit=75,variant=-1,last_observation=-1;
 bool active=false;
 int source_started=0,source_harvested=0,successor_started=0,cancelled=0,completed=0;
 int last_cancel_reason=0;
 void cancel(int reason){if(active){cancelled++;last_cancel_reason=reason;active=false;}}
 void observe(const dp7::View&o){
  if constexpr(P16_JOINT_BUNDLES==0)return;
  if(!active||o.step==last_observation)return;last_observation=o.step;
  for(int i=0;i<count;i++){
   auto&b=slots[i];const auto&t=o.own.tiles[b.pos];
   if(b.stage==0){
    if(dp7::plant(t)&&int(t.crop)==dp7::W&&t.planted_day>=created_day){b.source_birth=t.planted_day;b.stage=1;source_started++;}
    else if(o.day>created_day+1){cancel(1);return;}
   }else if(b.stage==1){
    bool same=dp7::plant(t)&&int(t.crop)==dp7::W&&t.planted_day==b.source_birth;
    if(!same){
     if(b.issued_harvest==o.step-1&&!dp7::plant(t)&&!dp7::animal(t)){
      b.stage=2;b.harvest_step=b.issued_harvest;b.deadline=std::min(28,b.harvest_step/24+3);source_harvested++;
     }else {cancel(2);return;}
    }
   }else if(b.stage==2){
    if(dp7::plant(t)&&int(t.crop)==b.next_kind&&t.planted_day>b.harvest_step/24){b.stage=3;successor_started++;}
    else if(dp7::animal(t)||dp7::plant(t)){cancel(3);return;}
    else if(o.day>b.deadline){cancel(4);return;}
   }
  }
  bool all=true;for(int i=0;i<count;i++)all&=slots[i].stage==3;
  if(all){active=false;completed++;}
 }
 void record(const dp7::View&o,const fastkag::PlayerAction&a){
  if constexpr(P16_JOINT_BUNDLES==0)return;
  if(!active)return;
  for(int i=0;i<count;i++){
   auto&b=slots[i];if(b.stage!=1)continue;
   for(size_t u=0;u<a.units.size();u++)if(a.units[u].op==fastkag::Op::HARVEST){
    int pos=dp7::cell(u?o.own.hands[u-1]:o.own.farmer);
    if(pos==b.pos&&dp7::plant(o.own.tiles[pos])&&o.own.tiles[pos].yield_units>0)b.issued_harvest=o.step;
   }
  }
 }
 int index(int pos)const{if(!active)return -1;for(int i=0;i<count;i++)if(slots[i].pos==pos&&slots[i].stage<3)return i;return -1;}
 int force_kind(int pos,int day)const{
  int i=index(pos);if(i<0)return -1;const auto&b=slots[i];
  if(b.stage==0)return dp7::W;
  if(b.stage==2&&day>b.harvest_step/24)return b.next_kind;
  return -1;
 }
 double seed_reserve(const dp7::View&o)const{
  if(!active)return 0;std::array<int,5>need{};
  for(int i=0;i<count;i++)if(slots[i].stage<2&&slots[i].next_kind>=0)need[slots[i].next_kind]++;
  double money=0;for(int k=0;k<5;k++)money+=std::max(0,need[k]-o.priv.seeds[k])*dp7::seed_price[k];
  return money;
 }
 int feed_cover(const dp7::View&o)const{
  if(!active)return 0;bool arrived=false;for(int i=0;i<count;i++)arrived|=slots[i].stage==2;
  if(!arrived)return 0;
  int n=0;for(const auto&t:o.own.tiles)n+=dp7::animal(t);
  return n*std::min(2,29-o.day);
 }
};
}
