#pragma once
#include "planner.hpp"
#ifndef R2_CROP_CLOCK_MODE
#define R2_CROP_CLOCK_MODE 0
#endif
namespace triad {
// Estimates removal timing, not intent or private inventory. Mature DIG and
// HARVEST may be observationally ambiguous; retain fallback/prior and audit.
struct ObservedCropClock {
 std::array<fastkag::Tile,100>previous{};
 std::array<std::array<int,32>,5>history{};
 std::array<int,5>count{},cursor{};
 int previous_step=-1;
 void observe(const dp7::View&o){
  if(previous_step==o.step)return;
  if(previous_step>=o.step||previous_step+1!=o.step){count={};cursor={};previous_step=-1;}
  if(previous_step>=0){
   int d=previous_step/24;
   for(int p=0;p<100;p++){
    const auto&a=previous[p];const auto&b=o.opponent.tiles[p];int k=int(a.crop);
    if(a.kind!=fastkag::TileKind::PLANT||(k!=dp7::W&&k!=dp7::C&&k!=dp7::M))continue;
    int age=d-a.planted_day;
    if(a.yield_units<=0||age<dp7::first[k]||age>=30)continue;
    if(a.max_lifespan_step>=0&&a.max_lifespan_step<=previous_step+1)continue;
    if((previous_step+1)%24==0&&!a.watered_today&&a.consecutive_unwatered>=1)continue;
    bool gone=b.kind==fastkag::TileKind::EMPTY||(b.kind==fastkag::TileKind::PLANT&&(b.crop!=a.crop||b.planted_day!=a.planted_day));
    if(gone){history[k][cursor[k]]=age;cursor[k]=(cursor[k]+1)%32;count[k]=std::min(32,count[k]+1);}
   }
  }
  std::copy(o.opponent.tiles.begin(),o.opponent.tiles.end(),previous.begin());previous_step=o.step;
 }
 std::array<std::array<double,30>,5> weights()const{
  std::array<std::array<double,30>,5>w{};
  for(int k:{dp7::W,dp7::C,dp7::M})if(count[k]>=3){
   // One fallback pseudo-observation; 32-event window follows changed behavior.
   // Buffer/minimum support control uncertainty, not an opponent-specific route.
   w[k][dp7::harvest_age[k]]=1;
   for(int n=0;n<count[k];n++)w[k][history[k][n]]+=1;
  }
  return w;
 }
};
}
