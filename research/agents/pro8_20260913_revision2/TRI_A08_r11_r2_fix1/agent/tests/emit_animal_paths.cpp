// Deterministic synthetic own-farm scenarios for official-rule cross-checks.
#include "policy/triad.hpp"
#include <iostream>
#include <iomanip>
using namespace competitive;
int main(){
 triad::Settings s;s.service=1;s.work_price=4;s.animal_work=1;s.discount=0;s.scenario=0;
 triad::Controller controller(s);
 int id=0;
 for(int day=24;day<=29;day++)for(int k=9;k<=11;k++)for(int h=0;h<2;h++)for(int b=0;b<held[k-9];b++)for(int y:{0,1,held[k-9]})for(int manure=0;manure<2;manure++)for(int pattern=0;pattern<4;pattern++){
  // Half of the noninitial cases also exercise already-completed own service.
  bool fed=(id%7==3),cared=(id%11==3);
  Flow px{};for(int d=0;d<30;d++)for(int i=0;i<9;i++)px[d][i]=pattern==0?1:pattern==1?(i==W?25:i==F?2:30):pattern==2?(i==W?25:i==F?100:200):(1+(d*37+i*71+39+9*d)%211);
  controller.model.day=day;Tile tile;tile.kind=TileKind::ANIMAL;tile.animal=Item(k);tile.placed_day=day-afirst[k-9];tile.consecutive_unfed=h;tile.pending_care_bonus=b;tile.yield_units=y;tile.fertilizer_available=manure;tile.fed_today=fed;tile.cared_today=cared;
  auto path=controller.animal_path(k,tile.placed_day,44,px,&tile);
  AnimalServiceDP dp;dp.solve(k,tile.placed_day,day,px,s.work_price);
  int hunger=h,bonus=b;bool alive=true;
  std::cout<<"{\"id\":"<<id++<<",\"day\":"<<day<<",\"kind\":"<<k<<",\"birth\":"<<tile.placed_day<<",\"hunger\":"<<h<<",\"bonus\":"<<b<<",\"yield\":"<<y<<",\"manure\":"<<manure<<",\"fed\":"<<fed<<",\"cared\":"<<cared<<",\"days\":[";
  for(int d=day;d<30;d++){
   int f=0,c=0;if(alive&&d<29){auto q=dp.choices[d][hunger][bonus];f=q.feed&&!(d==day&&fed);c=q.care&&!(d==day&&cared);}
   if(d!=day)std::cout<<",";
   std::cout<<"{\"day\":"<<d<<",\"feed\":"<<f<<",\"care\":"<<c<<",\"product\":"<<path.f[d][product[k-9]]<<",\"fertilizer\":"<<path.f[d][F]<<",\"wheat\":"<<path.f[d][W]<<",\"labor\":"<<std::setprecision(17)<<path.labor[d]<<"}";
   if(alive&&d<29){bool feeding=f||(d==day&&fed);hunger=feeding?0:hunger+1;if(hunger>=2){alive=false;continue;}bool tick=d+1-tile.placed_day>=afirst[k-9]&&(d+1-tile.placed_day-afirst[k-9])%ainterval[k-9]==0;if(tick)bonus=0;if(feeding&&(c||(d==day&&cared)))bonus=std::min(held[k-9]-1,bonus+1);}
  }
  std::cout<<"]}\n";
 }
}
