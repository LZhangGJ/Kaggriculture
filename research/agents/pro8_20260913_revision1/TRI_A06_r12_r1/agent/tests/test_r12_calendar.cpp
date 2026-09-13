#include "policy/search.hpp"
#include <cassert>
#include <iostream>
#include <random>
using namespace triad;using namespace dp7;
struct Fixture {Farm own,opp;PrivateState pr;Market market;std::vector<int8_t>shops{7};
 Fixture(){own.tiles.resize(100);opp.tiles.resize(100);own.money=1200;opp.money=1500;own.farmer=opp.farmer={4,4};own.unlocked_mask=opp.unlocked_mask=1;pr.inventories.resize(1);pr.inventory_order.resize(1);for(int i=0;i<9;i++){market.inventory[i]=9900;market.prices[i]=price(i,9900);}for(int pos=0;pos<100;pos++)if(quad(pos)!=0)own.tiles[pos].kind=opp.tiles[pos].kind=TileKind::LOCKED;}
 View view(int day=0){return {day*24,day,0,own,opp,pr,market,shops};}
};
int main(){
 std::mt19937 gen(120120);long long labels=0;
 for(int n=0;n<2000;n++){
  int begin=gen()%25,keep=gen()%8;std::vector<r12::Label>x;
  for(int i=0;i<8;i++)x.push_back({begin+int(gen()%(30-begin)),double(int(gen()%40)-20)});
  auto got=r12::solve(x,begin,keep);int expected=keep;
  for(int i=0;i<8;i++)if(x[i].value>x[expected].value+1e-8||(x[i].value==x[expected].value&&expected!=keep&&(x[i].finish<x[expected].finish||(x[i].finish==x[expected].finish&&i<expected))))expected=i;
  assert(got.index==expected);labels+=x.size();
 }
 Fixture f;auto o=f.view();Settings s;s.repeat=0;s.rotation=0;s.competition=2;s.a06_r12_calendar=1;triad::Controller c(s);c.model.day=0;c.model.demand(o);c.model.public_rival(o);Asset base;
 // Incumbent strawberry revenue and maintenance costs stay in EVERY label.
 for(int d=0;d<30;d++){base.f[d][S]=2+(d%4==0?9:0);base.f[d][W]=-3;base.labor[d]=22+(d%3==0?18:0);}
 base.fixed[0]=-700;c.model.value(o,base,&c.prices);auto dp=c.rotations_dp(12,c.prices,0);
 int checked=0;
 for(int k:{W,C,M})for(int paid=0;paid<2;paid++)for(int labor=0;labor<2;labor++){
  auto legacy=c.choose_crop(k,0,12,c.prices,dp);auto got=c.r12_select(o,c.model,base,k,0,12,c.prices,nullptr,legacy,paid,labor);
  auto val=[&](const Asset&a,int len){auto z=a;if(paid)z.fixed[0]+=z.first_cost;if(labor)z.labor[0]=0;auto all=base;add(all,z);assert(all.f[3][S]==base.f[3][S]);double v=c.model.value(o,all)-s.land_rent*len;assert(std::abs(r12::decompose(c.model,o,all).score-c.model.value(o,all))<1e-7);return v;};
  double best=val(legacy.a,legacy.length);int lo=k==M?10:2,hi=k==M?12:k==W?4:3;
  for(int len=lo;len<=hi;len++){auto a=c.crop(k,0,12,len,c.prices);best=std::max(best,val(a,len));}
  assert(std::abs(val(got.a,got.length)-best)<1e-7);checked++;
 }
 // OFF retains exact legacy outputs; no sunk input is mutated in the physical
 // asset returned to the real compiler, and no hypothetical sale pays a HIRE.
 auto legacy=c.choose_crop(M,0,12,c.prices,dp);c.s.a06_r12_calendar=0;
 auto off=c.choose_crop(M,0,12,c.prices,dp,&o,&base,&c.model,true,true);
 assert(legacy.length==off.length&&legacy.a.fixed==off.a.fixed&&legacy.a.f==off.a.f&&legacy.a.labor==off.a.labor);
 c.s.a06_r12_calendar=1;auto paid=c.choose_crop(M,0,12,c.prices,dp,&o,&base,&c.model,true,true);assert(paid.a.first_cost==seed_price[M]);
 // Both execution origins solve exactly the same cash-labelled calendar.
 auto pred=c;pred.call_origin=triad::Controller::CallOrigin::PublicPrediction;
 auto a=c.choose_crop(M,0,12,c.prices,dp,&o,&base,&c.model);auto b=pred.choose_crop(M,0,12,c.prices,dp,&o,&base,&pred.model);
 assert(a.length==b.length&&a.a.f==b.a.f&&a.a.fixed==b.a.fixed&&c.calendar12.last==pred.calendar12.last);
 // A paid successor due now is never delayed by calendar reconciliation.
 auto&t=f.own.tiles[12];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=-2;t.yield_units=5;
 c.book[12]={W,-2,0,2,S,true};c.core.triad_crop_age[12]=2;c.paths[12]=c.crop(W,-2,12,0,c.prices,&t);c.portfolio=base;add(c.portfolio,c.paths[12]);f.pr.seeds[S]=1;std::vector<int>free{12};auto before=c.portfolio;c.r12_reconcile_calendar(o,f.own,free);assert(c.core.triad_crop_age[12]==2&&c.portfolio.f==before.f);
 std::cout<<"PASS stopping DAG 2000 cases / "<<labels<<" labels; portfolio calendar exhaustive "<<checked<<"; decomposition, sunk inputs, OFF, live/roll equality and paid successor protection\n";
}
