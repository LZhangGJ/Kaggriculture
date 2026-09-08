#include "hybrid.hpp"
#include "inventory_dp.hpp"
#include "animal_service_dp.hpp"
#include "public_tape.hpp"
#include "reference_pick.hpp"
#include "executor/investment_candidates.hpp"
#include <iostream>
#include <cassert>
#include <functional>
using namespace fastkag;using namespace dp7;
int main(){
 Config ec;Simulator env(ec,123);auto f=env.farms()[0],r=env.farms()[1];auto pr=env.privates()[0];auto m=env.market();auto shops=env.shops();
 for(auto&t:f.tiles)if(t.kind!=TileKind::LOCKED)t=Tile{};
 f.money=0;f.farmer={4,4};Tile wheat;wheat.kind=TileKind::PLANT;wheat.crop=Item::WHEAT;wheat.planted_day=24;wheat.yield_units=3;wheat.watered_today=false;
 f.tiles[44]=wheat;View late{28*24,28,0,f,r,pr,m,shops};competitive::Planner p;p.plan(late);
 bool found=false;for(auto[pos,k]:p.core.target)if(pos==44&&k==W)found=true;assert(found);
 auto jobs=p.core.jobs(late);bool harvest=false;for(auto&j:jobs)for(auto&a:j.actions)harvest|=a.op==Op::HARVEST;assert(harvest);
 std::cout<<"PASS mature harvest retained without replant\n";
 f.tiles[44]=Tile{};pr.shed[CO]=1;View opening{0,0,0,f,r,pr,m,shops};competitive::Planner q;q.plan(opening);
 int ncow=0;for(auto[pos,k]:q.core.target)ncow+=k==CO;assert(ncow>=1);
 std::cout<<"PASS purchased animal retains placement intention\n";
 competitive::Planner z;z.day=0;for(auto&d:z.shadow)for(auto&x:d)x=100;
 auto cy=z.crop_cycle(W,0,44);assert(cy.f[4][W]==4);
 std::cout<<"PASS wheat unboosted life-cycle output\n";
 z.cfg.finite_fertilize=1;z.core.p.finite_fertilizer=true;z.core.p.crop_harvest_age[W]=3;for(auto&d:z.shadow){d[W]=50;d[F]=10;}
 cy=z.crop_cycle(W,0,44);assert(cy.f[3][W]==5&&cy.f[2][F]==-1);
 std::cout<<"PASS fertilized three-day wheat output and input cost\n";
 z.core.p.crop_harvest_age[M]=8;for(auto&d:z.shadow){d[M]=250;d[F]=10;}
 cy=z.crop_cycle(M,0,44);assert(cy.f[8][M]==6&&cy.f[6][F]==-1);
 std::cout<<"PASS fertilized eight-day melon output and input cost\n";

 // Repeated shops remain repeated and future shops draw from all eight types.
 shops={7,7};View repeated{144,6,0,f,r,pr,m,shops};competitive::Planner check;check.day=6;check.demand(repeated);
 assert(check.dem[6][WO]==25.);assert(check.dem[9][WO]==26.5);assert(check.dem[9][W]==4.75);
 std::cout<<"PASS repeated shops and IID future-demand expectation\n";
 // Mature finite crops may choose their next crop without erasing this harvest.
 f.money=3000;f.tiles[44]=wheat;f.tiles[44].planted_day=0;pr.seeds[C]=1;
 dp7::Controller rotation;rotation.day=4;rotation.target={{44,C}};View renew{96,4,0,f,r,pr,m,shops};
 auto renewjobs=rotation.jobs(renew);bool ordered=false;
 for(auto&j:renewjobs)if(j.pos==44){int harvestAt=-1,plantAt=-1,waterAfter=-1;for(int i=0;i<int(j.actions.size());i++){auto&a=j.actions[i];if(a.op==Op::HARVEST)harvestAt=i;if(a.op==Op::PLANT&&a.item==Item::CARROT)plantAt=i;if(a.op==Op::WATER&&plantAt>=0)waterAfter=i;}ordered=harvestAt>=0&&plantAt>harvestAt&&waterAfter>plantAt;}assert(ordered);
 std::cout<<"PASS finite renewal keeps old HARVEST before new PLANT and WATER\n";

 for(int item:{S,M,MI,WO})for(int stock=0;stock<12;stock++)for(int inv:{9800,10000,10400,12000}){
  std::vector<competitive::InventoryDP::Step> steps{{2,3,4},{1,2,8}};competitive::InventoryDP dp(item,steps,.8);
  auto got=dp.solve(0,stock,inv);double best=-1e100;
  for(int a=0;a<=stock;a++){auto[x,i1]=dp.sell(inv,a);auto[y,i2]=dp.sell(i1,3);
   for(int b=0;b<=stock-a+2;b++){auto[z,i3]=dp.sell(i2-4,b);auto[w,i4]=dp.sell(i3,2);double v=x-.8*y+z-.8*w+dp.sell(i4-8,stock-a+3-b).first;best=std::max(best,v);}}
  assert(std::abs(best-got.value)<1e-8);
 }
 std::cout<<"PASS inventory Bellman DP versus 192 exhaustive two-stage cases\n";
 competitive::InventoryDP floor(M,{},1.);auto sold=floor.sell(12000,10);assert(sold.first==10&&sold.second==12000);
 std::cout<<"PASS one-cash sales do not raise market inventory\n";
 competitive::Flow px{};for(auto&x:px){x[W]=1;x[WO]=1000;}
 competitive::AnimalServiceDP adp;adp.solve(SH,20,27,px,0);Tile sheep;sheep.animal=Item::SHEEP;sheep.kind=TileKind::ANIMAL;sheep.placed_day=20;sheep.consecutive_unfed=1;
 auto take=adp.first(27,sheep);assert(take.feed==1&&take.care==1);
 sheep.consecutive_unfed=0;sheep.pending_care_bonus=1;adp=competitive::AnimalServiceDP{};adp.solve(SH,20,28,px,0);take=adp.first(28,sheep);assert(take.feed==1&&take.care==0);
 std::cout<<"PASS care is banked after, not before, the next production tick\n";
 int service_cases=0;
 for(int kind:{G,CO,SH})for(int regime=0;regime<3;regime++){
  competitive::Flow oraclepx{};for(int d=26;d<30;d++){oraclepx[d][W]=regime?80:1;oraclepx[d][F]=regime==2?0:4;oraclepx[d][product[kind-9]]=25+(d%2)*80;}
  competitive::AnimalServiceDP test;test.solve(kind,18,26,oraclepx,3);
  std::function<double(int,int,int)> brute=[&](int d,int h,int p){if(d>=29)return 0.;double best=-1e100;int j=kind-9,age=d+1-18;bool tick=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
   for(int feed=0;feed<=1;feed++){int nextH=feed?0:h+1;if(nextH>=2){best=std::max(best,0.);continue;}for(int care=0;care<=feed;care++){if(care&&p>=held[j]-1&&!tick)continue;int qty=tick?1+(feed?p:0):0,np=std::min(held[j]-1,(tick?0:p)+care);double val=-feed*(oraclepx[d][W]+3)-care*3+qty*oraclepx[d+1][product[j]]+std::max(0.,oraclepx[d+1][F]-3)+brute(d+1,nextH,np);best=std::max(best,val);}}
   return best;};
  for(int h=0;h<2;h++)for(int p=0;p<held[kind-9];p++){assert(std::abs(test.value[26][h][p]-brute(26,h,p))<1e-8);service_cases++;}
 }
 std::cout<<"PASS animal service DP versus "<<service_cases<<" exhaustive three-day state cases\n";
 for(auto&x:px){x[W]=1000;x[WO]=1;}adp=competitive::AnimalServiceDP{};adp.solve(SH,20,28,px,0);take=adp.first(28,sheep);assert(take.feed==0&&take.care==0);
 dp7::Controller retire;retire.day=28;retire.target={{44,SH}};retire.animal_service_day=28;retire.service_feed.fill(0);retire.service_care.fill(0);sheep.consecutive_unfed=1;sheep.yield_units=1;f.tiles[44]=sheep;View retiring{672,28,0,f,r,pr,m,shops};bool hasharvest=false,hasfeed=false;for(auto&j:retire.jobs(retiring))for(auto&a:j.actions){hasharvest|=a.op==Op::HARVEST;hasfeed|=a.op==Op::FEED;}assert(hasharvest&&!hasfeed);
 std::cout<<"PASS retirement preserves even a one-unit final harvest\n";
 Simulator netenv;auto pp=netenv.privates()[0];pp.shed[M]=3;netenv.replace_private_for_public_counterfactual(0,pp);pp=netenv.privates()[1];pp.shed[M]=2;netenv.replace_private_for_public_counterfactual(1,pp);
 auto makeview=[&](){return View{netenv.step_count(),netenv.day(),netenv.hour(),netenv.farms()[0],netenv.farms()[1],netenv.privates()[0],netenv.market(),netenv.shops()};};
 competitive::PublicTape tape;PlayerAction oa,ea;oa.market={{Op::SELL,Item::MELON,2}};ea.market={{Op::SELL,Item::MELON,1}};auto before=makeview();tape.observe(before);tape.remember(before,oa);netenv.step({oa,ea});tape.observe(makeview());assert(tape.today[M]==1);
 std::cout<<"PASS public tape subtracts own fills and adds known town consumption\n";
 Simulator refenv;dp7::Controller ref(frozen_params()); // verified through live official traces separately
 for(int step=0;step<719;step++){View v{step,step/24,step%24,refenv.farms()[0],refenv.farms()[1],refenv.privates()[0],refenv.market(),refenv.shops()};
  if(step%24==0){auto fast=competitive::reference_pick(ref,v);auto menu=dp7branch::generate(ref,v);int chosen=0;if(v.day<29){auto&r=switch_plans[0][v.day];for(int n=0;n<int(menu.size());n++)if(menu[n].family==r.family&&menu[n].kind==r.kind&&menu[n].amount==r.amount){chosen=n;break;}}
   auto slow=menu[chosen].controller;if(fast.target!=slow.target||fast.planned_land!=slow.planned_land){std::cerr<<"MISMATCH day "<<v.day<<" recipe "<<(v.day<29?switch_plans[0][v.day].family:"KEEP")<<" selected "<<menu[chosen].family<<"\n";for(auto [p,k]:fast.target)for(auto[q,j]:slow.target)if(p==q&&k!=j)std::cerr<<p<<":"<<k<<" vs "<<j<<"\n";return 2;}assert(fast.queue.size()==slow.queue.size());for(size_t i=0;i<fast.queue.size();i++)assert(fast.queue[i].op==slow.queue[i].op&&fast.queue[i].item==slow.queue[i].item&&fast.queue[i].quantity==slow.queue[i].quantity);ref=std::move(slow);
  }refenv.step({ref.act(v),PlayerAction{}});
 }
 std::cout<<"PASS canonical reference selector matches full candidate selection through 719 steps\n";
}
