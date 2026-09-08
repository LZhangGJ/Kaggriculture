#include "../policy/proposals.hpp"
#include "../policy/public_flow_scenario.hpp"
#include <iostream>
#include <stdexcept>
#include <numeric>
using namespace triad;using namespace dp7;
#define CHECK(x) do{if(!(x))throw std::runtime_error(std::string(#x)+" line "+std::to_string(__LINE__));}while(0)
struct Fixture{
 fastkag::Simulator original;Farm own,other;PrivateState priv;Market market;std::vector<int8_t>shops;
 int day=0,hour=0;
 Fixture():original({},314159),own(original.farms()[0]),other(original.farms()[1]),priv(original.privates()[0]),market(original.market()){
  own.money=1e6;priv.shed[W]=40;priv.shed[F]=40;
 }
 View view()const{return{24*day+hour,day,hour,own,other,priv,market,shops};}
};
static void advance(fastkag::PublicFlowScenario&w,Action a){PlayerAction p;p.units={a};w.advance(p);}
static int total(const Asset&a,int i){double x=0;for(int d=0;d<30;d++)x+=a.f[d][i];return int(std::llround(x));}
void crop_calendar(){
 for(int birth:{0,4,10,18})for(int k=0;k<5;k++)for(int fertflag:{0,1}){
  int firstlen=k==W?2:k==C?2:k==M?10:first[k]+3*interval[k];
  int lastlen=k==W?4:k==C?3:k==M?12:firstlen;
  for(int len=firstlen;len<=lastlen;len++){
   if(birth+(ongoing(k)?first[k]:len)>29)continue;int finish=std::min(29,birth+len);
   Fixture f;f.day=birth;f.priv.seeds[k]=1;Settings st;st.crop_fert=fertflag;
   triad::Controller c(st);c.model.day=birth;Flow px{};for(auto&x:px){x.fill(200);x[F]=35;}
   auto predicted=c.crop(k,birth,44,finish,px);Flow flow{};fastkag::PublicFlowScenario world(f.view(),flow,0);
   advance(world,action(Op::PLANT,k));int harvest=0,usedfert=0;
   for(int d=birth;d<=finish;d++){
    auto v=world.view();auto t=v.own.tiles[44];bool water=false,fertilize=false;
    if(ongoing(k)){
     if(t.yield_units>0){harvest+=t.yield_units;advance(world,action(Op::HARVEST));}
     if(d<finish){OngoingMaintenanceDP md;md.kind=k;md.birth=birth;md.begin=d;md.mode=2;md.work=st.work_price+.10;
      for(int e=0;e<30;e++){md.price[e]=px[e][k];md.fert[e]=px[e][F];}
      bool tick=md.production(d+1);auto ch=md.first(d,t.consecutive_unwatered,std::clamp(int(t.fertilized_until_day)-d+1,0,3),t.consecutive_unwatered>=1||tick,tick&&t.fertilized_until_day<d&&fertflag);
      water=ch.water&&!t.watered_today;fertilize=ch.fertilize&&fertflag&&t.fertilized_until_day<d;
      if(d==birth&&!t.watered_today)water=true;
     }
    }else{int last=k==W?4:k==C?3:12;bool win=(d-birth)>=(last+1)/2&&(d-birth)<=last;
     water=!t.watered_today&&(t.consecutive_unwatered>=1||(win&&t.yield_units<(k==C?4:6)));
    }
    if(fertilize){advance(world,action(Op::PICKUP,F));advance(world,action(Op::FERTILIZE));usedfert++;}
    if(water)advance(world,action(Op::WATER));
    if(!ongoing(k)&&d==finish){harvest+=world.view().own.tiles[44].yield_units;advance(world,action(Op::HARVEST));}
    CHECK(!world.done());
    if(d<finish)while(world.view().day==d)advance(world,action(Op::PASS));
   }
   CHECK(harvest==total(predicted,k));CHECK(usedfert==-total(predicted,F));
  }
 }
}
void animal_calendar(){
 for(int birth:{0,5,17})for(int k=9;k<12;k++){
  Fixture f;f.day=birth;auto&t=f.own.tiles[44];t=Tile{};t.kind=TileKind::ANIMAL;t.animal=Item(k);t.placed_day=birth;
  Settings st;triad::Controller c(st);c.model.day=birth;Flow px{};for(auto&x:px){x.fill(200);x[W]=30;x[F]=50;}
  auto predicted=c.animal_path(k,birth,44,px);AnimalServiceDP dp;dp.solve(k,birth,birth,px,st.work_price);
  Flow flow{};fastkag::PublicFlowScenario world(f.view(),flow,0);int qty=0,manure=0,feeds=0;
  for(int d=birth;d<=29;d++){
   auto tile=world.view().own.tiles[44];if(!animal(tile))break;
   if(tile.yield_units>0){qty+=tile.yield_units;advance(world,action(Op::HARVEST));}
   if(tile.fertilizer_available){manure++;advance(world,action(Op::COLLECT_FERTILIZER));}
   if(d<29){auto ch=dp.first(d,tile);if(d==birth)ch.feed=ch.care=1;
    if(ch.feed){advance(world,action(Op::PICKUP,W));advance(world,action(Op::FEED));feeds++;}
    if(ch.care)advance(world,action(Op::CARE));
    while(world.view().day==d)advance(world,action(Op::PASS));
   }
  }
  CHECK(qty==total(predicted,product[k-9]));CHECK(manure==total(predicted,F));CHECK(feeds==-total(predicted,W));
 }
}
void resource_and_market(){
 Fixture f;f.priv.shed={};f.priv.shed[W]=10;f.priv.shed[F]=10;f.priv.shed[MI]=10;
 triad::Controller c;c.core.day=0;c.core.phase=3;c.core.plans.resize(1);auto&pl=c.core.plans[0];pl.a={action(Op::PICKUP,W,3),action(Op::PICKUP,F,2)};pl.target={44,44};
 f.priv.inventories[0][MI]=3;f.priv.inventory_order[0]={MI};PlayerAction out;out.units={action(Op::DROP)};
 c.settle_market(f.view(),out,false);Counts sold{};for(auto a:out.market)if(a.op==Op::SELL)sold[int(a.item)]+=a.quantity;
 CHECK(sold[MI]==13);CHECK(sold[W]==7);CHECK(sold[F]==8);CHECK(out.market.size()<=10);
 f.day=29;out.market.clear();out.units={action(Op::PASS)};c.settle_market(f.view(),out,false);sold={};for(auto a:out.market)if(a.op==Op::SELL)sold[int(a.item)]+=a.quantity;
 CHECK(sold[W]==10&&sold[F]==10&&sold[MI]==10);
}
void plant_water_chain(){
 Fixture f;dp7::Controller c;c.day=0;c.target={{44,S}};f.priv.seeds[S]=1;auto js=c.jobs(f.view());CHECK(js.size()==1);CHECK(js[0].actions.size()>=2);CHECK(js[0].actions[0].op==Op::PLANT);CHECK(js[0].actions[1].op==Op::WATER);
}
void funded_placement(){
 Fixture f;f.day=3;f.own.money=100;f.priv.shed={};f.priv.shed[CO]=1;f.priv.shed[W]=5;
 Settings st;st.repeat=0;triad::Controller c(st);c.book[34]={CO,-1,2,0,-1,true};c.plan(f.view());
 bool found=false;for(auto[p,k]:c.core.target)if(p==34&&k==CO)found=true;CHECK(found);CHECK(c.kept==1);
}
void funded_successor(){
 Fixture f;f.day=3;f.own.money=0;f.priv.shed={};f.priv.seeds[S]=1;
 auto&t=f.own.tiles[44];t=Tile{};t.kind=TileKind::PLANT;t.crop=Item(W);t.planted_day=0;t.yield_units=3;t.consecutive_unwatered=0;
 Settings st;st.rotation=0;st.repeat=0;st.preview=0;triad::Controller c(st);c.book[44]={W,0,2,4,S,true};c.plan(f.view());
 bool found=false;for(auto[p,k]:c.core.target)if(p==44&&k==S)found=true;CHECK(found);CHECK(c.core.triad_crop_age[44]==3);CHECK(c.kept==1);CHECK(c.book[44].successor==S);
}
void funded_successor_empty(){
 Fixture f;f.day=4;f.own.money=0;f.priv.shed={};f.priv.seeds[S]=1;
 Settings st;st.rotation=0;st.repeat=0;st.preview=0;triad::Controller c(st);c.book[44]={W,0,2,4,S,true};c.plan(f.view());
 bool found=false;for(auto[p,k]:c.core.target)if(p==44&&k==S)found=true;CHECK(found);CHECK(c.kept==1);CHECK(c.book[44].kind==S);
}
void exact_tour(){
 for(int n=2;n<=8;n++)for(bool ret:{false,true}){
  Route r(0,44);std::vector<int>idx(n);std::iota(idx.begin(),idx.end(),0);
  for(int i=0;i<n;i++){Job j;j.pos=(17*i+3)%50;j.actions={action(Op::WATER),action(Op::HARVEST)};j.out[W]=1;if(i%3==0)j.needs[W]=1;r.append(j);}
  int best=100000;do{std::vector<Job>js;for(int i:idx)js.push_back(r.jobs[i]);best=std::min(best,r.ordered(js,ret));}while(std::next_permutation(idx.begin(),idx.end()));
  auto out=triad_exact_tour(r,ret);CHECK(out.total(ret)==best);CHECK(out.jobs.size()==r.jobs.size());CHECK(out.needs==r.needs);
 }
}
void feature_boundary(){
 // Different hidden RNG futures and opponent private stock are NOT in View.
 // Identical public/own input therefore gives identical candidates/features.
 Fixture a,b;b.original=fastkag::Simulator({},99999);
 Settings st;st.portfolio_passes=6;st.repeat=0;triad::Controller c(st);
 auto x=generate_proposals(c,st,a.view(),true),y=generate_proposals(c,st,b.view());CHECK(x.size()==y.size());
 for(size_t i=0;i<x.size();i++){CHECK(x[i].features.x==y[i].features.x);CHECK(x[i].features.names.size()==x[i].features.x.size());
  for(const auto&name:x[i].features.names)CHECK(name!="seed"&&name!="opponent_id"&&name!="future"&&name!="opponent_private");}
}
int main(){try{crop_calendar();animal_calendar();resource_and_market();plant_water_chain();funded_placement();funded_successor();funded_successor_empty();exact_tour();feature_boundary();std::cout<<"PASS: deterministic crop/animal calendars, protected stock and same-turn sales, PLANT+WATER, funded cross-day placement + crop successor, exact tour vs brute force, public feature determinism\n";return 0;}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}}
