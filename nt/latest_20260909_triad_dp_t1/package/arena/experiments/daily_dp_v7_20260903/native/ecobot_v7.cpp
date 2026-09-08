// Exact public-opponent semantics, including its deliberately approximate
// valuations and fixed daily purchase/dispatch windows. Never our policy.
#include "ecobot_v7.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace eco7 {
namespace {
constexpr int LAST=29,LIQ=28;
constexpr int SEED[]{10,20,50,100,80},ACOST[]{300,400,500},AFIRST[]{4,8,6},INTERVAL[]{1,2,3};
constexpr int FIB[]{1,1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765};
template<class R,class X>bool has(const R&r,const X&x){return std::find(r.begin(),r.end(),x)!=r.end();}
double shape(int f,double x,double t){switch(f){case 0:return x;case 1:return x*x;case 2:return std::sqrt(std::max(0.,x));case 3:return std::log(1+std::max(0.,x));default:{double u=x/std::max(1.,t);return u+8*std::pow(std::max(0.,u-1),2);}}}
int hires(int count,int already){if(count<0||already<0||count+already>20)throw std::runtime_error("EcoBot hire range");return std::accumulate(FIB+already,FIB+already+count,0);}
Values demand(const std::vector<int8_t>&shops){
  Values d;d.fill(1);d[F]=0;constexpr int table[8][9]={{6,0,0,0,0,6,0,0,0},{6,0,0,6,0,6,0,0,0},{6,6,6,6,0,0,0,0,0},{6,0,0,6,0,0,6,0,0},{0,12,0,0,0,0,0,0,0},{6,0,6,0,0,0,6,0,0},{0,0,0,6,0,0,6,0,0},{0,0,0,0,0,0,0,12,0}};
  for(int shop:shops)if(shop>=0&&shop<8)for(int i=0;i<9;i++)d[i]+=table[shop][i];return d;
}
double shop_probability(int count,int horizon,int member_count){int n=0;for(int k=count+1;k<=8;k++)n+=k*3<=horizon;return n>0?1-std::pow(1-member_count/8.,n):0;}
int price(const Input&o,int item){return market_price(item,o.market.inventory[item]);}
int expected(const Input&o,int item,int lag,const Values&drift,double extra=0){double net=drift[item]-extra;int inv=o.market.inventory[item]-int(net*lag);return market_price(item,std::max(1,inv));}
double multiplier(const Input&o,int item,const Values&d){constexpr double t[]{400,450,200,100,300,332,122,105,200};if(d[item]>=12)return 1.6;if(d[item]>=6)return 1.3;if(10000-o.market.inventory[item]>.6*t[item])return 1.4;return 1;}
Score animal_score(const Input&o,int sp,int herd,int count,const Values&drift,const Values&d){
  int ai=sp-9,item=product(sp),lag=AFIRST[ai],interval=INTERVAL[ai],left=30-o.day-lag;double cost=ACOST[ai];if(left<=0)return {sp,item,cost,0,-1};
  int harvests=left/interval,units=harvests*interval;int p=expected(o,item,lag,drift,count*1.);double revenue=units*p*multiplier(o,item,d);
  double fert_revenue=(30-o.day)*double(price(o,F));double wd=drift[W]+double(herd+1);double feed_price=market_price(W,std::max(1,o.market.inventory[W]-int(wd*8)));
  double feed_cost=(30-o.day)*feed_price,labor=(30-o.day)*2.;double npv=revenue+fert_revenue-cost-feed_cost-labor;return {sp,item,cost,npv/std::max(1,30-o.day),npv};
}
Score fixed_score(const Input&o,int crop,const Values&drift,const Values&d,double supply=0){
  const auto&days=crop==S?STR_DAYS:TOM_DAYS;double cost=SEED[crop];int latest=crop==S?14:30-days[0]-2;if(o.day>latest)return {crop,crop,cost,0,-1};int count=0;for(int k:days)count+=o.day+k<=29;if(!count)return {crop,crop,cost,0,-1};
  double units=count*1.5;int p=expected(o,crop,days[0],drift,supply);double npv=units*p*multiplier(o,crop,d)-cost;int lifespan=std::min(30-o.day,days.back()+1);return {crop,crop,cost,npv/std::max(1,lifespan),npv};
}
Score replant_score(const Input&o,int crop,const Values&drift,const Values&d,double supply=0){
  double cost=SEED[crop];int grow=MAXAGE[crop];double units=crop==W?4:crop==C?3:6;if(crop==M&&o.day>18)return {crop,crop,cost,0,-1};int latest=crop==W?26:30-grow-1;if(o.day>latest)return {crop,crop,cost,0,-1};
  int p=expected(o,crop,grow,drift,supply);double daily=(units*p*multiplier(o,crop,d)-cost)/std::max(1,grow);return {crop,crop,cost,daily,daily*(30-o.day)};
}
double ap_cost(int item,int cows,int sheep,int geese,const Seeds&crops,const std::vector<int>&crop_order){
  auto ap=[](int i){return i==COW?3.3:i==SHEEP?3.25:i==G?4.:i==C?1.6:1.5;};double crop_ap=0;for(int crop:crop_order)crop_ap+=crops[crop]*ap(crop);
  double current=cows*3.3+sheep*3.25+geese*4.+crop_ap;current+=std::max(15,cows+sheep+geese)*1.5;int before=std::max(0,int(std::ceil(current/22.))-1),after=std::max(0,int(std::ceil((current+ap(item))/22.))-1);
  return after>before?double(hires(after,0)-hires(before,0)):0.;
}
void update_cull(const Input&o,const Values&drift,const Census&c,EvalState&s){
  if(o.day==s.cull_day)return;s.cull_day=o.day;if(o.day>=28){for(int sp:{COW,SHEEP,G})s.retained[sp]=0;return;}
  for(int sp:{COW,SHEEP,G}){int item=product(sp),lag=AFIRST[sp-9];double rate=1./INTERVAL[sp-9];int projected=std::max(1,o.market.inventory[item]-int(drift[item]*lag));double rev=market_price(item,projected)*rate+double(price(o,F));double cost=double(price(o,W))+2.;double profit=rev-cost;
    if(s.downsized[sp])continue;if(c.total(sp)<=0){s.negative[sp]=0;s.retained[sp]=999;continue;}
    if(profit<0){s.negative[sp]++;if(s.negative[sp]>=2){s.retained[sp]=std::max(1,int(std::ceil(c.total(sp)*.5)));s.downsized[sp]=true;s.negative[sp]=0;}else s.retained[sp]=999;}
    else{s.negative[sp]=0;s.retained[sp]=999;}
  }
}
int throttle(const Input&o,int item,int qty){if(o.day>=28||!(item==M||item==7||item==S||item==W))return qty;int p=price(o,item);if(p<=1)return qty;int lo=1,hi=qty,best=1;while(lo<=hi){int mid=(lo+hi)/2,next=market_price(item,o.market.inventory[item]+mid);if(double(p-next)/p<=.15){best=mid;lo=mid+1;}else hi=mid-1;}return best;}
std::pair<std::vector<Action>,double>sells(const Input&o,const FarmState&f,const Census&c,const Caps&caps){
  std::vector<Action>os;double rev=0;int due=0;for(auto&p:f.plants)due+=p.fert_due;int fert=std::max(0,o.priv.shed[F]-due-fert_due_tomorrow(f,o.day));
  if(fert>0){os.push_back(act(Op::SELL,F,fert));rev+=fert*price(o,F);}for(int i:{M,S,6,7,5,C,T})if(o.priv.shed[i]>0){int n=throttle(o,i,o.priv.shed[i]);if(n>0){os.push_back(act(Op::SELL,i,n));rev+=n*price(o,i);}}
  int ceiling=feed_thresholds(c,caps,o.day).second;if(o.priv.shed[W]>ceiling&&o.day>0){int n=throttle(o,W,o.priv.shed[W]-ceiling);if(n>0){os.push_back(act(Op::SELL,W,n));rev+=n*price(o,W);}}return {os,rev};
}
std::vector<Action>settle(const Input&o,const std::vector<Action>&os,double reserve){
  std::vector<Action>out;double cash=o.farm.money;int hired=o.farm.hires_today,quads=std::popcount(unsigned(o.farm.unlocked_mask));
  for(auto a:os){if(a.op==Op::SELL){cash+=a.quantity*price(o,int(a.item));out.push_back(a);continue;}
    if(a.op==Op::HIRE){double cost=hires(1,hired);if(cash-cost<0)continue;cash-=cost;hired++;out.push_back(a);continue;}
    if(a.op==Op::BUY_LAND){if(quads<1||quads>3)throw std::runtime_error("EcoBot land cost");double cost=quads==1?1000:quads==2?2000:4000;if(cash-cost<reserve)continue;cash-=cost;out.push_back(a);continue;}
    int item=int(a.item);double unit=a.op==Op::BUY_PRODUCT?price(o,item):a.op==Op::BUY_SEED?SEED[item]:ACOST[item-9];int n=std::min(a.quantity,int(std::floor(std::max(0.,cash-reserve)/unit)));if(n<=0)continue;cash-=n*unit;a.quantity=n;out.push_back(a);
  }if(cash<-.000001)throw std::runtime_error("EcoBot negative settlement");return out;
}
std::vector<Unit>get_units(const Input&o){std::vector<Unit>us{{{o.farm.farmer.x,o.farm.farmer.y},o.priv.inventories.at(0)}};for(int i=0;i<int(o.farm.hands.size());i++){auto p=o.farm.hands[i];us.push_back({{p.x,p.y},o.priv.inventories.at(i+1)});}return us;}
}
int market_price(int item,int inv){
  constexpr double base[]{25,35,60,120,250,50,160,200,100},t[]{400,450,200,100,300,332,122,105,200},below[]{.8,1,.4,.7,.2,.4,.6,.2,.4},above[]{.2,.7,.6,1.6,3.6,.2,1.6,3.2,.4};constexpr int bf[]{2,4,4,2,3,4,2,3,0},af[]{3,2,2,0,1,3,0,1,0};
  if(inv==10000)return int(std::nearbyint(base[item]));bool low=inv<10000;int func=low?bf[item]:af[item];double target=low?below[item]:above[item],x=low?10000-inv:inv-10000,denom=shape(func,t[item],t[item]),amp=denom>0?target*base[item]/denom:0;
  return std::max(1,int(std::nearbyint(base[item]+(low?1:-1)*amp*shape(func,x,t[item]))));
}
Decision evaluate(const Input&o,EvalState&s){
  int day=o.day,hour=o.hour,quads=std::popcount(unsigned(o.farm.unlocked_mask)),mask=o.farm.unlocked_mask;auto units=get_units(o);auto f=parse(o.farm,day);auto c=census(f,o.priv.shed,units);Decision out;
  out.pastures=needed_pastures(f,c);out.coops=needed_coops(f,c,out.pastures);auto d=demand(o.shops);
  if(day!=s.prev_day){if(s.prev_day>=0)for(int i=0;i<9;i++)s.observed[i]=double(s.prev_inv[i]-o.market.inventory[i]);s.prev_inv=o.market.inventory;s.prev_day=day;}
  Values drift{};for(int i=0;i<9;i++)drift[i]=std::max(d[i],.6*s.observed[i]);update_cull(o,drift,c,s);out.caps=s.retained;
  double reserve=4<=day&&day<28?200.:0.;std::vector<Action>os;
  auto finish=[&](){out.orders=settle(o,os,reserve);if(out.orders.size()>10)out.orders.resize(10);return out;};
  if(day==0&&hour==0&&o.priv.shed[SHEEP]==0&&f.animals.empty()){
    os={act(Op::HIRE),act(Op::HIRE),act(Op::HIRE),act(Op::HIRE),act(Op::BUY_ANIMAL,SHEEP,1),act(Op::BUY_ANIMAL,COW,3),act(Op::BUY_SEED,M,10),act(Op::BUY_SEED,W,8),act(Op::BUY_PRODUCT,W,4)};out.hints.grazers=4;out.hints.crops[M]=10;out.hints.crops[W]=8;return finish();
  }
  auto[so,revenue]=sells(o,f,c,out.caps);os=so;double budget=o.farm.money+revenue;int kept=kept_feedable(c,out.caps);
  if(kept>0&&day<28){int floor=feed_thresholds(c,out.caps,day).first,available=o.priv.shed[W];for(auto&u:units)available+=u.inv[W];if(available<floor){int want=floor-available;os.push_back(act(Op::BUY_PRODUCT,W,want));budget-=double(price(o,W)*want);}}
  int cows=c.total(COW),sheep=c.total(SHEEP),geese=c.total(G);Seeds crops{};std::vector<int>crop_order;
  int blocked=0,cluster=0;for(auto p:BLOCKED)blocked+=(mask&(1<<quad(p)))!=0;for(auto p:CLUSTER)cluster+=(mask&(1<<quad(p)))!=0;
  int pasture_cap=day>16?cows+sheep:cluster,coop_cap=quads*2,tile_capacity=quads*25-blocked;
  double milk_p=shop_probability(int(o.shops.size()),16,3),wool_p=shop_probability(int(o.shops.size()),16,1),egg_p=shop_probability(int(o.shops.size()),16,2);
  for(int iteration=0;iteration<60;iteration++){
    int total=cows+sheep+geese,grazers=cows+sheep;std::vector<Score>candidates;
    auto add=[&](Score score){double ap=ap_cost(score.kind,cows,sheep,geese,crops,crop_order);if(ap>0){score.daily-=ap;score.npv-=ap*std::max(1,30-day);}if(score.npv>0&&score.daily>0)candidates.push_back(score);};
    if(grazers<pasture_cap&&total<18){
      if(!s.downsized[COW]){int maxc=d[6]<6?(milk_p>=.5?4:2):d[6]<12?6:std::min(14,2+int(std::floor(d[6]/6.))*4);if(cows<out.caps[COW]&&cows<maxc)add(animal_score(o,COW,total,cows,drift,d));}
      if(!s.downsized[SHEEP]){int yarn=std::count(o.shops.begin(),o.shops.end(),int8_t(YARN));int base=yarn==0?(wool_p>=.5?4:2):yarn==1?4:std::min(8,yarn*4);bool expand=animal_score(o,COW,total,cows,drift,d).npv<=0&&(yarn>0||d[7]>=6);int maxs=expand?18:base;if(sheep<out.caps[SHEEP]&&sheep<maxs)add(animal_score(o,SHEEP,total,sheep,drift,d));}
    }
    int eggshops=0;for(int sh:o.shops)eggshops+=sh==BAKERY||sh==BRUNCH;
    if((eggshops>0||egg_p>=.5)&&geese<coop_cap&&total<18&&!s.downsized[G]){int floor=eggshops==0&&egg_p>=.5?2:0;int maxg=std::max(int(d[5]),floor);if(geese<out.caps[G]&&geese<maxg)add(animal_score(o,G,total,geese,drift,d));}
    int capacity=std::max(0,tile_capacity-pasture_cap-coop_cap-kept);int commercial=std::accumulate(crops.begin(),crops.end(),0);
    if(commercial<capacity){if(day<=14)add(fixed_score(o,S,drift,d,crops[S]*.46));if(day<=18)add(replant_score(o,M,drift,d,crops[M]*1.));if(d[T]>=6&&crops[T]<8)add(fixed_score(o,T,drift,d,crops[T]*.5));if(d[C]>=6&&crops[C]<8)add(replant_score(o,C,drift,d,crops[C]*1.));if(day<=26)add(replant_score(o,W,drift,d,crops[W]*1.));}
    if(candidates.empty())break;auto best=*std::max_element(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.daily<b.daily;});if(best.daily<=0)break;
    if(best.kind==COW)cows++;else if(best.kind==SHEEP)sheep++;else if(best.kind==G)geese++;else{if(!crops[best.kind])crop_order.push_back(best.kind);crops[best.kind]++;}
  }
  int total_needed=cows+sheep+geese+std::accumulate(crops.begin(),crops.end(),0)+kept,target_quads=quads;
  if(quads<3){double next=quads==1?1000:2000;int empty=0;for(auto p:f.empty)empty+=!has(BLOCKED,p);double usage=1.-double(empty)/std::max(1,tile_capacity);bool constrained=(total_needed>=tile_capacity&&usage>=.8)||empty<=2||cows+sheep>cluster||geese>quads*2;if(budget>=next+reserve&&constrained)target_quads=quads+1;}
  out.hints={cows+sheep,geese,crops};
  if(hour==0||hour==1){auto tasks=catalog(f,o.priv.shed,o.priv.seeds,out.pastures,out.coops,units,day,hour,out.hints,mask,out.caps);int count=required_hands(tasks,ready_premium(f,day),units,o.priv.shed,std::min(13,quads<3?11:13),day);os.insert(os.begin(),count,act(Op::HIRE));}
  if(hour==21){
    auto res=reserved(f,mask,out.hints,day);std::set<Pos>need(out.pastures.begin(),out.pastures.end());need.insert(out.coops.begin(),out.coops.end());
    int bought=0,grazers_bought=0;std::vector<Action>animals,land;
    for(auto[sp,target]:std::vector<std::pair<int,int>>{{COW,cows},{SHEEP,sheep},{G,geese}}){if(s.downsized[sp]||(sp==G&&quads<2))continue;int effective=std::min(target,out.caps[sp]);if(sp==G)effective=std::min(effective,coop_capacity(f)+int(res.size()));int deficit=effective-c.total(sp);if(deficit<=0)continue;int remaining=std::max(0,18-(c.all()+bought));int maxbuy=sp==G?remaining:std::min(remaining,std::max(0,18-(c.grazers()+grazers_bought)));int want=std::min(deficit,maxbuy);if(want>0){animals.push_back(act(Op::BUY_ANIMAL,sp,want));bought+=want;if(sp!=G)grazers_bought+=want;}}
    bool want_land=target_quads>quads;if((!(mask&2)&&want_land)||(!(mask&4)&&(mask&2)&&want_land)||(!(mask&8)&&(mask&4)&&target_quads>=4))land.push_back(act(Op::BUY_LAND));
    int empty=0,rsv=0;for(auto p:f.empty){if(!res.count(p)&&!need.count(p)&&!has(BLOCKED,p))empty++;if(res.count(p)&&!need.count(p))rsv++;}
    int wheat_committed=0;auto committed=[&](int crop){int n=o.priv.seeds[crop];for(auto&p:f.plants)n+=p.crop==crop;return n;};auto seed_buy=[&](int crop,int want){if(want<=0)return 0;os.push_back(act(Op::BUY_SEED,crop,want));return want;};
    if(4<=day&&day<=26){int current=committed(W),capacity=empty+rsv;if(current<kept&&capacity>0){int n=seed_buy(W,std::min(kept-current,capacity));int from_open=std::min(n,empty);empty-=from_open;rsv-=n-from_open;wheat_committed+=n;}}
    int straw_target=crops[S],total_straw=committed(S);if(day<=14&&total_straw<straw_target&&empty>0){int n=seed_buy(S,std::min(straw_target-total_straw,empty));empty=std::max(0,empty-n);}
    int straw_done=committed(S);for(int crop:crop_order){int target=crops[crop];if(crop==S||crop==W||target<=0)continue;
      if(straw_done<straw_target){double straw_npv=fixed_score(o,S,drift,d,straw_done*.46).npv;double crop_npv=ongoing(crop)?fixed_score(o,crop,drift,d).npv:replant_score(o,crop,drift,d).npv;if(straw_npv>crop_npv)continue;}
      int lag=ongoing(crop)?FIRST[crop]:MAXAGE[crop];if(day>30-lag-1)continue;int freeing=crop==C?maturing_tomorrow(f,crop):0,want=target+freeing-committed(crop);if(want>0){int n=seed_buy(crop,want);empty=std::max(0,empty-n);}
    }
    if(1<=day&&day<=26&&(day<26||hour<12)){int target=kept+crops[W],current=committed(W)+wheat_committed,freeing=maturing_tomorrow(f,W),already=o.priv.seeds[W]+wheat_committed;int room=target-current,capacity=empty+rsv,newplant=std::min(capacity,std::max(0,room)),replant=std::max(0,std::min(freeing,room+freeing));seed_buy(W,std::max(0,newplant+replant-already));}
    os.insert(os.end(),land.begin(),land.end());os.insert(os.end(),animals.begin(),animals.end());
  }return finish();
}
fastkag::PlayerAction Controller::act(const Input&o){
  if(o.step==0||(o.day==0&&o.hour==0))*this=Controller{};auto units=get_units(o);decision=evaluate(o,state);fastkag::PlayerAction out;out.market=decision.orders;
  if(o.hour==0){out.units.resize(units.size());return out;}
  if(plan_day!=o.day){auto f=parse(o.farm,o.day);int pending=std::max(0,int(o.farm.hires_today)-int(o.farm.hands.size()));for(auto&a:decision.orders)pending+=a.op==Op::HIRE;queues=day_plan(units,f,o.priv.shed,o.priv.seeds,o.day,o.hour,decision.pastures,decision.coops,decision.hints,o.farm.unlocked_mask,decision.caps,pending);plan_day=o.day;}
  for(int i=0;i<int(units.size());i++){if(i>=int(queues.size()))throw std::runtime_error("EcoBot unit has no route");auto&q=queues[i];out.units.push_back(q.empty()?Action{}:q[0]);if(!q.empty())q.erase(q.begin());}return out;
}
}
