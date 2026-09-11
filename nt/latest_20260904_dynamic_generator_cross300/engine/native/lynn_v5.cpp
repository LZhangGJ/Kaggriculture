#include "lynn_v5.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <numeric>
#include <tuple>

namespace lynn5 {
using namespace fastkag;
namespace {
constexpr int MI=6,WO=7,COW=10,SHEEP=11;
constexpr int purchases[]{88,150,169,176,313},quantities[]{1,2,1,1,1};
const std::array<std::vector<int>,5> steps{{{88,92,95},{150,152,153,156},{169,175,177},{176,180,183},{313,329,333}}};
int source_animal(int b){return b==4?SHEEP:COW;}
using Stock=std::array<int,12>;
int sum(const Stock& s){return std::accumulate(s.begin(),s.end(),0);}
int inv(const dp7::View&v,int i){return v.market.inventory[i]?v.market.inventory[i]:10000;}
Position pos(const Farm& f,int u){return u==0?f.farmer:f.hands.at(u-1);}
bool cross(Position p){return (p.x==4||p.x==5)&&(p.y==4||p.y==5);}
bool empty_target(const Farm&f){return f.tiles.at(35).kind==TileKind::PASTURE;}
bool same(const Action&a,const Action&b){return a.op==b.op&&a.item==b.item&&a.quantity==b.quantity;}
bool same_market(const PlayerAction&a,const PlayerAction&b){return a.market.size()==b.market.size()&&std::equal(a.market.begin(),a.market.end(),b.market.begin(),same);}
void align(PlayerAction&a,const dp7::View&v){a.units.resize(1+v.own.hands.size());}
double shape(int fn,double value,double scale){
  value=std::max(0.,value);
  switch(fn){case 0:return value;case 1:return value*value;case 2:return std::sqrt(value);case 3:return std::log1p(value);default:{double r=value/scale;double h=std::max(0.,r-1.);return r+8.*h*h;}}
}
int price(int item,int inventory){
  static constexpr double base[]{25,35,60,120,250,50,160,200,100};
  static constexpr double scale[]{400,450,200,100,300,332,122,105,200};
  static constexpr int below[]{2,4,4,2,3,4,2,3,0},above[]{3,2,2,0,1,3,0,1,0};
  static constexpr double bt[]{.8,1.,.4,.7,.2,.4,.6,.2,.4},at[]{.2,.7,.6,1.6,3.6,.2,1.6,3.2,.4};
  bool low=inventory<10000;int fn=low?below[item]:above[item];
  double amplitude=(low?bt[item]:at[item])*base[item]/shape(fn,scale[item],scale[item]);
  double delta=amplitude*shape(fn,std::abs(inventory-10000),scale[item]);
  return std::max(1,int(std::nearbyint(low?base[item]+delta:base[item]-delta)));
}
// Preserve the opponent's quote approximation: inventory increases even at $1.
int revenue(int item,int inventory,int quantity){int value=0;for(int q=0;q<quantity;q++)value+=price(item,inventory+q);return value;}
Stock project(const dp7::View&v,const PlayerAction&a){
  Stock s=v.priv.shed;
  for(size_t u=0;u<a.units.size()&&u<=v.own.hands.size();u++){
    if(!cross(pos(v.own,u)))continue;
    const auto c=a.units[u];int i=int(c.item),q=std::max(1,c.quantity);
    if(c.op==Op::PICKUP&&i>=0)s[i]-=std::min(q,s[i]);
    else if(c.op==Op::DROP&&u<v.priv.inventories.size()){
      // Python dictionary insertion order, NOT product enum order.
      for(int k:v.priv.inventory_order.at(u))s[k]+=std::min(v.priv.inventories[u][k],std::max(0,100-sum(s)));
    }else if(c.op==Op::PLACE&&i>=0){
      int held=u<v.priv.inventories.size()?v.priv.inventories[u][i]:0;
      s[i]+=std::min({q,held,std::max(0,100-sum(s))});
    }
  }return s;
}
PlayerAction fund(const dp7::View&v,PlayerAction a){
  double cash=v.own.money;int hires=std::max(0,int(v.own.hires_today));
  int unlocked=std::popcount(unsigned(v.own.unlocked_mask));Stock s=project(v,a);int load=sum(s);
  std::array<int,9> inventory;for(int i=0;i<9;i++)inventory[i]=inv(v,i);
  std::vector<Action> out;
  for(size_t n=0;n<a.market.size()&&n<10;n++){
    auto x=a.market[n];int i=int(x.item),request=std::max(0,x.quantity);
    if(x.op==Op::SELL&&i>=0&&i<9){int q=std::min(request,s[i]);cash+=revenue(i,inventory[i],q);inventory[i]+=q;s[i]-=q;load-=q;if(q){x.quantity=q;out.push_back(x);}continue;}
    if(x.op==Op::HIRE){double aa=1,b=1;for(int h=0;h<hires;h++){double t=aa;aa=b;b+=t;}if(cash>=aa){cash-=aa;hires++;out.push_back(x);}continue;}
    if(x.op==Op::BUY_LAND){int k=std::max(0,unlocked-1);constexpr int costs[]{1000,2000,4000};if(k<3&&cash>=costs[k]){cash-=costs[k];unlocked++;out.push_back(x);}continue;}
    if(x.op!=Op::BUY_SEED&&x.op!=Op::BUY_ANIMAL&&x.op!=Op::BUY_PRODUCT){out.push_back(x);continue;}
    int filled=0;
    for(int q=0;q<request;q++){
      bool storage=x.op!=Op::BUY_SEED;double cost=1e9;
      constexpr int seeds[]{10,20,50,100,80},animals[]{300,400,500};
      if(x.op==Op::BUY_SEED&&i>=0&&i<5)cost=seeds[i];
      else if(x.op==Op::BUY_ANIMAL&&i>=9&&i<12)cost=animals[i-9];
      else if(x.op==Op::BUY_PRODUCT&&i>=0&&i<9)cost=price(i,inventory[i]-1);
      if(cash<cost||(storage&&load>=100))break;
      cash-=cost;filled++;if(storage)load++;if(x.op==Op::BUY_PRODUCT)inventory[i]--;
      // Deliberately do not add purchased items to s: source only updates load.
    }
    if(filled){x.quantity=filled;out.push_back(x);}
  }a.market=std::move(out);return a;
}
std::pair<double,double> pressure(const dp7::View&v){
  int wool=0,dairy=0;
  for(int shop:v.shops){if(shop==7)wool+=2;if(shop==3||shop==5||shop==6)dairy++;}
  return {wool+v.market.prices[WO]/200.,dairy+v.market.prices[MI]/160.};
}
void assign(const dp7::View&v,int b,State&s){
  if(s.assignments[b]>=0)return;
  auto [wool,dairy]=pressure(v);int cows=0,sheep=0;
  for(auto&t:v.opponent.tiles){cows+=t.animal==Item::COW;sheep+=t.animal==Item::SHEEP;}
  double balance=double(cows-sheep)/std::max(1,cows+sheep),signal=wool-dairy+balance;
  bool mature=b!=0||v.shops.size()>=2||std::find(v.shops.begin(),v.shops.end(),7)!=v.shops.end();
  int source=source_animal(b),target=source,q=quantities[b];bool deferred=false,reused=false;
  if((b==2||b==3)&&s.window>=0){reused=true;if(s.window==SHEEP&&s.cow_to_sheep+q<=3){target=SHEEP;s.cow_to_sheep+=q;}}
  else if(source==COW&&signal>=1.&&mature&&s.cow_to_sheep+q<=3){target=SHEEP;s.cow_to_sheep+=q;s.deferred_bundle=-1;s.deferred_shops.clear();}
  else if(source==COW&&signal>=1.&&!mature){deferred=true;s.deferred_bundle=b;s.deferred_signal=signal;s.deferred_shops=v.shops;}
  else if(source==COW&&mature){s.deferred_bundle=-1;s.deferred_shops.clear();}
  else if(source==SHEEP&&wool-dairy<=-1.&&s.sheep_to_cow+q<=1){target=COW;s.sheep_to_cow+=q;}
  if(b==1){s.window=target;s.window_shops.assign(v.shops.begin(),v.shops.begin()+std::min(size_t(2),v.shops.size()));}
  s.assignments[b]=target;s.decisions.push_back({b,target,s.window,wool,dairy,double(v.market.prices[MI]),double(v.market.prices[WO]),balance,signal,mature,deferred,reused,v.shops});
}
void cancel(Delivery&d,std::string why){d.active=false;d.cancelled=true;d.reason=std::move(why);}
void deliver(const dp7::View&v,int step,PlayerAction&a,Delivery&d,int layer){
  if(!d.active||step<314||step>(layer==0?317:316))return;
  if(v.own.hands.size()!=11||a.units.size()!=12||v.priv.inventories.size()<12){cancel(d,"missing appended hand");return;}
  if(layer!=2&&!same(a.units.back(),Action{})){cancel(d,"parent claimed appended hand");return;}
  int y=layer==0?(step<=315?5:step==316?4:3):(step<=315?4:3);
  int animal=layer==2?SHEEP:d.animal;Stock expected{};if(step>314)expected[animal]=1;
  auto p=v.own.hands.back();
  if(p.x!=5||p.y!=y||v.priv.inventories[11]!=expected){
    std::string label=layer==0?"visible delivery mismatch at step ":layer==1?"visible corrected-delivery mismatch at step ":"visible diversified-delivery mismatch at step ";cancel(d,label+std::to_string(step));return;
  }
  bool place=step==(layer==0?317:316);
  if(place&&!empty_target(v.own)){cancel(d,"target pasture no longer empty");return;}
  a.units.back()=step==314?Action{Op::PICKUP,Item(animal),1}:place?Action{Op::PLACE,Item(animal),1}:Action{Op::NORTH};
  d.interventions.push_back(step);
}
void activate(const dp7::View&v,PlayerAction&a,Delivery&d){
  if(!empty_target(v.own)||v.own.hands.size()!=7||v.own.hires_today!=7||a.market.size()>=10||sum(v.priv.shed)>90)return;
  std::array<int,4> positions{};for(int u=0;u<8;u++){auto p=pos(v.own,u);if(!cross(p))return;positions[(p.y-4)*2+p.x-4]++;}
  if(positions!=std::array<int,4>{2,2,2,2})return;
  int found=-1,count=0,hires=0;
  for(size_t i=0;i<a.market.size();i++){auto x=a.market[i];if(x.op==Op::BUY_ANIMAL&&(x.item==Item::COW||x.item==Item::SHEEP)&&x.quantity==1){found=i;count++;}hires+=same(x,Action{Op::HIRE});}
  if(count!=1||hires!=3)return;int animal=int(a.market[found].item);
  if(v.own.money<(animal==COW?400:500)+89+1000)return;
  a.market[found].quantity=2;a.market.push_back({Op::HIRE});d.active=true;d.animal=animal;d.interventions.push_back(313);
}
void split(const dp7::View&v,PlayerAction&a,Delivery&d,const Delivery&parent){
  if(!parent.active||parent.animal!=COW||a.market.size()>=10)return;int count=0,found=-1;
  for(size_t i=0;i<a.market.size();i++){auto x=a.market[i];if(x.op==Op::BUY_ANIMAL&&x.item==Item::COW&&x.quantity==2){count++;found=i;}}
  if(count!=1)return;auto changed=a;changed.market[found].quantity=1;changed.market.push_back({Op::BUY_ANIMAL,Item::SHEEP,1});
  if(!same_market(fund(v,changed),changed))return;a=std::move(changed);d.active=true;d.interventions.push_back(313);
}
void terminal(const dp7::View&v,PlayerAction&a,bool complete){
  auto available=project(v,a);Stock sold{};std::array<int,9> slots;slots.fill(-1);
  if(complete){
    std::vector<Action> capped;for(size_t n=0;n<a.market.size()&&n<10;n++){
      auto x=a.market[n];int i=int(x.item);
      if(x.op==Op::SELL&&i>=0&&i<9){int q=std::min(std::max(0,x.quantity),available[i]);if(q<=0)continue;x.quantity=q;available[i]-=q;sold[i]+=q;slots[i]=int(capped.size());}capped.push_back(x);
    }a.market=std::move(capped);
    for(int i=0;i<9;i++)if(slots[i]>=0&&available[i]>0){a.market[slots[i]].quantity+=available[i];sold[i]+=available[i];available[i]=0;}
  }else for(auto x:a.market)if(x.op==Op::SELL&&int(x.item)>=0){int i=int(x.item);available[i]=std::max(0,available[i]-std::max(0,x.quantity));}
  std::vector<std::tuple<int,std::string,int,int>> candidates;
  for(int i=0;i<9;i++)if((complete||i==MI||i==WO)&&available[i]>0)candidates.emplace_back(revenue(i,inv(v,i)+sold[i],available[i]),item_name(i),available[i],i);
  std::sort(candidates.rbegin(),candidates.rend());for(auto [value,name,q,i]:candidates){if(a.market.size()>=10)break;a.market.push_back({Op::SELL,Item(i),q});}
}
}
PlayerAction Agent::source(const dp7::View&v,int step,const State&s)const{
  auto a=data.actions.at(std::clamp(step,0,int(data.actions.size())-1));
  for(int b=0;b<5;b++)if(s.assignments[b]>=0&&s.assignments[b]!=source_animal(b)&&std::find(steps[b].begin(),steps[b].end(),step)!=steps[b].end()){
    int src=source_animal(b),dst=s.assignments[b];
    for(auto&x:a.units)if((x.op==Op::PICKUP||x.op==Op::PLACE)&&int(x.item)==src)x.item=Item(dst);
    for(auto&x:a.market)if(x.op==Op::BUY_ANIMAL&&int(x.item)==src)x.item=Item(dst);
  }
  if(s.cow_to_sheep+s.sheep_to_cow){
    auto available=project(v,a);std::array<int,9> inventory{};inventory[MI]=inv(v,MI);inventory[WO]=inv(v,WO);auto [wool,dairy]=pressure(v);
    for(auto&x:a.market)if(x.op==Op::SELL&&(int(x.item)==MI||int(x.item)==WO)){
      auto score=[&](int i){int q=std::min(std::max(0,x.quantity),available[i]);return std::tuple{revenue(i,inventory[i],q),q,i==MI?dairy:wool,int(x.item)==i,std::string(item_name(i))};};
      int i=score(MI)>score(WO)?MI:WO;int q=std::min(std::max(0,x.quantity),available[i]);x.item=Item(i);available[i]-=q;inventory[i]+=q;
    }
  }return a;
}
PlayerAction Agent::act(const dp7::View&v,int seat,State&s)const{
  int step=std::clamp(v.step,0,719);
  if(s.seat!=seat||step==0||step<s.last){s=State{};s.seat=seat;}
  if(step<=s.base_last){s.weed.clear();s.base_last=-1;}
  for(auto&d:s.delivery)if(step<=d.last)d=Delivery{};
  for(int b=0;b<5;b++)if(step==purchases[b])assign(v,b,s);
  auto a=source(v,step,s);align(a,v);
  for(auto it=s.weed.begin();it!=s.weed.end();){int u=it->first,age=step-it->second.start;
    if(u>=int(a.units.size())){it=s.weed.erase(it);continue;}
    if(age==1)a.units[u]=it->second.intended;
    else if(age>=2&&age<=9){auto prior=source(v,step-1,s);a.units[u]=u<int(prior.units.size())?prior.units[u]:Action{};}
    else {it=s.weed.erase(it);continue;}++it;
  }
  for(size_t u=0;u<a.units.size();u++){
    auto x=a.units[u];if(s.weed.count(u)||(x.op!=Op::PLANT&&x.op!=Op::BUILD_PASTURE&&x.op!=Op::BUILD_COOP))continue;
    auto p=pos(v.own,u);if(v.own.tiles.at(p.y*10+p.x).kind==TileKind::WEED){s.weed[int(u)]={step,x};a.units[u]={Op::DIG};}
  }
  auto available=project(v,a);std::vector<Action> capped;
  for(auto x:a.market){int i=int(x.item);if(x.op==Op::SELL&&i>=0&&i<9){x.quantity=std::min(std::max(0,x.quantity),available[i]);available[i]-=x.quantity;if(!x.quantity)continue;}capped.push_back(x);}
  if(capped.size()>10)capped.resize(10);a.market=std::move(capped);
  std::vector<int> positions;std::vector<Action> sales;
  for(size_t i=0;i<a.market.size();i++)if(a.market[i].op==Op::SELL&&int(a.market[i].item)>=0&&int(a.market[i].item)<9){positions.push_back(i);sales.push_back(a.market[i]);}
  auto score=[&](Action x){int i=int(x.item),q=std::clamp(x.quantity,1,60),iv=inv(v,i);return std::tuple{revenue(i,iv,q)-revenue(i,iv+q,q),price(i,iv),std::string(item_name(i))};};
  std::stable_sort(sales.begin(),sales.end(),[&](Action x,Action y){return score(x)>score(y);});
  for(size_t i=0;i<sales.size();i++)a.market[positions[i]]=sales[i];a=fund(v,a);align(a,v);s.stages[0]=a;
  if(step==718)terminal(v,a,false);s.stages[1]=a;
  if(step==313)activate(v,a,s.delivery[0]);else deliver(v,step,a,s.delivery[0],0);s.stages[2]=a;
  if(step==313){if(s.delivery[0].active){s.delivery[1].active=true;s.delivery[1].animal=s.delivery[0].animal;s.delivery[1].interventions.push_back(step);}}
  else deliver(v,step,a,s.delivery[1],1);s.stages[3]=a;
  if(step==313)split(v,a,s.delivery[2],s.delivery[1]);else deliver(v,step,a,s.delivery[2],2);s.stages[4]=a;
  if(step==718)terminal(v,a,true);s.stages[5]=a;
  s.base_last=step;s.last=step;for(auto&d:s.delivery)d.last=step;return a;
}
}
