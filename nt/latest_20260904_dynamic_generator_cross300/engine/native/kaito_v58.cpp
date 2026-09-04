#include "kaito_v58.hpp"
#include <numeric>
#include <tuple>
#include <stdexcept>

namespace kaito58 {
using namespace fastkag;
namespace {
constexpr double base[]{25,35,60,120,250,50,160,200,100};
constexpr int shops[8][9]{
 {1,0,0,0,0,1,0,0,0}, {1,0,0,1,0,1,0,0,0},
 {1,1,1,1,0,0,0,0,0}, {1,0,0,1,0,0,1,0,0},
 {0,2,0,0,0,0,0,0,0}, {1,0,1,0,0,0,1,0,0},
 {0,0,0,1,0,0,1,0,0}, {0,0,0,0,0,0,0,2,0}};
int demand(const std::vector<int8_t>& sh,int step,int item) {
  int n=0;if(step%4==0)for(int s:sh)n+=shops[s][item];
  if(item!=8&&step%24==0)n++;return n;
}
double shape(int k,double value,double scale) {
  value=std::max(0.,value);
  if(k==0)return value;if(k==1)return value*value;
  if(k==2)return std::sqrt(value);if(k==3)return std::log1p(value);
  double n=value/scale;return n+8.*std::pow(std::max(0.,n-1.),2.);
}
int quote(int item,int inventory) {
  constexpr double scale[]{400,450,200,100,300,332,122,105,200};
  constexpr double below[]{.8,1,.4,.7,.2,.4,.6,.2,.4},above[]{.2,.7,.6,1.6,3.6,.2,1.6,3.2,.4};
  constexpr int bf[]{2,4,4,2,3,4,2,3,0},af[]{3,2,2,0,1,3,0,1,0};
  bool low=inventory<10000;int f=low?bf[item]:af[item];
  double amplitude=(low?below[item]:above[item])*base[item]/shape(f,scale[item],scale[item]);
  double delta=amplitude*shape(f,std::abs(inventory-10000),scale[item]);
  return std::max(1,int(std::nearbyint(low?base[item]+delta:base[item]-delta)));
}
double revenue(int item,int inventory,int q) {
  double out=0;for(int n=0;n<q;n++){int p=quote(item,inventory);out+=p;if(p>1)inventory++;}return out;
}
bool product(const Action&a){return a.op==Op::SELL&&int(a.item)>=0&&int(a.item)<9;}
Products sells(const PlayerAction&a) {
  Products out{};for(auto&x:a.market)if(product(x))out[int(x.item)]+=std::max(0,x.quantity);return out;
}
bool access(Position p){return (p.x==4||p.x==5)&&(p.y==4||p.y==5);}
int take(std::array<int,12>&inv,int item,int q){int n=std::min(std::max(0,q),std::max(0,inv[item]));inv[item]-=n;return n;}
// Preserve the source's conservative projection, not the engine's fuller one:
// no harvest credit, no cash clipping of BUY_PRODUCT, and FEED reads old tiles.
// Terminal helper has DIFFERENT semantics: it ignores PICKUP/FEED entirely.
std::array<int,12> project(const dp7::View&o,const PlayerAction&a,bool terminal=false) {
  auto shed=o.priv.shed;auto inventories=o.priv.inventories;
  for(size_t u=0;u<a.units.size()&&u<o.own.hands.size()+1&&u<inventories.size();u++) {
    auto pos=u?o.own.hands[u-1]:o.own.farmer;auto&t=o.own.tiles[pos.y*10+pos.x];
    auto x=a.units[u];auto&inv=inventories[u];int item=int(x.item);
    auto room=[&]{return std::max(0,100-std::accumulate(shed.begin(),shed.end(),0));};
    if(x.op==Op::DROP&&access(pos)) {
      // Original dict insertion order matters at the capacity boundary.
      for(int i:o.priv.inventory_order[u]){int n=std::min(std::max(0,inv[i]),room());shed[i]+=n;inv[i]=0;}
    } else if(!terminal&&x.op==Op::PICKUP&&access(pos)&&item>=0) {
      inv[item]+=take(shed,item,x.quantity);
    } else if(x.op==Op::PLACE&&item>=0) {
      bool put_animal=item>=9&&((item==9&&t.kind==TileKind::COOP)||(item!=9&&t.kind==TileKind::PASTURE));
      if(put_animal){if(!terminal)take(inv,item,1);continue;}
      if(access(pos))shed[item]+=take(inv,item,std::min(std::max(0,x.quantity),room()));
    } else if(!terminal&&x.op==Op::FEED&&t.kind==TileKind::ANIMAL&&!t.fed_today)take(inv,0,1);
  }
  return shed;
}
Products executable_net(const dp7::View&o,const PlayerAction&a) {
  auto shed=project(o,a);Products net{};
  for(auto x:a.market){int i=int(x.item);if(i<0||i>=9)continue;int q=std::max(0,x.quantity);
    if(x.op==Op::SELL){int n=std::min(q,std::max(0,shed[i]));shed[i]-=n;net[i]+=n;}
    else if(x.op==Op::BUY_PRODUCT&&(i==0||i==8)){
      int n=std::min(q,std::max(0,100-std::accumulate(shed.begin(),shed.end(),0)));shed[i]+=n;net[i]-=n;
    }
  }return net;
}
std::array<int,14> signature(const Farm&f) {
  std::array<int,14>c{};
  c[0]=int(f.hands.size());c[1]=3*std::popcount(unsigned(f.unlocked_mask));
  for(auto&t:f.tiles){if(t.kind==TileKind::PLANT)c[2+int(t.crop)]++;
    else if(t.kind==TileKind::ANIMAL)c[7+int(t.animal)-9]++;
    else if(t.kind==TileKind::PASTURE)c[10]++;else if(t.kind==TileKind::COOP)c[11]++;
    else if(t.kind==TileKind::WEED)c[12]++;}
  return c;
}
int distance(const dp7::View&o){auto a=signature(o.own),b=signature(o.opponent);int n=0;for(int i=0;i<14;i++)n+=std::abs(a[i]-b[i]);return n;}
std::array<double,9> exposure(const Farm&f,bool terminal=false) {
  std::array<double,9>x{};for(auto&t:f.tiles){if(t.kind==TileKind::PLANT)x[int(t.crop)]+=std::max(1,int(t.yield_units));
    if(t.kind==TileKind::ANIMAL){int i=t.animal==Item::COW?6:t.animal==Item::SHEEP?7:5;
      x[i]+=terminal?1+std::max(0,int(t.yield_units)):std::max(1,int(t.yield_units));
      if(t.fertilizer_available)x[8]++;
    }}return x;
}
void reorder(const dp7::View&o,PlayerAction&a) {
  struct Row{double score;size_t index;Action action;};std::vector<Row> rows;
  for(size_t n=0;n<a.market.size();n++){auto x=a.market[n];if(!product(x))continue;int i=int(x.item),q=std::max(0,x.quantity);
    int inventory=o.market.inventory[i];double score=q*std::max(0.,double(o.market.prices[i]-quote(i,inventory+q)));
    if(score>0){double daily=i==8?0.:1.;for(int sh:o.shops)daily+=6.*shops[sh][i];
      double excess=std::max(0.,double(inventory+q-10000));double recovery=excess/std::max(.25,daily);
      score*=1.+.25*std::min(1.,recovery/10.);
    }rows.push_back({score,n,x});
  }
  std::sort(rows.begin(),rows.end(),[](auto&a,auto&b){return a.score!=b.score?a.score>b.score:a.index<b.index;});
  size_t j=0;for(auto&x:a.market)if(product(x))x=rows[j++].action;
}
void terminal_market(const dp7::View&o,PlayerAction&a) {
  constexpr int order[]{3,4,6,7,5,2,1,0,8};constexpr double weight[]{1,1,1.3,2,3.6,1.5,2,3.2,1};
  auto shed=project(o,a,true);auto exp=exposure(o.opponent,true);
  struct Row{double score;int rank,item,q;};std::vector<Row>rows;
  for(int n=0;n<9;n++){int i=order[n],q=std::max(0,shed[i]);if(!q)continue;
    double score=(1.+exp[i])*weight[i]*std::max(1.,double(o.market.prices[i]))*std::log1p(q);
    rows.push_back({score,n,i,q});}
  std::sort(rows.begin(),rows.end(),[](auto&a,auto&b){return a.score!=b.score?a.score>b.score:a.rank<b.rank;});
  a.market.clear();for(auto&r:rows)a.market.push_back({Op::SELL,Item(r.item),r.q});
}
bool equal_tile(const Tile&a,const Tile&b) {
  if(a.kind!=b.kind)return false;
  if(a.kind==TileKind::PLANT)return std::tie(a.crop,a.planted_day,a.watered_today,a.consecutive_unwatered,a.yield_units,a.max_lifespan_step,a.fertilized_until_day)==std::tie(b.crop,b.planted_day,b.watered_today,b.consecutive_unwatered,b.yield_units,b.max_lifespan_step,b.fertilized_until_day);
  if(a.kind==TileKind::ANIMAL)return std::tie(a.animal,a.placed_day,a.yield_units,a.consecutive_unfed,a.fed_today,a.cared_today,a.fertilizer_available,a.pending_care_bonus)==std::tie(b.animal,b.placed_day,b.yield_units,b.consecutive_unfed,b.fed_today,b.cared_today,b.fertilizer_available,b.pending_care_bonus);
  return true;
}
bool mirror(const Farm&a,const Farm&b){
  if(a.money!=b.money||a.farmer.x!=b.farmer.x||a.farmer.y!=b.farmer.y||a.unlocked_mask!=b.unlocked_mask||a.hands.size()!=b.hands.size())return false;
  for(size_t u=0;u<a.hands.size();u++)if(a.hands[u].x!=b.hands[u].x||a.hands[u].y!=b.hands[u].y)return false;
  for(size_t k=0;k<a.tiles.size();k++)if(!equal_tile(a.tiles[k],b.tiles[k]))return false;return true;
}
}

PlayerAction Agent::policy(const dp7::View&o,int route,PolicyState&s) const {
  const int step=o.step;auto&trace=data.routes.at(route);auto c=[&](const char*k){return data.c(k);};
  if(step==0||step<s.last)s=PolicyState{};
  if(s.last==step-1&&s.last>=0){
    Products inferred{};int own_total=0,other_total=0,overlap=0;
    for(int i=0;i<9;i++){
      int cur=o.market.inventory[i]?o.market.inventory[i]:10000;
      int prev=s.inventory[i]?s.inventory[i]:cur;
      inferred[i]=std::clamp(cur-prev-s.net[i]+demand(s.shops,step-1,i),0,int(c("maximum_inferred_supply")));
      s.supply[i]=(1.-c("flow_alpha"))*s.supply[i]+c("flow_alpha")*inferred[i];
      if(i>=1&&i<=7){own_total+=std::max(0,s.net[i]);other_total+=inferred[i];overlap+=std::min(std::max(0,s.net[i]),inferred[i]);}
    }
    if(own_total>0||other_total>0){double agreement=2.*overlap/std::max(1.,double(own_total+other_total));
      double alpha=c(agreement<s.confidence?"adaptive_confidence_negative_alpha":"adaptive_confidence_alpha");
      s.confidence=std::clamp((1.-alpha)*s.confidence+alpha*agreement,0.,c("adaptive_confidence_ceiling"));s.evidence++;
    }else s.confidence*=c("adaptive_confidence_decay");
  }
  for(int i=0;i<9;i++)s.inventory[i]=o.market.inventory[i]?o.market.inventory[i]:10000;
  s.near_streak=distance(o)<=c("near_distance")?s.near_streak+1:0;
  if(s.near_streak>=c("near_streak_required"))s.near=true;
  PlayerAction a=trace.at(step);a.units.resize(o.own.hands.size()+1);
  for(auto it=s.weed.begin();it!=s.weed.end();){int u=it->first,age=step-it->second.start;
    if(u>=int(a.units.size())){it=s.weed.erase(it);continue;}
    if(age==1)a.units[u]=it->second.intended;
    else if(age>=2&&age<=9){auto&t=trace.at(step-1);a.units[u]=u<int(t.units.size())?t.units[u]:Action{};}
    else{it=s.weed.erase(it);continue;}++it;
  }
  for(int u=0;u<int(a.units.size());u++){auto x=a.units[u];if(s.weed.contains(u)||(x.op!=Op::BUILD_PASTURE&&x.op!=Op::PLANT))continue;
    auto pos=u?o.own.hands[u-1]:o.own.farmer;if(o.own.tiles[pos.y*10+pos.x].kind==TileKind::WEED){s.weed[u]={step,x};a.units[u]={Op::DIG};s.collisions++;}
  }
  reorder(o,a);
  if(auto it=s.due.find(step);it!=s.due.end()){
    auto debt=it->second;s.due.erase(it);std::vector<Action>kept;
    for(auto x:a.market){if(product(x)&&debt[int(x.item)]>0){int i=int(x.item),q=std::max(0,x.quantity),reduction=std::min(q,debt[i]);
      x.quantity=q-reduction;debt[i]-=reduction;s.repaid+=reduction;if(x.quantity<=0)continue;}kept.push_back(x);}
    if(step<718)for(int i=0;i<9;i++)if(debt[i]>0)s.due[step+1][i]+=debt[i];a.market=std::move(kept);
  }
  // Frozen config has defer=false and market_maker=false (asserted on load).
  double peak=*std::max_element(s.supply.begin(),s.supply.end()),deficit=o.opponent.money-o.own.money;
  if(step>=c("active_start")&&step<c("active_stop")&&a.market.size()<c("maximum_orders")&&
     ((s.near&&c("preempt_on_near"))||deficit>=c("preempt_money_deficit")||peak>=c("front_supply_threshold"))){
    auto stock=project(o,a);auto existing=sells(a);auto exp=exposure(o.opponent);int horizon=int(c("preempt_horizon"));
    struct Candidate{double delta,gross;int item,q;std::vector<std::pair<int,int>>allocations;};std::vector<Candidate>candidates;
    for(int i=1;i<=7;i++){int available=std::max(0,stock[i]-existing[i]);if(!available)continue;
      int remaining=std::min(available,int(c("preempt_maximum_batch"))),scheduled=0;std::vector<std::pair<int,int>>alloc;
      for(int f=step+1;f<=std::min(718,step+horizon)&&remaining>0;f++){
        int n=sells(trace[f])[i];auto d=s.due.find(f);if(d!=s.due.end())n-=d->second[i];n=std::min(remaining,std::max(0,n));
        if(n){alloc.emplace_back(f,n);remaining-=n;scheduled+=n;}}
      if(!scheduled||double(o.market.prices[i])/base[i]<c("preempt_minimum_price_ratio"))continue;
      double forecast=horizon*(s.supply[i]+c("preempt_exposure_scale")*exp[i]);
      if(s.near){double weight=c("near_scheduled_supply_weight");if(c("adaptive_near_schedule_weight"))
        weight=s.evidence>=c("adaptive_confidence_minimum_events")&&s.confidence>=c("adaptive_confidence_activation_threshold")?s.confidence:0.;
        forecast+=std::max(0.,weight)*scheduled;}
      int dem=0;for(int f=step;f<step+horizon;f++)dem+=demand(o.shops,f,i);
      int inventory=o.market.inventory[i]?o.market.inventory[i]:10000;
      int future=int(std::nearbyint(inventory+forecast-dem));double now=revenue(i,inventory,scheduled),later=revenue(i,future,scheduled);
      if(now-later>=c("preempt_minimum_delta"))candidates.push_back({now-later,now,i,scheduled,alloc});
    }
    std::sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){if(a.delta!=b.delta)return a.delta>b.delta;if(a.gross!=b.gross)return a.gross>b.gross;return std::string(item_name(a.item))>item_name(b.item);});
    int remaining=int(c("preempt_maximum_batch"));std::vector<Action>additions;
    for(auto&r:candidates){if(a.market.size()+additions.size()>=c("maximum_orders")||remaining<=0)break;int q=std::min(r.q,remaining),left=q;
      additions.push_back({Op::SELL,Item(r.item),q});for(auto [f,n]:r.allocations){int k=std::min(left,n);if(k){s.due[f][r.item]+=k;left-=k;}if(!left)break;}
      remaining-=q;s.preempt_units+=q;}
    if(!additions.empty()){a.market.insert(a.market.begin(),additions.begin(),additions.end());s.preempt_turns++;}
  }
  reorder(o,a);double front_peak=*std::max_element(s.supply.begin()+1,s.supply.begin()+8);
  if(c("front_existing_sells")&&((c("front_on_near")&&s.near)||deficit>=c("front_money_deficit")||front_peak>=c("front_supply_threshold")))
    std::stable_partition(a.market.begin(),a.market.end(),[](auto x){return product(x)&&int(x.item)>=1&&int(x.item)<=7;});
  if(step==718)terminal_market(o,a);
  if(a.market.size()>size_t(c("maximum_orders")))a.market.resize(int(c("maximum_orders")));
  s.net=executable_net(o,a);s.shops=o.shops;s.last=step;s.emitted=a;return a;
}

int Agent::select(const dp7::View&o,State&s) const {
  int step=o.step,first=o.shops.empty()?-1:o.shops[0],second=o.shops.size()<2?-1:o.shops[1];
  int base_route=step<72?0:first==7?1:first==4&&step>=144&&(second==7||second==4)?2:0;
  auto counts=signature(o.opponent);
  std::array<int,7> sig{int(std::nearbyint(o.own.money)),int(std::nearbyint(o.opponent.money)),o.market.inventory[0],counts[8],counts[9],counts[2],counts[6]};
  if(step==72&&s.mode.empty()){
    using Key=std::pair<int,std::array<int,7>>;
    static const std::map<Key,std::string> modes{
      {{2,{141,193,9974,3,2,7,12}},"farmers-recovery"},{{0,{145,195,9975,3,2,7,12}},"recovery"},
      {{6,{145,193,9975,3,2,7,12}},"recovery"},{{0,{142,142,9973,3,2,7,12}},"recovery"},
      {{3,{142,191,9974,3,2,7,12}},"recovery"},{{3,{145,195,9975,3,2,7,12}},"ice-minimax"},
      {{0,{141,193,9974,3,2,7,12}},"bakery-second-shop"},{{5,{141,193,9974,3,2,7,12}},"pizza-second-shop"}};
    auto it=modes.find({first,sig});s.mode=it==modes.end()?"base":it->second;
  }
  if(s.mode=="farmers-recovery")return step>=144&&second==6?4:3;
  if(s.mode=="recovery")return 3;if(s.mode=="ice-minimax")return 7;
  if(s.mode=="bakery-second-shop")return step>=144&&second==7?8:base_route;
  if(s.mode=="pizza-second-shop")return step>=144&&second==4?9:base_route;
  if(step==96&&first==7)s.known_yarn=sig==std::array<int,7>{214,214,9982,4,2,7,12}||sig==std::array<int,7>{215,125,9979,4,2,7,12};
  if(step>=96&&s.known_yarn)return 6;
  if(step>=72&&first==7){s.mirror_streak=mirror(o.own,o.opponent)?s.mirror_streak+1:0;
    if(step==360)s.clone=s.mirror_streak>=240;if(step>=360&&s.clone)return 5;}
  return base_route;
}
PlayerAction Agent::act(const dp7::View&o,int seat,State&s) const {
  if(seat<0||seat>1||o.step<0||o.step>718)throw std::invalid_argument("Kaito state bounds");
  if(seat!=s.seat){s=State{};s.seat=seat;}
  // Router <= reset differs from the child controllers' < reset, as in source.
  if(o.step==0||o.step<=s.last){s.mode.clear();s.mirror_streak=0;s.clone=false;s.known_yarn=false;}
  s.last=o.step;
  // Eager native warm-up: all independent children see every real observation.
  // Python builds one/turn, replaying missed observations. No child shares
  // mutable state; by its first visible call the two histories are identical.
  for(int r=0;r<10;r++)policy(o,r,s.policies[r]);
  s.selected=select(o,s);return s.policies[s.selected].emitted;
}
}
