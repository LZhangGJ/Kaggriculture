#include "boatlee_v29.hpp"

namespace boatlee29 {
using namespace fastkag;
namespace {
int pickup(const PlayerAction& a,int item) {
  int n=0;for(auto& x:a.units)if(x.op==Op::PICKUP&&int(x.item)==item)n+=std::max(0,x.quantity);return n;
}
int sell(const PlayerAction& a,int item) {
  int n=0;for(auto& x:a.market)if(x.op==Op::SELL&&int(x.item)==item)n+=std::max(0,x.quantity);return n;
}
int count(const Farm& f,int kind) {
  int n=0;for(auto& t:f.tiles) {
    if(kind==dp7::S)n+=t.kind==TileKind::PLANT&&int(t.crop)==kind;
    else n+=t.kind==TileKind::ANIMAL&&int(t.animal)==kind;
  }return n;
}
void add_sell(PlayerAction& a,int item,int quantity) {
  if(quantity<=0)return;
  auto it=std::find_if(a.market.begin(),a.market.end(),[&](const auto& x){return x.op==Op::SELL&&int(x.item)==item;});
  if(it!=a.market.end())it->quantity=std::max(0,it->quantity)+quantity;
  else if(a.market.size()<10)a.market.insert(a.market.begin(),{Op::SELL,Item(item),quantity});
  if(a.market.size()>10)a.market.resize(10);
}
}
int Agent::demand(const std::vector<int8_t>& shops,int item) const {
  int n=0;for(int shop:shops)if(shop>=0&&shop<8)n+=data.shop_demand[shop][item];return n;
}
void Agent::weed(const dp7::View& o,int step,PlayerAction& a,State& s) const {
  a.units.resize(o.own.hands.size()+1);
  if(step==0||step<s.weed_last)s.active.clear();
  s.weed_last=step;
  for(auto it=s.active.begin();it!=s.active.end();) {
    int u=it->first,age=step-it->second.start;
    if(u>=int(a.units.size())){it=s.active.erase(it);continue;}
    if(age==1)a.units[u]=it->second.intended;
    else if(age>=2&&age<=1+data.weed_replay_steps) {
      auto& trace=data.actions[std::clamp(step-1,0,int(data.actions.size())-1)];
      a.units[u]=u<int(trace.units.size())?trace.units[u]:Action{};
    } else {it=s.active.erase(it);continue;}
    ++it;
  }
  for(int u=0;u<int(a.units.size());u++) {
    auto intended=a.units[u];if(s.active.contains(u)||(intended.op!=Op::BUILD_PASTURE&&intended.op!=Op::PLANT))continue;
    auto p=u?o.own.hands[u-1]:o.own.farmer;
    if(p.x<0||p.x>=10||p.y<0||p.y>=10||o.own.tiles[p.y*10+p.x].kind!=TileKind::WEED)continue;
    s.active[u]={step,intended};a.units[u]=Action{Op::DIG};
  }
}
void Agent::observe(const dp7::View& o,int step,State& s) const {
  if(step==0||step<=s.am_last) {
    s.am_last=step;s.last_inventory={};s.last_sold={};s.last_shops.clear();
    s.pressure={};s.added={};s.near_mirror=-1;
  }
  if(s.am_last==step-1)for(int i:data.items) {
    int delta=o.market.inventory[i]-s.last_inventory[i];
    int consumption=(i!=dp7::F&&(step-1)%24==0)?1:0;
    if((step-1)%4==0)consumption+=demand(s.last_shops,i);
    int external=delta+consumption-s.last_sold[i];
    s.pressure[i]=std::max(0.0,s.pressure[i]*0.72+std::min(24.0,std::max(0.0,double(external))));
  }
  s.am_last=step;
  if(s.near_mirror<0&&step>=data.c("mirror_latch_step")) {
    int distance=0;for(int i:{dp7::CO,dp7::SH,dp7::S})distance+=std::abs(count(o.own,i)-count(o.opponent,i));
    s.near_mirror=distance<=data.c("mirror_composition_distance")&&o.own.hands.size()==o.opponent.hands.size()
      &&std::popcount(unsigned(o.own.unlocked_mask))==std::popcount(unsigned(o.opponent.unlocked_mask))
      &&std::abs(o.own.money-o.opponent.money)<=data.c("mirror_money_distance");
  }
}
void Agent::market(const dp7::View& o,int step,PlayerAction& a,State& s) const {
  observe(o,step,s);
  int total=0;for(auto n:o.priv.shed)total+=std::max(0,n);
  if(step>=data.c("start_step"))for(int i:data.items) {
    int stock=std::max(0,o.priv.shed[i]),reserved_pickup=pickup(a,i);
    int scheduled=std::min(std::max(0,stock-reserved_pickup),sell(a,i));
    int unscheduled=std::max(0,stock-reserved_pickup-scheduled);
    int reserve=step<528?int(data.c("reserve_early")):step<600?int(data.c("reserve_mid")):
      step<648?int(data.c("reserve_late")):step<684?int(data.c("reserve_tail")):step<704?2:0;
    if(total>=data.c("shed_hard"))reserve=0;
    else if(total>=data.c("shed_soft"))reserve=std::max(0,reserve-int(data.c("shed_reserve_cut")));
    int shop_demand=demand(o.shops,i);
    if(step<684)reserve+=std::min(int(data.c("demand_reserve_cap")),shop_demand*int(data.c("demand_reserve")));
    int excess=std::max(0,unscheduled-reserve);if(excess<=0)continue;
    int max_extra=int(data.c(s.near_mirror==1?"mirror_max_extra_per_item":"max_extra_per_item"));
    int budget=std::max(0,max_extra-s.added[i]);if(budget<=0)continue;
    double ratio=double(o.market.prices[i])/data.base_price[i];
    int kind=i==dp7::S?dp7::S:i==dp7::MI?dp7::CO:dp7::SH;
    int capacity_delta=count(o.opponent,kind)-count(o.own,kind);
    double gate=data.c("price_gate");
    if(step<684)gate+=std::min(data.c("demand_gate_cap"),shop_demand*data.c("demand_gate"));
    if(step>=648)gate-=0.12;
    if(step>=684)gate-=0.18;
    if(s.pressure[i]>=data.c("pressure_trigger"))gate-=data.c("pressure_gate_cut");
    if(capacity_delta>=data.c("capacity_trigger"))gate-=data.c("capacity_gate_cut");
    bool urgent=total>=data.c("shed_hard")||step>=704;
    if(!urgent&&ratio<std::max(0.20,gate))continue;
    int tranche=int(data.c("tranche"));
    if(s.pressure[i]>=data.c("pressure_trigger"))tranche+=int(data.c("pressure_tranche"));
    if(total>=data.c("shed_soft"))tranche+=int(data.c("shed_tranche"));
    if(step>=684)tranche+=int(data.c("tail_tranche"));
    int quantity=std::min({excess,std::max(1,tranche),budget});
    add_sell(a,i,quantity);
    // Preserve original accounting even when ten existing orders prevent adding.
    s.added[i]+=quantity;
  }
  for(int i:data.items) {
    s.last_sold[i]=std::min(std::max(0,o.priv.shed[i]),sell(a,i));
    s.last_inventory[i]=o.market.inventory[i];
  }
  s.last_shops=o.shops;
}
PlayerAction Agent::act(const dp7::View& o,int seat,State& s) const {
  if(seat<0||seat>1)throw std::invalid_argument("Boatlee seat");
  if(s.seat!=seat){s=State{};s.seat=seat;}
  int step=std::clamp(o.step,0,int(data.actions.size())-1);
  auto a=data.actions.at(step);weed(o,step,a,s);
  // Frozen _FR_ITEMS is empty; its repayment ledger can never become nonempty.
  // The data loader rejects any version that enables that different branch.
  market(o,step,a,s);a.units.resize(o.own.hands.size()+1);return a;
}
}
