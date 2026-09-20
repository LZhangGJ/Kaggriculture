// Licensed under the Apache License, Version 2.0.
// Clean-room typed implementation of kaggle-environments 1.32.7 kaggriculture.
#include "simulator.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fastkag {
namespace {
struct CropDef { int seed, first_day, max_day, interval, max_yield; bool ongoing; };
constexpr CropDef CROPS[N_CROPS] = {
 {10,2,4,0,6,false},{20,2,3,0,4,false},{50,8,8,1,4,true},
 {100,10,10,2,4,true},{80,10,12,0,6,false}
};
struct AnimalDef { int cost; TileKind structure; int first_day,interval,max_held,product; };
constexpr AnimalDef ANIMALS[N_ANIMALS] = {
 {300,TileKind::COOP,4,1,4,5},{400,TileKind::PASTURE,8,2,6,6},
 {500,TileKind::PASTURE,6,3,6,7}
};
enum Shape { LINEAR,SQ,SQRT,LOG,HINGE };
struct MarketDef { int base,I0,T; Shape below,above; double bt,at; };
constexpr MarketDef MP[N_PRODUCTS] = {
 {25,10000,400,SQRT,LOG,.8,.2},{35,10000,450,HINGE,SQRT,1.,.7},
 {60,10000,200,HINGE,SQRT,.4,.6},{120,10000,100,SQRT,LINEAR,.7,1.6},
 {250,10000,300,LOG,SQ,.2,3.6},{50,10000,332,HINGE,LOG,.4,.2},
 {160,10000,122,SQRT,LINEAR,.6,1.6},{200,10000,105,LOG,SQ,.2,3.2},
 {100,10000,200,LINEAR,LINEAR,.4,.4}
};
constexpr const char* ITEMS[N_ITEMS] = {"WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"};
constexpr const char* SHOPS[8] = {"BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE"};
constexpr int LAND_PRICE[3]={1000,2000,4000};

double shape(Shape f,double x,double T) {
 x=std::max(0.,x);
 switch(f){case LINEAR:return x;case SQ:return x*x;case SQRT:return std::sqrt(x);
 case LOG:return std::log1p(x);case HINGE:{double u=x/T;return u+8.*std::pow(std::max(0.,u-1.),2);}}
 return x;
}
int shed_sum(const PrivateState& p){int n=0;for(int x:p.shed)n+=x;return n;}
int fib(int n){int a=1,b=1;while(n--){int c=a+b;a=b;b=c;}return a;}
void ensure_inventory(PrivateState& p,int idx){while((int)p.inventories.size()<=idx){p.inventories.push_back({});p.inventory_order.push_back({});}}
void erase_inventory_key(PrivateState& p,int idx,int item){auto&o=p.inventory_order[idx];o.erase(std::remove(o.begin(),o.end(),item),o.end());}
void inventory_add(PrivateState& p,int idx,int item,int n){if(n<=0)return;ensure_inventory(p,idx);auto&inv=p.inventories[idx];if(inv[item]==0)p.inventory_order[idx].push_back((int8_t)item);inv[item]+=n;}
bool inventory_take(PrivateState& p,int idx,int item,int n){ensure_inventory(p,idx);auto&inv=p.inventories[idx];if(n<=0||inv[item]<n)return false;inv[item]-=n;if(inv[item]==0)erase_inventory_key(p,idx,item);return true;}
}

const char* item_name(int i){return i>=0&&i<N_ITEMS?ITEMS[i]:"";}
const char* shop_name(int i){return i>=0&&i<8?SHOPS[i]:"";}

// CPython _randommodule.c compatible MT19937 seeding/output for non-negative int seeds.
void PythonRandom::init_genrand(uint32_t s){mt_[0]=s;for(int mti=1;mti<624;mti++)mt_[mti]=1812433253U*(mt_[mti-1]^(mt_[mti-1]>>30))+mti;index_=624;}
void PythonRandom::init_by_array(const uint32_t* key,int len){
 init_genrand(19650218U);int i=1,j=0,k=std::max(624,len);
 for(;k;k--){mt_[i]=(mt_[i]^((mt_[i-1]^(mt_[i-1]>>30))*1664525U))+key[j]+j;i++;j++;if(i>=624){mt_[0]=mt_[623];i=1;}if(j>=len)j=0;}
 for(k=623;k;k--){mt_[i]=(mt_[i]^((mt_[i-1]^(mt_[i-1]>>30))*1566083941U))-i;i++;if(i>=624){mt_[0]=mt_[623];i=1;}}
 mt_[0]=0x80000000U;
}
void PythonRandom::reseed(uint64_t s){uint32_t key[2]={(uint32_t)s,(uint32_t)(s>>32)};init_by_array(key,key[1]?2:1);}
uint32_t PythonRandom::genrand_uint32(){
 static constexpr uint32_t mag[2]={0,0x9908b0dfU};
 if(index_>=624){int kk;for(kk=0;kk<227;kk++){uint32_t y=(mt_[kk]&0x80000000U)|(mt_[kk+1]&0x7fffffffU);mt_[kk]=mt_[kk+397]^(y>>1)^mag[y&1];}
 for(;kk<623;kk++){uint32_t y=(mt_[kk]&0x80000000U)|(mt_[kk+1]&0x7fffffffU);mt_[kk]=mt_[kk-227]^(y>>1)^mag[y&1];}
 uint32_t y=(mt_[623]&0x80000000U)|(mt_[0]&0x7fffffffU);mt_[623]=mt_[396]^(y>>1)^mag[y&1];index_=0;}
 uint32_t y=mt_[index_++];y^=y>>11;y^=(y<<7)&0x9d2c5680U;y^=(y<<15)&0xefc60000U;y^=y>>18;return y;
}
double PythonRandom::random(){uint32_t a=genrand_uint32()>>5,b=genrand_uint32()>>6;return (a*67108864.0+b)*(1.0/9007199254740992.0);}
uint32_t PythonRandom::getrandbits(int k){if(k<=0)return 0;if(k<=32)return genrand_uint32()>>(32-k);throw std::runtime_error("getrandbits >32 unsupported");}
uint32_t PythonRandom::randbelow(uint32_t n){int k=32-__builtin_clz(n);uint32_t r;do r=getrandbits(k);while(r>=n);return r;}

Simulator::Simulator(Config c,uint64_t seed):cfg_(c){reset(seed);}
int Simulator::quadrant(int x,int y)const{int h=cfg_.board_size/2;return (y<h?0:2)+(x<h?0:1);}
Position Simulator::default_spawn()const{int h=cfg_.board_size/2;return Position{(int16_t)(h-1),(int16_t)(h-1)};}
bool Simulator::shed_adjacent(Position p)const{int h=cfg_.board_size/2;return (p.x==h-1||p.x==h)&&(p.y==h-1||p.y==h);}
void Simulator::reset(uint64_t seed){
 seed_=seed;step_=0;done_=false;shops_.clear();last_end_of_day_overflow_={0,0};
 for(int p=0;p<2;p++){
  farms_[p]=Farm{};farms_[p].money=cfg_.starting_money;farms_[p].farmer=default_spawn();farms_[p].tiles.resize(cfg_.board_size*cfg_.board_size);
  for(int y=0;y<cfg_.board_size;y++)for(int x=0;x<cfg_.board_size;x++)farms_[p].tiles[tile_index(x,y)].kind=quadrant(x,y)==0?TileKind::EMPTY:TileKind::LOCKED;
  privates_[p]=PrivateState{};privates_[p].inventories.resize(1);privates_[p].inventory_order.resize(1);
 }
 for(int i=0;i<N_PRODUCTS;i++){market_.inventory[i]=MP[i].I0;market_.prices[i]=MP[i].base;}
}
Position Simulator::spawn_hand(const Farm& f)const{
 int h=cfg_.board_size/2;Position ps[4]={{(int16_t)(h-1),(int16_t)(h-1)},{(int16_t)h,(int16_t)(h-1)},{(int16_t)(h-1),(int16_t)h},{(int16_t)h,(int16_t)h}};int c[4]={};
 auto add=[&](Position p){for(int i=0;i<4;i++)if(p.x==ps[i].x&&p.y==ps[i].y)c[i]++;};add(f.farmer);for(auto p:f.hands)add(p);
 int b=0;for(int i=1;i<4;i++)if(c[i]<c[b])b=i;return ps[b];
}
int Simulator::market_price(int i,int inv)const{
 const auto&p=MP[i];double price;
 if(inv<p.I0){double amp=p.bt*p.base/shape(p.below,p.T,p.T);price=p.base+amp*shape(p.below,p.I0-inv,p.T);}
 else{double amp=p.at*p.base/shape(p.above,p.T,p.T);price=p.base-amp*shape(p.above,inv-p.I0,p.T);}
 return std::max(1,(int)std::nearbyint(price));
}
void Simulator::refresh_prices(){for(int i=0;i<N_PRODUCTS;i++)market_.prices[i]=market_price(i,market_.inventory[i]);}

void Simulator::apply_unit(int p,int idx,const Action&a,int day){
 Farm& f=farms_[p];PrivateState& pr=privates_[p];
 if(idx<0||idx>(int)f.hands.size())return;Position pos=idx?f.hands[idx-1]:f.farmer;
 ensure_inventory(pr,idx);auto& inv=pr.inventories[idx];
 auto move=[&](int dx,int dy){int nx=pos.x+dx,ny=pos.y+dy;if(nx>=0&&ny>=0&&nx<cfg_.board_size&&ny<cfg_.board_size){Position q{(int16_t)nx,(int16_t)ny};if(idx)f.hands[idx-1]=q;else f.farmer=q;}};
 if(a.op==Op::NORTH){move(0,-1);return;}if(a.op==Op::SOUTH){move(0,1);return;}if(a.op==Op::EAST){move(1,0);return;}if(a.op==Op::WEST){move(-1,0);return;}if(a.op==Op::PASS)return;
 Tile& t=f.tiles[tile_index(pos.x,pos.y)];int item=(int)a.item,n=std::max(0,a.quantity);
 if(a.op==Op::DROP){if(!shed_adjacent(pos))return;for(int i:pr.inventory_order[idx]){int room=std::max(0,cfg_.shed_capacity-shed_sum(pr));int take=std::min(inv[i],room);pr.shed[i]+=take;inv[i]=0;}pr.inventory_order[idx].clear();return;}
 if(a.op==Op::PICKUP){if(!shed_adjacent(pos)||item<0||item>=N_ITEMS||n<=0)return;n=std::min(n,pr.shed[item]);if(n<=0)return;pr.shed[item]-=n;inventory_add(pr,idx,item,n);return;}
 if(a.op==Op::PLACE){
  if(item>=9&&item<12&&t.kind==ANIMALS[item-9].structure&&t.animal==Item::NONE){if(inventory_take(pr,idx,item,1)){t=Tile{};t.kind=TileKind::ANIMAL;t.animal=(Item)item;t.placed_day=day;}return;}
  if(shed_adjacent(pos)&&item>=0&&item<N_ITEMS&&n>0){n=std::min(n,inv[item]);n=std::min(n,std::max(0,cfg_.shed_capacity-shed_sum(pr)));if(n>0){inventory_take(pr,idx,item,n);pr.shed[item]+=n;}}return;
 }
 if(t.kind==TileKind::LOCKED)return;
 if(a.op==Op::PLANT){if(item<0||item>=N_CROPS||t.kind!=TileKind::EMPTY||pr.seeds[item]<=0)return;pr.seeds[item]--;t=Tile{};t.kind=TileKind::PLANT;t.crop=(Item)item;t.planted_day=day;t.consecutive_unwatered=1;t.yield_units=CROPS[item].ongoing?0:1;t.max_lifespan_step=CROPS[item].ongoing?-1:(day+CROPS[item].max_day+1)*cfg_.turns_per_day;return;}
 if(a.op==Op::WATER){if(t.kind!=TileKind::PLANT||t.watered_today)return;t.watered_today=true;auto&c=CROPS[(int)t.crop];int age=day-t.planted_day,ws=(c.max_day+1)/2;if(!c.ongoing&&age>=ws&&age<=c.max_day)t.yield_units=std::min(c.max_yield,t.yield_units+(t.fertilized_until_day>=day?2:1));return;}
 if(a.op==Op::HARVEST){if(t.yield_units<=0)return;if(t.kind==TileKind::PLANT){auto&c=CROPS[(int)t.crop];if(day-t.planted_day<c.first_day)return;inventory_add(pr,idx,(int)t.crop,t.yield_units);t.yield_units=0;if(!c.ongoing)t=Tile{};}else if(t.kind==TileKind::ANIMAL){inventory_add(pr,idx,ANIMALS[(int)t.animal-9].product,t.yield_units);t.yield_units=0;}return;}
 if(a.op==Op::FERTILIZE){if(t.kind==TileKind::PLANT&&inventory_take(pr,idx,8,1)){t.fertilized_until_day=std::max<int>(t.fertilized_until_day,day+2);}return;}
 if(a.op==Op::DIG){if(t.kind!=TileKind::EMPTY&&t.kind!=TileKind::ANIMAL)t=Tile{};return;}
 if(a.op==Op::BUILD_COOP){if(t.kind==TileKind::EMPTY)t.kind=TileKind::COOP;return;}
 if(a.op==Op::BUILD_PASTURE){if(t.kind==TileKind::EMPTY)t.kind=TileKind::PASTURE;return;}
 if(a.op==Op::FEED){if(t.kind==TileKind::ANIMAL&&!t.fed_today&&inventory_take(pr,idx,0,1)){t.fed_today=true;}return;}
 if(a.op==Op::COLLECT_FERTILIZER){if(t.kind==TileKind::ANIMAL&&t.fertilizer_available){t.fertilizer_available=false;inventory_add(pr,idx,8,1);}return;}
 if(a.op==Op::CARE){if(t.kind==TileKind::ANIMAL&&!t.cared_today)t.cared_today=true;return;}
}

bool Simulator::commit_unit(Op op,int item,int price,int p){auto&f=farms_[p];auto&pr=privates_[p];
 if(op==Op::SELL){if(item<0||item>=N_PRODUCTS||pr.shed[item]<=0)return false;pr.shed[item]--;f.money+=price;if(price>1)market_.inventory[item]++;return true;}
 if(op==Op::BUY_PRODUCT){if((item!=0&&item!=8)||f.money<price||shed_sum(pr)>=cfg_.shed_capacity)return false;f.money-=price;pr.shed[item]++;market_.inventory[item]--;return true;}
 if(op==Op::BUY_SEED){if(item<0||item>=N_CROPS||f.money<price)return false;f.money-=price;pr.seeds[item]++;return true;}
 if(op==Op::BUY_ANIMAL){if(item<9||item>=12||f.money<price||shed_sum(pr)>=cfg_.shed_capacity)return false;f.money-=price;pr.shed[item]++;return true;}return false;
}

void Simulator::process_market(const std::array<PlayerAction,2>& aa){
 size_t ml=std::min<size_t>(cfg_.max_market_orders,std::max(aa[0].market.size(),aa[1].market.size()));
 for(int p=0;p<2;p++){last_market_fills_[p].assign(aa[p].market.size(),0);last_market_cash_shortfalls_[p].assign(aa[p].market.size(),0.);}
 for(size_t oi=0;oi<ml;oi++){
  Action os[2];bool active[2]={false,false};for(int p=0;p<2;p++)if(oi<aa[p].market.size()){os[p]=aa[p].market[oi];active[p]=os[p].quantity>0;}
  for(int p=0;p<2;p++)if(active[p]&&(os[p].op==Op::HIRE||os[p].op==Op::BUY_LAND)){
   auto&f=farms_[p];auto&pr=privates_[p];if(os[p].op==Op::HIRE){int cost=cfg_.farm_hand_cost_mult*fib(f.hires_today);if(f.money>=cost){f.money-=cost;f.hires_today++;f.hands.push_back(spawn_hand(f));pr.inventories.push_back({});pr.inventory_order.push_back({});last_market_fills_[p][oi]=1;}else last_market_cash_shortfalls_[p][oi]=std::max(0.,cost-f.money);}
   else{int extras=__builtin_popcount((unsigned)f.unlocked_mask)-1;if(extras<3&&f.money>=LAND_PRICE[extras]){f.money-=LAND_PRICE[extras];int q=extras+1;f.unlocked_mask|=1<<q;for(int y=0;y<cfg_.board_size;y++)for(int x=0;x<cfg_.board_size;x++)if(quadrant(x,y)==q&&f.tiles[tile_index(x,y)].kind==TileKind::LOCKED)f.tiles[tile_index(x,y)]=Tile{};last_market_fills_[p][oi]=1;}else if(extras<3)last_market_cash_shortfalls_[p][oi]=std::max(0.,LAND_PRICE[extras]-f.money);}
   active[p]=false;
  }
  int remaining[2]={os[0].quantity,os[1].quantity};
  while(active[0]||active[1]){
   int price[2]={};bool quoted[2]={};
   for(int p=0;p<2;p++)if(active[p]&&remaining[p]>0){int i=(int)os[p].item;if(os[p].op==Op::SELL&&i>=0&&i<N_PRODUCTS){price[p]=market_price(i,market_.inventory[i]);quoted[p]=true;}else if(os[p].op==Op::BUY_PRODUCT&&(i==0||i==8)){price[p]=market_price(i,market_.inventory[i]-1);quoted[p]=true;}else if(os[p].op==Op::BUY_SEED&&i>=0&&i<N_CROPS){price[p]=CROPS[i].seed;quoted[p]=true;}else if(os[p].op==Op::BUY_ANIMAL&&i>=9&&i<12){price[p]=ANIMALS[i-9].cost;quoted[p]=true;}else active[p]=false;}
   if(!quoted[0]&&!quoted[1])break;bool any=false;for(int p=0;p<2;p++)if(quoted[p]){if(commit_unit(os[p].op,(int)os[p].item,price[p],p)){remaining[p]--;last_market_fills_[p][oi]++;any=true;if(remaining[p]<=0)active[p]=false;}else{auto&f=farms_[p];auto&pr=privates_[p];bool cash_limited=(os[p].op==Op::BUY_SEED&&f.money<price[p])||(os[p].op==Op::BUY_ANIMAL&&shed_sum(pr)<cfg_.shed_capacity&&f.money<price[p]);if(cash_limited)last_market_cash_shortfalls_[p][oi]=std::max(0.,remaining[p]*double(price[p])-f.money);active[p]=false;}}if(!any)break;
  }
  refresh_prices();
 }
}

void Simulator::town_consume(int s){
 if(s%std::max(1,cfg_.town_shop_sell_interval)==0)for(int sh:shops_){auto take=[&](int i,int n=1){market_.inventory[i]-=n;};switch(sh){case 0:take(5);take(0);break;case 1:take(5);take(0);take(3);break;case 2:take(0);take(1);take(2);take(3);break;case 3:take(3);take(6);take(0);break;case 4:take(1,2);break;case 5:take(6);take(2);take(0);break;case 6:take(3);take(6);break;case 7:take(7,2);break;}}
 if(s%std::max(1,cfg_.town_center_sell_interval)==0)for(int i=0;i<8;i++)market_.inventory[i]--;
 refresh_prices();
}
void Simulator::decay_plants(int s){for(auto&f:farms_)for(auto&t:f.tiles)if(t.kind==TileKind::PLANT&&t.max_lifespan_step>=0&&s>=t.max_lifespan_step&&(s-t.max_lifespan_step)%2==0){if(--t.yield_units<=0){t=Tile{};t.kind=TileKind::WEED;}}}
void Simulator::end_of_day(int day){PythonRandom rng((seed_*1000003ULL)^uint64_t(day));
 for(int p=0;p<2;p++){auto&f=farms_[p];auto&pr=privates_[p];int nd=day+1;
  for(auto&t:f.tiles){if(t.kind==TileKind::PLANT){bool watered=t.watered_today;t.consecutive_unwatered=watered?0:t.consecutive_unwatered+1;t.watered_today=false;if(t.consecutive_unwatered>=2){t=Tile{};t.kind=TileKind::WEED;}else{auto&c=CROPS[(int)t.crop];if(c.ongoing){int ds=nd-t.planted_day-c.first_day;if(ds>=0&&ds%c.interval==0){int pc=ds/c.interval+1;if(pc<=c.max_yield){bool fert=watered&&t.fertilized_until_day>=day;t.yield_units=std::min(c.max_yield,t.yield_units+(fert?2:1));if(pc==c.max_yield)t.max_lifespan_step=(nd+1)*cfg_.turns_per_day;}}}}}
   else if(t.kind==TileKind::ANIMAL){t.consecutive_unfed=t.fed_today?0:t.consecutive_unfed+1;if(t.consecutive_unfed>=2){TileKind k=ANIMALS[(int)t.animal-9].structure;t=Tile{};t.kind=k;}else{auto&a=ANIMALS[(int)t.animal-9];int ds=nd-t.placed_day-a.first_day;if(ds>=0&&ds%a.interval==0){int bonus=t.fed_today?t.pending_care_bonus:0;t.yield_units=std::min(a.max_held,t.yield_units+1+bonus);t.pending_care_bonus=0;}if(t.cared_today&&t.fed_today)t.pending_care_bonus++;t.fertilizer_available=true;t.fed_today=false;t.cared_today=false;}}
  }
  for(auto&t:f.tiles)if(t.kind==TileKind::EMPTY&&rng.random()<cfg_.weed_spawn_chance)t.kind=TileKind::WEED;
  for(size_t u=0;u<pr.inventories.size();u++){auto&inv=pr.inventories[u];for(int i:pr.inventory_order[u]){int take=std::min(inv[i],std::max(0,cfg_.shed_capacity-shed_sum(pr)));pr.shed[i]+=take;last_end_of_day_overflow_[p]+=std::max(0,inv[i]-take);inv[i]=0;}}
  f.farmer=default_spawn();f.hands.clear();f.hires_today=0;pr.inventories.clear();pr.inventories.resize(1);pr.inventory_order.clear();pr.inventory_order.resize(1);
 }
 int nd=day+1;if(nd>0&&nd%std::max(1,cfg_.town_shop_unlock_interval)==0&&shops_.size()<8)shops_.push_back((int8_t)rng.randbelow(8));
}
void Simulator::apply_unit_phase(const std::array<PlayerAction,2>& aa){int day=step_/cfg_.turns_per_day;
 for(int p=0;p<2;p++){std::array<int,5>demand{};for(auto&a:aa[p].units)if(a.op==Op::PLANT&&(int)a.item>=0&&(int)a.item<5)demand[(int)a.item]++;std::array<bool,5>blocked{};for(int i=0;i<5;i++)blocked[i]=demand[i]>privates_[p].seeds[i];
  size_t n=std::min(aa[p].units.size(),farms_[p].hands.size()+1);for(size_t u=0;u<n;u++){Action a=aa[p].units[u];if(a.op==Op::PLANT&&(int)a.item>=0&&(int)a.item<5&&blocked[(int)a.item])a.op=Op::PASS;apply_unit(p,u,a,day);}}
}
Simulator Simulator::preview_unit_phase(const std::array<PlayerAction,2>& aa)const{Simulator out=*this;if(!out.done_)out.apply_unit_phase(aa);return out;}
void Simulator::step(const std::array<PlayerAction,2>& aa){if(done_)return;last_end_of_day_overflow_={0,0};int day=step_/cfg_.turns_per_day;
 apply_unit_phase(aa);
 process_market(aa);town_consume(step_);decay_plants(step_);if((step_+1)%cfg_.turns_per_day==0)end_of_day(day);step_++;if(step_-1>=cfg_.episode_steps-2)done_=true;
}
} // namespace fastkag
