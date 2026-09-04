// Observation-only C ABI. No opponent identity, future seed or replay tape input.
#include "investment_candidates.hpp"
#include "frozen.hpp"
#include <cstdint>
#include <memory>
#include <string>
using namespace fastkag;
namespace {
struct Cursor {
 const double* p;size_t n,i=0;
 double next(){if(i>=n)throw std::runtime_error("truncated observation");return p[i++];}
 int integer(){double x=next();if(!std::isfinite(x)||x!=int(x))throw std::runtime_error("invalid integer");return int(x);}
 int count(){int x=integer();if(x<0||x>4096)throw std::runtime_error("invalid count");return x;}
 Position position(){return {int16_t(integer()),int16_t(integer())};}
 Farm farm(){Farm f;f.money=next();f.farmer=position();int hands=count();for(int j=0;j<hands;j++)f.hands.push_back(position());
  f.unlocked_mask=integer();f.hires_today=integer();f.tiles.resize(100);
  for(auto&t:f.tiles){t.kind=TileKind(integer());t.crop=Item(integer());t.animal=Item(integer());t.planted_day=integer();t.placed_day=integer();
   t.yield_units=integer();t.consecutive_unwatered=integer();t.consecutive_unfed=integer();t.fertilized_until_day=integer();t.pending_care_bonus=integer();
   t.max_lifespan_step=integer();t.watered_today=integer();t.fed_today=integer();t.cared_today=integer();t.fertilizer_available=integer();}
  return f;
 }
 PrivateState priv(){PrivateState p;for(auto&x:p.shed)x=integer();for(auto&x:p.seeds)x=integer();int n=count();p.inventories.resize(n);p.inventory_order.resize(n);
  for(int j=0;j<n;j++){for(auto&x:p.inventories[j])x=integer();int m=count();for(int k=0;k<m;k++)p.inventory_order[j].push_back(integer());}return p;}
};
std::unique_ptr<dp7::Controller> controllers[2];int last[2]{-1,-1};std::string error;
}
extern "C" int dp27_abi(){return 1;}
extern "C" const char*dp27_error(){return error.c_str();}
extern "C" void dp27_reset(){for(int s=0;s<2;s++){controllers[s].reset();last[s]=-1;}}
extern "C" int dp27_act(const double*input,size_t length,int32_t*output,size_t capacity){
 try {
  Cursor r{input,length};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  if(seat<0||seat>1||step<0||day!=step/24||hour!=step%24)throw std::runtime_error("invalid clock/seat");
  auto a=r.farm(),b=r.farm();auto priv=r.priv();Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
  std::vector<int8_t>shops;int count=r.count();for(int j=0;j<count;j++)shops.push_back(r.integer());
  if(r.i!=length)throw std::runtime_error("observation trailing data");
  dp7::View view{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  if(step==0||!controllers[seat]||step<last[seat]){controllers[seat]=std::make_unique<dp7::Controller>(frozen_params());last[seat]=-1;}
  if(step!=last[seat]+1)throw std::runtime_error("nonsequential calls require full persistent state");
  auto&own=*controllers[seat];
  if(hour==0&&day<=28){
   auto menu=dp7branch::generate(own,view);if(menu.empty()||menu[0].family!="KEEP")throw std::runtime_error("KEEP missing");
   auto&recipe=frozen_plan[day];int chosen=0;
   for(int i=0;i<int(menu.size());i++)if(menu[i].family==recipe.family&&menu[i].kind==recipe.kind&&menu[i].amount==recipe.amount){chosen=i;break;}
   own=menu[chosen].controller;
  }
  auto action=own.act(view);size_t size=2+3*(action.units.size()+action.market.size());if(size>capacity)throw std::runtime_error("action buffer overflow");
  output[0]=int(action.units.size());output[1]=int(action.market.size());int at=2;
  auto put=[&](const Action&a){output[at++]=int(a.op);output[at++]=int(a.item);output[at++]=a.quantity;};
  for(auto&a:action.units)put(a);for(auto&a:action.market)put(a);last[seat]=step;return int(size);
 }catch(const std::exception&e){error=e.what();return -1;}
}
