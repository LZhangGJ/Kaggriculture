// Reuse the validated observation decoder only; legacy entry is never called.
#define dp27_abi legacy_dp27_abi
#define dp27_error legacy_dp27_error
#define dp27_reset legacy_dp27_reset
#define dp27_act legacy_dp27_act
#include "../../../submission/pending_dp27_fixed_macro_dynamic_20260904/bridge.cpp"
#undef dp27_abi
#undef dp27_error
#undef dp27_reset
#undef dp27_act
#include "switch_model.hpp"
namespace {
int active[2]{-1,-1},switched_at[2]{-1,-1},missing_count[2]{};
std::array<float,146>last_features[2]{};
}
extern "C" const char*s8_error(){return error.c_str();}
extern "C" void s8_reset(){legacy_dp27_reset();for(int i=0;i<2;i++){active[i]=-1;switched_at[i]=-1;missing_count[i]=0;last_features[i].fill(0);}}
extern "C" int s8_target(int seat){return active[seat];}
extern "C" int s8_switch_day(int seat){return switched_at[seat];}
extern "C" int s8_features(int seat,float*out){for(int i=0;i<146;i++)out[i]=last_features[seat][i];return 146;}
extern "C" int s8_act(const double*input,size_t length,const float*base_features,size_t feature_length,int32_t*output,size_t capacity){
 try{
  Cursor r{input,length};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  if(seat<0||seat>1||step<0||day!=step/24||hour!=step%24)throw std::runtime_error("clock/seat");
  auto a=r.farm(),b=r.farm();auto priv=r.priv();Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
  std::vector<int8_t>shops;int count=r.count();for(int j=0;j<count;j++)shops.push_back(r.integer());if(r.i!=length)throw std::runtime_error("trailing input");
  dp7::View view{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  if(step==0||!controllers[seat]||step<last[seat]){controllers[seat]=std::make_unique<dp7::Controller>(frozen_params());last[seat]=-1;active[seat]=switch_opening;switched_at[seat]=-1;missing_count[seat]=0;}
  if(step!=last[seat]+1)throw std::runtime_error("nonsequential observation");auto&own=*controllers[seat];
  if(hour==0&&switched_at[seat]<0)for(auto&rule:switch_rules)if(rule.day==day){
   if(feature_length!=129)throw std::runtime_error("129 observed features required");auto&x=last_features[seat];for(int i=0;i<129;i++)x[i]=base_features[i];
   dp7::Counts targets{};for(auto[pos,k]:own.target)if(k>=0&&k<12)targets[k]++;int at=129;for(int v:targets)x[at++]=v;
   x[at++]=own.phase;x[at++]=own.queue.size();x[at++]=own.planned_land;x[at++]=own.feed_stock_target;x[at++]=missing_count[seat];
   for(float v:x)if(!std::isfinite(v))throw std::runtime_error("nonfinite feature");int node=0;while(rule.left[node]>=0)node=x[rule.feature[node]]<=rule.threshold[node]?rule.left[node]:rule.right[node];
   int next=rule.target[node];if(next!=active[seat]){active[seat]=next;switched_at[seat]=day;}break;
  }
  if(hour==0&&day<=28){auto menu=dp7branch::generate(own,view);if(menu.empty()||menu[0].family!="KEEP")throw std::runtime_error("KEEP absent");
   auto&recipe=switch_plans.at(active[seat]).at(day);int choice=0;bool found=false;
   for(int i=0;i<int(menu.size());i++)if(menu[i].family==recipe.family&&menu[i].kind==recipe.kind&&menu[i].amount==recipe.amount){choice=i;found=true;break;}
   missing_count[seat]+=!found;own=menu[choice].controller;
  }
  auto action=own.act(view);size_t size=2+3*(action.units.size()+action.market.size());if(size>capacity)throw std::runtime_error("output too small");
  output[0]=action.units.size();output[1]=action.market.size();int at=2;auto put=[&](const Action&a){output[at++]=int(a.op);output[at++]=int(a.item);output[at++]=a.quantity;};
  for(auto&a:action.units)put(a);for(auto&a:action.market)put(a);last[seat]=step;return size;
 }catch(const std::exception&e){error=e.what();return -1;}
}
