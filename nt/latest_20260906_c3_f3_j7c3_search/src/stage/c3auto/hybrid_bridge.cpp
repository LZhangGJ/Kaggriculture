#include "hybrid.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <cstring>
#include <limits>
#include <type_traits>
#include <sstream>
using namespace fastkag;
namespace {
struct Cursor {
 const double* p;size_t n,i=0;
 double next(){if(i>=n)throw std::runtime_error("truncated observation");return p[i++];}
 int integer(){double x=next();if(!std::isfinite(x)||x<std::numeric_limits<int>::min()||x>std::numeric_limits<int>::max()||x!=int(x))throw std::runtime_error("invalid integer");return int(x);}
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

std::string error,debug;std::unique_ptr<competitive::Hybrid> planners[2];competitive::Config cfg;
}
extern "C" const char*mp_error(){return error.c_str();}
extern "C" void mp_reset(){for(auto&p:planners)p.reset();}
extern "C" int mp_config(const double*x,int n){try{
 static_assert(std::is_trivially_copyable_v<competitive::Config>);
 static_assert(sizeof(competitive::Config)==49*sizeof(double));
 if(!x||n!=49)throw std::invalid_argument("expected exactly49 ordered parameters");
 for(int i=0;i<n;i++)if(!std::isfinite(x[i]))throw std::invalid_argument("non-finite parameter");
 competitive::Config candidate;std::memcpy(&candidate,x,sizeof(candidate));
 if(candidate.new_limit<0||candidate.new_limit>100||candidate.expansion<0||candidate.expansion>20||candidate.service_rounds<0||candidate.service_rounds>8||candidate.rotation_edits<0||candidate.rotation_edits>100||candidate.inventory_dp<0||candidate.inventory_dp>48)throw std::invalid_argument("unsafe search bound");
 if(candidate.investment_blend<0||candidate.investment_blend>1||candidate.investment_scope<1||candidate.investment_scope>2||candidate.investment_timing<0||candidate.investment_timing>2||candidate.investment_start<0||candidate.investment_start>29||candidate.investment_margin<0||candidate.investment_margin>1e6||candidate.crop_dp<0||candidate.crop_dp>3||candidate.crop_start<0||candidate.crop_start>29||candidate.crop_gain<0||candidate.crop_gain>1e6||candidate.crop_action_cost<0||candidate.crop_action_cost>1000||candidate.crop_fert_floor<0||candidate.crop_fert_floor>10000||candidate.crop_max_changes<0||candidate.crop_max_changes>100||candidate.crop_timing<0||candidate.crop_timing>1||candidate.crop_price_mode<0||candidate.crop_price_mode>1)throw std::invalid_argument("unsafe C3 control range");
 if(candidate.investment_consistent<0||candidate.investment_consistent>1||candidate.investment_paid_labor<0||candidate.investment_paid_labor>1)throw std::invalid_argument("unsafe consistent investment range");
 if(candidate.investment_crop_consistent<0||candidate.investment_crop_consistent>1)throw std::invalid_argument("unsafe crop consistency range");
 if(candidate.calendar_samples<0||candidate.calendar_samples>100)throw std::invalid_argument("unsafe observation calendar minimum");
 cfg=candidate;mp_reset();error.clear();return 0;
 }catch(const std::exception&e){error=e.what();return -1;}}
extern "C" const char*mp_debug(int seat){
 if(seat<0||seat>1||!planners[seat]){debug="{}";return debug.c_str();}
 auto&h=*planners[seat];auto&c=h.core;std::ostringstream s;
 s<<"{\"full_plan_calls\":"<<h.model.full_plan_calls<<",\"legacy_choose_calls\":"<<c.legacy_choose_calls
  <<",\"reference_consults\":"<<h.reference_consults<<",\"full_crop_assets\":"<<h.full_crop_assets
  <<",\"full_crop_investments\":"<<h.full_crop_investments<<",\"full_price_updates\":"<<h.full_price_updates
  <<",\"portfolio_evaluations\":"<<h.model.portfolios<<",\"opening_animals\":"<<c.p.opening_animals.size()
  <<",\"opening_crops\":"<<std::accumulate(c.p.opening_crops.begin(),c.p.opening_crops.end(),0)
  <<",\"autonomous_start\":"<<(c.p.autonomous_start?"true":"false")<<",\"new_limit\":"<<h.cfg.new_limit
  <<",\"intraday_admission\":"<<(c.p.intraday_admission?"true":"false")
  <<",\"crop_changes\":"<<h.crop_changes<<",\"crop_checks\":"<<h.crop_checks<<",\"investment_checks\":"<<h.investment_checks<<",\"day\":"<<c.day<<",\"edits\":"<<h.edits<<",\"service_day\":"<<c.animal_service_day<<",\"sale_changes\":"<<h.sale_changes<<",\"uses_team_J7_reference\":"<<(C3_AUTONOMOUS?"false":"true")<<",\"targets\":[";
 bool sep=false;for(auto[pos,k]:c.target){if(sep)s<<",";sep=true;s<<"["<<pos<<","<<k<<"]";}s<<"],\"feed\":[";
 for(int p=0;p<100;p++){if(p)s<<",";s<<int(c.service_feed[p]);}s<<"],\"care\":[";
 for(int p=0;p<100;p++){if(p)s<<",";s<<int(c.service_care[p]);}s<<"],\"forecast_today_prices\":[";
 for(int i=0;i<9;i++){if(i)s<<",";s<<h.model.shadow[std::clamp(c.day,0,29)][i];}s<<"]}";debug=s.str();return debug.c_str();}

extern "C" int mp_act(const double*input,size_t length,int32_t*output,size_t capacity){try{
Cursor r{input,length};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();if(seat<0||seat>1||step<0||day!=step/24||hour!=step%24)throw std::runtime_error("clock/seat");
auto a=r.farm(),b=r.farm();auto priv=r.priv();Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int count=r.count();for(int j=0;j<count;j++)shops.push_back(r.integer());if(r.i!=length)throw std::runtime_error("trailing input");
dp7::View view{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};if(step==0||!planners[seat]||step<=planners[seat]->core.last_step)planners[seat]=std::make_unique<competitive::Hybrid>(cfg);
auto action=planners[seat]->act(view);size_t size=2+3*(action.units.size()+action.market.size());if(size>capacity)throw std::runtime_error("output size");output[0]=action.units.size();output[1]=action.market.size();int at=2;auto put=[&](const Action&a){output[at++]=int(a.op);output[at++]=int(a.item);output[at++]=a.quantity;};for(auto&a:action.units)put(a);for(auto&a:action.market)put(a);return size;
}catch(const std::exception&e){error=e.what();return -1;}}
