// Offline diagnostics: reads only legal current View and our controller state.
// Include the actual production bridge; this library is NOT the submitted binary.
#include "bridge.cpp"
extern "C" const char* td_land_audit(void*p,const double*input,size_t count){
 auto&h=*static_cast<Handle*>(p);
 try{
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
  for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
  std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
  if(r.i!=count)throw std::runtime_error("trailing audit input");
  dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  auto&c=h.policy.live;int owned=std::popcount(unsigned(v.own.unlocked_mask));
  std::ostringstream j;j.precision(17);
  auto curve=[&](const auto&x){j<<"[";for(size_t i=0;i<x.size();i++){if(i)j<<",";j<<x[i];}j<<"]";};
  auto asset=[&](const competitive::Asset&x){j<<"{\"kind\":"<<x.kind<<",\"end\":"<<x.end<<",\"first_cost\":"<<x.first_cost<<",\"fixed\":";curve(x.fixed);j<<",\"labor\":";curve(x.labor);j<<",\"flow\":[";for(int d=0;d<30;d++){if(d)j<<",";curve(x.f[d]);}j<<"]}";};
  j<<"{\"step\":"<<step<<",\"day\":"<<day<<",\"owned\":"<<owned<<",\"planned_land\":"<<c.core.planned_land<<",\"targets\":[";
  bool sep=false;competitive::Asset stripped=c.portfolio;
  for(auto[pos,k]:c.core.target){if(sep)j<<",";sep=true;
   j<<"{\"pos\":"<<pos<<",\"kind\":"<<k<<",\"quad\":"<<dp7::quad(pos)<<",\"locked\":"<<(v.own.tiles[pos].kind==fastkag::TileKind::LOCKED)<<",\"length\":"<<c.length[pos]<<",\"successor\":"<<c.successor[pos]<<",\"crop_age\":"<<c.core.triad_crop_age[pos]<<",\"path\":";asset(c.paths[pos]);j<<"}";
   if(dp7::quad(pos)>=owned)competitive::add(stripped,c.paths[pos],-1);
  }
  double land_cost=0;for(int q=owned;q<c.core.planned_land;q++)land_cost+=dp7::next_land_cost(q);
  stripped.fixed[day]+=land_cost;
  j<<"],\"land_cost\":"<<land_cost<<",\"value_with_land\":"<<c.model.value(v,c.portfolio)<<",\"value_strip_newland_paths_and_refund\":"<<c.model.value(v,stripped)<<",\"portfolio\":";asset(c.portfolio);j<<"}";
  h.text=j.str();return h.text.c_str();
 }catch(const std::exception&e){h.text=std::string("ERROR: ")+e.what();return h.text.c_str();}
}
