#include "../candidate/policy/bridge.cpp"
extern "C" const char* audit_services(void*p,const double*input,size_t count){
 auto&h=*static_cast<Handle*>(p);auto&c=h.policy.live;
 Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
 auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();std::vector<int8_t>shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
 dp7::View o{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
 auto jobs=c.core.jobs(o);std::ostringstream out;out.precision(17);
 out<<"{\"day\":"<<day<<",\"step\":"<<step<<",\"predicted\":"<<c.predicted<<",\"labor\":"<<c.portfolio.labor[day]<<",\"wages\":"<<c.model.wages(c.portfolio.labor[day])<<",\"paths\":[";
 bool sep=false;for(int pos=0;pos<100;pos++){
  const auto&t=o.own.tiles[pos];if(!dp7::plant(t)&&!dp7::animal(t))continue;
  if(sep)out<<",";sep=true;
  auto&p=c.paths[pos];out<<"{\"pos\":"<<pos<<",\"kind\":"<<(dp7::animal(t)?int(t.animal):int(t.crop))<<",\"birth\":"<<(dp7::animal(t)?t.placed_day:t.planted_day)<<",\"path_kind\":"<<p.kind<<",\"end\":"<<p.end<<",\"service_feed\":"<<int(c.core.service_feed[pos])<<",\"service_care\":"<<int(c.core.service_care[pos])<<",\"water\":"<<int(c.core.crop_water[pos])<<",\"fertilize\":"<<int(c.core.crop_fertilize[pos])<<",\"first_labor\":"<<p.labor[day]<<",\"path_f\":[";
  for(int d=0;d<30;d++){if(d)out<<",";out<<"[";for(int i=0;i<9;i++){if(i)out<<",";out<<p.f[d][i];}out<<"]";}
  out<<"],\"jobs\":[";bool js=false;for(const auto&j:jobs)if(j.pos==pos)for(auto ac:j.actions){if(js)out<<",";js=true;out<<int(ac.op);}out<<"]}";
 }out<<"]}";h.text=out.str();return h.text.c_str();
}
