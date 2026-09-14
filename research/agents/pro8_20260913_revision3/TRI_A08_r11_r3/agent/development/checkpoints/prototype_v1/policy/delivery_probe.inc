// Read-only test object: verifies the EXACT linked proposal, no agent mutation.
extern "C" const char*td_delivery_proposal_snapshot(const double*input,size_t count,
 const int32_t*raw,size_t raw_count,double shadow,int authorize){
 static thread_local std::string result;
 try{
  if(!input||!raw||!std::isfinite(shadow)||shadow<0)throw std::runtime_error("invalid argument");
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  if(step<0||step>=719||day!=step/24||hour!=step%24||seat<0||seat>1)throw std::runtime_error("clock/seat");
  auto fa=r.farm(),fb=r.farm();auto priv=r.priv();Market market;
  for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
  std::vector<int8_t>shops;int n=r.count();if(n>8)throw std::runtime_error("shops");
  for(int j=0;j<n;j++){int i=r.integer();if(i<0||i>7)throw std::runtime_error("shop id");shops.push_back(i);}
  if(r.i!=count)throw std::runtime_error("trailing observation");
  const auto&own=seat?fb:fa;const auto&opponent=seat?fa:fb;
  if(priv.inventories.size()!=own.hands.size()+1||priv.inventories.size()>16)throw std::runtime_error("units");
  dp7::View v{step,day,hour,own,opponent,priv,market,shops};
  dp7::Controller ctl;ctl.day=day;ctl.phase=3;ctl.p.midroute_delivery=true;ctl.p.action_shadow=shadow;
  ctl.delivery_observation_step=authorize?step:-1;
  size_t k=0;auto next=[&](){if(k>=raw_count)throw std::runtime_error("truncated plan");return raw[k++];};
  int units=next();if(units!=int(priv.inventories.size()))throw std::runtime_error("plan units");
  ctl.plans.resize(units);
  for(auto&p:ctl.plans){int size=next();if(size<0||size>128)throw std::runtime_error("plan length");
   for(int j=0;j<size;j++){int op=next(),i=next(),q=next(),t=next();
    if(op<0||op>17||i< -1||i>11||q<0||q>10000||t< -1||t>99)throw std::runtime_error("plan atom");
    p.a.push_back(dp7::action(Op(op),i,q));p.target.push_back(t);
   }
  }if(k!=raw_count)throw std::runtime_error("trailing plan");
  auto c=dp7::deliveryhandoff::propose(ctl,v);std::ostringstream s;s.precision(17);
  s<<"{\"selected\":"<<(c.selected?"true":"false")<<",\"donor\":"<<c.donor<<",\"receiver\":"<<c.receiver
   <<",\"plot\":"<<c.plot<<",\"depot\":"<<c.depot<<",\"quantity\":"<<c.quantity<<",\"extra_actions\":"<<c.extra_actions
   <<",\"arrival\":"<<c.arrival<<",\"overflow_before\":"<<c.overflow_before<<",\"overflow_after\":"<<c.overflow_after
   <<",\"candidates\":"<<c.candidates<<",\"deadline_rejects\":"<<c.deadline_rejects<<",\"order_rejects\":"<<c.order_rejects
   <<",\"capacity_rejects\":"<<c.capacity_rejects<<",\"economy_rejects\":"<<c.economy_rejects
   <<",\"conditional_quote\":"<<c.conditional_quote<<",\"score\":"<<c.score;
  auto dump=[&](const char*name,const std::vector<dp7::Plan>&ps){s<<",\""<<name<<"\":[";
   for(size_t u=0;u<ps.size();u++){if(u)s<<",";s<<"[";const auto&p=ps[u];
    for(size_t j=0;j<p.a.size();j++){if(j)s<<",";auto a=p.a[j];s<<"["<<int(a.op)<<","<<int(a.item)<<","<<a.quantity<<","<<p.target[j]<<"]";}s<<"]";}s<<"]";};
  dump("base",c.base);dump("best",c.best);s<<"}";result=s.str();
 }catch(const std::exception&e){result=std::string("{\"error\":\"")+e.what()+"\"}";}
 return result.c_str();
}
