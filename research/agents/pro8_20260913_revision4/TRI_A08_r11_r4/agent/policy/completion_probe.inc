// Read-only test seam: fresh isolated Controller; never mutates a live agent.
extern "C" const char*td_completion_snapshot(const double*input,size_t count,
 const int32_t*raw,size_t raw_count,const int32_t*job,size_t job_count,double shadow,int mode){
 static thread_local std::string text;
 try{
  if(!input||!raw||!job||!std::isfinite(shadow)||shadow<0||mode<0||mode>2)throw std::runtime_error("invalid argument");
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  if(step<0||step>=719||day!=step/24||hour!=step%24||seat<0||seat>1)throw std::runtime_error("clock/seat");
  auto a=r.farm(),b=r.farm();auto priv=r.priv();Market market;for(auto&x:market.inventory)x=r.integer();for(auto&x:market.prices)x=r.integer();
  std::vector<int8_t>shops;int n=r.count();if(n>8)throw std::runtime_error("shops");for(int i=0;i<n;i++){int x=r.integer();if(x<0||x>7)throw std::runtime_error("shop");shops.push_back(x);}
  if(r.i!=count)throw std::runtime_error("trailing input");
  dp7::View v{step,day,hour,seat?b:a,seat?a:b,priv,market,shops};
  if(priv.inventories.size()!=v.own.hands.size()+1||priv.inventories.size()>16)throw std::runtime_error("units");
  size_t at=0;auto take=[&](){if(at>=raw_count)throw std::runtime_error("truncated plan");return raw[at++];};
  int units=take();if(units!=int(priv.inventories.size()))throw std::runtime_error("plan units");std::vector<dp7::Plan>ps(units);
  for(auto&p:ps){int k=take();if(k<0||k>128)throw std::runtime_error("plan length");for(int i=0;i<k;i++){
   int op=take(),item=take(),q=take(),target=take();if(op<0||op>17||item< -1||item>11||q<0||q>10000||target< -1||target>99)throw std::runtime_error("plan atom");
   p.a.push_back(dp7::action(Op(op),item,q));p.target.push_back(target);
  }}if(at!=raw_count)throw std::runtime_error("trailing plan");
  at=0;auto get=[&](){if(at>=job_count)throw std::runtime_error("truncated job");return job[at++];};dp7::Job whole,part;
  whole.pos=part.pos=get();if(whole.pos<0||whole.pos>=100)throw std::runtime_error("plot");
  for(auto*j:{&whole,&part}){int k=get();if(k<0||k>16)throw std::runtime_error("job length");for(int i=0;i<k;i++){int op=get(),item=get(),q=get();if(op<0||op>17||item< -1||item>11||q<=0||q>10000)throw std::runtime_error("job action");j->actions.push_back(dp7::action(Op(op),item,q));}}
  if(at!=job_count)throw std::runtime_error("trailing job");
  dp7::Controller c;c.day=day;c.phase=3;c.p.action_shadow=shadow;c.p.stepwise_recoordination=true;c.p.shared_task_atoms_v2=true;c.p.fix_logistics=true;
  auto base=ps;auto old=dp7::completion::audit(c,v,base);bool selected=false;
  if(mode==0)selected=dp7::completion::insert(c,v,ps,whole,part);
  else if(mode==1)selected=dp7::completion::close(c,v,ps);
  else {c.plans=ps;for(int pos=0;pos<100;pos++){auto&t=v.own.tiles[pos];if(dp7::plant(t))c.target.emplace_back(pos,int(t.crop));else if(dp7::animal(t))c.target.emplace_back(pos,int(t.animal));}
   c.recoordinate(v);ps=c.plans;for(size_t u=0;u<ps.size();u++)selected|=!dp7::Controller::same_remaining(ps[u],base[u]);}
  auto end=dp7::completion::audit(c,v,ps);std::ostringstream s;s.precision(17);s<<"{\"enabled\":"<<A08_COMPLETE_ROLLING_ROUTES<<",\"selected\":"<<(selected?"true":"false");
  auto audit=[&](const char*name,const dp7::completion::Audit&a){s<<",\""<<name<<"\":{\"known\":"<<a.known<<",\"shape\":"<<a.shape<<",\"executed\":"<<a.executed<<",\"cash\":"<<a.cash<<",\"work\":"<<a.work<<",\"waits\":"<<a.waits<<",\"failed\":"<<a.failed<<",\"ticks\":"<<a.ticks;
   auto arr=[&](const char*n,const auto&xs){s<<",\""<<n<<"\":[";for(size_t i=0;i<xs.size();i++){if(i)s<<",";s<<xs[i];}s<<"]";};arr("sold",a.sold);arr("stranded",a.stranded);arr("deposited",a.deposited);s<<"}";};audit("before",old);audit("after",end);
  auto plans=[&](const char*name,const auto&xs){s<<",\""<<name<<"\":[";for(size_t u=0;u<xs.size();u++){if(u)s<<",";s<<"[";auto&p=xs[u];for(size_t k=p.index;k<p.a.size();k++){if(k!=p.index)s<<",";auto a=p.a[k];s<<"["<<int(a.op)<<","<<int(a.item)<<","<<a.quantity<<","<<p.target[k]<<"]";}s<<"]";}s<<"]";};plans("base",base);plans("best",ps);
  s<<",\"candidates\":"<<c.completion_candidates<<",\"evaluations\":"<<c.completion_evaluations<<",\"horizon_rejects\":"<<c.completion_horizon_rejects<<",\"resource_rejects\":"<<c.completion_resource_rejects<<",\"precedence_rejects\":"<<c.completion_precedence_rejects<<",\"execution_rejects\":"<<c.completion_execution_rejects<<",\"value_rejects\":"<<c.completion_value_rejects<<",\"bounded_skips\":"<<c.completion_bounded_skips<<"}";text=s.str();
 }catch(const std::exception&e){text=std::string("{\"error\":\"")+e.what()+"\"}";}
 return text.c_str();
}
extern "C" const char*td_completion_json(void*p){
 auto&h=*static_cast<Handle*>(p);const auto&c=h.policy.live.core;std::ostringstream s;s.precision(17);
 s<<"{\"enabled\":"<<A08_COMPLETE_ROLLING_ROUTES<<",\"checks\":"<<c.completion_checks<<",\"candidates\":"<<c.completion_candidates<<",\"evaluations\":"<<c.completion_evaluations<<",\"applied\":"<<c.completion_applied<<",\"closed\":"<<c.completion_closed
  <<",\"horizon_rejects\":"<<c.completion_horizon_rejects<<",\"resource_rejects\":"<<c.completion_resource_rejects<<",\"precedence_rejects\":"<<c.completion_precedence_rejects<<",\"execution_rejects\":"<<c.completion_execution_rejects<<",\"value_rejects\":"<<c.completion_value_rejects<<",\"bounded_skips\":"<<c.completion_bounded_skips
  <<",\"last_step\":"<<c.completion_last_step<<",\"last_unit\":"<<c.completion_last_unit<<",\"last_pos\":"<<c.completion_last_pos<<",\"conditional_cash_delta\":"<<c.completion_last_cash_delta<<",\"work_cost\":"<<c.completion_last_work_cost<<"}";
 h.text=s.str();return h.text.c_str();
}
