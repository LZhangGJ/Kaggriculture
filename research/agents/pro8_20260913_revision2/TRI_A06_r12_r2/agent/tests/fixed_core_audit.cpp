// Test-only library. Replays the IMMUTABLE parent to acquire a real own policy
// context, then copies its unchanged executor to compile actual fixture stocks.
// Does not implement or score an opponent, RNG, complete game, or future replay.
#ifndef AUDIT_PARENT_BRIDGE
#error Supply the immutable parent bridge.cpp path at compile time.
#endif
#include AUDIT_PARENT_BRIDGE
extern "C" const char* audit_fixed_routes(void* p,const double* input,size_t count) {
 auto& h=*static_cast<Handle*>(p);
 try {
  Cursor r{input,count};int step=r.integer(),day=r.integer(),hour=r.integer(),seat=r.integer();
  auto a=r.farm(),b=r.farm();auto priv=r.priv();fastkag::Market market;
  for(auto& x:market.inventory)x=r.integer();for(auto& x:market.prices)x=r.integer();
  std::vector<int8_t> shops;int n=r.count();for(int i=0;i<n;i++)shops.push_back(r.integer());
  if(r.i!=count||day!=h.policy.live.core.day||hour<1||hour>23)throw std::runtime_error("same-day fixture only");
  dp7::View v{step,day,hour,seat==0?a:b,seat==0?b:a,priv,market,shops};
  auto core=h.policy.live.core;core.compile(v);
  std::ostringstream j;j<<"{\"day\":"<<day<<",\"hour\":"<<hour<<",\"drop_count\":"<<core.actual_drop<<",\"routes\":[";
  for(size_t u=0;u<core.plans.size();++u){if(u)j<<',';j<<'[';const auto& plan=core.plans[u];
   for(size_t k=plan.index;k<plan.a.size();++k){if(k>plan.index)j<<',';auto x=plan.a[k];j<<'['<<int(x.op)<<','<<int(x.item)<<','<<x.quantity<<','<<plan.target[k]<<']';}j<<']';}
  j<<"]}";h.text=j.str();return h.text.c_str();
 }catch(const std::exception& e){h.text=std::string("ERROR: ")+e.what();return h.text.c_str();}
}
