#include "r3/r3_checks.cpp"
extern "C" const char*td_fix1_tests(void*ptr,const double*in,size_t n){auto&h=*static_cast<Handle*>(ptr);h.text=with_view(ptr,in,n,[](Handle&h,const dp7::View&v){
 std::ostringstream j;j<<"{\"cases\":[";int cases=0;
 for(int d:{0,14,15,25,28})for(int have:{0,11,20,25})for(int selected:{0,20}){
  auto f=v.own;auto p=v.priv;f.money=100000;p.shed[dp7::F]=have;p.shed[dp7::W]=0;
  dp7::View ob{d*24,d,0,f,v.opponent,p,v.market,v.shops};auto c=h.policy.live.core;c.day=d;c.planned_land=std::popcount(unsigned(f.unlocked_mask));dp7::Job task;task.needs[dp7::F]=selected;task.needs[dp7::W]=3;
  auto orders=c.orders(ob,{task});int got=0,wheat=0;
  for(auto x:orders){if(x.a.op==fastkag::Op::BUY_PRODUCT&&int(x.a.item)==dp7::F){got+=x.a.quantity;need(x.priority==4,"fert priority changed");}if(x.a.op==fastkag::Op::BUY_PRODUCT&&int(x.a.item)==dp7::W){wheat+=x.a.quantity;need(x.priority==0,"feed priority changed");}}
  need(got==std::max(0,selected-have),"selected late fertilizer omitted or surplus bought");need(wheat==3,"feed omitted");
  auto poor=c.admit(ob,orders,0);need(poor.empty(),"zero money admitted purchase");auto paid=c.admit(ob,orders,100000);int paid_f=0;for(auto x:paid)if(x.a.op==fastkag::Op::BUY_PRODUCT&&int(x.a.item)==dp7::F)paid_f+=x.a.quantity;need(paid_f==got,"funded deficit not admitted");
  if(cases++)j<<",";j<<"{\"day\":"<<d<<",\"stock\":"<<have<<",\"selected\":"<<selected<<",\"bought\":"<<got<<",\"pass\":true}";
 }
 j<<"],\"case_count\":"<<cases<<",\"new_games\":0}";return j.str();});return h.text.c_str();}
