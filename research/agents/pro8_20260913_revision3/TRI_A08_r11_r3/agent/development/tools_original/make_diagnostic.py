from pathlib import Path
import shutil
b=Path('/mnt/data/r3_work');d=b/'diagnostic';shutil.copytree(b/'source_input'/'agent',d)
p=d/'policy/executor/policy.hpp';s=p.read_text();s=s.replace('#include <string>','#include <string>\n#include <sstream>')
s=s.replace('struct Plan{Acts a;', 'inline thread_local const void* delivery_audit_target=nullptr;\ninline thread_local std::string delivery_audit_json="{}";\nstruct Plan{Acts a;')
fn=r'''
 void audit_delivery(const View&o)const{
  if(delivery_audit_target!=this)return;
  std::ostringstream s;s.precision(17);
  s<<"{\"step\":"<<o.step<<",\"phase\":"<<phase<<",\"midroute\":"<<p.midroute_delivery<<",\"safety\":"<<p.shed_safety<<",\"pressure\":"<<p.triad_delivery_pressure<<",\"forecast\":"<<expected_auto_deposit(o)<<",\"units\":[";
  auto counts=[&](const Counts&c){s<<"[";for(int i=0;i<12;i++){if(i)s<<",";s<<c[i];}s<<"]";};
  auto plan=[&](const Plan&pl){s<<"[";for(size_t k=pl.index;k<pl.a.size();k++){if(k!=pl.index)s<<",";s<<"["<<int(pl.a[k].op)<<","<<int(pl.a[k].item)<<","<<pl.a[k].quantity<<","<<pl.target[k]<<"]";}s<<"]";};
  for(size_t u=0;u<plans.size()&&u<o.priv.inventories.size();u++){
   if(u)s<<",";const auto&pl=plans[u];Counts need{};bool drop=false,mixed=false;
   for(size_t k=pl.index;k<pl.a.size();k++){auto a=pl.a[k];if(a.op==Op::DROP)drop=true;if(a.op==Op::FEED)need[W]++;if(a.op==Op::FERTILIZE)need[F]++;if(a.op==Op::PLACE&&int(a.item)>=9)need[int(a.item)]++;}
   for(int i=0;i<12;i++)if(o.priv.inventories[u][i]>0&&(need[i]>0||i>=9))mixed=true;
   int stale=0;auto fixed=repair_plan(o,int(u),pl,stale);
   s<<"{\"u\":"<<u<<",\"pos\":"<<cell(u?o.own.hands[u-1]:o.own.farmer)<<",\"drop\":"<<drop<<",\"mixed\":"<<mixed<<",\"need\":";counts(need);s<<",\"cargo\":";counts(o.priv.inventories[u]);s<<",\"raw\":";plan(pl);s<<",\"repair\":";plan(fixed);s<<"}";
  }s<<"]}";delivery_audit_json=s.str();
 }
'''
s=s.replace(' void dispatch_midroute(const View&o){',fn+'\n void dispatch_midroute(const View&o){\n  audit_delivery(o);')
p.write_text(s)
p=d/'policy/bridge.cpp';s=p.read_text();s=s.replace('try{*out=h.policy.act(*v);return 0;}', 'try{dp7::delivery_audit_target=&h.policy.live.core;dp7::delivery_audit_json="{}";*out=h.policy.act(*v);dp7::delivery_audit_target=nullptr;return 0;}');s+='\nextern "C" const char*td_delivery_audit(void*){return dp7::delivery_audit_json.c_str();}\n';p.write_text(s)
