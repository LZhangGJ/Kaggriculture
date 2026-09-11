#pragma once
// Offline only: included in a copied header tree, never in production policy.
#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <cstdint>
namespace dp7::packprofile {
struct Group {uint64_t calls=0,repeated=0;double compute_seconds=0.;};
struct Stats {
 std::map<std::string,Group> groups;
 std::map<std::vector<int>,uint64_t> seen;
 uint64_t key_ints=0,uncatalogued=0;
};
inline thread_local Stats* active=nullptr;
inline thread_local const char* context="act";
struct Context {
 const char* prior;explicit Context(const char*name):prior(context){context=name;}
 ~Context(){context=prior;}
};
struct Call {
 Group*g=nullptr;std::chrono::steady_clock::time_point start;
 template<class Jobs>Call(const Jobs&jobs,const std::vector<int>&starts,int budget,bool ret,bool insertion,bool regret){
  if(!active)return;
  g=&active->groups[context];g->calls++;
  std::vector<int>key{budget,int(ret),int(insertion),int(regret),int(starts.size())};
  key.insert(key.end(),starts.begin(),starts.end());key.push_back(int(jobs.size()));
  for(const auto&j:jobs){
   key.insert(key.end(),{j.pos,j.priority,int(j.crop),int(j.actions.size())});
   for(auto a:j.actions)key.insert(key.end(),{int(a.op),int(a.item),a.quantity});
   for(auto v:j.needs)key.push_back(v);for(auto v:j.seeds)key.push_back(v);for(auto v:j.out)key.push_back(v);
  }
  auto it=active->seen.find(key);
  if(it!=active->seen.end()){it->second++;g->repeated++;}
  else if(active->seen.size()<20000){active->key_ints+=key.size();active->seen.emplace(std::move(key),1);}
  else active->uncatalogued++;
  start=std::chrono::steady_clock::now();
 }
 ~Call(){if(g)g->compute_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();}
};
struct Active {
 Stats*prior;explicit Active(Stats&s):prior(active){active=&s;}
 ~Active(){active=prior;}
};
}
