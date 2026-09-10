#pragma once
// Exact task-input memoization. Real controllers reset at day/game boundaries;
// conditional copies share the current context. No identity or future inputs.
#include <unordered_map>
#include <vector>
#include <cstdint>
namespace dp7::packmemo {
using Result=std::pair<std::vector<Route>,int>;
struct Hash {
 size_t operator()(const std::vector<int>&key)const noexcept{
  uint64_t h=1469598103934665603ULL;
  for(int x:key){h^=uint32_t(x);h*=1099511628211ULL;}return size_t(h);
 }
};
inline std::vector<int> key(const std::vector<Job>&jobs,const std::vector<int>&starts,int budget,bool ret,bool insertion,bool regret,bool incremental=false){
 std::vector<int>k;k.reserve(8+starts.size()+jobs.size()*48);
 k.insert(k.end(),{budget,int(ret),int(insertion),int(regret),int(incremental),int(starts.size())});
 k.insert(k.end(),starts.begin(),starts.end());k.push_back(int(jobs.size()));
 for(const auto&j:jobs){
  k.insert(k.end(),{j.pos,j.priority,int(j.crop),int(j.actions.size())});
  for(auto a:j.actions)k.insert(k.end(),{int(a.op),int(a.item),a.quantity});
  k.insert(k.end(),j.needs.begin(),j.needs.end());k.insert(k.end(),j.seeds.begin(),j.seeds.end());k.insert(k.end(),j.out.begin(),j.out.end());
 }return k;
}
struct Entry {Result result;int trials=0,improvements=0;};
struct Cache {
 std::unordered_map<std::vector<int>,Entry,Hash> entries;
 uint64_t calls=0,hits=0,misses=0,not_stored=0,accounted_bytes=0;
 static constexpr uint64_t byte_limit=128ULL*1024*1024;
 void remember(std::vector<int>&&k,const Result&result,int trials,int improvements){
  // Conservative bookkeeping for key/result payloads and map node overhead.
  uint64_t bytes=256+k.capacity()*sizeof(int)+result.first.size()*sizeof(Route);
  for(const auto&r:result.first){bytes+=r.jobs.size()*sizeof(Job);for(const auto&j:r.jobs)bytes+=j.actions.size()*sizeof(Action);}
  if(entries.size()>=8192||bytes>byte_limit-accounted_bytes){not_stored++;return;}
  entries.emplace(std::move(k),Entry{result,trials,improvements});accounted_bytes+=bytes;
 }
};
inline thread_local Cache* active=nullptr;
struct Scope {
 Cache*prior;explicit Scope(Cache*cache):prior(active){active=cache;}
 ~Scope(){active=prior;}
};
}
