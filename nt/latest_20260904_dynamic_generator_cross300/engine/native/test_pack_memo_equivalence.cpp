#include "policy.hpp"
#include <random>
#include <iostream>
#include <stdexcept>
using namespace dp7;
struct Reference:Controller {
#include "reference_regret.inc"
};
bool same_job(const Job&a,const Job&b){
 if(a.pos!=b.pos||a.priority!=b.priority||a.crop!=b.crop||a.needs!=b.needs||a.seeds!=b.seeds||a.out!=b.out||a.actions.size()!=b.actions.size())return false;
 for(size_t k=0;k<a.actions.size();k++)if(!Controller::same_action(a.actions[k],b.actions[k]))return false;return true;
}
bool same(const packmemo::Result&a,const packmemo::Result&b){
 if(a.second!=b.second||a.first.size()!=b.first.size())return false;
 for(size_t i=0;i<a.first.size();i++){
  const auto&x=a.first[i];const auto&y=b.first[i];
  if(x.unit!=y.unit||x.start!=y.start||x.end!=y.end||x.cost!=y.cost||x.needs!=y.needs||x.has_output!=y.has_output||x.jobs.size()!=y.jobs.size())return false;
  for(size_t j=0;j<x.jobs.size();j++)if(!same_job(x.jobs[j],y.jobs[j]))return false;
 }return true;
}
void require(bool yes,const char*message){if(!yes)throw std::runtime_error(message);}
int main(){
 std::mt19937 random(487211);int comparisons=0,key_checks=0,hit_checks=0;
 for(int sample=0;sample<768;sample++){
  Reference c;c.p.incremental_regret_cost=true;c.p.regret_schedule=sample%2;c.p.regret_compile=(sample/2)%2;c.p.regret_hire_estimate=(sample/4)%2;
  int n=sample<128?sample%8:random()%81;std::vector<Job>jobs;
  for(int j=0;j<n;j++){
   Job x;x.pos=random()%100;x.priority=int(random()%5)-1;x.crop=random()%2;
   for(int k=0,count=1+random()%5;k<count;k++)x.actions.push_back(action(Op(8+random()%10),int(random()%13)-1,1+random()%3));
   for(int k=0;k<12;k++){x.needs[k]=(random()%8==0)?1+random()%3:0;x.seeds[k]=(random()%16==0)?1:0;x.out[k]=(random()%10==0)?1+random()%4:0;}jobs.push_back(x);
  }
  std::vector<int>starts;for(int u=0,count=random()%16;u<count;u++)starts.push_back(random()%100);
  int budget=int(random()%36)-1;bool ret=random()%2,insertion=random()%2,hire=random()%2;
  auto expected=c.regret_pack_reference(jobs,starts,budget,ret);auto actual=c.regret_pack(jobs,starts,budget,ret);
  require(same(expected,actual),"incremental regret changes exact schedule");comparisons++;
  auto plain=c.pack(jobs,starts,budget,ret,insertion,hire);packmemo::Cache cache;
  {packmemo::Scope scope(&cache);
   auto first=c.pack(jobs,starts,budget,ret,insertion,hire);int trials=c.regret_trials,better=c.regret_improvements;
   auto second=c.pack(jobs,starts,budget,ret,insertion,hire);
   require(same(plain,first)&&same(first,second),"memo changes schedule");
   require(cache.hits==1&&cache.misses==1,"exact repetition not memoized");
   const auto&entry=cache.entries.begin()->second;
   require(c.regret_trials-trials==entry.trials&&c.regret_improvements-better==entry.improvements,"counter effects lost");hit_checks++;
  }
  require(packmemo::active==nullptr,"context leaked");
  cache=packmemo::Cache{};cache.accounted_bytes=packmemo::Cache::byte_limit;
  {packmemo::Scope scope(&cache);require(same(plain,c.pack(jobs,starts,budget,ret,insertion,hire)),"full cache changes schedule");}
  require(cache.not_stored==1&&cache.entries.empty(),"cache byte limit not respected");comparisons++;
 }
 Job j;j.pos=44;j.actions={action(Op::WATER)};std::vector<Job>jobs{j};std::vector<int>starts{44,45};
 auto key=packmemo::key(jobs,starts,24,false,false,true);
 auto changed=[&](auto other){require(key!=other,"key omitted an input");key_checks++;};
 for(int field=0;field<43;field++){
  auto copy=jobs;
  if(field==0)copy[0].pos++;else if(field==1)copy[0].priority++;else if(field==2)copy[0].crop=!copy[0].crop;
  else if(field==3)copy[0].actions[0].op=Op::CARE;else if(field==4)copy[0].actions[0].item=Item(W);
  else if(field==5)copy[0].actions[0].quantity++;else if(field==6)copy[0].actions.push_back(action(Op::HARVEST));
  else if(field<19)copy[0].needs[field-7]++;else if(field<31)copy[0].seeds[field-19]++;else copy[0].out[field-31]++;
  changed(packmemo::key(copy,starts,24,false,false,true));
 }
 changed(packmemo::key(jobs,{45,44},24,false,false,true));changed(packmemo::key(jobs,starts,23,false,false,true));
 changed(packmemo::key(jobs,starts,24,true,false,true));changed(packmemo::key(jobs,starts,24,false,true,true));
 changed(packmemo::key(jobs,starts,24,false,false,false));
 changed(packmemo::key(jobs,starts,24,false,false,true,true));
 std::cout<<"{\"status\":\"PASS\",\"schedule_comparisons\":"<<comparisons<<",\"memo_hit_checks\":"<<hit_checks<<",\"key_input_checks\":"<<key_checks<<"}\n";
}
