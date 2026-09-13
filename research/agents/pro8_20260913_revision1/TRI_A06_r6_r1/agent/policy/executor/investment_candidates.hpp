#pragma once
// Public-state project alternatives. No simulator/opponent handle/seed enters
// this generator. Offline suffix evaluation lives in a separate translation unit.
#include "policy.hpp"
#include <set>
namespace dp7branch {
struct Candidate {
 std::string family;int kind=-1,amount=0;dp7::Controller controller;
 double executable_score=0;int proposed=0,conditionally_started=0;
 std::vector<std::pair<int,int>>edits;
 Candidate(std::string f,int k,int q,const dp7::Controller&c):family(std::move(f)),kind(k),amount(q),controller(c){}
};
inline std::vector<Candidate>generate(const dp7::Controller&incoming,const dp7::View&actual){
 using namespace dp7;
 if(actual.hour!=0)throw std::invalid_argument("investment branch requires day boundary");
 auto keep=incoming;keep.new_day(actual);
 std::optional<Farm>projected;int released=0;
 if(keep.p.plan_zero_expiry){projected=actual.own;released=Controller::project_zero_expiry(*projected,actual.step);}
 View planning{actual.step,actual.day,actual.hour,projected?*projected:actual.own,actual.opponent,actual.priv,actual.market,actual.shops};
 std::vector<int>free,proposed,idle;
 for(auto[pos,k]:keep.target)if(!plant(planning.own.tiles[pos])&&!animal(planning.own.tiles[pos])){
  free.push_back(pos);(k<0?idle:proposed).push_back(pos);
 }
 auto order=[](int a,int b){return std::tuple(near(a),snake(a))<std::tuple(near(b),snake(b));};
 for(auto*v:{&free,&proposed,&idle})std::stable_sort(v->begin(),v->end(),order);
 std::vector<Candidate>result;std::set<std::vector<int>>seen;
 auto append=[&](std::string family,int kind,int amount,Controller c){
  if(family!="KEEP"){
   Counts counts{};for(auto[pos,k]:c.target)if(k>=0)counts[k]++;
   auto&p=c.p;Counts limits{75,75,p.max_tomato,p.max_strawberry,p.max_melon,0,0,0,0,p.max_geese,p.max_cows,p.max_sheep};
   for(int k:{0,1,2,3,4,9,10,11})if(counts[k]>std::max(limits[k],c.counts(planning)[k]))return;
   if(counts[G]+counts[CO]+counts[SH]>std::max(p.max_animals,c.counts(planning)[G]+c.counts(planning)[CO]+c.counts(planning)[SH]))return;
   // No pointless land purchase after withdrawing all proposed use of it.
   int owned=std::popcount(unsigned(actual.own.unlocked_mask));bool uses_new=false;
   for(auto[pos,k]:c.target)uses_new|=k>=0&&quad(pos)>=owned;
   if(!uses_new)c.planned_land=owned;
   c.prepare_orders(planning,actual,released);
  }
  std::vector<int>key{c.planned_land,c.phase};for(auto[pos,k]:c.target){key.push_back(pos);key.push_back(k);}
  for(auto&a:c.queue){key.push_back(int(a.op));key.push_back(int(a.item));key.push_back(a.quantity);}
  if(!seen.insert(key).second)return;
  Candidate item(family,kind,amount,c);auto preview=c.preview_bundle(planning);
  item.executable_score=preview.score;item.proposed=preview.proposed;item.conditionally_started=preview.started;
  for(size_t i=0;i<c.target.size();i++)if(c.target[i]!=keep.target[i])item.edits.push_back(c.target[i]);
  // All existing living assets keep exactly the same target. Scheduling and
  // funding consequences still need real simulation; they are not guaranteed.
  for(auto[pos,k]:keep.target)if(plant(planning.own.tiles[pos])||animal(planning.own.tiles[pos])){
   auto found=std::find(c.target.begin(),c.target.end(),std::pair(pos,k));
   if(found==c.target.end())throw std::runtime_error("candidate overwrote an existing asset target");
  }
  result.push_back(std::move(item));
 };
 append("KEEP",-1,0,keep);
 if(free.empty()||actual.day>=29)return result;
 auto assign=[](Controller&c,int pos,int k){for(auto&x:c.target)if(x.first==pos){x.second=k;return;}throw std::runtime_error("unknown investment slot");};
 auto empty=keep;for(int pos:free)assign(empty,pos,-1);append("DEFER_NEW",-1,0,empty);
 for(int kind:{0,1,2,3,4,9,10,11}){
  if(kind>=9&&(actual.day>keep.p.latest_animal_day||actual.day+afirst[kind-9]>29))continue;
  if(kind<5&&actual.day+keep.h_age(kind)>29)continue;
  std::set<int>quantities{1,2,4,8,int(free.size())};
  for(int q:quantities)if(q<=int(free.size())){auto c=empty;for(int i=0;i<q;i++)assign(c,free[i],kind);append("REBUILD_NEW",kind,q,c);}
  for(int q:{1,4}){
   if(q<=int(proposed.size())){auto c=keep;for(int i=0;i<q;i++)assign(c,proposed[i],kind);append("REPLACE_NEW",kind,q,c);}
   if(q<=int(idle.size())){auto c=keep;for(int i=0;i<q;i++)assign(c,idle[i],kind);append("EXPAND_NEW",kind,q,c);}
  }
 }
 return result;
}
}
