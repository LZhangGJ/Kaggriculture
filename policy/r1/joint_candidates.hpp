#pragma once
// Structural exchanges, not an opponent-specific opening. The prefilter below
// is only a conditional stream estimate. Real admission is the common event
// rollout, including sourcing feed, wages, delivery and successor PLANT/WATER.
namespace triad {
struct JointCandidate {int id=16;Controller policy;int removed=0,feeders=0,age=0,next=-1;double hint=0;};
inline std::vector<JointCandidate> joint_candidates(const Controller&live,const Controller&keep,const View&o){
 std::vector<JointCandidate> out;
 if constexpr(P16_JOINT_BUNDLES==0)return out;
 if(live.joint.active||o.day+4>=29)return out;
 std::vector<int>removable,empty;int total_animals=0,actual_animals=0;
 for(const auto&t:o.own.tiles)actual_animals+=animal(t);
 for(auto[pos,k]:keep.core.target){
  const auto&t=o.own.tiles[pos];if(k>=9)total_animals++;
  bool vacant=!animal(t)&&!plant(t)&&t.kind!=TileKind::LOCKED;
  if(!vacant)continue;
  if(k<0)empty.push_back(pos);
  if(k>=9&&!keep.book[pos].funded&&keep.paths[pos].first_cost>0){
   int paid=o.priv.shed[k];for(const auto&iv:o.priv.inventories)paid+=iv[k];
   if(paid==0)removable.push_back(pos);
  }
 }
 if(removable.empty())return out;
 // Least valuable unpurchased animal removed first, ties are geometric only.
 std::stable_sort(removable.begin(),removable.end(),[&](int a,int b){
  auto aa=keep.portfolio,bb=keep.portfolio;add(aa,keep.paths[a],-1);add(bb,keep.paths[b],-1);
  return std::tuple(keep.model.value(o,aa),-near(a),-a)>std::tuple(keep.model.value(o,bb),-near(b),-b);
 });
 std::stable_sort(empty.begin(),empty.end(),[](int a,int b){return std::tuple(near(a),a)<std::tuple(near(b),b);});
 std::set<std::tuple<int,int,int,int>>seen;
 for(auto[remove,nfeed]:std::array<std::pair<int,int>,3>{{{1,1},{1,2},{2,2}}}){
  if(int(removable.size())<remove||total_animals-remove<actual_animals)continue;
  std::vector<int>positions(removable.begin(),removable.begin()+std::min(remove,nfeed));
  for(int p:empty)if(int(positions.size())<nfeed)positions.push_back(p);
  if(int(positions.size())<nfeed)continue;
  // Enumerate feeder duration and a SINGLE successor cycle. No infinite
  // repeat credit; staggered arrival is next dawn, never same-day financing.
  struct Option{double value=-1e100;int age=-1,kind=-1;};Option best;
  for(int age=2;age<=4;age++)for(int k=0;k<5;k++){
   int next=o.day+age+1;if(next>=29)continue;
   auto portfolio=keep.portfolio;for(int i=0;i<remove;i++)add(portfolio,keep.paths[removable[i]],-1);
   bool okay=true;
   for(int pos:positions){
    auto first=keep.crop(W,o.day,pos,o.day+age,keep.path_prices());
    // For the hint only: feeder receipts cannot fund an earlier purchase.
    first.f[o.day+age+1][W]+=first.f[o.day+age][W];first.f[o.day+age][W]=0;
    add(portfolio,first);
    auto path=keep.choose_crop(k,next,pos,keep.path_prices(),keep.rotations_dp(pos,keep.path_prices(),next));
    if(path.kind<0){okay=false;break;}
    add(portfolio,path.a);
   }
   if(!okay)continue;
   double value=keep.model.value(o,portfolio);
   if(value>best.value+1e-6)best={value,age,k};
  }
  if(best.kind<0||!seen.insert({remove,nfeed,best.age,best.kind}).second)continue;
  Controller p=live;p.configure(keep.s);
  auto history=live.joint;p.joint=JointBundleState{};
  p.joint.source_started=history.source_started;p.joint.source_harvested=history.source_harvested;
  p.joint.successor_started=history.successor_started;p.joint.cancelled=history.cancelled;p.joint.completed=history.completed;
  p.joint.active=true;p.joint.count=nfeed;p.joint.created_day=o.day;p.joint.animal_limit=total_animals-remove;p.joint.variant=16+int(out.size());
  for(int i=0;i<nfeed;i++){auto&b=p.joint.slots[i];b.pos=positions[i];b.source_age=best.age;b.next_kind=best.kind;}
  p.plan(o);
  bool all=true;
  for(int pos:positions){auto it=std::find_if(p.core.target.begin(),p.core.target.end(),[&](auto x){return x.first==pos;});all&=it!=p.core.target.end()&&it->second==W;}
  int now_animals=0;for(auto[pos,k]:p.core.target)now_animals+=k>=9;
  all&=now_animals<=p.joint.animal_limit;
  if(all)out.push_back({16+int(out.size()),std::move(p),remove,nfeed,best.age,best.kind,best.value});
 }
 return out;
}
}

