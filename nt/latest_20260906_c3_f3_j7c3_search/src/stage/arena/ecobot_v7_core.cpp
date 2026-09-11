#include "ecobot_v7_core.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

namespace eco7 {
namespace {
template<class Range,class T>bool has(const Range&r,const T&t){return std::find(r.begin(),r.end(),t)!=r.end();}
constexpr int INF=100000000;
constexpr const char*ITEMS[]{"WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"};
constexpr const char*OPS[]{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
std::string task_action_repr(const std::vector<Action>&as){
  std::string s="(";for(size_t i=0;i<as.size();i++){if(i)s+=", ";auto&a=as[i];s+="['";s+=OPS[int(a.op)];s+="'";if(a.item!=Item::NONE){s+=", '";s+=ITEMS[int(a.item)];s+="'";}s+="]";}
  if(as.size()==1)s+=",";return s+")";
}
std::vector<Action>harvest_actions(bool watered,int day){if(!watered&&day<29)return {act(Op::WATER),act(Op::HARVEST)};return {act(Op::HARVEST)};}
bool planted(const FarmState&f,Pos p){return std::any_of(f.plants.begin(),f.plants.end(),[&](auto&t){return t.pos==p;});}
std::optional<std::vector<Action>>build_actions(Pos p,const FarmState&f,Op build){
  if(has(f.weeds,p))return std::vector<Action>{act(Op::DIG),act(build)};
  for(auto&t:f.plants)if(t.pos==p){if(t.yield>0||(!ongoing(t.crop)&&t.age>=FIRST[t.crop]-1))return {};return std::vector<Action>{act(Op::DIG),act(build)};}
  if(has(f.empty,p))return std::vector<Action>{act(build)};return {};
}
int route_cost(const Route&r){int n=0;Pos p=r.start;for(auto&s:r.stops){n+=dist(p,s.pos)+int(s.actions.size());p=s.pos;}return n;}
std::pair<Pos,int>pickup_tile(Pos prev,std::optional<Pos>next){
  Pos best=SHEDS[0];int cost=INF,base=next?dist(prev,*next):0;
  for(auto p:SHEDS){int n=dist(prev,p)+(next?dist(p,*next):0)-base;if(n<cost){cost=n;best=p;}}return {best,cost};
}
int drop_cost(Pos p){return pickup_tile(p,{}).second+1;}
std::vector<Route>init_routes(const std::vector<Unit>&units){std::vector<Route>rs;for(auto&u:units){Route r;r.start=u.pos;for(int i=0;i<12;i++)r.carried[i]=std::max(0,u.inv[i]);rs.push_back(r);}return rs;}
void plan_routes(std::vector<Route>&routes,const std::vector<Task>&tasks,Stock&shed,const std::vector<int>&budgets,bool final_drop){
  int urgency=INF;for(auto&t:tasks){if(t.urgency>urgency)throw std::runtime_error("EcoBot catalog urgency");
    if(t.urgency<urgency)for(auto&r:routes)r.from=int(r.stops.size());urgency=t.urgency;
    std::optional<Candidate>best;int ri=-1;
    for(int i=0;i<int(routes.size());i++){auto&r=routes[i];auto c=eval_route(r,t,shed);if(!c)continue;
      int reserve=final_drop?drop_cost(c->at>=int(r.stops.size())?t.pos:r.stops.back().pos):0;
      if(r.cost+c->cost+reserve>budgets.at(i))continue;
      if(!best||c->cost<best->cost){best=c;ri=i;}}
    if(best)commit(routes[ri],t,*best,shed);
  }
}
void premium_rounds(std::vector<Route>&routes,std::vector<Plant>pool,int day,const std::vector<int>&budgets){
  while(!pool.empty()){
    std::vector<Plant>block{pool.back()};pool.pop_back();
    if(!pool.empty()){int nearest=0;for(int i=1;i<int(pool.size());i++)if(dist(block[0].pos,pool[i].pos)<dist(block[0].pos,pool[nearest].pos))nearest=i;block.push_back(pool[nearest]);pool.erase(pool.begin()+nearest);}
    int best_cost=INF,ri=-1;std::vector<Plant>best_order;
    for(int i=0;i<int(routes.size());i++){
      auto&r=routes[i];Pos tail=r.stops.empty()?r.start:r.stops.back().pos;int cost=INF;std::vector<Plant>order;
      for(int rev=0;rev<(block.size()==1?1:2);rev++){auto trial=block;if(rev)std::reverse(trial.begin(),trial.end());Pos pos=tail;int n=0;for(auto&p:trial){n+=dist(pos,p.pos)+int(harvest_actions(p.watered,day).size());pos=p.pos;}n+=drop_cost(pos);if(n<cost){cost=n;order=trial;}}
      if(r.cost+cost>budgets.at(i))continue;if(cost<best_cost){best_cost=cost;ri=i;best_order=order;}}
    if(ri<0)continue;auto&r=routes[ri];if(r.upto<0)r.upto=int(r.stops.size());
    for(auto&p:best_order)r.stops.push_back({p.pos,harvest_actions(p.watered,day),std::pair{p.crop,p.yield}});
    r.stops.push_back({pickup_tile(best_order.back().pos,{}).first,{act(Op::DROP)},{}});r.cost=route_cost(r);
  }
}
void append_drops(std::vector<Route>&routes,const std::vector<int>&budgets){
  for(int i=0;i<int(routes.size());i++){auto&r=routes[i];if(r.stops.empty()||r.stops.back().actions.back().op==Op::DROP)continue;
    bool cargo=false;for(auto&s:r.stops)for(auto&a:s.actions)if(a.op==Op::HARVEST||a.op==Op::COLLECT_FERTILIZER)cargo=true;for(int k=0;k<9;k++)if(r.carried[k]>0)cargo=true;
    if(!cargo)continue;auto pos=r.stops.back().pos;if(r.cost+drop_cost(pos)>budgets.at(i))continue;
    if(has(SHEDS,pos))r.stops.back().actions.push_back(act(Op::DROP));else r.stops.push_back({pickup_tile(pos,{}).first,{act(Op::DROP)},{}});r.cost=route_cost(r);
  }
}
}
int dist(Pos a,Pos b){return std::abs(a.first-b.first)+std::abs(a.second-b.second);}
int quad(Pos p){return (p.first>=5?1:0)+(p.second>=5?2:0);}
FarmState parse(const fastkag::Farm&f,int day){
  FarmState out;for(int y=0;y<10;y++)for(int x=0;x<10;x++){
    auto&t=f.tiles.at(y*10+x);Pos pos{x,y};using fastkag::TileKind;
    if(t.kind==TileKind::LOCKED)continue;out.unlocked_count++;
    if(t.kind==TileKind::EMPTY){out.empty.push_back(pos);continue;}
    if(t.kind==TileKind::WEED){out.weeds.push_back(pos);continue;}
    if(t.kind==TileKind::PASTURE||t.kind==TileKind::COOP||t.kind==TileKind::ANIMAL){
      if(t.animal==Item::NONE){(t.kind==TileKind::COOP?out.coops:out.pastures).push_back(pos);}
      else out.animals.push_back({pos,int(t.animal),t.fed_today,t.consecutive_unfed,t.cared_today,t.fertilizer_available,t.yield_units});continue;}
    if(t.kind!=TileKind::PLANT||int(t.crop)<0||int(t.crop)>=5)throw std::runtime_error("EcoBot invalid crop/tile");
    int crop=int(t.crop),age=day-t.planted_day;bool due=false,bonus=false,expired=age>=MAXAGE[crop]+1;
    if(ongoing(crop)){const auto&days=crop==S?STR_DAYS:TOM_DAYS;int next=-1;for(int d:days)if(d>=age+1){next=d;break;}
      if(next>=0){bool reachable=day+1<29,tomorrow=next==age+1;due=reachable&&tomorrow&&t.fertilized_until_day<day+1;bonus=reachable&&tomorrow&&(t.fertilized_until_day>=day+1||due);}}
    else bonus=BONUS[crop]<=age&&age<=MAXAGE[crop];
    bool water=!t.watered_today&&day<29&&(t.consecutive_unwatered>=1||bonus);
    out.plants.push_back({pos,crop,age,t.watered_today,water,t.yield_units,due,t.fertilized_until_day,expired});
  }return out;
}
Census census(const FarmState&f,const Stock&shed,const std::vector<Unit>&units){Census c;c.shed=shed;for(auto&a:f.animals)c.field[a.species]++;for(auto&u:units)for(int sp:{COW,SHEEP,G})c.carried[sp]+=u.inv[sp];return c;}
int kept_feedable(const Census&c,const Caps&caps){int n=0;for(int sp:{COW,SHEEP,G})n+=std::min(c.field[sp]+c.shed[sp],caps[sp]);return n;}
std::pair<int,int>feed_thresholds(const Census&c,const Caps&caps,int day){if(day>=28)return {0,0};int n=kept_feedable(c,caps);return {std::max(4,n*2),std::max(16,n*4)};}
int fert_due_tomorrow(const FarmState&f,int day){if(day+2>=29)return 0;int n=0;for(auto&p:f.plants){if(!ongoing(p.crop)||p.expired||p.fert_until>=day+2)continue;const auto&ds=p.crop==S?STR_DAYS:TOM_DAYS;if(has(ds,p.age+2))n++;}return n;}
int maturing_tomorrow(const FarmState&f,int crop){if(crop<0||crop>=5||ongoing(crop))throw std::runtime_error("EcoBot maturing ongoing");int n=0;for(auto&p:f.plants)if(p.crop==crop&&p.age+1>=MAXAGE[crop])n++;return n;}
int coop_capacity(const FarmState&f){int n=int(f.coops.size());for(auto&a:f.animals)if(a.species==G)n++;return n;}
std::set<Pos>reserved(const FarmState&f,int quads,const Hints&h,int day){
  int q=std::popcount(unsigned(quads)),gn=0,sn=0;if(day<=16)gn=std::max(h.grazers,q==1?4:std::min(18,q*6));if(day<28&&q>=2)sn=std::max(0,h.geese-coop_capacity(f));
  std::set<Pos>r;for(auto p:CLUSTER){if(int(r.size())>=gn+sn)break;if(quads&(1<<quad(p)))r.insert(p);}return r;
}
std::vector<Pos>needed_pastures(const FarmState&f,const Census&c){std::vector<Pos>r;for(auto p:CLUSTER){if(int(r.size()+f.pastures.size())+c.field[COW]+c.field[SHEEP]>=c.grazers())break;if(has(f.empty,p)||has(f.weeds,p)||planted(f,p))r.push_back(p);}return r;}
std::vector<Pos>needed_coops(const FarmState&f,const Census&c,const std::vector<Pos>&exclude){std::vector<Pos>r;for(auto p:CLUSTER){if(has(exclude,p))continue;if(int(r.size()+f.coops.size())+c.field[G]>=c.total(G))break;if(has(f.empty,p)||has(f.weeds,p)||planted(f,p))r.push_back(p);}return r;}
bool ready(const Plant&p,int day){return p.yield>0&&(ongoing(p.crop)||p.age>=MAXAGE[p.crop]||day>=28);}
std::vector<Plant>ready_premium(const FarmState&f,int day){std::vector<Plant>r;for(auto&p:f.plants)if(!p.expired&&premium(p.crop)&&ready(p,day))r.push_back(p);return r;}

std::vector<Task>catalog(const FarmState&f,const Stock&shed,const Seeds&seeds,const std::vector<Pos>&pastures,const std::vector<Pos>&coops,const std::vector<Unit>&units,int day,int hour,const Hints&h,int quads,const Caps&caps){
  std::vector<Task>out;std::set<Pos>kept,needed(pastures.begin(),pastures.end());needed.insert(coops.begin(),coops.end());
  for(int sp:{COW,SHEEP,G}){std::vector<Animal>as;for(auto&a:f.animals)if(a.species==sp)as.push_back(a);std::stable_sort(as.begin(),as.end(),[](auto&a,auto&b){return std::tuple{a.fed,dist(a.pos,SHEDS[0])}<std::tuple{b.fed,dist(b.pos,SHEDS[0])};});for(int i=0;i<std::min(std::max(0,caps[sp]),int(as.size()));i++)kept.insert(as[i].pos);}
  Stock carried{},field{},waiting{};for(auto&u:units)for(int i=0;i<12;i++)carried[i]+=u.inv[i];for(auto&a:f.animals)field[a.species]++;
  int wheat=shed[W]+carried[W],fert=shed[F]+carried[F];auto res=reserved(f,quads,h,day);
  for(auto&a:f.animals){Pos p=a.pos;auto produce=std::pair{product(a.species),a.yield};
    if(!kept.count(p)){if(a.yield>0)out.push_back({80,p,{act(Op::HARVEST)},{},produce});if(a.fertilizer)out.push_back({68,p,{act(Op::COLLECT_FERTILIZER)},{},std::pair{F,1}});continue;}
    if(!a.fed&&wheat>0)out.push_back({a.unfed>=1?100:70,p,{act(Op::FEED)},{W},{}});
    if(!a.cared)out.push_back({65,p,{act(Op::CARE)},{},{}});
    if(a.fertilizer)out.push_back({68,p,{act(Op::COLLECT_FERTILIZER)},{},std::pair{F,1}});
    if(a.yield>0)out.push_back({80,p,{act(Op::HARVEST)},{},produce});
  }
  for(int sp:{COW,SHEEP,G})waiting[sp]=field[sp]<caps[sp]?shed[sp]+carried[sp]:0;
  auto bypos=[](Pos a,Pos b){return std::tuple{dist(a,SHEDS[0]),a.second,a.first}<std::tuple{dist(b,SHEDS[0]),b.second,b.first};};
  for(int kind=0;kind<2;kind++){auto spots=kind?f.coops:f.pastures;int n=kind?waiting[G]:waiting[COW]+waiting[SHEEP];std::stable_sort(spots.begin(),spots.end(),bypos);for(int i=0;i<std::min(n,int(spots.size()));i++)out.push_back({95,spots[i],{act(Op::PLACE)},kind?std::vector<int>{G}:std::vector<int>{COW,SHEEP},{}});}
  std::vector<int>targets;if(day<=26){if(day>=21){if(seeds[W]>0)targets.push_back(W);}else {for(int crop:(day<=12?std::vector<int>{M,S,W}:std::vector<int>{S,M,W}))if(seeds[crop]>0&&day<=(crop==M?18:crop==S?14:26))targets.push_back(crop);}}
  std::array<int,100>freed_assignment;freed_assignment.fill(-1);
  if(hour<23){std::vector<Pos>open,rsv;std::set<Pos>future,freed;for(auto p:res)if(!needed.count(p))future.insert(p);
    for(auto p:f.empty)if(!has(BLOCKED,p)){if(!future.count(p)&&!needed.count(p))open.push_back(p);if(future.count(p))rsv.push_back(p);}
    for(auto&p:f.plants){if(ongoing(p.crop)||p.expired||premium(p.crop)||!ready(p,day)||needed.count(p.pos))continue;freed.insert(p.pos);(future.count(p.pos)?rsv:open).push_back(p.pos);}
    std::stable_sort(open.begin(),open.end(),bypos);std::stable_sort(rsv.begin(),rsv.end(),bypos);
    std::vector<std::tuple<int,int,int>>queue;for(int crop:{M,S})if(has(targets,crop))queue.emplace_back(crop,72,seeds[crop]);for(int crop:{C,T})queue.emplace_back(crop,71,std::min(seeds[crop],h.crops[crop]));for(int crop:targets)if(!premium(crop))queue.emplace_back(crop,71,seeds[crop]);
    for(auto[crop,urg,limit]:queue){if(open.empty()&&rsv.empty())break;int remaining=limit;for(auto*pool:(crop==W?std::vector<std::vector<Pos>*>{&open,&rsv}:std::vector<std::vector<Pos>*>{&open})){if(remaining<=0||pool->empty())continue;int n=std::min(remaining,int(pool->size()));for(int i=0;i<n;i++){Pos pos=pool->at(i);if(freed.count(pos))freed_assignment[pos.second*10+pos.first]=crop;else out.push_back({urg,pos,{act(Op::PLANT,crop),act(Op::WATER)},{},{}});}pool->erase(pool->begin(),pool->begin()+n);remaining-=n;}}
  }
  for(auto&p:f.plants){
    if(p.expired)out.push_back({30,p.pos,{act(Op::DIG)},{},{}});
    else if(ready(p,day)){if(!premium(p.crop)){auto actions=harvest_actions(p.watered,day);int crop=freed_assignment[p.pos.second*10+p.pos.first];if(crop>=0){actions.push_back(act(Op::PLANT,crop));actions.push_back(act(Op::WATER));}std::vector<int>need;if(ongoing(p.crop)&&p.fert_due&&fert>0){actions.insert(actions.begin(),act(Op::FERTILIZE));need={F};}out.push_back({80,p.pos,actions,need,std::pair{p.crop,p.yield}});}}
    else if(p.water_needed&&p.fert_due&&fert>0)out.push_back({77,p.pos,{act(Op::WATER),act(Op::FERTILIZE)},{F},{}});
    else if(p.water_needed)out.push_back({75,p.pos,{act(Op::WATER)},{},{}});
    else if(p.fert_due&&fert>0)out.push_back({77,p.pos,{act(Op::FERTILIZE)},{F},{}});
  }
  if(day<=26)for(auto p:f.weeds)if(!has(BLOCKED,p))out.push_back({20,p,{act(Op::DIG)},{},{}});
  for(int kind=0;kind<2;kind++){auto&spots=kind?coops:pastures;int left=std::max(0,kind?waiting[G]-int(f.coops.size()):waiting[COW]+waiting[SHEEP]-int(f.pastures.size())),n=0;
    for(auto p:spots){if(n>=left)break;auto actions=build_actions(p,f,kind?Op::BUILD_COOP:Op::BUILD_PASTURE);if(!actions)continue;actions->push_back(act(Op::PLACE));out.push_back({95,p,*actions,kind?std::vector<int>{G}:std::vector<int>{COW,SHEEP},{}});n++;}}
  std::stable_sort(out.begin(),out.end(),[](auto&a,auto&b){return std::tuple{-a.urgency,dist(a.pos,SHEDS[0]),a.pos.second,a.pos.first,task_action_repr(a.actions)}<std::tuple{-b.urgency,dist(b.pos,SHEDS[0]),b.pos.second,b.pos.first,task_action_repr(b.actions)};});return out;
}
std::optional<Candidate>eval_route(const Route&r,const Task&t,const Stock&shed){
  int n=int(r.stops.size()),limit=r.upto<0?n:r.upto,floor=r.from;std::vector<int>pref(n+1,INF),pk(n+1,-1);int best=INF,bk=-1;
  for(int k=0;k<n;k++){if(k>=floor){Pos prev=k==0?r.start:r.stops[k-1].pos;int d=pickup_tile(prev,r.stops[k].pos).second+1;if(d<best){best=d;bk=k;}}pref[k+1]=best;pk[k+1]=bk;}
  int earliest=INF;for(int p:r.pickup)if(p>=0)earliest=std::min(earliest,p);auto items=t.need.empty()?std::vector<int>{-1}:t.need;std::optional<Candidate>bc;
  for(int i=floor;i<=limit;i++){Pos prev=i==0?r.start:r.stops[i-1].pos;std::optional<Pos>next=i<n?std::optional{r.stops[i].pos}:std::nullopt;int dt=dist(prev,t.pos)+(next?dist(t.pos,*next)-dist(prev,*next):0)+int(t.actions.size());
    for(int item:items){Candidate c{dt,i,item,Mode::PLAIN};
      if(item<0){}else if(r.carried[item]>0)c.mode=Mode::CARRIED;else if(shed[item]<=0)continue;else if(r.pickup[item]>=0&&r.pickup[item]<i)c.mode=Mode::BUMP;else if(earliest<i){c.mode=Mode::EXTEND;c.cost++;}
      else {int d=pref[i],k=pk[i];std::optional<Pos>p;auto[pdet,det]=pickup_tile(prev,t.pos);det++;if(det<=d){d=det;k=i;p=pdet;}if(d>=INF)continue;c.mode=Mode::NEW;c.cost+=d;c.pickup_idx=k;c.pickup_pos=p;}
      if(!bc||c.cost<bc->cost)bc=c;
    }}return bc;
}
void commit(Route&r,const Task&t,const Candidate&c,Stock&shed){
  int n=int(r.stops.size()),insert=c.at,item=c.item;
  if(c.mode==Mode::NEW){Pos pos=c.pickup_pos?*c.pickup_pos:pickup_tile(c.pickup_idx==0?r.start:r.stops.at(c.pickup_idx-1).pos,r.stops.at(c.pickup_idx).pos).first;r.stops.insert(r.stops.begin()+c.pickup_idx,{pos,{act(Op::PICKUP,item,1)},{}});insert++;shed[item]--;}
  else if(c.mode==Mode::BUMP){for(auto&a:r.stops.at(r.pickup[item]).actions)if(a.op==Op::PICKUP&&int(a.item)==item){a.quantity++;break;}shed[item]--;}
  else if(c.mode==Mode::EXTEND){int j=INF;for(int p:r.pickup)if(p>=0)j=std::min(j,p);r.stops.at(j).actions.push_back(act(Op::PICKUP,item,1));shed[item]--;}
  else if(c.mode==Mode::CARRIED)r.carried[item]--;
  auto actions=t.actions;for(auto&a:actions)if(a.op==Op::PLACE&&a.item==Item::NONE&&item>=0)a.item=Item(item);
  r.stops.insert(r.stops.begin()+insert,{t.pos,actions,t.produces});if(r.upto>=0)r.upto+=int(r.stops.size())-n;r.cost=route_cost(r);r.pickup.fill(-1);
  for(int i=0;i<int(r.stops.size());i++)for(auto&a:r.stops[i].actions)if(a.op==Op::PICKUP&&r.pickup[int(a.item)]<0)r.pickup[int(a.item)]=i;
}
std::vector<Route>solve(const std::vector<Task>&tasks,const std::vector<Plant>&premium,const std::vector<Unit>&units,const Stock&shed,const std::vector<int>&budgets,int day){
  auto routes=init_routes(units);auto stock=shed;std::vector<Task>critical,rest;for(auto&t:tasks)(t.urgency>85?critical:rest).push_back(t);
  plan_routes(routes,critical,stock,budgets,false);premium_rounds(routes,premium,day,budgets);plan_routes(routes,rest,stock,budgets,day==29);append_drops(routes,budgets);return routes;
}
std::vector<Action>materialize(const Route&r){std::vector<Action>q;Pos p=r.start;for(auto&s:r.stops){while(p!=s.pos){int dx=s.pos.first-p.first,dy=s.pos.second-p.second;Op op;if(std::abs(dx)>=std::abs(dy)){op=dx>0?Op::EAST:Op::WEST;p.first+=dx>0?1:-1;}else{op=dy>0?Op::SOUTH:Op::NORTH;p.second+=dy>0?1:-1;}q.push_back(act(op));}q.insert(q.end(),s.actions.begin(),s.actions.end());}return q;}
int required_hands(const std::vector<Task>&tasks,const std::vector<Plant>&premium,const std::vector<Unit>&units,const Stock&shed,int max_hands,int day){
  if(tasks.empty()&&premium.empty())return 0;int current=int(units.size()),maxteam=max_hands+1;if(current>=maxteam)return 0;
  int budget=std::max(1,23-(day==29?2:0)),target=int(tasks.size()+premium.size()),best=0;
  for(int team=current;team<=maxteam;team++){auto us=units;for(int i=current;i<team;i++)us.push_back({SHEDS[i%4],{}});auto rs=solve(tasks,premium,us,shed,std::vector<int>(us.size(),budget),day);int inserted=0,active=0;for(auto&r:rs){bool work=false;for(auto&s:r.stops)if(!s.actions.empty()&&s.actions[0].op!=Op::PICKUP&&s.actions[0].op!=Op::DROP){inserted++;work=true;}active+=work;}
    if(inserted==target)return team-current;if(active==team)best=team-current;
  }return best;
}
std::vector<std::vector<Action>>day_plan(const std::vector<Unit>&units,const FarmState&f,const Stock&shed,const Seeds&seeds,int day,int hour,const std::vector<Pos>&pastures,const std::vector<Pos>&coops,const Hints&h,int quads,const Caps&caps,int pending_hires){
  int effective=std::max(1,hour),budget=std::max(0,24-effective-(day==29?2:0));auto us=units;std::vector<int>budgets(units.size(),budget);
  for(int i=0;i<pending_hires;i++){us.push_back({SHEDS[(units.size()+i)%4],{}});budgets.push_back(budget-1);}
  auto routes=init_routes(us);if(budget>0&&!us.empty())routes=solve(catalog(f,shed,seeds,pastures,coops,us,day,effective,h,quads,caps),ready_premium(f,day),us,shed,budgets,day);
  std::vector<std::vector<Action>>out;for(int i=0;i<int(routes.size());i++){auto q=materialize(routes[i]);if(int(q.size())>budgets[i])throw std::runtime_error("EcoBot day-plan exceeds budget");out.push_back(std::move(q));}return out;
}
}
