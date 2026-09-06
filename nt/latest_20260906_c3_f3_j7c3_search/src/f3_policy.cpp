#define DP_LIBRARY
#include "stage/f3/agent.cpp"
#include "stage/arena/vendor/simulator.hpp"
#include "rl_common.hpp"
#include "f3_config.hpp"
namespace {
Tile convert(const fastkag::Tile&t){
 Tile r;using K=fastkag::TileKind;
 r.kind=t.kind==K::LOCKED?-1:t.kind==K::EMPTY?0:t.kind==K::WEED?1:t.kind==K::PLANT?2:
    (t.kind==K::COOP||(t.kind==K::ANIMAL&&int(t.animal)==9))?3:4;
 r.item=t.kind==K::PLANT?int(t.crop):int(t.animal);r.placed=t.kind==K::PLANT?t.planted_day:t.placed_day;
 r.yield=t.yield_units;r.unwatered=t.consecutive_unwatered;r.unfed=t.consecutive_unfed;
 r.pending=t.pending_care_bonus;r.fertilized=t.fertilized_until_day;r.lifespan=t.max_lifespan_step;
 r.water=t.watered_today;r.feed=t.fed_today;r.care=t.cared_today;r.manure=t.fertilizer_available;return r;
}
Obs observe(const econrl::Input&i){
 Obs o;o.step=i.step;o.day=i.day;o.hour=i.hour;o.seat=i.seat;o.money=i.own->money;o.unlocked=std::popcount(unsigned(i.own->unlocked_mask));
 auto cell=[](fastkag::Position p){return int(p.y)*10+p.x;};o.positions.push_back(cell(i.own->farmer));
 for(auto p:i.own->hands)o.positions.push_back(cell(p));
 for(int p=0;p<100;p++){o.tiles[p]=convert(i.own->tiles[p]);o.opponent_tiles[p]=convert(i.opponent->tiles[p]);}
 for(int k=0;k<12;k++)o.shed[k]=i.priv->shed[k];for(int k=0;k<5;k++)o.seeds[k]=i.priv->seeds[k];
 for(auto&v:i.priv->inventories){Stock x{};for(int k=0;k<12;k++)x[k]=v[k];o.invs.push_back(x);}
 for(int k=0;k<9;k++){o.prices[k]=i.market->prices[k];o.market[k]=i.market->inventory[k];o.params[k]={BASE[k],THR[k],10000,LT[k],HT[k],LO[k],HI[k]};}
 const char*shops[]{"BAKERY","BRUNCH_SPOT","FARMERS_MARKET","ICE_CREAM_SHOP","PET_CAFE","PIZZA_SHOP","SMOOTHIE_SHOP","YARN_STORE"};
 for(int s:*i.shops)o.shops.push_back(shops[s]);return o;
}
fastkag::Action atom(const J&j){
 constexpr const char*ops[]{"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
 fastkag::Action a;auto&v=j.as_array();string op(v[0].as_string());int id=0;for(;id<24&&op!=ops[id];id++);if(id==24)throw runtime_error("F3 op");a.op=fastkag::Op(id);
 if(v.size()>1&&v[1].is_string())a.item=fastkag::Item(itemid(string(v[1].as_string())));
 if(v.size()>2)a.quantity=v[2].is_int64()?v[2].as_int64():v[2].as_double();else if(v.size()>1&&!v[1].is_string())a.quantity=v[1].as_int64();return a;
}
struct Policy {
 Context ctx;econrl::Session rl;
 Policy(const char*p,uint64_t seed,int mode):rl(p,seed,mode){ctx.agent.s.update(bj::parse(f3_config));}
 fastkag::PlayerAction act(const econrl::Input&i){
  auto o=observe(i);Config config;auto&a=ctx.agent;auto&m=a.m;
  if(o.day!=m.day){
   a.plan(o,config);rl.plan_calls++;
   econrl::Decision d;d.day=o.day;d.mask[0]=1;d.global=econrl::features(i,m.projects,o.unlocked+int(m.land));d.candidates[0]=econrl::candidate(0,-1,-1,-1,0,0);
   int held[12]{};for(int k=9;k<12;k++){held[k]=o.shed[k];for(auto&v:o.invs)held[k]+=v[k];}
   set<int>protected_positions;for(auto[p,k]:m.projects)if(k>=9&&held[k]>0){held[k]--;protected_positions.insert(p);}
   int pos=-1,old=-1,index=-1;
   for(int p=0;p<100;p++)if(o.tiles[p].kind!=-1&&!o.tiles[p].active()&&!protected_positions.count(p)){
    if(pos<0||tuple(dist(p,access(p)),Y(p),X(p))<tuple(dist(pos,access(pos)),Y(pos),X(pos)))pos=p;
   }
   for(int n=0;n<int(m.projects.size());n++)if(m.projects[n].first==pos){index=n;old=m.projects[n].second;break;}
   array<int,econrl::A>kind{-1,0,1,2,3,4,9,10,11,-1};
   array<Memory,econrl::A>options;options[0]=m;
   Economy eco(o,config,a.s);double liquid=o.money;for(int k=0;k<9;k++)liquid+=o.shed[k]*o.prices[k];
   auto cost=[](int k){return k<0?0:costs[k];};
   if(pos>=0&&o.day<29)for(int j=1;j<econrl::A;j++){
    int k=kind[j];if(k==old||cost(k)-cost(old)>liquid)continue;
    if(k>=9&&o.day+ANI[k-9].first>29)continue;
    auto copy=m;
    Flow newflow{};
    if(k>=9){auto ap=a.animalplan(k,o.day,o.day,nullptr,m.prices);copy.animalfeed[pos]=ap.feed;copy.animalcare[pos]=ap.care;copy.haspolicy[pos]=false;newflow=ap.flow;}
    else if(k>=0){auto cp=eco.newcrop(k,m.prices);if(!cp.valid)continue;copy.policy[pos]=cp.variant;copy.haspolicy[pos]=true;newflow=cp.flow;}
    else copy.haspolicy[pos]=false;
    auto projects=copy.projects;
    if(index>=0){if(k<0)projects.erase(projects.begin()+index);else projects[index].second=k;}
    else if(k>=0)projects.emplace_back(pos,k);
    // Atomically remove the EXACT originally accepted stream (including every
    // repeat cycle and all feed/fertilizer inputs), then insert this proposal.
    copy.forecast=copy.forecast_ledger.replace(o.day,pos,old,k,newflow);
    copy.setprojects(projects);copy.prices=eco.value(copy.forecast).second;
    copy.scheduled=false;copy.routes.clear();copy.backlog.clear();copy.couriers.clear();
    options[j]=std::move(copy);d.mask[j]=1;int quantity=0;for(auto[p,z]:projects)quantity+=z==k;
    d.candidates[j]=econrl::candidate(j,old,k,pos,cost(k)-cost(old),quantity);
    if(k>=0){
     Flow projected{};double work=0;
     if(k>=9){auto ap=a.animalplan(k,o.day,o.day,nullptr,m.prices);projected=ap.flow;work=a.assetwork(k,pos)*(29-o.day);}
     else{auto cp=eco.newcrop(k,m.prices);projected=cp.flow;work=a.assetwork(k,pos,cp.variant)*std::max(1,std::min(29-o.day,cp.variant.length));}
     double gross=0,inputs=0;int first=30;
     for(int day=o.day;day<30;day++)for(int item=0;item<9;item++){double q=projected[day][item];if(q>0){gross+=q*o.prices[item];first=std::min(first,day);}else inputs-=q*o.prices[item];}
     auto&f=d.candidates[j];f[27]=(first-o.day)/30.f;f[28]=gross/100000.;f[29]=work/1000.;f[30]=cost(k)/1000.;f[31]=inputs/10000.;
    }
   }
   int choice=rl.select(d);if(choice){m=std::move(options[choice]);d.changed=1;d.pos=pos;d.oldkind=old;d.newkind=kind[choice];
    if(m.newitem[pos]!=kind[choice])throw runtime_error("F3 decision target mismatch");}
   rl.records.push_back(d);
  }
  auto result=a.execute(o,config);fastkag::PlayerAction out;out.units.push_back(atom(result.at("farmer")));
  for(auto&u:result.at("hands").as_array())out.units.push_back(atom(u));
  for(auto&v:result.at("market").as_array())out.market.push_back(atom(v));rl.execute_calls++;return out;
 }
};
void*create(const char*f,uint64_t s,int m){return new Policy(f,s,m);}
void destroy(void*p){delete static_cast<Policy*>(p);}
fastkag::PlayerAction act(void*p,const econrl::Input&i){return static_cast<Policy*>(p)->act(i);}
econrl::Session*session(void*p){return &static_cast<Policy*>(p)->rl;}
}
extern "C" econrl::Api economic_policy_api(){return {create,destroy,act,session};}
