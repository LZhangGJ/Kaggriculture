#ifdef RL_C3_AUTO
#include "stage/c3auto/hybrid.hpp"
#include "c3auto_config.hpp"
#else
#include "stage/c3j7/hybrid.hpp"
#include "c3j7_config.hpp"
#endif
#include "rl_common.hpp"
namespace {
using namespace fastkag;using namespace econrl;
struct Policy {
 competitive::Hybrid h{frozen_config()};Session rl;
 Policy(const char*file,uint64_t seed,int mode):rl(file,seed,mode){}
 PlayerAction act(const Input&i){
  dp7::View o{i.step,i.day,i.hour,*i.own,*i.opponent,*i.priv,*i.market,*i.shops};
  h.observe_calendar(o);if(h.cfg.tape>0)h.tape.observe(o);
  if(o.day!=h.core.day){
   h.plan(o);h.renewal_plan(o);if(h.cfg.crop_timing>.5)h.crop_plan(o);rl.plan_calls++;
#ifndef RL_C3_AUTO
   rl.reference_calls+=o.day<29;
#else
   if(h.reference_consults||h.model.full_plan_calls!=rl.plan_calls)throw std::runtime_error("C3 autonomous call path");
#endif
   Decision d;d.day=o.day;d.mask[0]=1;d.global=features(i,h.core.target,h.core.planned_land);d.candidates[0]=candidate(0,-1,-1,-1,0,0);
   auto targets=h.core.target;int qs[12]{};for(auto[p,k]:targets)if(k>=0)qs[k]++;
   int held[12]{};for(int k=9;k<12;k++){held[k]=o.priv.shed[k];for(auto&v:o.priv.inventories)held[k]+=v[k];}
   int index=-1;for(int n=0;n<int(targets.size());n++){
    auto[p,k]=targets[n];auto&t=o.own.tiles[p];
    if(t.kind==TileKind::PLANT||t.kind==TileKind::ANIMAL||t.kind==TileKind::LOCKED||h.core.plant_not_before[p]>o.day)continue;
    if(k>=9&&held[k]>0){held[k]--;continue;}
    if(index<0||std::tuple(dp7::near(p),p)<std::tuple(dp7::near(targets[index].first),targets[index].first))index=n;
   }
   int pos=index<0?-1:targets[index].first,old=index<0?-1:targets[index].second;
   std::array<int,A>kind{-1,0,1,2,3,4,9,10,11,-1};
   auto price=[](int k){constexpr int p[12]{10,20,50,100,80,0,0,0,0,300,400,500};return k<0?0:p[k];};
   double liquid=o.own.money;for(int k=0;k<9;k++)liquid+=o.priv.shed[k]*o.market.prices[k];
   if(index>=0&&o.day<29)for(int a=1;a<A;a++){
    int k=kind[a];if(k==old)continue;
    if(k>=9&&o.day+dp7::afirst[k-9]>29)continue;
    if(k>=0&&k<5&&o.day+(dp7::ongoing(k)?dp7::first[k]:h.core.h_age(k))>29)continue;
    auto&p=h.core.p;int caps[12]{75,75,p.max_tomato,p.max_strawberry,p.max_melon,0,0,0,0,p.max_geese,p.max_cows,p.max_sheep};
    if(k>=0&&qs[k]+1>caps[k])continue;
    if(k>=9&&old<9&&qs[9]+qs[10]+qs[11]>=p.max_animals)continue;
    if(price(k)-price(old)>liquid)continue;
    d.mask[a]=1;d.candidates[a]=candidate(a,old,k,pos,price(k)-price(old),k>=0?qs[k]+1:0);
    if(k>=0){
     auto estimate=k>=9?h.model.animal_stream(k,o.day,pos):h.crop_investment(o,h.core,k,o.day,pos);
     double gross=0,work=0,inputs=0;int first=30;
     for(int day=o.day;day<30;day++){work+=estimate.labor[day];for(int item=0;item<9;item++){
      double q=estimate.f[day][item];if(q>0){gross+=q*o.market.prices[item];first=std::min(first,day);}else inputs-=q*o.market.prices[item];}}
     auto&f=d.candidates[a];f[27]=(first-o.day)/30.f;f[28]=gross/100000.;f[29]=work/1000.;f[30]=estimate.first_cost/1000.;f[31]=inputs/10000.;
    }
   }
   int choice=rl.select(d);
   if(choice){
    int k=kind[choice];h.core.target[index].second=k;
    if(k<0)h.core.plant_not_before[pos]=o.day+1;
    std::optional<Farm> projected;int released=0;
    if(h.core.p.plan_zero_expiry){projected=o.own;released=dp7::Controller::project_zero_expiry(*projected,o.step);}
    dp7::View planning{o.step,o.day,o.hour,projected?*projected:o.own,o.opponent,o.priv,o.market,o.shops};
    h.core.prepare_orders(planning,o,released);
    if(h.core.target[index].second!=k)throw std::runtime_error("C3 decision overwritten during compile");
    d.pos=pos;d.oldkind=old;d.newkind=k;d.changed=1;
   }
   rl.records.push_back(d);
  }
  h.install_investment(o);auto a=h.market_control(o,h.core.act(o));if(h.cfg.tape>0)h.tape.remember(o,a);
  rl.execute_calls++;return a;
 }
};
void*create(const char*f,uint64_t s,int m){return new Policy(f,s,m);}
void destroy(void*p){delete static_cast<Policy*>(p);}
PlayerAction act(void*p,const Input&i){return static_cast<Policy*>(p)->act(i);}
Session*session(void*p){return &static_cast<Policy*>(p)->rl;}
}
extern "C" econrl::Api economic_policy_api(){return {create,destroy,act,session};}
