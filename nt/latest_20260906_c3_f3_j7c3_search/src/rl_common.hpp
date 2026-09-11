#pragma once
#include <array>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <chrono>
#include <numeric>
// Included AFTER each frozen policy's simulator header. No live Simulator is
// ever passed across this boundary; only observation-visible fields.
namespace econrl {
constexpr int G=128,C=32,A=10,H=128,I=G+C;
constexpr int ACTOR=H*I+H+H*H+H+H+1,CRITIC=H*G+H+H*H+H+H+1,NW=ACTOR+CRITIC;
struct Input {
 int step,day,hour,seat;
 const fastkag::Farm*own;const fastkag::Farm*opponent;
 const fastkag::PrivateState*priv;const fastkag::Market*market;
 const std::vector<int8_t>*shops;
};
using Global=std::array<float,G>;using Feature=std::array<float,C>;
struct Decision {
 Global global{};std::array<Feature,A>candidates{};std::array<float,A>mask{},probability{};
 int day=0,choice=0,pos=-1,oldkind=-1,newkind=-1;
 float logp=0,value=0;int changed=0;
};
struct Network {
 std::vector<float>w;
 Network():w(NW,0){}
 explicit Network(const std::string&path):w(NW){
  std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("model open");
  f.read(reinterpret_cast<char*>(w.data()),NW*sizeof(float));
  if(f.gcount()!=NW*sizeof(float)||f.peek()!=EOF)throw std::runtime_error("model size");
  for(float x:w)if(!std::isfinite(x))throw std::runtime_error("model finite");
 }
 float mlp(const float*x,int dim,int offset)const{
  float a[H],b[H];auto*p=w.data()+offset;
  for(int j=0;j<H;j++){float v=p[H*dim+j];for(int k=0;k<dim;k++)v+=p[j*dim+k]*x[k];a[j]=std::tanh(v);}
  p+=H*dim+H;
  for(int j=0;j<H;j++){float v=p[H*H+j];for(int k=0;k<H;k++)v+=p[j*H+k]*a[k];b[j]=std::tanh(v);}
  p+=H*H+H;float v=p[H];for(int j=0;j<H;j++)v+=p[j]*b[j];return v;
 }
 void evaluate(Decision&d)const{
  float x[I],scores[A],mx=-1e30f;
  std::copy(d.global.begin(),d.global.end(),x);
  for(int j=0;j<A;j++){
   std::copy(d.candidates[j].begin(),d.candidates[j].end(),x+G);
   scores[j]=d.mask[j]?mlp(x,I,0)+(j==0?3.58351893846f:0.f):-1e30f;
   mx=std::max(mx,scores[j]);
  }
  float total=0;for(int j=0;j<A;j++)total+=d.probability[j]=d.mask[j]?std::exp(scores[j]-mx):0;
  for(float&x:d.probability)x/=total;d.value=mlp(d.global.data(),G,ACTOR);
 }
};
inline Global features(const Input&o,const std::vector<std::pair<int,int>>&targets,int land){
 Global x{};int at=0;auto add=[&](double v){if(at>=G)throw std::runtime_error("feature overflow");x[at++]=float(std::clamp(v,-10.,10.));};
 add(o.day/29.);add(o.hour/23.);add(o.own->money/100000.);add(o.opponent->money/100000.);
 add((o.own->money-o.opponent->money)/100000.);add(o.own->hands.size()/15.);add(o.opponent->hands.size()/15.);
 add(std::popcount(unsigned(o.own->unlocked_mask))/4.);add(std::popcount(unsigned(o.opponent->unlocked_mask))/4.);
 for(int i=0;i<12;i++)add(o.priv->shed[i]/100.);
 for(int i=0;i<5;i++)add(o.priv->seeds[i]/100.);
 for(int i=0;i<12;i++){int n=0;for(auto&v:o.priv->inventories)n+=v[i];add(n/100.);}
 for(int i=0;i<9;i++)add(o.market->prices[i]/300.);
 for(int i=0;i<9;i++)add((o.market->inventory[i]-10000)/1000.);
 for(int s=0;s<8;s++)add(std::find(o.shops->begin(),o.shops->end(),s)!=o.shops->end());
 for(auto*f:{o.own,o.opponent}){
  int counts[12]{},yield[9]{},weed=0,water=0,feed=0;
  for(auto&t:f->tiles){
   if(t.kind==fastkag::TileKind::PLANT){counts[int(t.crop)]++;yield[int(t.crop)]+=t.yield_units;water+=!t.watered_today;}
   if(t.kind==fastkag::TileKind::ANIMAL){counts[int(t.animal)]++;yield[int(t.animal)-4]+=t.yield_units;feed+=!t.fed_today;}
   weed+=t.kind==fastkag::TileKind::WEED;
  }
  for(int k:{0,1,2,3,4,9,10,11})add(counts[k]/25.);
  for(int y:yield)add(y/100.);add(weed/25.);add(water/25.);add(feed/25.);
 }
 int desired[12]{};for(auto[p,k]:targets)if(k>=0)desired[k]++;
 for(int k:{0,1,2,3,4,9,10,11})add(desired[k]/25.);add(land/4.);
 add(std::log1p(std::max(0.,o.own->money))/12.);
 add(std::log1p(std::max(0.,o.opponent->money))/12.);
 return x;
}
inline Feature candidate(int index,int oldkind,int kind,int pos,double cost_delta,double new_quantity,int land_delta=0){
 Feature x{};x[index]=1;
 int ks[]{0,1,2,3,4,9,10,11};
 for(int j=0;j<8;j++)x[10+j]=(kind==ks[j])-(oldkind==ks[j]);
 x[18]=cost_delta/1000.;x[19]=new_quantity/25.;x[20]=land_delta/4.;
 x[21]=pos<0?0:(pos%10)/9.;x[22]=pos<0?0:(pos/10)/9.;
 x[23]=kind>=9;x[24]=oldkind>=9;x[25]=kind<0;x[26]=oldkind<0;
 return x;
}
struct Session {
 Network net;std::mt19937_64 rng;int mode;std::vector<Decision>records;
 double model_seconds=0;int64_t plan_calls=0,reference_calls=0,execute_calls=0;
 Session(const std::string&path,uint64_t seed,int m):net(path.empty()?Network():Network(path)),rng(seed),mode(m){}
 int select(Decision&d){
  auto t=std::chrono::steady_clock::now();net.evaluate(d);
  if(mode==0)d.choice=0;
  else if(mode==1)d.choice=int(std::max_element(d.probability.begin(),d.probability.end())-d.probability.begin());
  else if(mode>=10){d.choice=mode-10;if(d.choice>=A||!d.mask[d.choice])d.choice=0;}
  else{std::array<double,A>w{};for(int j=0;j<A;j++)w[j]=(mode==3?d.mask[j]:d.probability[j]);d.choice=std::discrete_distribution<int>(w.begin(),w.end())(rng);}
  d.logp=std::log(std::max(1e-30f,d.probability[d.choice]));
  model_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();return d.choice;
 }
};
struct Api {
 void*(*create)(const char*,uint64_t,int);void(*destroy)(void*);
 fastkag::PlayerAction(*act)(void*,const Input&);
 Session*(*session)(void*);
};
}
