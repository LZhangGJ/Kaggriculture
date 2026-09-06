// Native conditional portfolio DP and joint-neighbourhood search.
// Only numeric forecasts constructed from the current observation are supplied.
// No simulator, filesystem, random seed, replay, or opponent program is accessed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <limits>
#include <unordered_map>
using std::vector; using std::array;
constexpr int H=30, J=9, S=390, G=8;
static const double baseprice[J]={25,35,60,120,250,50,160,200,100};
static const double threshold[J]={400,450,200,100,300,332,122,105,200};
static const double lt[J]={.8,1,.4,.7,.2,.4,.6,.2,.4};
static const double ht[J]={.2,.7,.6,1.6,3.6,.2,1.6,3.2,.4};
static const int lf[J]={2,4,4,2,3,4,2,3,0},hf[J]={3,2,2,0,1,3,0,1,0};
inline double shape(int f,double x,double t){
 if(f==0)return x;if(f==1)return x*x;if(f==2)return std::sqrt(x);if(f==3)return std::log1p(x);
 double u=x/t;return u+8*std::pow(std::max(0.,u-1),2);
}
struct Context{
 int day; const double *dm,*initial,*cfg; double discounts[H],loamp[J],hiamp[J],fib[23],wages[23],sigma[H][J]; double uncertainty;
 Context(int d,const double *dem,const double *ini,const double *config,double risk=0):day(d),dm(dem),initial(ini),cfg(config),uncertainty(risk){
  const double var[J]={8.4375,17.4375,6.75,9.,0.,6.75,8.4375,15.75,0.};
  for(int z=0;z<H;z++){
   discounts[z]=std::exp(-cfg[0]*(z-d));double v=0;
   for(int unlock=3;unlock<=24;unlock+=3)if(unlock>d&&unlock<=z)v+=(z-unlock+1)*(z-unlock+1);
   for(int j=0;j<J;j++)sigma[z][j]=risk*std::sqrt(3*v*var[j]);
  }
  fib[0]=fib[1]=1;wages[0]=0;for(int i=2;i<23;i++)fib[i]=fib[i-1]+fib[i-2];
  for(int i=0;i<22;i++)wages[i+1]=wages[i]+fib[i];
  for(int i=0;i<J;i++){loamp[i]=baseprice[i]*lt[i]/shape(lf[i],threshold[i],threshold[i]);hiamp[i]=baseprice[i]*ht[i]/shape(hf[i],threshold[i],threshold[i]);}
 }
 double price(int j,double stock)const{
  double x=stock-10000.;
  double p=x<0?baseprice[j]+loamp[j]*shape(lf[j],-x,threshold[j]):baseprice[j]-hiamp[j]*shape(hf[j],x,threshold[j]);
  return std::max(1.,std::nearbyint(p));
 }
 double expected_price(int d,int j,double stock)const{
  double s=sigma[d][j];if(s==0)return price(j,stock);
  return (price(j,stock-s)+4*price(j,stock)+price(j,stock+s))/6.;
 }
 double score(const double* s,double *minprefix=nullptr,int cash_horizon=12)const{
  double inv[J];std::memcpy(inv,initial,sizeof(inv));double val=0,cumulative=0,low=0;
  for(int d=day;d<H;d++){
   double need=std::min(19.,std::max(0.,(s[330+d]+cfg[1]*s[360+d])/cfg[2]-1));
   int lo=(int)need;double wage=wages[lo]+(need-lo)*fib[lo];
   double daily=-(cfg[3]*wage+(1-cfg[3])*s[330+d]*cfg[4]);
   for(int j=0;j<J;j++){
    double q=s[d*J+j]-(j==0?s[270+d]:j==8?s[300+d]:0);inv[j]-=dm[d*J+j];
    if(q!=0){double p=0;for(double f:{.125,.375,.625,.875})p+=expected_price(d,j,inv[j]+f*q);daily+=q*p*.25;}
    inv[j]+=q;
   }
   val+=daily*discounts[d];
   if(d<day+cash_horizon){cumulative+=daily;low=std::min(low,cumulative);}
  }
  if(minprefix)*minprefix=low;
  return val;
 }
};
extern "C" double ledger_value(int day,const double*s,const double*dm,const double*initial,const double*cfg){return Context(day,dm,initial,cfg).score(s);}
extern "C" int ledger_plan(int day,int ng,int T,int B,int vacant,int unlocked,int rounds,int local_steps,
 const double*base,const double*st,const int*cu,const int*maxn,const double*capital,const double*dm,const double*initial,const double*cfg,
 int*answer,double*meta){
 if(ng>G||T>100||B>5000||ng<1)return -1;
 Context ctx(day,dm,initial,cfg,cfg[8]);const double baseval=ctx.score(base);int evals=1;
 const int land[3]={1000,2000,4000};const int unit=(int)cfg[5];const double haircut=cfg[6],relax=cfg[7];
 using Counts=array<int,G>;Counts previous{},best{};vector<Counts> seeds(1);
 array<double,S> aggregate,ss;std::copy(base,base+S,aggregate.begin());double bestval=0;int bestextra=0;
 auto landcost=[&](int ex){int n=0;for(int i=0;i<ex;i++)n+=land[unlocked-1+i];return n;};
 const int eval_budget=(int)cfg[9];
 std::unordered_map<uint64_t,array<double,3>>cache;cache.reserve(8192);
 auto value=[&](const Counts &c,int extra){
  uint64_t key=0;for(int k=0;k<ng;k++)key|=(uint64_t)c[k]<<(7*k);
  auto found=cache.find(key);array<double,3>entry;
  if(found!=cache.end())entry=found->second;
  else{
   std::copy(base,base+S,ss.begin());double cost=0;
   for(int k=0;k<ng;k++)if(c[k]){cost+=c[k]*capital[k];for(int j=0;j<S;j++)ss[j]+=c[k]*st[k*S+j];}
   evals++;double low=0,upfront=0;double intrinsic=ctx.score(ss.data(),&low,(int)cfg[13])-baseval-cost;
   for(int k=0;k<ng;k++)upfront+=c[k]*cu[k]*unit;
   entry={intrinsic,low,upfront};cache.emplace(key,entry);
  }
  double lc=landcost(extra);double v=entry[0]-lc;
  if(cfg[11]>0)v-=cfg[11]*std::max(0.,cfg[12]-(cfg[10]-entry[2]-lc+entry[1]));
  return extra?v*haircut:v;
 };
 const int W=B+1,N=(T+1)*W;
 for(int it=0;it<rounds;it++){
  vector<double>dp(N,-1e30);std::fill(dp.begin(),dp.begin()+W,0.);
  vector<vector<uint8_t>>hist;hist.reserve(ng);
  for(int k=0;k<ng;k++){
   array<double,S>other;double sc=it?relax:0.;for(int j=0;j<S;j++)other[j]=base[j]+sc*(aggregate[j]-base[j]-previous[k]*st[k*S+j]);
   double v0=ctx.score(other.data());evals++;
   vector<double> vals(maxn[k]+1);
   for(int n=1;n<=maxn[k];n++){for(int j=0;j<S;j++)ss[j]=other[j]+n*st[k*S+j]; vals[n]=ctx.score(ss.data())-v0-n*capital[k];evals++;}
   vector<double>next=dp;vector<uint8_t>choice(N);
   for(int n=1;n<=maxn[k];n++){
    if(vals[n]<=0)continue;int shift=n*cu[k];
    for(int t=n;t<=T;t++){
     int dst=t*W+shift,src=(t-n)*W;int count=W-shift;
     for(int b=0;b<count;b++){double v=dp[src+b]+vals[n];if(v>next[dst+b]){next[dst+b]=v;choice[dst+b]=n;}}
    }
   }
   dp.swap(next);hist.push_back(std::move(choice));
  }
  double roundval=-1e99;Counts roundc{};bool any=false;
  for(int ex=0;ex<=4-unlocked;ex++){
   int b=B-(landcost(ex)+unit-1)/unit;if(b<0)continue;
   int cap=std::min(T,vacant+25*ex),tbest=0;for(int t=1;t<=cap;t++)if(dp[t*W+b]>dp[tbest*W+b])tbest=t;
   Counts c{};int t=tbest;
   for(int k=ng-1;k>=0;k--){int n=hist[k][t*W+b];c[k]=n;t-=n;b-=n*cu[k];}
   double v=value(c,ex);seeds.push_back(c);
   if(v>bestval){bestval=v;best=c;bestextra=ex;}
   if(v>roundval){roundval=v;roundc=c;any=true;}
  }
  if(!any||roundc==previous)break;
  previous=roundc;std::copy(base,base+S,aggregate.begin());for(int k=0;k<ng;k++)for(int j=0;j<S;j++)aggregate[j]+=previous[k]*st[k*S+j];
 }
 // Joint search corrects conditional-DP externalities. Every move is checked
 // against the SAME budget and land constraints; no hidden rollouts are used.
 if(local_steps>0){
  seeds.push_back(best);std::sort(seeds.begin(),seeds.end());seeds.erase(std::unique(seeds.begin(),seeds.end()),seeds.end());
  auto feasible=[&](const Counts& c){int count=0,cash=0;for(int k=0;k<ng;k++){if(c[k]<0||c[k]>maxn[k])return -1;count+=c[k];cash+=c[k]*cu[k];}
   if(count>T)return -1;int ex=std::max(0,(count-vacant+24)/25);if(ex>4-unlocked||cash+(landcost(ex)+unit-1)/unit>B)return -1;return ex;};
  // Optional deterministic restarts: isolated-industry maxima and the best
  // coarse two-industry portfolio. They are evaluated on current forecasts,
  // never selected from seed-indexed routes or stored replays.
  if(cfg[15]>0){
   for(int k=0;k<ng;k++){
    Counts bc{};double bv=0;
    for(int n=1;n<=maxn[k];n++){Counts c{};c[k]=n;int e=feasible(c);if(e<0)continue;double v=value(c,e);if(v>bv){bv=v;bc=c;}}
    seeds.push_back(bc);
   }
   for(int k=0;k<ng;k++)for(int l=k+1;l<ng;l++){
    Counts bc{};double bv=0;
    for(int n:{1,2,4,8,12,16,24,32})for(int m:{1,2,4,8,12,16,24,32}){
     Counts c{};c[k]=n;c[l]=m;int e=feasible(c);if(e<0)continue;
     double v=value(c,e);if(v>bv){bv=v;bc=c;}
    }
    seeds.push_back(bc);
   }
   std::sort(seeds.begin(),seeds.end());seeds.erase(std::unique(seeds.begin(),seeds.end()),seeds.end());
   std::stable_sort(seeds.begin(),seeds.end(),[&](const Counts&a,const Counts&b){int ea=feasible(a),eb=feasible(b);return (ea<0?-1e99:value(a,ea))>(eb<0?-1e99:value(b,eb));});
   if(seeds.size()>(size_t)cfg[15])seeds.resize((int)cfg[15]);
   for(auto c:seeds){int e=feasible(c);if(e<0)continue;double v=value(c,e);if(v>bestval){bestval=v;best=c;bestextra=e;}}
  }
  for(auto start:seeds){
   if(evals>eval_budget)break;
   int ex=feasible(start);if(ex<0)continue;double curval=value(start,ex);Counts cur=start;
   for(int iter=0;iter<local_steps;iter++){
    if(evals>eval_budget)break;
    Counts bc=cur;double bv=curval;int be=ex;
    auto test=[&](Counts &c){int ce=feasible(c);if(ce<0)return;double v=value(c,ce);if(v>bv+1e-7){bv=v;bc=c;be=ce;}};
    // Single increments, decrements, exchanges, and cash-unlocking multi-adds.
    for(int k=0;k<ng;k++){
     for(int delta:{-1,1,2,4}){Counts c=cur;c[k]+=delta;test(c);}
     if(cur[k])for(int l=0;l<ng;l++)if(l!=k){
      for(int add:{1,2,4}){Counts c=cur;c[k]--;c[l]+=add;test(c);}
     }
    }
    if(cfg[14]>0){
     for(int k=0;k<ng;k++)for(int rem:{2,4,8}){
      Counts c=cur;c[k]-=rem;test(c);
      if(cur[k]>=rem)for(int l=0;l<ng;l++)if(l!=k)for(int add:{1,2,4,8}){
       c=cur;c[k]-=rem;c[l]+=add;test(c);
      }
     }
    }
    if(bc==cur)break;cur=bc;curval=bv;ex=be;
   }
   if(curval>bestval){bestval=curval;best=cur;bestextra=ex;}
  }
 }
 for(int k=0;k<ng;k++)answer[k]=best[k];answer[ng]=bestextra;
 meta[0]=bestval;meta[1]=evals;return 0;
}

// Small-route Held--Karp DP plus cross-worker relocation. The task is attached
// to a tile, not to the worker that originally planted/placed that asset.
extern "C" int ledger_routes(int nt,int nu,int limit,int terminal,int iterations,
 const int* xy,const int* ops,const double* values,const int* needs,
 const int* units,const int* inventories,int* routes,double* diagnostics){
 if(nt>100||nu>24||nt<0||nu<1)return -1;
 constexpr int R=12;using Route=vector<int>;vector<Route>rr(nu);
 for(int u=0;u<nu;u++)for(int j=0;j<100&&routes[u*100+j]>=0;j++)rr[u].push_back(routes[u*100+j]);
 auto distance=[](int x,int y,int xx,int yy){return std::abs(x-xx)+std::abs(y-yy);};
 auto sd=[&](int x,int y){return std::max(0,4-x)+std::max(0,x-5)+std::max(0,4-y)+std::max(0,y-5);};
 auto dd=[&](int a,int b){return distance(xy[2*a],xy[2*a+1],xy[2*b],xy[2*b+1]);};
 auto first=[&](int u,int t,bool refill){
  int x=units[2*u],y=units[2*u+1],xx=xy[2*t],yy=xy[2*t+1];
  if(!refill)return distance(x,y,xx,yy);int b=10000;
  for(int sx=4;sx<=5;sx++)for(int sy=4;sy<=5;sy++)b=std::min(b,distance(x,y,sx,sy)+distance(sx,sy,xx,yy));return b;
 };
 auto resources=[&](int u,const Route&r){array<int,R>req{};for(int t:r)for(int k=0;k<R;k++)req[k]+=needs[t*R+k];int n=0;for(int k=0;k<R;k++)n+=req[k]>inventories[u*R+k];return n;};
 auto cost=[&](int u,const Route&r){if(r.empty())return 0;int missing=resources(u,r);int c=missing+first(u,r[0],missing>0);for(int j=0;j<(int)r.size();j++){c+=ops[r[j]];if(j)c+=dd(r[j-1],r[j]);}if(terminal)c+=sd(xy[2*r.back()],xy[2*r.back()+1])+1;return c;};
 auto reorder=[&](int u,Route&r){
  int n=(int)r.size();if(n<=1)return;int missing=resources(u,r);int original=cost(u,r);
  if(n<=11){
   int masks=1<<n;vector<int> dp(masks*n,100000),pr(masks*n,-1);
   for(int j=0;j<n;j++)dp[(1<<j)*n+j]=first(u,r[j],missing>0);
   for(int mask=1;mask<masks;mask++)for(int j=0;j<n;j++)if(mask&(1<<j)){
    int before=mask^(1<<j);if(!before)continue;
    for(int k=0;k<n;k++)if(before&(1<<k)){
     int val=dp[before*n+k]+dd(r[k],r[j]);if(val<dp[mask*n+j]){dp[mask*n+j]=val;pr[mask*n+j]=k;}
    }
   }
   int best=-1,bestcost=100000;for(int j=0;j<n;j++){
    int c=dp[(masks-1)*n+j]+(terminal?sd(xy[2*r[j]],xy[2*r[j]+1])+1:0);
    if(c<bestcost){bestcost=c;best=j;}
   }
   Route opt(n);int mask=masks-1,last=best;for(int p=n-1;p>=0;p--){opt[p]=r[last];int prev=pr[mask*n+last];mask^=1<<last;last=prev;}
   if(cost(u,opt)<original)r=opt;
  }else{
   for(int it=0;it<3;it++){bool changed=false;for(int i=0;i<n;i++)for(int j=i+1;j<n;j++){
    Route a=r;std::reverse(a.begin()+i,a.begin()+j+1);int c=cost(u,a);if(c<original){r=a;original=c;changed=true;}
   }if(!changed)break;}
  }
 };
 for(int u=0;u<nu;u++)reorder(u,rr[u]);
 // Drop only jobs that still cannot fit after exact reordering. Usually these
 // are stale-route/resource detours from a mid-day observation.
 for(int u=0;u<nu;u++)while(!rr[u].empty()&&cost(u,rr[u])>limit){
  int worst=0;double merit=1e100;
  for(int j=0;j<(int)rr[u].size();j++){Route r=rr[u];int t=r[j];r.erase(r.begin()+j);double m=values[t]/std::max(1,cost(u,rr[u])-cost(u,r));if(m<merit){merit=m;worst=j;}}
  rr[u].erase(rr[u].begin()+worst);
 }
 auto add_dropped=[&](){
  vector<int>has(nt,0);for(auto&r:rr)for(int t:r)has[t]=1;vector<int> order;
  for(int t=0;t<nt;t++)if(!has[t])order.push_back(t);
  std::stable_sort(order.begin(),order.end(),[&](int a,int b){return values[a]>values[b];});
  for(int t:order){int bu=-1,bj=-1;double bm=1e100;
   for(int u=0;u<nu;u++){int old=cost(u,rr[u]);for(int j=0;j<=(int)rr[u].size();j++){
    Route r=rr[u];r.insert(r.begin()+j,t);int c=cost(u,r);if(c>limit)continue;double m=c-old+.01*c*c;
    if(m<bm){bm=m;bu=u;bj=j;}
   }}if(bu>=0)rr[bu].insert(rr[bu].begin()+bj,t);
  }
 };
 add_dropped();int relocations=0;
 for(int it=0;it<iterations;it++){
  double bestgain=1e-6;int bu=-1,bv=-1,bi=-1,bj=-1;vector<int>c(nu);for(int u=0;u<nu;u++)c[u]=cost(u,rr[u]);
  auto objective=[](int x){return x+.04*x*x;};
  for(int u=0;u<nu;u++)for(int i=0;i<(int)rr[u].size();i++){
   Route ru=rr[u];int t=ru[i];ru.erase(ru.begin()+i);int uc=cost(u,ru);
   for(int v=0;v<nu;v++)if(v!=u)for(int j=0;j<=(int)rr[v].size();j++){
    Route rv=rr[v];rv.insert(rv.begin()+j,t);int vc=cost(v,rv);if(vc>limit)continue;
    double gain=objective(c[u])+objective(c[v])-objective(uc)-objective(vc);
    if(gain>bestgain){bestgain=gain;bu=u;bv=v;bi=i;bj=j;}
   }
  }
  if(bu<0)break;int t=rr[bu][bi];rr[bu].erase(rr[bu].begin()+bi);rr[bv].insert(rr[bv].begin()+bj,t);relocations++;add_dropped();
 }
 for(int u=0;u<nu;u++)reorder(u,rr[u]);add_dropped();
 std::fill(routes,routes+nu*100,-1);int count=0,total=0,maxload=0;
 for(int u=0;u<nu;u++){for(int j=0;j<(int)rr[u].size();j++)routes[u*100+j]=rr[u][j];count+=rr[u].size();int c=cost(u,rr[u]);total+=c;maxload=std::max(maxload,c);}
 diagnostics[0]=nt-count;diagnostics[1]=total;diagnostics[2]=maxload;diagnostics[3]=relocations;return 0;
}

// Storage-constrained SELL_NOW/HOLD DP under visible expected consumption.
// The market state follows cumulative own sales exactly within this forecast;
// no future shop realization or competing private inventory is supplied.
extern "C" int ledger_sale(int day,int horizon,int capacity,double interest,double carry_fee,
 const int* quantities,const double* initial,const double* incoming,const double* demand,int* keep,double* meta){
 if(day<0||day>=30||capacity<0||capacity>100||horizon<1)return -1;
 horizon=std::min(horizon,30-day);double cfg[8]={0,1.85,18,1,3,10,.85,1};Context ctx(day,demand,initial,cfg);
 const double discount=std::exp(-interest);const int W=capacity+1;vector<vector<double>> options(J);
 for(int j=0;j<J;j++){
  int q0=quantities[j];if(q0<0||q0>1000)return -2;
  options[j].resize(std::min(capacity,q0)+1);
  vector<int> produced(horizon),inc(horizon);vector<double> eaten(horizon);
  produced[0]=q0;inc[0]=q0;eaten[0]=0;
  for(int t=1;t<horizon;t++){
   inc[t]=std::max(0,(int)std::nearbyint(incoming[(day+t)*J+j]));
   produced[t]=produced[t-1]+inc[t];eaten[t]=eaten[t-1]+demand[(day+t-1)*J+j];
  }
  vector<double>next(W,0.),cur(W,0.);
  for(int t=horizon-1;t>=0;t--){
   vector<double> pref(produced[t]+1);for(int s=0;s<produced[t];s++)pref[s+1]=pref[s]+ctx.price(j,initial[j]-eaten[t]+s);
   for(int prev=0;prev<=capacity;prev++){
    int q=t==0?q0:prev+inc[t];if(q>produced[t]){cur[prev]=-1e30;continue;}
    int maxk=(t==horizon-1)?0:std::min(capacity,q);double best=-1e30;
    for(int k=0;k<=maxk;k++){
     double val=pref[produced[t]-k]-pref[produced[t]-q]+discount*next[k]-carry_fee*k;
     if(val>best)best=val;
     if(t==0)options[j][k]=val;
    }
    cur[prev]=best;if(t==0)break;
   }
   if(t==0&&horizon==1)for(int k=1;k<(int)options[j].size();k++)options[j][k]=-1e30;
   next.swap(cur);
  }
 }
 vector<double>dp(W,0);vector<vector<int>>hist;
 for(int j=0;j<J;j++){
  vector<double>next(W,-1e30);vector<int>ch(W,0);
  for(int c=0;c<=capacity;c++)for(int k=0;k<(int)options[j].size()&&k<=c;k++){
   double v=dp[c-k]+options[j][k];if(v>next[c]){next[c]=v;ch[c]=k;}
  }
  hist.push_back(std::move(ch));dp.swap(next);
 }
 int c=capacity;for(int j=J-1;j>=0;j--){keep[j]=hist[j][c];c-=keep[j];}
 meta[0]=dp[capacity];return 0;
}
