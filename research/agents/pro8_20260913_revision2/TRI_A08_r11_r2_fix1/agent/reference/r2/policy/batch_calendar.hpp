#pragma once
// Included inside competitive. Bounded, observation-only warehouse calendar.
// Stages: uncollected on a plot, carried by an observed worker, warehouse.
// It schedules a conditional route (not future truth, nor an executor rollout).
// No bag-size limit exists in the official game. Only the shed has capacity 100.
struct BatchContext {
 int day=0,hour=0,units=1,hires=0;
 std::array<int,16>position{},route_end{},route_busy{};
 std::array<double,9>market_price{},shed{};
 double money=0,animal_stock=0;
 void observe(const View&o,const dp7::Controller&c){
  day=o.day;hour=o.hour;units=std::min(16,1+int(o.own.hands.size()));hires=o.own.hires_today;money=o.own.money;
  route_busy.fill(0);position.fill(44);route_end.fill(44);
  for(int i=0;i<9;i++){market_price[i]=o.market.prices[i];shed[i]=o.priv.shed[i];}
  animal_stock=o.priv.shed[9]+o.priv.shed[10]+o.priv.shed[11];
  for(int u=0;u<units;u++){
   position[u]=cell(u?o.own.hands[u-1]:o.own.farmer);route_end[u]=position[u];
   // Preserve the already committed route prefix. At dawn no old-day plan
   // applies, and the ordinary service-load reservation is used instead.
   if(c.day==o.day&&c.phase==3&&u<int(c.plans.size())){
    const auto&p=c.plans[u];
    for(int k=p.index;k<int(p.a.size());k++){
     ++route_busy[u];route_end[u]=p.target[k];
     if(p.a[k].op==Op::DROP)break;
    }
   }
  }
 }
};
struct CashCalendar {
 Flow trade{},opening{},same_day{},overnight{},unrealized{};
 Curve workers{},service_work{};
};
inline CashCalendar batch_calendar(const Asset&a,const BatchContext&ctx,double labor_hours,int max_hands){
 CashCalendar out;out.trade=a.f;
 if(a.batches.empty())return out;
 // Signed path replacements are netted by (day, physical location, item).
 // This prevents a removed incumbent batch from surviving a DP substitution.
 std::array<std::array<std::array<double,9>,100>,30>jobs{};
 std::array<std::array<double,9>,16>carried{};
 std::array<std::array<double,100>,30>service_jobs{};
 for(const auto&b:a.batches){
  if(b.day<ctx.day||b.day>=30||b.item>=9)continue;
  if(b.item<0){if(b.pos>=0&&b.pos<100)service_jobs[b.day][b.pos]+=b.quantity;continue;}
  out.trade[b.day][b.item]-=b.quantity;
  if(b.stage==1){if(b.pos>=0&&b.pos<16)carried[b.pos][b.item]+=b.quantity;}
  else if(b.pos>=0&&b.pos<100)jobs[b.day][b.pos][b.item]+=b.quantity;
 }
 for(int d=ctx.day;d<30;d++){
  int first_hour=d==ctx.day?ctx.hour:0,last_hour=d==29?22:23;
  int n=std::clamp(int(std::ceil(std::max(1.,a.labor[d])/std::max(1.,labor_hours))),1,std::min(16,max_hands+1));
  if(d==ctx.day){
   // New helpers must be affordable BEFORE uncollected output is sold.
   double money=ctx.money+a.fixed[d];
   for(int i=0;i<9;i++)money+=(a.f[d][i]-a.physical_output[d][i])*ctx.market_price[i];
   int affordable=ctx.units;
   for(int h=ctx.hires;affordable<n && h<int(fib.size());h++){
    if(money<fib[h])break;money-=fib[h];++affordable;
   }
   n=std::max(ctx.units,std::min(n,affordable));n=std::min(n,16);
  }
  double service=0;for(int p=0;p<100;p++)service+=std::max(0.,service_jobs[d][p]);
  out.workers[d]=n;out.service_work[d]=service;
  std::array<double,16>tick{};std::array<int,16>pos{};
  std::array<std::array<double,9>,16>cargo{};
  int pickup_actions=0;for(int i:{W,F})pickup_actions+=(a.f[d][i]-a.physical_output[d][i])<-1e-8;
  for(int u=0;u<n;u++){
   pos[u]=d==ctx.day&&u<ctx.units?ctx.position[u]:depot[u%4];
   tick[u]=first_hour+pickup_actions;
   if(d==ctx.day&&u<ctx.units&&ctx.route_busy[u]>0){
    tick[u]=std::max(tick[u],double(first_hour+ctx.route_busy[u]));pos[u]=ctx.route_end[u];
   }
   if(d==ctx.day)cargo[u]=carried[u];
  }
  // Precompute the immutable task counts; keep the original global
  // earliest-return selection and tie order exactly (no route-policy change).
  std::array<int,100>task_ops{};std::vector<int>active;
  for(int p=0;p<100;p++){
   int ops=int(std::ceil(std::max(0.,service_jobs[d][p])));double q=ops;
   for(double x:jobs[d][p]){ops+=x>1e-8;q+=std::max(0.,x);}
   task_ops[p]=ops;if(q>=1e-8)active.push_back(p);
  }
  std::array<bool,100>used{};
  for(int count=0;count<int(active.size());count++){
   int bp=-1,bu=-1;double best=1e90;
   for(int p:active)if(!used[p]){
    int ops=task_ops[p];
    for(int u=0;u<n;u++){
     double end=tick[u]+dist(pos[u],p)+ops;
     // Prefer a route capable of a same-day warehouse return, with no seed,
     // replay, shop realization or opponent identity in the ordering.
     double score=end+near(p)+1;
     if(score<best){best=score;bp=p;bu=u;}
    }
   }
   if(bp<0)break;used[bp]=true;
   int ops=task_ops[bp];
   double finish=tick[bu]+dist(pos[bu],bp)+ops;
   if(finish<=last_hour+1){
    tick[bu]=finish;pos[bu]=bp;
    for(int i=0;i<9;i++)cargo[bu][i]+=std::max(0.,jobs[d][bp][i]);
   }else{
    // Still on the plot: do not label it carried or warehouse. The next-day
    // retry is conditional on survival and re-planning, not an extra crop.
    for(int i=0;i<9;i++)if(jobs[d][bp][i]>1e-8){
     if(d<29)jobs[d+1][bp][i]+=jobs[d][bp][i];else out.unrealized[d][i]+=jobs[d][bp][i];
    }
   }
  }
  // Multiple worker drops share ONE shed. Sales occur after the whole unit
  // phase. Stagger a congested drop by an hour rather than selling phantom
  // overflow; residual carried stock uses the official EOD handoff.
  std::array<double,24>occupied{};
  double overnight_room=100-(d==ctx.day?std::clamp(ctx.animal_stock,0.,100.):0.);
  for(int u=0;u<n;u++){
   double q=0;for(double x:cargo[u])q+=x;if(q<1e-8)continue;
   int arrival=int(std::ceil(tick[u]+near(pos[u]))); // DROP at this hour
   double held=std::min(q,100.);
   while(arrival<=last_hour&&occupied[arrival]+held>100.+1e-8)arrival++;
   bool same=arrival<=last_hour;
   double room=same?std::max(0.,100.-occupied[arrival]):std::max(0.,overnight_room);
   for(int i=0;i<9;i++){
    double take=std::min(room,cargo[u][i]);room-=take;
    if(same){out.trade[d][i]+=take;out.same_day[d][i]+=take;occupied[arrival]+=take;}
    else if(d<29){out.trade[d+1][i]+=take;out.opening[d+1][i]+=take;out.overnight[d][i]+=take;overnight_room-=take;}
    else out.unrealized[d][i]+=take;
    out.unrealized[d][i]+=cargo[u][i]-take;
   }
  }
 }
 return out;
}
