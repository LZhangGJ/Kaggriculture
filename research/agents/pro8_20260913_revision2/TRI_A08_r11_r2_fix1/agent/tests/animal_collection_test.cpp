// Focused conditional-model tests, not a tournament or route proof.
#include "policy/triad.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
using namespace competitive;
static long checks=0, brute_cases=0, legacy_value_mismatches=0, path_cases=0, choice_states=0;
static void check(bool b,const char*why){++checks;if(!b)throw std::runtime_error(why);}
// Independent finite-horizon enumeration of allowed feed/care choices. Every
// next-day credited product/manure must incur its real collection action.
static double brute(int k,int birth,int d,int hunger,int bonus,const Flow&px,double work){
 if(d==29)return 0.;int j=k-9;double best=-1e100;
 for(int feed=0;feed<2;feed++)for(int care=0;care<=feed;care++){
  int h=feed?0:hunger+1;if(h==2){best=std::max(best,0.);continue;}
  int age=d+1-birth;bool prod=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
  int q=prod?std::min(held[j],1+(feed?bonus:0)):0;
  int next_bonus=std::min(held[j]-1,(prod?0:bonus)+care);
  double v=-(feed+care)*work-feed*px[d][W]
      +q*px[d+1][product[j]]+px[d+1][F]-work*(1+(q>0));
  v+=brute(k,birth,d+1,h,next_bonus,px,work);best=std::max(best,v);
 }return best;
}
// Every stored score must belong to its stored action, not merely equal an
// independently optimal value somewhere else in the table. Read all successor
// scores from the DP but compute action execution/reward independently here.
static void check_choices(const AnimalServiceDP&dp,int k,int birth,int day,const Flow&px,double work){
 const int j=k-9;
 for(int d=day;d<=29;++d)for(int h=0;h<2;++h)for(int b=0;b<held[j];++b){
  ++choice_states;auto q=dp.choices[d][h][b];const double got=dp.value[d][h][b];
  check(q.value==got,"choice.value and DP value differ");
  check((q.feed==0||q.feed==1)&&(q.care==0||q.care==1)&&q.care<=q.feed,"illegal stored service action");
  if(d==29){check(got==0&&q.feed==0&&q.care==0,"terminal choice/value not zero");continue;}
  const bool tick=d+1-birth>=afirst[j]&&(d+1-birth-afirst[j])%ainterval[j]==0;
  const auto action_value=[&](int feed,int care){
   int nh=feed?0:h+1;if(nh>=2)return 0.;
   int count=tick?1+(feed?b:0):0;
   int nb=std::min(held[j]-1,(tick?0:b)+care);
   double immediate=-feed*px[d][W]-(feed+care)*work+count*px[d+1][product[j]];
   if(A08_ANIMAL_COLLECTION_LABOR)immediate+=px[d+1][F]-work*(1+int(count>0));
   else immediate+=std::max(0.,px[d+1][F]-work);
   return immediate+dp.value[d+1][nh][nb];
  };
  check(!(q.care&&b==held[j]-1&&!tick),"stored dominated/forbidden care");
  check(std::abs(got-action_value(q.feed,q.care))<1e-8,"stored action does not realize its value");
  int tf=0,tc=0;double optimum=action_value(0,0);
  for(int feed=0;feed<=1;++feed)for(int care=0;care<=feed;++care){
   if(care&&b==held[j]-1&&!tick)continue;
   double v=action_value(feed,care);
   if(v>optimum+1e-9){optimum=v;tf=feed;tc=care;}
  }
  check(q.feed==tf&&q.care==tc,"stored action is not the tie-ordered argmax");
  check(std::abs(got-optimum)<1e-8,"stored action/value not jointly optimal");
  if(A08_ANIMAL_COLLECTION_LABOR)check(std::abs(got-brute(k,birth,d,h,b,px,work))<1e-8,"choice-state optimum differs from full enumeration");
 }
}
static Flow prices(int pattern){
 Flow p{};
 for(int d=0;d<30;d++)for(int i=0;i<9;i++){
  if(pattern==0)p[d][i]=1.;
  else if(pattern==1)p[d][i]=(i==W?25:i==F?2:30);
  else if(pattern==2)p[d][i]=(i==W?25:i==F?100:200);
  else p[d][i]=1+((d*37+i*71+pattern*13+pattern*d*3)%211);
 }return p;
}
int main(){try{
 for(int k=9;k<=11;k++)for(int horizon=1;horizon<=5;horizon++)for(int phase=0;phase<4;phase++)for(int pattern=0;pattern<8;pattern++){
  int day=29-horizon,birth=std::max(0,day-afirst[k-9]+phase);Flow p=prices(pattern);AnimalServiceDP dp;dp.solve(k,birth,day,p,4.);check_choices(dp,k,birth,day,p,4.);
  for(int h=0;h<2;h++)for(int b=0;b<held[k-9];b++){
   double truth=brute(k,birth,day,h,b,p,4.);double got=dp.value[day][h][b];++brute_cases;
   if(std::abs(truth-got)>1e-8)++legacy_value_mismatches;
   if(A08_ANIMAL_COLLECTION_LABOR)check(std::abs(truth-got)<1e-8,"DP differs from independent enumeration");
  }
 }
 triad::Settings settings;settings.service=1;settings.work_price=4;settings.animal_work=1;settings.discount=0;settings.scenario=0;
 triad::Controller c(settings);
 long labor_mismatch=0, cashflow_mismatch=0;
 for(int day=24;day<=29;day++)for(int k=9;k<=11;k++)for(int h=0;h<2;h++)for(int b=0;b<held[k-9];b++)for(int y:{0,1,held[k-9]})for(int manure=0;manure<2;manure++)for(int pattern=0;pattern<4;pattern++){
  c.model.day=day;Flow p=prices(pattern);Tile t;t.kind=TileKind::ANIMAL;t.animal=Item(k);t.placed_day=day-afirst[k-9];t.consecutive_unfed=h;t.pending_care_bonus=b;t.yield_units=y;t.fertilizer_available=manure;
  auto a=c.animal_path(k,t.placed_day,44,p,&t);AnimalServiceDP dp;dp.solve(k,t.placed_day,day,p,4.);++path_cases;
  double actual=0,expected=dp.value[day][h][b]+y*p[day][product[k-9]]+manure*p[day][F]-(int(y>0)+manure)*4;
  int hunger=h,bonus=b;
  for(int d=day;d<30;d++){
   int feed=0,care=0;if(d<29 && d<a.end){auto q=dp.choices[d][hunger][bonus];feed=q.feed;care=q.care;}
   double need=feed+care+(a.f[d][product[k-9]]>0)+(a.f[d][F]>0);
   if(std::abs(a.labor[d]-need)>1e-8)++labor_mismatch;
   if(A08_ANIMAL_COLLECTION_LABOR)check(std::abs(a.labor[d]-need)<1e-8,"labor not on physical collection day");
   for(int i=0;i<9;i++)actual+=a.f[d][i]*p[d][i];actual-=4*a.labor[d];
   if(d<29&&d<a.end){hunger=feed?0:hunger+1;if(hunger>=2)continue;bool prod=(d+1-t.placed_day>=afirst[k-9] && (d+1-t.placed_day-afirst[k-9])%ainterval[k-9]==0);if(prod)bonus=0;if(feed&&care)bonus=std::min(held[k-9]-1,bonus+1);}
  }
  if(std::abs(expected-actual)>1e-8)++cashflow_mismatch;
  if(A08_ANIMAL_COLLECTION_LABOR)check(std::abs(expected-actual)<1e-8,"DP value and labor-booked lifecycle stream disagree");
 }
 // Economic negative control: no feed if the only next-day output cannot
 // pay feed plus the collection labor it actually requires.
 Flow p{};for(auto&d:p){d.fill(30);d[W]=25;d[F]=2;}
 AnimalServiceDP poor;poor.solve(9,21,28,p,4.);
 if(A08_ANIMAL_COLLECTION_LABOR){check(poor.choices[28][1][0].feed==0,"unprofitable collection extension accepted");check(poor.value[28][1][0]==0,"retirement value not zero");}
 else check(poor.choices[28][1][0].feed==1,"legacy reproducer no longer triggers");
 for(auto&d:p)d[E]=100;
 AnimalServiceDP rich;rich.solve(9,21,28,p,4.);check(rich.choices[28][1][0].feed==1,"positive-return maintenance lost");
 if(A08_ANIMAL_COLLECTION_LABOR)check(rich.value[28][1][0]==65,"profitable service exact value wrong");
 check_choices(rich,9,21,28,p,4.);
 check(A08_SALE_FLOOR_DP==1 && A08_SALE_SCHEDULE_DP==1 && A08_PAID_CONTINUATION==1 && A08_LAND_DP_MODE==2 && A08_CROP_PORTFOLIO_MODE==1 && A08_COMPLETE_CROP_STOPS==1,"ancestral feature flags changed");
 if(!A08_ANIMAL_COLLECTION_LABOR)check(legacy_value_mismatches>0&&labor_mismatch>0&&cashflow_mismatch>0,"legacy defect not detected");
 std::cout<<"{\"status\":\"PASS\",\"collection_labor_enabled\":"<<A08_ANIMAL_COLLECTION_LABOR<<",\"checks\":"<<checks<<",\"brute_force_cases\":"<<brute_cases<<",\"path_cases\":"<<path_cases<<",\"choice_states\":"<<choice_states<<",\"dp_value_mismatches\":"<<legacy_value_mismatches<<",\"labor_day_mismatches\":"<<labor_mismatch<<",\"stream_value_mismatches\":"<<cashflow_mismatch<<",\"new_games\":0}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<" after "<<checks<<" checks\n";return 1;}}
