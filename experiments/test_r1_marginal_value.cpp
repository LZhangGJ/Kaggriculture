#include "../policy/r1/triad.hpp"
#include <iostream>
#include <stdexcept>

static void require(bool value,const char*message){if(!value)throw std::runtime_error(message);}
int main(){
 using namespace competitive;
 fastkag::Simulator env({},2610500000ULL);
 int checks=0;double worst=0;
 // Compare the optimized, single-product suffix calculation against the
 // independent full portfolio objective, including persistent cash shortfalls.
 for(int day:{0,11,28,29})for(double money:{0.,3000.,25000.}){
  auto farm=env.farms()[0];farm.money=money;
  auto market=env.market();for(int i=0;i<9;i++)market.inventory[i]=10000+(i-3)*31;
  dp7::View v{day*24,day,0,farm,env.farms()[1],env.privates()[0],market,env.shops()};
  Planner model;model.day=day;model.cfg.risk=1.3;model.cfg.competition=2.;model.cfg.discount=.017;
#if R2_SALE_CLOCK_MODE >= 2
  for(int i=0;i<9;i++)model.rival_early_share[i]=.1+.1*i;
#endif
  Asset a;
  for(int d=day;d<30;d++){
   a.fixed[d]=-900.;a.labor[d]=70.+d;
   for(int i=0;i<9;i++){model.dem[d][i]=1+(i+d)%7;model.rival[d][i]=(i+d)%9-2.5;a.f[d][i]=(i*3+d)%13-4.25;}
  }
  for(double normalization:{0.,.023}){
   auto values=MarginalValue::compute(model,v,a,normalization);
   for(int d=day;d<30;d++)for(int i=0;i<9;i++){
    Asset plus=a,minus=a;plus.f[d][i]+=2;minus.f[d][i]-=2;
    double reference=(model.value(v,plus)-model.value(v,minus))/4*std::pow(1+normalization,d-day);
    double error=std::abs(values[d][i]-reference);worst=std::max(worst,error);
    require(error<1e-6*(1+std::abs(reference)),"marginal differs from full objective");checks++;
   }
   for(int d=0;d<day;d++)for(double value:values[d])require(value==0,"past flow has value");
  }
 }
 // A one-period, no-rival problem must reduce to actual incremental revenue,
 // rather than the post-sale market quote. All nine commodities are covered.
 {
  Planner model;model.day=29;model.cfg.risk=0;model.cfg.competition=0;
  auto farm=env.farms()[0];farm.money=5000;
  dp7::View v{696,29,0,farm,env.farms()[1],env.privates()[0],env.market(),env.shops()};
  Asset a;for(int i=0;i<9;i++)a.f[29][i]=30;
  auto mu=MarginalValue::compute(model,v,a);
  for(int i=0;i<9;i++){
   double x=10000,y=10000;
   double expected=(Planner::trade(i,x,32)-Planner::trade(i,y,28))/4;
   require(std::abs(mu[29][i]-expected)<1e-9,"single-period marginal incorrect");
  }
 }
 // A supply perturbation must propagate to its own future economic signal,
 // without creating cross-product effects when the liquidity penalty is off.
 {
  Planner model;model.day=29;model.cfg.risk=0;model.cfg.competition=1;
  dp7::View v{696,29,0,env.farms()[0],env.farms()[1],env.privates()[0],env.market(),env.shops()};
  Asset a;Flow quote{};model.value(v,a,&quote);
  auto before=MarginalValue::compute(model,v,a);
  a.f[29][dp7::M]=120;auto after=MarginalValue::compute(model,v,a);
  require(after[29][dp7::M]<before[29][dp7::M],"own supply did not reduce its marginal value");
  require(after[29][dp7::C]==before[29][dp7::C],"unrelated commodity changed without liquidity coupling");
 }
 // Discounted Bellman service values agree with exhaustive action sequences.
 {
  Flow px{};for(int d=26;d<30;d++){px[d][dp7::W]=20;px[d][dp7::E]=80;px[d][dp7::F]=30;}
  AnimalServiceDP dp;double beta=.8;dp.solve(dp7::G,20,26,px,4,beta);
  std::function<double(int,int,int)> brute=[&](int d,int hunger,int bonus){
   if(d==29)return 0.;double best=-1e100;
   for(int feed=0;feed<=1;feed++){
    int nh=feed?0:hunger+1;if(nh>=2){best=std::max(best,0.);continue;}
    for(int care=0;care<=feed;care++){
     int np=care;double value=-feed*24-care*4+beta*((1+(feed?bonus:0))*80+26+brute(d+1,nh,np));
     best=std::max(best,value);
    }
   }return best;
  };
  require(std::abs(dp.value[26][0][0]-brute(26,0,0))<1e-8,"service DP discount inconsistency");
 }
 // Execution must retain today's service from the accepted crop stream even
 // after another project changes the marginal table.
 {
  triad::Settings settings;settings.marginal_value=1;settings.crop_fert=1;
  triad::Controller controller(settings);controller.model.day=1;
  auto farm=env.farms()[0];auto&t=farm.tiles[0];
  t.kind=fastkag::TileKind::PLANT;t.crop=fastkag::Item(dp7::S);t.planted_day=0;
  t.consecutive_unwatered=1;t.fertilized_until_day=-1;
  Flow px{};for(int d=0;d<30;d++){px[d][dp7::S]=100;px[d][dp7::F]=1;}
  auto path=controller.crop(dp7::S,0,0,29,px,&t,nullptr,-1,-1,&controller.forecast_crop_service[0]);
  auto action=controller.forecast_crop_service[0];
  require(action[0]==1,"dry incumbent's service was not forecast");
  require(path.f[1][dp7::F]==-action[1],"forecast fertilizer and flow differ");
  for(auto&day:controller.generation_values)day.fill(-10000);
  dp7::View v{24,1,0,farm,env.farms()[1],env.privates()[0],env.market(),env.shops()};
  controller.set_service(v);
  require(controller.core.crop_water[0]==action[0]&&controller.core.crop_fertilize[0]==action[1],
          "refreshed prices silently changed committed crop service");
 }
 std::cout<<"{\"status\":\"PASS\",\"derivative_checks\":"<<checks<<",\"worst_error\":"<<worst<<"}\n";
}
