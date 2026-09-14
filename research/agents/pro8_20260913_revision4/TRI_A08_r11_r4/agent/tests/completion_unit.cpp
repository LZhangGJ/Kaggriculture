#include "../policy/executor/policy.hpp"
#include <iostream>
#include <stdexcept>
#include <tuple>
using namespace dp7;
static long checks=0,cases=0,selected=0,closed=0;
void require(bool x,const char*what){++checks;if(!x)throw std::runtime_error(what);}
int mirror(int c,int r){int x=c%10,y=c/10;if(r&1)x=9-x;if(r&2)y=9-y;return y*10+x;}
struct Fixture{
 fastkag::Farm f,other;fastkag::PrivateState priv;fastkag::Market market;std::vector<int8_t>shops;Controller ctl;int step,plot;
 Fixture(int reflection,int tick,int distance,int qty,int held,int inventory,double shadow):step(tick),plot(mirror(45+distance,reflection)){
  int home=mirror(45,reflection);f.tiles.resize(100);other.tiles.resize(100);f.money=50000;f.farmer={int16_t(home%10),int16_t(home/10)};f.unlocked_mask=15;
  auto&t=f.tiles[plot];t.kind=TileKind::PLANT;t.crop=Item::TOMATO;t.planted_day=10;t.yield_units=qty;t.watered_today=true;t.fertilized_until_day=30;
  other=f;priv.inventories.resize(1);priv.inventory_order.resize(1);priv.inventories[0][E]=held;if(held)priv.inventory_order[0]={E};
  for(int i=0;i<9;i++){market.inventory[i]=10000;market.prices[i]=price(i,10000);}market.inventory[T]=inventory;market.prices[T]=price(T,inventory);
  ctl.day=29;ctl.phase=3;ctl.p.action_shadow=shadow;ctl.p.stepwise_recoordination=true;ctl.p.shared_task_atoms_v2=true;ctl.p.fix_logistics=true;
  ctl.target={{plot,T}};ctl.plans.resize(1);ctl.plans[0].a={action(Op::DROP)};ctl.plans[0].target={home};
 }
 View view()const{return {step,29,step%24,f,other,priv,market,shops};}
 Job job()const{Job j;j.pos=plot;j.actions={action(Op::HARVEST)};j.out[T]=f.tiles[plot].yield_units;return j;}
};
int main(){try{
 for(int r=0;r<4;r++)for(int t:{696,706,710,713,714,715,716,717,718})for(int d:{1,2,3})for(int q:{1,4})for(int held:{0,1,99})for(int inv:{9990,10070})for(double shadow:{0.,2.,10000.}){
  Fixture x(r,t,d,q,held,inv,shadow);auto v=x.view();auto before=x.ctl.plans;auto baseline=completion::audit(x.ctl,v,before);auto j=x.job();auto plans=before;
  bool accepted=completion::insert(x.ctl,v,plans,j,j);cases++;
  if(accepted){selected++;auto end=completion::audit(x.ctl,v,plans);require(completion::preserves(baseline,end),"selected plan effects not executable");
   require(completion::within(v,plans),"selected plan exceeds final action718");require(exchange::resources(v,plans),"selected plan spends unavailable resources");
   require(end.cash-baseline.cash>shadow*std::max(0,end.work-baseline.work)+1e-6,"choice not backed by its conditional net cash");
   require(end.sold[T]==q,"missing physical output sale");require(!completion::open_load(v,0,plans[0]),"selected route unclosed");
   require(end.stranded[T]==0&&end.stranded[E]==0,"selected output stranded");
   require(x.ctl.completion_last_cash_delta==end.cash-baseline.cash,"recorded value not from chosen plan");
   require(x.ctl.completion_last_work_cost==shadow*std::max(0,end.work-baseline.work),"chosen work value mismatch");
  }else require(Controller::same_remaining(plans[0],before[0]),"rejected alternative mutates old commitments");
 }
 // Reproduce the actual shared branch. These must fail with the legacy mode.
 Fixture shortday(0,715,1,2,1,10000,2);auto v=shortday.view();shortday.ctl.recoordinate(v);auto end=completion::audit(shortday.ctl,v,shortday.ctl.plans);
 require(end.sold[T]==2&&end.stranded[T]==0,"actual recoordinate appended output after last DROP");
 require(shortday.ctl.plans[0].a.size()==4&&shortday.ctl.plans[0].a.back().op==Op::DROP,"actual branch missing full return length");
 // Closing a pre-existing bad tail is separate from admitting missing work.
 Fixture bad(0,714,1,2,1,10000,2);int pos=45;Controller::walk(bad.ctl.plans[0],pos,46);bad.ctl.plans[0].a.push_back(action(Op::HARVEST));bad.ctl.plans[0].target.push_back(46);
 auto routes=bad.ctl.plans;require(completion::close(bad.ctl,bad.view(),routes),"profitable complete tail repair rejected");closed++;
 require(routes[0].a.size()==5&&routes[0].a.back().op==Op::DROP,"second return unaccounted");
 Fixture small(0,716,1,2,1,10000,2);auto old=small.ctl.plans;require(!completion::insert(small.ctl,small.view(),old,small.job(),small.job()),"one tick short admitted");
 Fixture floor(0,715,1,2,1,100000,2);old=floor.ctl.plans;require(!completion::insert(floor.ctl,floor.view(),old,floor.job(),floor.job()),"price floor hides labor cost");
 // A blocked other worker is not a blanket veto on a feasible positive return.
 Fixture separate(0,718,1,2,1,10000,4);separate.f.hands={{0,0}};separate.priv.inventories.push_back({});separate.priv.inventories[1][MI]=1;separate.priv.inventory_order.push_back({MI});std::vector<Plan>empty(2);
 require(completion::close(separate.ctl,separate.view(),empty),"unreachable peer vetoes profitable tail");auto solo=completion::audit(separate.ctl,separate.view(),empty);
 require(solo.sold[E]==1&&solo.stranded[MI]==1&&solo.executed,"individual closure not physically reachable");
 // Explicit per-item preservation, not total-value-only acceptance.
 auto a=completion::audit(shortday.ctl,v,shortday.ctl.plans),b=a;b.stranded[T]++;
 require(!completion::preserves(a,b),"stranded negative control accepted");b=a;b.sold[T]--;
 require(!completion::preserves(a,b),"missing realized output negative control accepted");b=a;b.executed=false;
 require(!completion::preserves(a,b),"failed execution negative control accepted");
 require(selected>0&&closed>0,"no positive behavior tested");
 std::cout<<"PASS cases="<<cases<<" selected="<<selected<<" closed="<<closed<<" checks="<<checks<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL checks="<<checks<<" cases="<<cases<<": "<<e.what()<<"\n";return 1;}}
