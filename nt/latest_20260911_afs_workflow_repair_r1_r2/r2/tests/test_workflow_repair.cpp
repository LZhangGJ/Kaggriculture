#include "../policy/executor/policy.hpp"
#include <iostream>
#include <stdexcept>

using namespace dp7;
static int checks=0;
static void require(bool ok,const char*message){checks++;if(!ok)throw std::runtime_error(message);}

struct Fixture {
 Farm own,opponent;PrivateState priv;Market market;std::vector<int8_t>shops;
 int day=2,hour=21;
 Fixture(){
  own.tiles.resize(100);opponent.tiles.resize(100);own.farmer={4,4};opponent.farmer={4,4};
  own.unlocked_mask=opponent.unlocked_mask=15;priv.inventories.resize(1);priv.inventory_order.resize(1);
  for(int i=0;i<9;i++)market.prices[i]=price(i,0);
 }
 View view()const{return {day*24+hour,day,hour,own,opponent,priv,market,shops};}
 Controller controller()const{Controller c;c.day=day;c.phase=3;c.plans.resize(priv.inventories.size());c.workflow.day=day;c.workflow.captured=true;return c;}
};

int main(){try{
 require(P16_AFS_WORKFLOW_REPAIR==1,"repair test must compile enabled");
 for(int crop=0;crop<5;crop++)for(int pos:{34,43,45,54})for(int hour:{21,22}){
  Fixture x;x.hour=hour;x.priv.seeds[crop]=1;auto c=x.controller();c.target={{pos,crop}};
  c.workflow.required[Controller::task_key(pos,action(Op::PLANT,crop))]=1;
  c.workflow.required[Controller::task_key(pos,action(Op::WATER))]=1;
  workflow::inspect(c,x.view());auto proof=workflow::verify(x.view(),c.plans);
  if(hour==21)require(proof.valid&&workflow::covers(proof.completed,workflow::debt(c.workflow)),"complete crop chain restored");
  else require(c.plans[0].a.empty()&&c.workflow.unresolved>0,"late crop chain rejected atomically");
 }
 for(int pos:{34,43,45,54}){
  Fixture x;x.hour=19;x.priv.shed[W]=1;auto&t=x.own.tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item::GOOSE;t.placed_day=0;t.yield_units=2;
  auto c=x.controller();c.target={{pos,G}};int from=44;Controller::walk(c.plans[0],from,pos);
  c.plans[0].a.push_back(action(Op::HARVEST));c.plans[0].target.push_back(pos);
  for(auto op:{Op::HARVEST,Op::FEED,Op::CARE})c.workflow.required[Controller::task_key(pos,action(op))]=1;
  workflow::inspect(c,x.view());auto proof=workflow::verify(x.view(),c.plans);
  require(proof.valid&&workflow::covers(proof.completed,workflow::debt(c.workflow)),"animal chain restored");
  require(proof.harvested.at(Controller::task_key(pos,action(Op::HARVEST)))==2,"animal harvest preserved");
  require(c.plans[0].a.front().op==Op::PICKUP,"animal feed pickup inserted");
 }
 for(int pos:{34,43,45,54})for(int milk_price:{1,100,300}){
  Fixture x;x.market.prices[MI]=milk_price;x.priv.seeds[T]=1;auto c=x.controller();c.target={{pos,T}};
  auto plant_key=Controller::task_key(pos,action(Op::PLANT,T)),water_key=Controller::task_key(pos,action(Op::WATER));
  c.workflow.required={{plant_key,1},{water_key,1}};Plan witness;int from=44;Controller::walk(witness,from,pos);
  witness.a.push_back(action(Op::PLANT,T));witness.target.push_back(pos);witness.a.push_back(action(Op::WATER));witness.target.push_back(pos);
  c.workflow.witness={witness};workflow::inspect(c,x.view());auto proof=workflow::verify(x.view(),c.plans);
  require(proof.valid&&workflow::covers(proof.completed,workflow::debt(c.workflow))&&c.workflow.witness_used==1,"saved witness reused independent of price");
 }
 {
  Fixture x;x.priv.seeds[T]=1;Plan plan{{action(Op::WATER),action(Op::PLANT,T)},{44,44},0};
  require(!workflow::verify(x.view(),{plan}).valid,"water before plant rejected");
 }
 {
  Fixture x;x.day=29;x.hour=22;x.priv.seeds[T]=1;Plan plan{{action(Op::PLANT,T),action(Op::WATER)},{44,44},0};
  require(!workflow::verify(x.view(),{plan}).valid,"chain beyond terminal rejected");
 }
 {
  Fixture x;x.hour=18;x.priv.seeds[T]=1;auto&t=x.own.tiles[43];t.kind=TileKind::PLANT;t.crop=Item::WHEAT;t.planted_day=0;t.yield_units=1;t.consecutive_unwatered=1;
  auto c=x.controller();c.target={{43,W},{45,T}};int from=44;Controller::walk(c.plans[0],from,43);c.plans[0].a.push_back(action(Op::WATER));c.plans[0].target.push_back(43);
  c.workflow.required[Controller::task_key(43,action(Op::WATER))]=1;c.workflow.required[Controller::task_key(45,action(Op::PLANT,T))]=1;c.workflow.required[Controller::task_key(45,action(Op::WATER))]=1;
  workflow::inspect(c,x.view());auto proof=workflow::verify(x.view(),c.plans);
  require(proof.valid&&workflow::covers(proof.completed,workflow::debt(c.workflow)),"existing work preserved while chain restored");
  require(c.target==std::vector<std::pair<int,int>>({{43,W},{45,T}}),"economic targets unchanged");
 }
 {
  Fixture x;x.hour=3;auto&t=x.own.tiles[44];t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.placed_day=0;t.consecutive_unfed=1;
  x.priv.inventories[0][W]=1;x.priv.inventory_order[0].push_back(W);auto c=x.controller();auto key=Controller::task_key(44,action(Op::FEED));c.workflow.required[key]=1;
  Acts scheduled{action(Op::FEED)};PlayerAction out;out.units={action(Op::NORTH)};
  workflow::protect(c,x.view(),scheduled,out);
  require(out.units[0].op==Op::FEED&&c.workflow.output_restores==1,"certified atom restored after late override");
 }
 {
  // A later recoordination must not spend the only wheat on a new target and
  // silently abandon the already certified target.
  Fixture x;x.hour=3;x.priv.shed[W]=1;
  for(int pos:{44,45}){auto&t=x.own.tiles[pos];t.kind=TileKind::ANIMAL;t.animal=Item::COW;t.placed_day=0;t.consecutive_unfed=1;}
  auto c=x.controller();auto due=Controller::task_key(44,action(Op::FEED));c.workflow.required[due]=1;
  Plan witness;witness.a={action(Op::PICKUP,W,1),action(Op::FEED)};witness.target={44,44};
  c.workflow.witness={witness};c.workflow.latest_cover={witness};
  c.plans[0].a={action(Op::PICKUP,W,1),action(Op::FEED)};c.plans[0].target={44,45};
  workflow::inspect(c,x.view());auto proof=workflow::verify(x.view(),c.plans);
  require(proof.valid&&workflow::covers(proof.completed,workflow::debt(c.workflow)),"certified target wins impossible shared-input conflict");
  require(proof.completed.find(Controller::task_key(45,action(Op::FEED)))==proof.completed.end(),"new target cannot steal certified input");
 }
 std::cout<<"{\"status\":\"PASS\",\"checks\":"<<checks<<"}\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAILED after "<<checks<<": "<<e.what()<<"\n";return 1;}}
