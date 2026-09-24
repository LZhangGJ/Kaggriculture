#include "../policy/triad.hpp"
#include <iostream>
#include <cstdlib>
using namespace dp7;
static int checks=0;
void check(bool ok,const char* text){++checks;if(!ok){std::cerr<<"FAIL "<<text<<"\n";std::exit(1);}}
int buys(const std::vector<Order>&xs){int n=0;for(auto x:xs)if(x.a.op==Op::BUY_PRODUCT&&int(x.a.item)==F)n+=x.a.quantity;return n;}
int fneeds(const std::vector<Job>&js){int n=0;for(auto&j:js)n+=j.needs[F];return n;}
int main(){
 Farm own,rival;own.tiles.resize(100);rival.tiles.resize(100);own.unlocked_mask=15;own.farmer={4,4};own.money=35909;
 PrivateState priv;priv.inventories.resize(1);priv.inventory_order.resize(1);
 Market market;market.inventory.fill(10000);market.inventory[F]=10301;for(int k=0;k<9;k++)market.prices[k]=price(k,market.inventory[k]);
 std::vector<int8_t>shops;View v{480,20,0,own,rival,priv,market,shops};
 Controller c;c.day=20;c.crop_service_day=20;c.p.procure_service_inputs=true;c.p.renew_ongoing=true;
 for(int pos=0;pos<22;pos++){auto&t=own.tiles[pos];t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=7;t.fertilized_until_day=18;t.consecutive_unwatered=1;
  c.target.push_back({pos,S});c.crop_birth[pos]=7;c.crop_kind[pos]=S;c.crop_water[pos]=c.crop_fertilize[pos]=1;}
 priv.shed[F]=13;auto js=c.jobs(v);check(fneeds(js)==22,"22 approved services");check(c.approved_ongoing_fertilizer_gap(v,js)==9,"13 stock -> 9 resource-degraded services");
 check(buys(c.orders(v,js))==9,"late preparation bridges selected shortfall");
 auto a=c.admit(v,c.orders(v,js),own.money);check(buys(a)==9,"adequate actual money admits nine");
 double total=0;for(auto x:a)total+=c.cost(v,x.a);check(total>0&&total<own.money,"purchase charged actual increasing unit cost");
 auto pr=priv;for(auto x:a)if(x.a.op==Op::BUY_PRODUCT)pr.shed[int(x.a.item)]+=x.a.quantity;
 View funded{480,20,0,own,rival,pr,market,shops};check(fneeds(c.reserve(funded,js))==22,"funded reservation retains all services");
 auto no=c.admit(v,c.orders(v,js),0);check(no.empty(),"no cash cannot buy");
 for(int cash=0;cash<600;cash++){auto aa=c.admit(v,c.orders(v,js),cash);double cc=0;for(auto x:aa)cc+=c.cost(v,x.a);check(cc<=cash,"partial buy cannot overspend");check(buys(aa)<=9,"partial buy cannot exceed selected gap");}
 auto before=c.resource_degraded;for(int i=0;i<8;i++)c.approved_ongoing_fertilizer_gap(v,js);check(c.resource_degraded==before,"read-only reservation check");
 priv.shed[F]=22;check(c.approved_ongoing_fertilizer_gap(v,js)==0,"fully stocked negative");check(buys(c.orders(v,js))==0,"no speculative stock buffer");priv.shed[F]=13;
 c.crop_service_day=19;check(c.approved_ongoing_fertilizer_gap(v,js)==0,"stale service day");c.crop_service_day=20;
 c.day=19;check(c.approved_ongoing_fertilizer_gap(v,js)==0,"clock disagreement");c.day=20;
 c.p.procure_service_inputs=false;check(c.approved_ongoing_fertilizer_gap(v,js)==0,"procurement disabled");c.p.procure_service_inputs=true;
 for(int pos=0;pos<22;pos++)c.crop_fertilize[pos]=0;check(c.approved_ongoing_fertilizer_gap(v,js)==0,"rejected service cannot be re-created");
 auto rejected=c.jobs(v);check(fneeds(rejected)==0&&buys(c.orders(v,rejected))==0,"rejected economic plan buys no fertilizer");
 for(int pos=0;pos<22;pos++)c.crop_fertilize[pos]=1;
 auto malformed=js;for(auto&j:malformed){j.actions.erase(std::remove_if(j.actions.begin(),j.actions.end(),[](auto a){return a.op==Op::WATER;}),j.actions.end());}
 check(c.approved_ongoing_fertilizer_gap(v,malformed)==0,"no bonus without water or planned water");
 for(int pos=0;pos<22;pos++)own.tiles[pos].watered_today=true;check(c.approved_ongoing_fertilizer_gap(v,malformed)==9,"already watered ongoing crop can still receive bonus");
 for(int pos=0;pos<22;pos++)own.tiles[pos].watered_today=false;
 for(auto&j:malformed)j.actions.push_back(action(Op::WATER));for(auto&j:malformed)j.actions.push_back(action(Op::DIG));check(c.approved_ongoing_fertilizer_gap(v,malformed)==0,"replacement is not incumbent maintenance");
 for(int pos=0;pos<22;pos++)c.crop_birth[pos]=6;check(c.approved_ongoing_fertilizer_gap(v,js)==0,"replanted identity mismatch");
 for(int pos=0;pos<22;pos++)c.crop_birth[pos]=7;
 // Date / species / birth boundaries, including day 15 and the last day.
 for(int kind:{T,S})for(int day=0;day<30;day++)for(int birth=0;birth<30;birth++){
  Controller z;z.day=day;z.crop_service_day=day;z.p.procure_service_inputs=true;z.target={{0,kind}};z.crop_birth[0]=birth;z.crop_kind[0]=kind;z.crop_water[0]=z.crop_fertilize[0]=1;
  auto saved=own.tiles[0];auto&t=own.tiles[0];t.crop=Item(kind);t.planted_day=birth;t.fertilized_until_day=-1;t.watered_today=false;
  auto pv=priv;pv.shed[F]=0;View vv{day*24,day,0,own,rival,pv,market,shops};Job j{};j.pos=0;j.crop=true;j.priority=0;j.needs[F]=1;j.actions={action(Op::FERTILIZE),action(Op::WATER)};
  int due=day+1-birth-first[kind];int want=(day>=15&&day<29&&due>=0&&due%interval[kind]==0&&due/interval[kind]<4)?1:0;
  check(z.approved_ongoing_fertilizer_gap(vv,{j})==want,"all ongoing date boundaries");own.tiles[0]=saved;
 }
 // The exact resource order matters: do not buy to fill finite-only gaps.
 Controller z;z.day=20;z.crop_service_day=20;z.p.procure_service_inputs=true;z.crop_birth[0]=7;z.crop_kind[0]=S;z.crop_fertilize[0]=z.crop_water[0]=1;
 Job ongoingj{};ongoingj.pos=0;ongoingj.priority=0;ongoingj.crop=true;ongoingj.needs[F]=1;ongoingj.actions={action(Op::FERTILIZE),action(Op::WATER)};
 Job finitej=ongoingj;finitej.pos=99;finitej.priority=5;own.tiles[99].kind=TileKind::PLANT;own.tiles[99].crop=Item::MELON;own.tiles[99].planted_day=10;priv.shed[F]=1;
 check(z.approved_ongoing_fertilizer_gap(v,{ongoingj,finitej})==0,"unrelated finite shortage not purchased");finitej.priority=-1;
 check(z.approved_ongoing_fertilizer_gap(v,{ongoingj,finitej})==1,"higher priority finite reservation honored");
 priv.shed[F]=0;priv.inventories[0][F]=10;check(z.approved_ongoing_fertilizer_gap(v,{ongoingj})==1,"future cargo not credited as warehouse inputs");priv.inventories[0][F]=0;
 check(z.approved_ongoing_fertilizer_gap(v,{finitej})==0,"finite-only negative");
 // Price-based negative and positive selection come from the unchanged DP.
 triad::Settings settings;settings.work_price=4;settings.crop_fert=1;triad::Controller ec(settings);ec.core.day=20;ec.core.target={{0,S}};
 for(int d=0;d<30;d++){ec.prices[d][S]=1;ec.prices[d][F]=40;}
 ec.set_service(v);check(ec.core.crop_fertilize[0]==0,"floor-price production is not worth fertilizer");check(buys(ec.core.orders(v,ec.core.jobs(v)))==0,"economically rejected service remains rejected");
 for(int d=0;d<30;d++)ec.prices[d][S]=219;ec.set_service(v);check(ec.core.crop_fertilize[0]==1,"high-value ongoing service selected by existing DP");check(buys(ec.core.orders(v,ec.core.jobs(v)))==1,"selected high-value service has preparation supply");
 // Former cutoff's early path is unchanged even without service metadata.
 c.day=v.day=14;v.step=336;c.crop_service_day=-1;priv.shed[F]=13;check(buys(c.orders(v,js))==9,"pre-day15 legacy procurement unchanged");
 c.day=v.day=29;v.step=696;c.crop_service_day=29;check(buys(c.orders(v,js))==0,"no terminal maintenance procurement");
 std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"scope\":\"approved ongoing fertilizer preparation, price rejection and resource/cash/date negatives\"}\n";
}
