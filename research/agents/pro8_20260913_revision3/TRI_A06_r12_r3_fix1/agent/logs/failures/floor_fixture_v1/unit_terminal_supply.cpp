#include "../policy/triad.hpp"
#include <iostream>
#include <cstdlib>
#include <limits>
using namespace dp7;
static int checks=0;
void ck(bool x,const char*m){++checks;if(!x){std::cerr<<"FAIL "<<m<<"\n";std::exit(1);}}
struct Fixture {
 Farm own,rival;PrivateState priv;Market market;std::vector<int8_t> shops;triad::Controller c;
 Fixture(int crop_inventory,int fertilizer_inventory):c([](){triad::Settings s;s.max_hands=0;s.max_land=4;s.intraday=0;s.a06_r11_recovery=0;s.delay_sale=-1;return s;}()){
  own.tiles.resize(100);rival.tiles.resize(100);own.unlocked_mask=15;own.farmer={4,4};own.money=1000;
  priv.inventories.resize(1);priv.inventory_order.resize(1);market.inventory.fill(10000);market.inventory[S]=crop_inventory;market.inventory[F]=fertilizer_inventory;
  for(int k=0;k<9;k++)market.prices[k]=price(k,market.inventory[k]);
  auto&t=own.tiles[34];t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=15;t.fertilized_until_day=26;t.consecutive_unwatered=1;
  c.core.day=28;c.core.crop_service_day=28;c.core.target={{34,S}};c.core.crop_birth[34]=15;c.core.crop_kind[34]=S;c.core.crop_water[34]=c.core.crop_fertilize[34]=1;
 }
 View view(int hour=19){return{672+hour,28,hour,own,rival,priv,market,shops};}
 int buys(){int q=0;for(auto a:c.core.queue)if(a.op==Op::BUY_PRODUCT&&int(a.item)==F)q+=a.quantity;return q;}
};
int main(){
 const double v[]={-100.,-1.,0.,1e-7,1.,10.,100.};
 for(double a:v)for(double b:v)for(double c:v)for(double d:v)
  ck(triad::Controller::terminal_supply_profitable({a,b},{c,d})==(a>1e-6&&b>1e-6&&c>1e-6&&d>1e-6),"both own and relative cash positive in both scenarios");
 ck(!triad::Controller::terminal_supply_profitable({1.,1.},{1.,-1.}),"own profit that subsidizes rival is rejected");
 ck(!triad::Controller::terminal_supply_profitable({1.,-1.},{1.,1.}),"own loss not admitted through relative score");
 ck(!triad::Controller::terminal_supply_profitable({std::numeric_limits<double>::infinity(),1.},{1.,1.}),"infinite forecast rejected");
 ck(!triad::Controller::terminal_supply_profitable({1.,1.},{std::numeric_limits<double>::quiet_NaN(),1.}),"NaN forecast rejected");
 {Fixture f(0,10000);auto v=f.view();auto cash=f.own.money;auto shed=f.priv.shed;f.c.prepare_with_ongoing_supply(v,v,0);
  std::cout<<"positive_fixture "<<f.buys()<<" "<<f.c.ongoing_terminal_last<<"\n";
  ck(f.c.ongoing_terminal_checks==1,"terminal positive invokes endpoint check");ck(f.buys()==1,"reachable high-price terminal bonus retained");
  ck(f.c.ongoing_terminal_rejected==0&&f.c.ongoing_terminal_errors==0,"positive has no rejection/error");
  ck(f.own.money==cash&&f.priv.shed==shed,"scenario cannot mutate actual funds/stock");}
 {Fixture f(10000,10000);auto v=f.view();f.c.prepare_with_ongoing_supply(v,v,0);
  std::cout<<"floor_fixture "<<f.buys()<<" "<<f.c.ongoing_terminal_last<<"\n";
  ck(f.c.ongoing_terminal_checks==1,"floor case is physically reachable");ck(f.buys()==0&&f.c.ongoing_terminal_rejected==1,"completed but zero net-cash bonus rejected");}
 {Fixture f(0,10000);f.own.money=0;auto v=f.view();f.c.prepare_with_ongoing_supply(v,v,0);ck(f.buys()==0,"no financing cannot buy");ck(f.c.ongoing_terminal_checks==0,"no unjustified terminal simulation when no purchase");}
 {Fixture f(0,10000);f.priv.shed[F]=1;auto v=f.view();f.c.prepare_with_ongoing_supply(v,v,0);ck(f.buys()==0&&f.c.ongoing_terminal_checks==0,"existing resource never repurchased");}
 std::cout<<"{\"checks\":"<<checks<<",\"failed\":0,\"scope\":\"terminal executed-cash admission, positive/negative economics, real deadline, cash and stock\"}\n";
}
