#include "../policy/executor/policy.hpp"
#include <iostream>
#include <map>
#include <cstdlib>
using namespace dp7;
static int checks = 0;
static void check(bool ok, const char* what) {
 ++checks; if (!ok) { std::cerr << "FAIL " << what << "\n"; std::exit(1); }
}
using Key=std::tuple<int,int,int>;
static Key key(const Action& a) { return {int(a.op),int(a.item),a.quantity}; }
static bool equal(const Acts& a,const Acts& b) {
 if(a.size()!=b.size())return false;
 for(size_t i=0;i<a.size();++i)if(key(a[i])!=key(b[i]))return false;
 return true;
}
static void emit(const Acts& a) {
 const char* names[]={"PASS","NORTH","SOUTH","EAST","WEST","DROP","PICKUP","PLACE","PLANT","WATER","HARVEST","FERTILIZE","DIG","BUILD_COOP","BUILD_PASTURE","FEED","COLLECT_FERTILIZER","CARE","HIRE","BUY_LAND","BUY_SEED","BUY_PRODUCT","BUY_ANIMAL","SELL"};
 const char* items[]={"WHEAT","CARROT","TOMATO","STRAWBERRY","MELON","EGG","MILK","WOOL","FERTILIZER","GOOSE","COW","SHEEP"};
 std::cout << '[';bool comma=false;
 for(auto x:a){if(comma)std::cout<<',';comma=true;std::cout<<"[\""<<names[int(x.op)]<<"\"";if(int(x.item)>=0)std::cout<<",\""<<items[int(x.item)]<<"\","<<x.quantity;std::cout<<']';}std::cout<<']';
}
int main() {
 Farm own,rival;own.tiles.resize(100);rival.tiles.resize(100);
 PrivateState priv;Market market;std::vector<int8_t> shops;
 View v{216,9,0,own,rival,priv,market,shops};
 auto& t=own.tiles[0];t.kind=TileKind::PLANT;t.crop=Item::STRAWBERRY;t.planted_day=7;t.consecutive_unwatered=1;
 Job j{};j.pos=0;j.crop=true;j.actions={action(Op::WATER)};std::vector<Job> js{j};
 Acts sales{action(Op::SELL,MI,14),action(Op::SELL,F,9)};
 std::vector<Order> purchases{{action(Op::BUY_PRODUCT,W,12),0},{action(Op::BUY_LAND),1},{action(Op::BUY_ANIMAL,G,3),2},{action(Op::BUY_SEED,S,4),3},{action(Op::BUY_SEED,W,3),3},{action(Op::BUY_SEED,T,1),3}};
 own.money=482;
 check(a06::reserve_ongoing_wages(v,js,sales,3500,7,33),"observed-risk positive");
 own.money=3533;check(!a06::reserve_ongoing_wages(v,js,sales,3500,7,33),"cash funded equality");
 own.money=482;check(!a06::reserve_ongoing_wages(v,js,{},3500,7,33),"no unconfirmed sale");
 check(!a06::reserve_ongoing_wages(v,js,sales,3500,0,0),"no planned hire");
 check(!a06::reserve_ongoing_wages(v,{},sales,3500,7,33),"no job");
 for(int kind=0;kind<5;++kind)for(int birth=0;birth<30;++birth)for(int day=0;day<30;++day)for(int dry=0;dry<2;++dry)for(int wet=0;wet<2;++wet){
  t.crop=Item(kind);t.planted_day=birth;t.consecutive_unwatered=dry;t.watered_today=wet;v.day=day;
  bool want=false;
  if((kind==T||kind==S)&&dry&&!wet)for(int wave=0;wave<4;++wave){int due=birth+(kind==T?8:10)+wave*(kind==T?1:2);want|=due>day&&due<=29;}
  check(a06::ongoing_survival_water(v,js)==want,"all crop/date/dry/wet boundaries");
 }
 t.crop=Item::STRAWBERRY;t.planted_day=7;t.consecutive_unwatered=1;t.watered_today=false;v.day=9;
 for(auto op:{Op::HARVEST,Op::FERTILIZE,Op::PASS}){js[0].actions={action(op)};check(!a06::ongoing_survival_water(v,js),"no WATER action");}
 js[0].actions={action(Op::DIG),action(Op::PLANT,S),action(Op::WATER)};check(!a06::ongoing_survival_water(v,js),"new replacement water is not survival");
 js[0].actions={action(Op::FERTILIZE),action(Op::WATER)};check(a06::ongoing_survival_water(v,js),"fertilize then water remains incumbent maintenance");
 t.kind=TileKind::WEED;check(!a06::ongoing_survival_water(v,js),"already dead");t.kind=TileKind::PLANT;
 js[0].pos=-1;check(!a06::ongoing_survival_water(v,js),"negative position");js[0].pos=100;check(!a06::ongoing_survival_water(v,js),"position upper bound");js[0].pos=0;
 for(int h=0;h<=14;++h){
  Acts old=sales;for(auto x:purchases)old.push_back(x.a);for(int i=0;i<h;++i)old.push_back(action(Op::HIRE));
  auto legacy=a06::preparation_with_wage_reserve(sales,purchases,h,false);check(equal(old,legacy),"negative path byte-for-byte order");
  auto sorted=a06::preparation_with_wage_reserve(sales,purchases,h,true);
  std::map<Key,int> counts;for(auto a:old)counts[key(a)]++;for(auto a:sorted)counts[key(a)]--;for(auto[k,n]:counts)check(n==0,"no order/quantity additions or deletions");
  check(sorted.size()==old.size(),"same queue length and max-orders setup budget");
  for(size_t i=0;i<sales.size();++i)check(key(sorted[i])==key(sales[i]),"sale order unchanged");
  check(key(sorted[2])==key(purchases[0].a),"current feed before wages");
  for(int i=0;i<h;++i)check(sorted[3+i].op==Op::HIRE,"contiguous reserved hires");
  for(size_t i=1;i<purchases.size();++i)check(key(sorted[2+h+i])==key(purchases[i].a),"discretionary purchases stable");
 }
 auto extras=purchases;extras.push_back({action(Op::BUY_PRODUCT,W,20),4});extras.push_back({action(Op::BUY_PRODUCT,F,4),4});
 auto order=a06::preparation_with_wage_reserve(sales,extras,7,true);
 check(order[order.size()-2].quantity==20&&order.back().quantity==4,"buffer and fertilizer not promoted before wages");
 auto legacy=a06::preparation_with_wage_reserve(sales,purchases,7,false);
 auto protectedq=a06::preparation_with_wage_reserve(sales,purchases,7,true);
 std::cout << "{\"checks\":"<<checks<<",\"failed\":0,\"original_orders\":";emit(legacy);std::cout<<",\"protected_orders\":";emit(protectedq);std::cout<<"}\n";
}
