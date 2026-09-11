#include "policy.hpp"
#include <iostream>
// Synthetic observed assets. No opponent rollout, actual future or labels.
int main(){
 using namespace dp7;fastkag::Simulator env(fastkag::Config{},77403);bool first_row=true;
 std::cout<<"{\"status\":\"CONDITIONAL_REPRESENTATION_AUDIT_NOT_STRENGTH\",\"rows\":[";
 for(int k:portfolio::kinds){
  auto own=env.farms()[0];auto priv=env.privates()[0];auto&t=own.tiles[44];
  if(k>=9){t.kind=TileKind::ANIMAL;t.animal=Item(k);t.placed_day=0;}
  else{t.kind=TileKind::PLANT;t.crop=Item(k);t.planted_day=0;}
  View v{0,0,0,own,env.farms()[1],priv,env.market(),env.shops()};Controller c;c.day=0;
  auto current=portfolio::physical_context(c,v);auto proposed=portfolio::delivery(c,k,0).cash;
  int item=k>=9?product[k-9]:k;double a=0,b=0;
  for(int d=0;d<30;d++){a+=current.own[d][item];b+=proposed.quantity[d][item];}
  if(!first_row)std::cout<<",";first_row=false;
  std::cout<<"{\"kind\":\""<<name(k)<<"\",\"existing_remaining_output\":"<<a<<",\"new_remaining_output\":"<<b<<"}";
 }
 std::cout<<"]}\n";
}
