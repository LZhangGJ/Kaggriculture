#include "planner.hpp"
#include <cassert>

int main(){
 competitive::Planner p;p.day=12;p.use_rival_forecast=true;p.rival_forecast[17][7]=41.5;
 fastkag::Farm own,rival;own.tiles.resize(100);rival.tiles.resize(100);
 fastkag::PrivateState priv;fastkag::Market market;std::vector<int8_t>shops;
 dp7::View view{288,12,0,own,rival,priv,market,shops};
 p.public_rival(view);
 assert(p.rival[17][7]==41.5);
}
