#include "policy/proposals.hpp"
#include <iostream>
int main(){fastkag::Simulator e({},101);dp7::View v{0,0,0,e.farms()[0],e.farms()[1],e.privates()[0],e.market(),e.shops()};triad::Controller c;c.plan(v);auto f=triad::portfolio_features(v,c,c,0,true);std::cout<<"[";for(size_t i=0;i<f.names.size();i++){if(i)std::cout<<",";std::cout<<"\""<<f.names[i]<<"\"";}std::cout<<"]\n";}
