#include "../policy/search_selection.hpp"
#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
int main(){
 int checks=0;auto check=[&](bool ok){++checks;if(!ok)throw std::runtime_error("selector test failed");};
 {triad::MarginSelection s(50);s.observe(0,1000);s.observe(1,1050);check(s.winner==0);s.observe(2,1051);check(s.winner==2);check(s.gain()==51);}
 {triad::MarginSelection s(0);s.observe(0,-50000);s.observe(1,-50000);check(s.winner==0);s.observe(2,-49999);check(s.winner==2);}
 std::mt19937 rng(417);std::uniform_int_distribution<int>v(-100000,100000);
 for(int n=0;n<12000;++n){std::vector<double>a;int count=2+rng()%20;for(int i=0;i<count;++i)a.push_back(v(rng));
  double margin=rng()%201,offset=v(rng);triad::MarginSelection x(margin),y(margin),z(0);
  for(int i=0;i<count;++i){x.observe(i,a[i]);y.observe(i,a[i]+offset);z.observe(i,a[i]);}
  check(x.winner==y.winner);check(z.winner==int(std::max_element(a.begin(),a.end())-a.begin()));
  check(x.winner==0||x.gain()>margin);
 }
 for(double bad:{-1.,std::numeric_limits<double>::infinity()}){bool threw=false;try{triad::MarginSelection x(bad);}catch(const std::invalid_argument&){threw=true;}check(threw);}
 std::cout<<"checks="<<checks<<" failures=0\n";
}
