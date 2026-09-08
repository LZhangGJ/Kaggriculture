// Standalone animal forecast diagnostic. No Python, ctypes, opponent, or game loop.
#include "../t1/policy/triad.hpp"
#include <iostream>
#include <iomanip>
int main() {
    triad::Settings s;
    s.work_price=4; s.animal_work=1; s.service=1;
    triad::Controller p(s);
    competitive::Flow prices{};
    for (int d=0;d<30;d++) for (int i=0;i<9;i++) prices[d][i]=dp7::price(i,10000);
    std::cout<<std::setprecision(17);
    for (int k=9;k<12;k++) {
        auto a=p.animal_path(k,0,44,prices);
        std::cout<<"animal "<<k<<" cost "<<a.first_cost<<" end "<<a.end<<"\n";
        for(int d=0;d<30;d++) {
            std::cout<<d<<" "<<a.labor[d]<<" "<<a.fixed[d];
            for (double x:a.f[d]) std::cout<<" "<<x;
            std::cout<<"\n";
        }
    }
}
