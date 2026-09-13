#define A08_SALE_SCHEDULE_DP 1
#include "../policy/sale_schedule_dp.hpp"
#include <iostream>
int main(){int item,stock,own,rival,order;while(std::cin>>item>>stock>>own>>rival>>order){
 triad::sale_dp::Problem p;p.item=item;p.stock=stock;p.quantity=std::max(1,own);p.steps=1;p.rival[0]=rival;
 if(!triad::sale_dp::supported(p)){std::cout<<"UNSUPPORTED\n";continue;}
 triad::sale_dp::Quotes q(p);auto c=q.trade(stock,own,rival,triad::sale_dp::Ordering(order));
 std::cout<<c.own<<" "<<c.rival<<" "<<triad::sale_dp::next_stock(item,stock,own,rival,triad::sale_dp::Ordering(order))<<"\n";
}return 0;}
