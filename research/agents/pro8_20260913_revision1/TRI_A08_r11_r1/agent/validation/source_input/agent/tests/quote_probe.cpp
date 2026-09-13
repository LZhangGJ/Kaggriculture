#include "../policy/sale_schedule_dp.hpp"
extern "C" int quote_pair(int item,int stock,int ours,int theirs,int order,double*out){
 if(item<0||item>8||ours<0||theirs<0||ours>100||theirs>100||order<0||order>2)return -1;
 auto r=triad::sale_dp::reference_trade(item,stock,ours,theirs,triad::sale_dp::Ordering(order));out[0]=r.own;out[1]=r.rival;out[2]=stock;return 0;
}
