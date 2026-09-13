#include "policy/planner.hpp"
extern "C" {
double probe_legacy(int item,double stock,double quantity,double*out){double p=competitive::Planner::quote(item,stock,quantity);if(quantity<0||p>1)stock+=quantity;*out=stock;return quantity*p;}
double probe_floor_safe(int item,double stock,double quantity,double*out){double cash;if(quantity>0 && stock+quantity>competitive::ConditionalMarket::saturation(item))cash=competitive::ConditionalMarket::execute(item,stock,quantity);else {double p=competitive::Planner::quote(item,stock,quantity);if(quantity<0||p>1)stock+=quantity;cash=quantity*p;}*out=stock;return cash;}
double probe_exact(int item,double stock,double quantity,double*out){double cash=competitive::ConditionalMarket::execute(item,stock,quantity);*out=stock;return cash;}
double probe_price(int item,double stock){return dp7::price(item,stock);}
double probe_saturation(int item){return competitive::ConditionalMarket::saturation(item);}
}
