from pathlib import Path
p=Path('/mnt/data/TRI_A08_work/candidate/policy/sale_schedule_dp.hpp')
s=p.read_text()
s=s.replace('#include <sstream>','#include <sstream>\n#include <functional>\n#include <cmath>')
s=s.replace('// Certified non-floor region gives an exact inventory lattice indexed by\n// cumulative sold units. Saturating-price paths deliberately retain r10.', '''// The parent r11 used a cumulative-units inventory lattice and rejected every
// window whose upper supply bound reached the $1 floor. That handed a congested
// market back to the no-rival local hold rule. At the floor, actual sold units
// still earn cash, but do NOT increase market inventory. Keep the old fast
// lattice off-floor and use an explicit (tick, remaining, stock) state on-floor.
constexpr int MAX_FLOOR_STOCK_SPAN=1024;
inline bool non_saturating(const Problem&p){
 long long high=static_cast<long long>(p.stock)+p.quantity;
 for(int t=0;t<p.steps;t++)high+=p.rival[t];
 return high<=20000 && dp7::price(p.item,double(high))>1;
}''')
s=s.replace('return p.item>=1&&p.item<=7&&dp7::price(p.item,double(high))>1;', '''if(p.item<1||p.item>7)return false;
 if(dp7::price(p.item,double(high))<=1){
  const int floor=int(competitive::ConditionalMarket::saturation(p.item));
  const long long reachable_hi=std::min(high,static_cast<long long>(std::max(p.stock,floor+1)));
  if(reachable_hi-low+1>MAX_FLOOR_STOCK_SPAN)return false;
 }
 return true;''')
old='inline Schedule solve(const Problem&p,const Quotes&q,bool traffic,Ordering ordering,int*states=nullptr){'
new='''// Exact stock transition for both serial queue orderings and same-slot
// lockstep. A final paired quote above $1 may advance stock one unit beyond
// the serial saturation threshold; retaining that overshoot is essential.
inline int next_stock(int item,int stock,int own,int rival,Ordering order){
 const int floor=int(competitive::ConditionalMarket::saturation(item));
 if(stock>=floor)return stock;
 if(order!=LOCKSTEP)return stock+std::min(own+rival,floor-stock);
 const int paired=std::min(own,rival);
 stock+=2*std::min(paired,(floor-stock+1)/2);
 return stock+std::min(std::abs(own-rival),std::max(0,floor-stock));
}
inline Schedule solve_saturating(const Problem&p,const Quotes&q,bool traffic,Ordering ordering,int*states=nullptr){
 const double NEG=-1e100;
 int demand=0,rivals=0;
 for(int t=0;t<p.steps;t++){demand+=p.demand[t];rivals+=traffic?p.rival[t]:0;}
 const int lo=p.stock-demand;
 const int floor=int(competitive::ConditionalMarket::saturation(p.item));
 const int hi=std::min(p.stock+p.quantity+rivals,std::max(p.stock,floor+1));
 const int width=hi-lo+1;
 // supported() has already bounded this table. It is private to one decision
 // and only reachable states are expanded, never hypothetical future output.
 if(width<1||width>MAX_FLOOR_STOCK_SPAN)throw std::runtime_error("sale floor state span");
 const size_t count=size_t(p.steps)*(p.quantity+1)*width;
 std::vector<double>memo(count,std::numeric_limits<double>::quiet_NaN());
 std::vector<int16_t>take(count,-1);
 auto index=[&](int t,int rem,int stock){
  if(stock<lo||stock>hi)throw std::runtime_error("sale floor stock bounds");
  return (size_t(t)*(p.quantity+1)+rem)*width+stock-lo;
 };
 std::function<double(int,int,int)> value=[&](int t,int rem,int stock)->double{
  if(t==p.steps)return rem==0?0:NEG;
  const size_t ix=index(t,rem,stock);
  if(!std::isnan(memo[ix]))return memo[ix];
  if(states)(*states)++;
  double best=NEG;int chosen=0;
  const int enemy=traffic?p.rival[t]:0;
  int low=t==0?p.minimum_now:0;if(t==p.steps-1)low=rem;
  for(int amount=low;amount<=rem;amount++){
   const auto cash=q.trade(stock,amount,enemy,ordering);
   const int after=next_stock(p.item,stock,amount,enemy,ordering)-p.demand[t];
   const double future=value(t+1,rem-amount,after);
   if(future<NEG/2)continue;
   const double score=cash.own-p.competition*cash.rival+future;
   if(score>best+1e-9){best=score;chosen=amount;}
  }
  take[ix]=int16_t(chosen);memo[ix]=best;return best;
 };
 value(0,p.quantity,p.stock);
 Schedule result{};int rem=p.quantity,stock=p.stock;
 for(int t=0;t<p.steps;t++){
  int amount=take[index(t,rem,stock)];
  if(amount<0)throw std::runtime_error("sale floor unvisited decision");
  result[t]=amount;rem-=amount;
  stock=next_stock(p.item,stock,amount,traffic?p.rival[t]:0,ordering)-p.demand[t];
 }
 return result;
}
inline Schedule solve(const Problem&p,const Quotes&q,bool traffic,Ordering ordering,int*states=nullptr){
 if(!non_saturating(p))return solve_saturating(p,q,traffic,ordering,states);'''
assert old in s;s=s.replace(old,new)
s=s.replace('stock+=a[t]+rival-p.demand[t];','stock=next_stock(p.item,stock,a[t],rival,order)-p.demand[t];')
s=s.replace('long observed_slots_appended=0;','long observed_slots_appended=0;\n long floor_eligible=0,floor_changed=0;')
s=s.replace('eligible++;auto result=sale_dp::select(p);states+=result.states;','''eligible++;const bool saturating=!sale_dp::non_saturating(p);floor_eligible+=saturating;
   auto result=sale_dp::select(p);states+=result.states;''')
s=s.replace('log<<"{\\"item\\":"<<i<<",\\"warehouse_surplus_after_current_units\\":"<<amount', 'log<<"{\\"item\\":"<<i<<",\\"saturating_market_dp\\":"<<(saturating?"true":"false")<<",\\"warehouse_surplus_after_current_units\\":"<<amount')
s=s.replace('if(now>sell[i]){changed++;advanced_units', 'if(now>sell[i]){changed++;floor_changed+=saturating;advanced_units')
p.write_text(s)
# Only telemetry, not decision logic, changes in the enclosing controller.
p=Path('/mnt/data/TRI_A08_work/candidate/policy/triad.hpp');s=p.read_text()
s=s.replace('<<",\\"sale_schedule_floor_fallback\\":"<<sale_schedule.skip_floor', '<<",\\"sale_schedule_floor_fallback\\":"<<sale_schedule.skip_floor<<",\\"sale_schedule_floor_eligible\\":"<<sale_schedule.floor_eligible<<",\\"sale_schedule_floor_changed\\":"<<sale_schedule.floor_changed')
p.write_text(s)
