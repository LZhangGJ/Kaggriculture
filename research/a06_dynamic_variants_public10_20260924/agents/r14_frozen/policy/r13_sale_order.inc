// Only reorders existing SELL slots. It never changes quantities, input
// reservations, non-SELL positions, or observes the rival's private inventory.
// Public harvests give an uncertain upper bound, not a known warehouse.
void order_sales(const View&o,PlayerAction&out)const {
 std::vector<int>slots;Acts sales;
 for(int j=0;j<int(out.market.size());j++)if(out.market[j].op==Op::SELL){slots.push_back(j);sales.push_back(out.market[j]);}
 const int n=sales.size();if(n<2)return;
 int mode=int(base.r13_sale_order);
 std::array<int,9>rq{};std::array<double,9>loss{};
 for(int i=1;i<=7;i++){
  double daily=std::max(0.,live.model.rival[o.day][i]*base.supply);
  rq[i]=std::clamp(int(std::round(std::min(ledger.upper[i],std::max(6.,daily*.6)))),0,80);
 }
 auto duel=[&](int i,int q,int r,int relation){
  int inv=o.market.inventory[i];double a=0,b=0;
  auto sell=[&](int count,double&cash){for(int j=0;j<count;j++){int p=dp7::price(i,inv);cash+=p;if(p>1)inv++;}};
  if(relation<0){sell(q,a);sell(r,b);}else if(relation>0){sell(r,b);sell(q,a);}else {
   for(int j=0;j<std::max(q,r);j++){int p=dp7::price(i,inv);if(j<q){a+=p;inv+=p>1;}if(j<r){b+=p;inv+=p>1;}}
  }
  return a-base.competition*b;
 };
 if(mode<=2){
  auto score=[&](const Action&a){int i=int(a.item),q=std::max(0,a.quantity);
   if(mode==2)return dp7::revenue(i,o.market.inventory[i],q);
   return duel(i,q,rq[i],-1)-duel(i,q,rq[i],1);
  };
  std::stable_sort(sales.begin(),sales.end(),[&](const auto&a,const auto&b){return score(a)>score(b)+1e-9;});
 }else{
  std::vector<int>rival;for(int i=1;i<=7;i++)if(rq[i]>0)rival.push_back(i);
  if(rival.empty())return;
  auto byprice=rival,byvalue=rival;
  std::stable_sort(byprice.begin(),byprice.end(),[&](int a,int b){return o.market.prices[a]>o.market.prices[b];});
  std::stable_sort(byvalue.begin(),byvalue.end(),[&](int a,int b){return dp7::revenue(a,o.market.inventory[a],rq[a])>dp7::revenue(b,o.market.inventory[b],rq[b]);});
  double benefit[9][9]{};
  for(int k=0;k<n;k++)for(int j=0;j<n;j++){
   int i=int(sales[k].item),q=std::max(0,sales[k].quantity);
   for(const auto&order:{byprice,byvalue}){
    auto found=std::find(order.begin(),order.end(),i);
    if(found==order.end())continue;
    int pos=std::distance(order.begin(),found);
    int rel=(slots[j]>pos)-(slots[j]<pos);
    benefit[k][j]+=.375*duel(i,q,rq[i],rel);
   }
   // Reserve 25% weight for unknown rival queue order rather than treating
   // the two finite public-order hypotheses as certain.
   for(int pos=0;pos<int(rival.size());pos++){
    int rel=(slots[j]>pos)-(slots[j]<pos);
    benefit[k][j]+=.25/rival.size()*duel(i,q,rq[i],rel);
   }
  }
  std::vector<double>best(1<<n,-1e100);std::vector<int>pick(1<<n,-1);best[0]=0;
  for(int mask=0;mask<(1<<n);mask++){
   int j=std::popcount(unsigned(mask));if(j>=n)continue;
   for(int k=0;k<n;k++)if(!(mask&(1<<k))){int next=mask|(1<<k);double v=best[mask]+benefit[k][j];
    // Deterministic tiny preference for preserving the original order.
    v-=1e-8*std::abs(k-j);
    if(v>best[next]+1e-10){best[next]=v;pick[next]=k;}
   }
  }
  Acts chosen(n);int mask=(1<<n)-1;
  for(int j=n-1;j>=0;j--){int k=pick[mask];if(k<0)return;chosen[j]=sales[k];mask^=1<<k;}
  sales=std::move(chosen);
 }
 for(int j=0;j<n;j++)out.market[slots[j]]=sales[j];
}
