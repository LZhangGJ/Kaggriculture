
// Storage is a bounded public-information alternative, not a price forecast
// guarantee. Keep only goods already in the post-unit shed; inputs and the
// complete existing return wave reserve capacity first. No new purchase is
// made by this module, and the terminal day never retains stock.
void storage_filter(const View&o,const fastkag::PrivateState&priv,const Acts&buys,
                    const Counts&reserve,Counts&sell){
 if(s.r15_storage<=0||o.day>=28||o.own.money<3000||!buys.empty())return;
 int autoload=std::max(0,core.expected_auto_deposit(o)-sum(sell));
 int room=std::clamp(100-core.p.shed_safety-autoload,0,50);
 int postload=sum(priv.shed);for(const auto&b:priv.inventories)postload+=sum(b);
 room=std::min(room,std::max(0,90-postload+sum(sell)));
 if(room<=0)return;
 struct Offer{int item,quantity;double per_unit,gain;};std::vector<Offer>offers;
 auto known=PublicTradeLedger::consumption((o.step/24+1)*24,o.shops);
 // Exact known daily shop demand (no unseen future-shop draw).
 static const std::array<std::vector<int>,8>shop_products{{{E,W},{E,W,S},{W,C,T,S},{S,MI,W},{C,C},{MI,T,W},{S,MI},{WO,WO}}};
 std::array<double,9>demand{};for(int i=0;i<8;i++)demand[i]=1;
 for(int sh:o.shops)for(int i:shop_products[sh])demand[i]+=6;
 for(int item=1;item<8;item++){
  int n=std::min(sell[item],room);if(n<=0||demand[item]<=1)continue;
  int ceiling=s.r15_storage>=2?100:20;
  if(o.market.prices[item]>ceiling)continue;storage_checks++;
  auto payoff=[&](int wait,int q,double rival_scale){
   double stock=o.market.inventory[item],ours=0,enemy=0;
   if(wait==0)ours=ConditionalMarket::execute(item,stock,q);
   int end=std::min(29,o.day+5);
   for(int d=o.day;d<=end;d++){
    double part=d==o.day?double(24-o.hour)/24.:1.;
    // Other present holdings are sold by the unmodified sale compiler.
    if(d==o.day)ConditionalMarket::execute(item,stock,std::max(0,sell[item]-q));
    double r=std::max(0.,model.rival[d][item])*s.supply*rival_scale*part;
    stock-=demand[item]*part*.5;
    enemy+=ConditionalMarket::execute(item,stock,r*.5);
    if(wait>0&&d==o.day+wait)ours+=ConditionalMarket::execute(item,stock,q)/std::pow(1.02,wait);
    // The conditional own portfolio is not extra physical inventory.
    if(d>o.day)ConditionalMarket::execute(item,stock,std::max(0.,portfolio.f[d][item]));
    enemy+=ConditionalMarket::execute(item,stock,r*.5);
    stock-=demand[item]*part*.5;
   }
   return ours-s.competition*enemy;
  };
  double best=0;int bestq=0;
  for(int q:{std::min(n,8),std::min(n,24),n})for(int h=1;h<=std::min(5,29-o.day);h++){
   double robust=1e100;
   for(double scale:{.7,1.3})robust=std::min(robust,payoff(h,q,scale)-payoff(0,q,scale));
   // Explicit storage/action opportunity cost prevents tiny quote churn.
   robust-=q*h*2.;
   if(robust>best+1e-6){best=robust;bestq=q;}
  }
  if(bestq>0)offers.push_back({item,bestq,best/bestq,best});
 }
 std::stable_sort(offers.begin(),offers.end(),[](const auto&a,const auto&b){return a.per_unit>b.per_unit;});
 for(const auto&v:offers){int n=std::min(v.quantity,room);if(n<=0)break;sell[v.item]-=n;room-=n;storage_holds++;storage_units+=n;storage_conditional_gain+=n*v.per_unit;}
}
