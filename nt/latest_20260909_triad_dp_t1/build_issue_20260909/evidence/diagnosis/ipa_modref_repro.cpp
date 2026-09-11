// Reduced from T1's animal forecast. No game, wrapper, heap alias, or casts.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
using Flow=std::array<std::array<double,9>,30>;
constexpr int held[3]{4,6,6},first[3]{4,8,6},interval[3]{1,2,3},product[3]{5,6,7};
struct DP {
    struct Choice {int feed=0,care=0;double value=0;};
    Choice choices[30][2][6]{};
    double value[30][2][6]{};
    __attribute__((noinline)) void solve(int kind,int placed,int day,const Flow&prices,double work) {
        int j=kind-9,cap=held[j]-1;
        for(int d=28;d>=day;d--) {
            int age=d+1-placed;bool tick=age>=first[j]&&(age-first[j])%interval[j]==0;
            for(int h=0;h<2;h++)for(int p=0;p<=cap;p++) {
                Choice best{0,0,-1e100};
                for(int feed=0;feed<=1;feed++) {
                    int nh=feed?0:h+1;
                    if(nh>=2){if(0>best.value)best={0,0,0};continue;}
                    for(int care=0;care<=feed;care++) {
                        if(care&&p>=cap&&!tick)continue;
                        int quantity=tick?1+(feed?p:0):0,np=std::min(cap,(tick?0:p)+care);
                        double v=-feed*(prices[d][0]+work)-care*work+quantity*prices[d+1][product[j]]+std::max(0.,prices[d+1][8]-work)+value[d+1][nh][np];
                        if(v>best.value+1e-9)best={feed,care,v};
                    }
                }
                choices[d][h][p]=best;value[d][h][p]=best.value;
            }
        }
    }
};
__attribute__((noinline)) int forecast(int kind,int start,const Flow&prices,double work,bool fresh) {
    DP dp;dp.solve(kind,start,start,prices,work);
    int hunger=0,bonus=0,count=0,j=kind-9;
    for(int d=start;d<29;d++) {
        auto c=dp.choices[d][hunger][bonus];
        if(fresh&&d==start)c.feed=c.care=1;
        count+=c.feed;
        hunger=c.feed?0:hunger+1;
        if(hunger>=2)break;
        int age=d+1-start;
        if(age>=first[j]&&(age-first[j])%interval[j]==0)bonus=0;
        if(c.feed&&c.care)bonus=std::min(held[j]-1,bonus+1);
    }
    return count;
}
int main(int argc,char**argv) {
    Flow prices{};
    const double p[9]{25,35,60,120,250,50,160,200,100};
    for(auto&day:prices)std::copy(p,p+9,day.begin());
    int start=argc>1?std::atoi(argv[1]):0;
    for(int k=9;k<12;k++)std::printf("%d %d\n",k,forecast(k,start,prices,4.,true));
}
