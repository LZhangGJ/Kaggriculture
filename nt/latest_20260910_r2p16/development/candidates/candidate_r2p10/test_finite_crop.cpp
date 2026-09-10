#include "policy/triad.hpp"
#include <iostream>
int main(){using namespace competitive;int k,birth,begin,finish,y,dry,fert,wet,enable;double p,f,work;
 std::cout.precision(17);
 while(std::cin>>k>>birth>>begin>>finish>>y>>dry>>fert>>wet>>p>>f>>work>>enable){
  triad::Settings settings;settings.work_price=work;settings.crop_fert=enable;
  triad::Controller c(settings);c.model.day=begin;c.core.day=begin;
  fastkag::Tile t;t.kind=fastkag::TileKind::PLANT;t.crop=fastkag::Item(k);t.planted_day=birth;t.yield_units=y;t.consecutive_unwatered=dry;t.fertilized_until_day=fert;t.watered_today=wet;
  Flow prices{};for(auto&d:prices){d[k]=p;d[F]=f;}
  bool z=false;auto a=c.crop(k,birth,22,finish,prices,&t,&z);
  double total=0,fertilizers=0;for(int d=begin;d<30;d++){total+=a.f[d][k];fertilizers-=a.f[d][F];}
  std::cout<<z<<' '<<c.core.water_due(t)<<' '<<total<<' '<<fertilizers;
  for(int d=0;d<30;d++)std::cout<<' '<<a.f[d][F];
  std::cout<<'\n'<<std::flush;
 }
}
