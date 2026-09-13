#pragma once
#include <limits>
#include "executor/observation_view.hpp"
using namespace fastkag;
struct Cursor {
 const double* p;size_t n,i=0;
 double next(){if(i>=n)throw std::runtime_error("truncated observation");return p[i++];}
 int integer(){double x=next();if(!std::isfinite(x)||x<std::numeric_limits<int>::min()||x>std::numeric_limits<int>::max()||x!=int(x))throw std::runtime_error("invalid integer");return int(x);}
 int count(){int x=integer();if(x<0||x>4096)throw std::runtime_error("invalid count");return x;}
 Position position(){return {int16_t(integer()),int16_t(integer())};}
 Farm farm(){Farm f;f.money=next();f.farmer=position();int hands=count();for(int j=0;j<hands;j++)f.hands.push_back(position());
  f.unlocked_mask=integer();f.hires_today=integer();f.tiles.resize(100);
  for(auto&t:f.tiles){t.kind=TileKind(integer());t.crop=Item(integer());t.animal=Item(integer());t.planted_day=integer();t.placed_day=integer();
   t.yield_units=integer();t.consecutive_unwatered=integer();t.consecutive_unfed=integer();t.fertilized_until_day=integer();t.pending_care_bonus=integer();
   t.max_lifespan_step=integer();t.watered_today=integer();t.fed_today=integer();t.cared_today=integer();t.fertilizer_available=integer();}
  return f;
 }
 PrivateState priv(){PrivateState p;for(auto&x:p.shed)x=integer();for(auto&x:p.seeds)x=integer();int n=count();p.inventories.resize(n);p.inventory_order.resize(n);
  for(int j=0;j<n;j++){for(auto&x:p.inventories[j])x=integer();int m=count();for(int k=0;k<m;k++)p.inventory_order[j].push_back(integer());}return p;}
};

