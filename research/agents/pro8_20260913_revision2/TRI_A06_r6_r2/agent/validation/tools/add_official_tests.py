from pathlib import Path
r=Path('/mnt/data/a06_r6_r2_work/candidate/tests')
f=r/'land_choice_checks.cpp';s=f.read_text()
s=s.replace('bool same_book(const triad::Controller&a,const triad::Controller&b){', '''std::vector<double> own_signature(const fastkag::Simulator&s){
 const auto&f=s.farms()[0];const auto&p=s.privates()[0];const auto&m=s.market();
 std::vector<double>z{f.money,double(f.farmer.x),double(f.farmer.y),double(f.hands.size())};
 for(auto xy:f.hands){z.push_back(xy.x);z.push_back(xy.y);}z.push_back(f.unlocked_mask);z.push_back(f.hires_today);
 for(const auto&t:f.tiles){for(double x:{double(t.kind),double(t.crop),double(t.animal),double(t.planted_day),double(t.placed_day),double(t.yield_units),double(t.consecutive_unwatered),double(t.consecutive_unfed),double(t.fertilized_until_day),double(t.pending_care_bonus),double(t.max_lifespan_step),double(t.watered_today),double(t.fed_today),double(t.cared_today),double(t.fertilizer_available)})z.push_back(x);}
 for(auto x:p.shed)z.push_back(x);for(auto x:p.seeds)z.push_back(x);z.push_back(p.inventories.size());
 for(size_t i=0;i<p.inventories.size();i++){for(auto x:p.inventories[i])z.push_back(x);z.push_back(p.inventory_order[i].size());for(auto x:p.inventory_order[i])z.push_back(x);}
 for(auto x:m.inventory)z.push_back(x);for(auto x:m.prices)z.push_back(x);return z;
}
bool same_book(const triad::Controller&a,const triad::Controller&b){''')
s=s.replace('int ticks=0,land_actions=0,new_projects=0;std::vector<fastkag::PlayerAction>acts;', 'int ticks=0,land_actions=0,new_projects=0;std::vector<fastkag::PlayerAction>acts;std::vector<double>prefix;')
s=s.replace('auto act=test.act(vv);need(act.market.size()', 'auto act=test.act(vv);if(t==0){fastkag::ObservedDayScenario observed(vv);auto after_units=observed.project_units(act.units,-1);prefix=own_signature(after_units.project_own_market(0,act.market));}need(act.market.size()')
s=s.replace('j<<"]}";\n   }', '''j<<"],\\"first_prefix_signature\\":[";for(size_t i=0;i<prefix.size();i++){if(i)j<<",";j<<prefix[i];}j<<"]}";
   }''')
f.write_text(s)
