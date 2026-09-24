// Inside Controller. Optional context is explicit so no residual portfolio or
// day-start cash is accidentally reused in the live remaining-flow evaluator.
CropPath r12_select(const View&o,const Planner&forecast,const Asset&base,
 int k,int birth,int pos,const Flow&px,const Tile*existing,
 const CropPath&keep,bool already_paid=false,bool today_labor_paid=false)const {
 if(s.a06_r12_calendar<=0||ongoing(k)||k<0||k>=5||s.repeat>0||keep.kind<0)return keep;
 const int earliest=std::max(o.day,birth+first[k]);
 const int latest=std::min(29,birth+(k==W?4:k==C?3:12));
 if(earliest>latest)return keep;
 std::vector<CropPath>options{keep};
 for(int finish=earliest;finish<=latest;++finish){
  auto a=crop(k,birth,pos,finish,px,existing);
  options.push_back({a,k,finish-birth,0});
 }
 auto whole=[&](const CropPath&q){auto a=q.a;
  if(already_paid&&a.first_cost>0){a.fixed[birth]+=a.first_cost;}
  if(today_labor_paid)a.labor[o.day]=0;
  auto all=base;add(all,a);return all;
 };
 std::vector<r12::Label>labels;labels.reserve(options.size());
 for(const auto&q:options){const int finish=birth+q.length;
  // Existing path occupancy starts now; a new project's starts at birth.
  double rent=s.land_rent*std::max(0,finish-std::max(o.day,birth));
  labels.push_back({finish,forecast.value(o,whole(q))-rent});
 }
 auto decision=r12::solve(labels,o.day,0);if(decision.index<0)return keep;
 calendar12.calls++;calendar12.labels+=labels.size();
 const bool changed=decision.index!=0;
 calendar12.changes+=changed;calendar12.existing_changes+=changed&&existing;
 // Retain bounded data, not a growing trajectory in the live agent.
 // Verbose diagnostic decomposition removed; selection and counters unchanged.
 auto result=options[decision.index];result.value=labels[decision.index].value;return result;
}

// Coordinate only existing finite crops against all incumbent streams. KEEP
// includes their already-paid inputs; a funded successor due today is binding
// and is never displaced. No animal/strawberry service is removed or zeroed.
void r12_reconcile_calendar(const View&o,const Farm&farm,std::vector<int>&free){
 if(s.a06_r12_calendar<=0||s.repeat>0)return;
 for(int pos=0;pos<100;++pos){const auto&t=farm.tiles[pos];
  if(!plant(t)||ongoing(int(t.crop)))continue;
  const int k=int(t.crop),birth=t.planted_day;
  if(book[pos].successor>=0&&o.priv.seeds[book[pos].successor]>0&&o.day>=birth+first[k])continue;
  auto base=portfolio;add(base,paths[pos],-1);
  CropPath keep{paths[pos],k,core.triad_crop_age[pos],0};
  auto chosen=r12_select(o,model,base,k,birth,pos,prices,&t,keep,true,false);
  const int finish=birth+chosen.length;
  portfolio=base;add(portfolio,chosen.a);paths[pos]=chosen.a;
  core.triad_crop_age[pos]=chosen.length;book[pos].length=chosen.length;release[pos]=finish;
#if R2_FINITE_FERTILIZER
  finite_first_fertilizer[pos]=chosen.a.f[o.day][F]<0;
#endif
  free.erase(std::remove(free.begin(),free.end(),pos),free.end());
  if(finish==o.day&&o.day<29)free.push_back(pos);
  model.value(o,portfolio,&prices);
 }
}
