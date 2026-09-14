std::pair<std::vector<Route>,int>regret_pack_incremental(const std::vector<Job>&js,const std::vector<int>&starts,int budget,bool ret,bool complete_only=false)const{
 std::vector<Route>rs;for(int i=0;i<int(starts.size());i++)rs.emplace_back(i,starts[i]);
 // Only the route that receives a job changes. Keep all other task/route
 // insertion answers, including their original tie breaks, exactly intact.
 struct Slot {int version=-1,delta=100000,total=100000,at=-1;};
 std::vector<Slot>slots(js.size()*rs.size());std::vector<int>versions(rs.size(),0);
 std::vector<bool>done(js.size(),false);int remaining=js.size(),dropped=0;
 while(remaining){
  int priority=100000;for(size_t i=0;i<js.size();i++)if(!done[i])priority=std::min(priority,js[i].priority);
  int selected=-1,unit=-1,index=-1;std::tuple<int,int,int,int>regret_key;
  for(size_t i=0;i<js.size();i++)if(!done[i]&&js[i].priority==priority){
   int first=100000,second=100000,best_unit=-1,best_index=-1,best_total=100000;
   for(auto&r:rs){
    auto&cached=slots[i*rs.size()+r.unit];
    if(cached.version!=versions[r.unit]){
     cached={versions[r.unit],100000,100000,-1};
     const auto&job=js[i];int fixed=r.cost+int(job.actions.size());
     for(int k=0;k<12;k++)fixed+=(job.needs[k]>0&&r.needs[k]<=0);
     const bool returning=ret&&(r.has_output||output(job.out));
     const int old=r.total(ret),n=int(r.jobs.size());
     for(int at=0;at<=n;at++){
      int before=at?r.jobs[at-1].pos:r.start;
      int total=fixed+dist(before,job.pos);
      if(at<n)total+=dist(job.pos,r.jobs[at].pos)-dist(before,r.jobs[at].pos);
      total+=returning?near(at==n?job.pos:r.end)+1:0;
      int delta=total-old;
      if(total<=budget&&std::tuple(delta,total,at)<std::tuple(cached.delta,cached.total,cached.at))
       {cached.delta=delta;cached.total=total;cached.at=at;}
     }
    }
    int local=cached.delta,local_at=cached.at,local_total=cached.total;
    if(local_at<0)continue;
    if(std::tuple(local,local_total,r.unit)<std::tuple(first,best_total,best_unit))
     {second=first;first=local;best_total=local_total;best_unit=r.unit;best_index=local_at;}
    else second=std::min(second,local);
   }
   if(best_unit<0){
    // Completion caller discards every partial alternative; stop the same
    // greedy construction as soon as its failure is irreversible.
    if(complete_only)return {rs,1};
    done[i]=true;remaining--;dropped++;continue;
   }
   auto key=std::tuple(second==100000?100000:second-first,first,int(js[i].actions.size()),-int(i));
   if(selected<0||key>regret_key){selected=i;unit=best_unit;index=best_index;regret_key=key;}
  }
  if(selected<0)continue;
  auto&route=rs[unit];auto ordered=route.jobs;ordered.insert(ordered.begin()+index,js[selected]);
  Route updated(route.unit,route.start);for(auto&j:ordered)updated.append(j);route=std::move(updated);versions[unit]++;
  done[selected]=true;remaining--;
 }return {rs,dropped};
}
