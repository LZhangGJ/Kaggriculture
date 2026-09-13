from pathlib import Path
p=Path('/mnt/data/fix1_work/candidate/tests/animal_collection_test.cpp')
s=p.read_text();s=s.replace('static long checks=0, brute_cases=0, legacy_value_mismatches=0, path_cases=0;', 'static long checks=0, brute_cases=0, legacy_value_mismatches=0, path_cases=0, choice_states=0;')
pos=s.index('static Flow prices(')
add='''// Every stored score must belong to its stored action, not merely equal an
// independently optimal value somewhere else in the table. Read all successor
// scores from the DP but compute action execution/reward independently here.
static void check_choices(const AnimalServiceDP&dp,int k,int birth,int day,const Flow&px,double work){
 const int j=k-9;
 for(int d=day;d<=29;++d)for(int h=0;h<2;++h)for(int b=0;b<held[j];++b){
  ++choice_states;auto q=dp.choices[d][h][b];const double got=dp.value[d][h][b];
  check(q.value==got,"choice.value and DP value differ");
  check((q.feed==0||q.feed==1)&&(q.care==0||q.care==1)&&q.care<=q.feed,"illegal stored service action");
  if(d==29){check(got==0&&q.feed==0&&q.care==0,"terminal choice/value not zero");continue;}
  const bool tick=d+1-birth>=afirst[j]&&(d+1-birth-afirst[j])%ainterval[j]==0;
  const auto action_value=[&](int feed,int care){
   int nh=feed?0:h+1;if(nh>=2)return 0.;
   int count=tick?1+(feed?b:0):0;
   int nb=std::min(held[j]-1,(tick?0:b)+care);
   double immediate=-feed*px[d][W]-(feed+care)*work+count*px[d+1][product[j]];
   if(A08_ANIMAL_COLLECTION_LABOR)immediate+=px[d+1][F]-work*(1+int(count>0));
   else immediate+=std::max(0.,px[d+1][F]-work);
   return immediate+dp.value[d+1][nh][nb];
  };
  check(!(q.care&&b==held[j]-1&&!tick),"stored dominated/forbidden care");
  check(std::abs(got-action_value(q.feed,q.care))<1e-8,"stored action does not realize its value");
  int tf=0,tc=0;double optimum=action_value(0,0);
  for(int feed=0;feed<=1;++feed)for(int care=0;care<=feed;++care){
   if(care&&b==held[j]-1&&!tick)continue;
   double v=action_value(feed,care);
   if(v>optimum+1e-9){optimum=v;tf=feed;tc=care;}
  }
  check(q.feed==tf&&q.care==tc,"stored action is not the tie-ordered argmax");
  check(std::abs(got-optimum)<1e-8,"stored action/value not jointly optimal");
  if(A08_ANIMAL_COLLECTION_LABOR)check(std::abs(got-brute(k,birth,d,h,b,px,work))<1e-8,"choice-state optimum differs from full enumeration");
 }
}
'''
s=s[:pos]+add+s[pos:]
s=s.replace('AnimalServiceDP dp;dp.solve(k,birth,day,p,4.);','AnimalServiceDP dp;dp.solve(k,birth,day,p,4.);check_choices(dp,k,birth,day,p,4.);')
s=s.replace('if(A08_ANIMAL_COLLECTION_LABOR)check(rich.value[28][1][0]==65,"profitable service exact value wrong");','if(A08_ANIMAL_COLLECTION_LABOR)check(rich.value[28][1][0]==65,"profitable service exact value wrong");\n check_choices(rich,9,21,28,p,4.);')
s=s.replace('<<",\\\"dp_value_mismatches\\\":"', '<<",\\\"choice_states\\\":"<<choice_states<<",\\\"dp_value_mismatches\\\":"')
p.write_text(s)
