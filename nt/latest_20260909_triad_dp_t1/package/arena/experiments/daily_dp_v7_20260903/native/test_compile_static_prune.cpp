// Reuse the existing mechanism fixture, but never invoke its main routine.
#define main s5d_existing_fixture_main
#include "test_compile_choices.cpp"
#undef main
#include "compile_static_prune_prototype.hpp"
#include <chrono>
int main(){try{
 int cases=0,old_evals=0,new_evals=0;double old_seconds=0,new_seconds=0;
 for(int hands=0;hands<=7;hands++)for(int money:{0,20,20000})for(bool cached:{false,true}){
  Fixture f(hands);f.f.money=money;f.c.p.compile_replant_choices=true;f.c.p.compile_bounded_rollout=true;f.c.p.exact_schedule_cache=cached;
  f.c.compile_base(f.view());auto a=f.c,b=f.c;
  auto t=std::chrono::steady_clock::now();compilechoice::compare(a,f.view());old_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  t=std::chrono::steady_clock::now();compileprune::compare(b,f.view());new_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();
  check(compilechoice::same(a,b),"pruning changed selected semantics");
  check(a.actual_drop==b.actual_drop&&a.resource_degraded==b.resource_degraded&&a.unresolved_overflow==b.unresolved_overflow,"pruning changed diagnostics used by actual execution");
  check(a.compile_choice_changes==b.compile_choice_changes,"pruning changed switch count");
  check(b.compile_choice_evaluations<=a.compile_choice_evaluations,"pruning increased scenario count");
  old_evals+=a.compile_choice_evaluations;new_evals+=b.compile_choice_evaluations;cases++;
 }
 {Fixture f;f.c.p.compile_consequence=true;f.c.p.compile_replant_choices=true;f.c.p.compile_bounded_rollout=true;f.c.compile_base(f.view());auto a=f.c,b=f.c;
  compilechoice::compare(a,f.view());compileprune::compare(b,f.view());check(compilechoice::same(a,b)&&a.compile_choice_evaluations==b.compile_choice_evaluations,"economic scoring must not use static pruning");}
 check(new_evals<old_evals,"prototype never reduced work");
 std::cout<<"{\"status\":\"PASS_ISOLATED_PROTOTYPE_NOT_PRODUCTION\",\"cases\":"<<cases<<",\"checks\":"<<checks
  <<",\"original_evaluations\":"<<old_evals<<",\"pruned_evaluations\":"<<new_evals<<",\"original_seconds\":"<<old_seconds<<",\"pruned_seconds\":"<<new_seconds<<"}\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
