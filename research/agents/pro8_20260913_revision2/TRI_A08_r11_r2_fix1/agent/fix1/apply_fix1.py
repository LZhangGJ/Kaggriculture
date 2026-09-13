from pathlib import Path
import shutil
r=Path('/mnt/data/fix1_work');p=r/'candidate';old=r/'original_r2'
s=(old/'policy/animal_service_dp.hpp').read_text()
a=s.index(' void solve('); b=s.index(' Choice first(')
new=''' // Keep the Bellman table writer at a real optimized call boundary. Central
 // GCC13 -O3 testing of r2 observed feed=0 with value=65 for a state whose
 // unique optimal action is feed=1. Local GCC14/Clang do not reproduce that
 // failure; this is a targeted code-generation workaround, not a UB diagnosis.
 // noipa prevents caller-specific clones/inlining, NOT optimization of this
 // function's body. The production build remains -O3.
#if defined(__GNUC__) && !defined(__clang__)
 __attribute__((noipa))
#elif defined(__clang__)
 __attribute__((noinline))
#endif
 void solve(int kind,int placed,int day,const Flow&prices,double work){
  const int j=kind-9,cap=held[j]-1;
  for(int d=28;d>=day;--d){
   const int age=d+1-placed;
   const bool tick=age>=afirst[j]&&(age-afirst[j])%ainterval[j]==0;
   for(int h=0;h<2;++h)for(int p=0;p<=cap;++p){
    // Preserve the original action order and 1e-9 tie rule. Select a single
    // immutable candidate index, then commit its action AND value together;
    // do not carry three independent winner fields through nested loops.
    const auto evaluate=[&](int feed,int care)->Choice{
     const int nh=feed?0:h+1;
     if(nh>=2)return {0,0,0};
     const int quantity=tick?1+(feed?p:0):0;
     const int np=std::min(cap,(tick?0:p)+care);
     const double collection=A08_ANIMAL_COLLECTION_LABOR
       ? (quantity>0 ? work : 0.) + work : 0.;
     const double manure=A08_ANIMAL_COLLECTION_LABOR
       ? prices[d+1][F] : std::max(0.,prices[d+1][F]-work);
     const double v=-feed*(prices[d][W]+work)-care*work
       +quantity*prices[d+1][product[j]]+manure-collection+value[d+1][nh][np];
     return {feed,care,v};
    };
    const Choice candidates[3]={evaluate(0,0),evaluate(1,0),
      (p<cap||tick)?evaluate(1,1):Choice{0,0,-1e100}};
    int selected=0;
    for(int i=1;i<3;++i)
     if(candidates[i].value>candidates[selected].value+1e-9)selected=i;
    const Choice winner=candidates[selected];
    choices[d][h][p]=winner;
    value[d][h][p]=winner.value;
   }
  }
 }
'''
(p/'policy/animal_service_dp.hpp').write_text(s[:a]+new+s[b:])
# Old reports remain byte-for-byte under history/r2, no stale PASS at the root.
hist=p/'history/r2';hist.mkdir(parents=True,exist_ok=True)
for name in ['README.md','VALIDATION_REPORT.md','BUILD.md','IDENTITY.json','RESOURCE_AND_TIMING.json','DELIVERY_CHECKPOINT.json','PREARCHIVE_CHECKPOINT.json','PACKAGE_SHA256SUMS.txt','SOURCE_SHA256SUMS.txt']:
 if (p/name).exists():shutil.move(p/name,hist/name)
# All inherited evidence/tests/raw records remain; immutable r2 production is
# separately preserved in addition to the existing immutable r1 reference.
ref=p/'reference/r2';ref.mkdir(parents=True,exist_ok=True)
for name in ['main.py','build.py','policy']:
 if (old/name).is_dir():shutil.copytree(old/name,ref/name)
 else:shutil.copy2(old/name,ref/name)
for name in ['main.py','build.py']:
 path=p/name;s=path.read_text().replace('TRI_A08_r11_r2','TRI_A08_r11_r2_fix1').replace('tri_a08_r11_r2','tri_a08_r11_r2_fix1');path.write_text(s)
b=p/'build.py';s=b.read_text().replace("'parent_native_sha256':'bb2f248f689b7a75042c7bf4897700099f1f6d1980b1abc38442db0a1bbadc4c'", "'parent_native_sha256':'afca1d69fde38a0474546b70ecee38911477548dfde9aadc685c58a2a0b88a2f'")
s=s.replace("'recovery_base':'TRI_A08_r11_r1_from_751c049fc9d765319264cddad8a5a4544059ec15101354d2fcc69e874ec670ac'", "'recovery_base':'TRI_A08_r11_r2_from_2f1557875d550000ae492c3814c84c78d5402e99b3a754308dc3907aa32b2178'")
b.write_text(s)
# Fixed scripts point only at the explicitly named fix1 native/receipt.
for name in ['tests/run_checks.py','tests/test_entry_contract.py','development/compile_unit.py']:
 path=p/name;path.write_text(path.read_text().replace('tri_a08_r11_r2','tri_a08_r11_r2_fix1'))
for name in ['tri_a08_r11_r2.so','tri_a08_r11_r2.BUILD.json']:
 (p/'policy'/name).unlink()
# Preserve original test source as central-reported failing artifact.
shutil.copy2(old/'tests/animal_collection_test.cpp',hist/'animal_collection_test.cpp')
