from pathlib import Path
p=Path(__file__).resolve().parents[1]/'candidate'
f=p/'policy/animal_service_dp.hpp';s=f.read_text();s=s.replace('namespace competitive {','''#ifndef A08_ANIMAL_COLLECTION_LABOR
#define A08_ANIMAL_COLLECTION_LABOR 1
#endif
static_assert(A08_ANIMAL_COLLECTION_LABOR==0 || A08_ANIMAL_COLLECTION_LABOR==1);
namespace competitive {''',1)
old='''double v=-feed*(prices[d][W]+work)-care*work+quantity*prices[d+1][product[j]]+std::max(0.,prices[d+1][F]-work)+value[d+1][nh][np];'''
new='''// Every credited future output must be collected. The live executor
      // harvests positive animal yield and always collects available manure;
      // neither is a free/optional cash credit, including on the last day.
      const double collection = A08_ANIMAL_COLLECTION_LABOR
        ? (quantity>0 ? work : 0.) + work : 0.;
      const double manure = A08_ANIMAL_COLLECTION_LABOR
        ? prices[d+1][F] : std::max(0.,prices[d+1][F]-work);
      double v=-feed*(prices[d][W]+work)-care*work
        +quantity*prices[d+1][product[j]]+manure-collection+value[d+1][nh][np];'''
assert old in s;s=s.replace(old,new);f.write_text(s)
f=p/'policy/triad.hpp';s=f.read_text()
old='''else {a.f[day][product[j]]+=t->yield_units;a.f[day][F]+=t->fertilizer_available;}
  AnimalServiceDP dp;'''
new='''else {
   a.f[day][product[j]]+=t->yield_units;a.f[day][F]+=t->fertilizer_available;
   // Already produced is not already collected. Match today's HARVEST and
   // COLLECT_FERTILIZER atoms, even when no maintenance day remains.
   if constexpr(A08_ANIMAL_COLLECTION_LABOR)
    a.labor[day]+=(int(t->yield_units>0)+int(t->fertilizer_available))*s.animal_work;
  }
  AnimalServiceDP dp;'''
assert old in s;s=s.replace(old,new)
old='''a.f[d][W]-=f;a.labor[d]+=(f+z+1+.20*near(pos))*s.animal_work;service_work(a,d,pos,f+z);'''
new='''a.f[d][W]-=f;
   a.labor[d]+=(f+z+(A08_ANIMAL_COLLECTION_LABOR?0:1)+.20*near(pos))*s.animal_work;
   service_work(a,d,pos,f+z);'''
assert old in s;s=s.replace(old,new)
old='''if(tick){a.f[d+1][product[j]]+=1+(feeding?bonus:0);a.labor[d+1]+=1;bonus=0;}'''
new='''if(tick){a.f[d+1][product[j]]+=1+(feeding?bonus:0);a.labor[d+1]+=A08_ANIMAL_COLLECTION_LABOR?s.animal_work:1.;bonus=0;}'''
assert old in s;s=s.replace(old,new)
old='''a.f[d+1][F]+=1;
  }return realize(std::move(a),pos);'''
new='''a.f[d+1][F]+=1;
   // Manure produced at dawn d+1 consumes collection work on d+1, not
   // on d. This also charges terminal-day collection instead of losing it.
   if constexpr(A08_ANIMAL_COLLECTION_LABOR)a.labor[d+1]+=s.animal_work;
  }return realize(std::move(a),pos);'''
assert old in s;s=s.replace(old,new)
f.write_text(s)
f=p/'main.py';s=f.read_text().replace('TRI_A08_r11_r1: exact-price-floor sale-window extension (not accepted).','TRI_A08_r11_r2: collection-work-consistent animal service valuation (not accepted).').replace('_tri_a08_r11_r1_codec','_tri_a08_r11_r2_codec').replace('policy/tri_a08_r11_r1.so','policy/tri_a08_r11_r2.so');f.write_text(s)
f=p/'build.py';s=f.read_text().replace('Portable TRI_A08_r11_r1 builder (parent: exact A08_r11).','Portable TRI_A08_r11_r2 builder (parent: exact TRI_A08_r11_r1).').replace("p=argparse.ArgumentParser();","p=argparse.ArgumentParser();p.add_argument('--animal-collection-labor',type=int,choices=[0,1],default=1);").replace('policy/tri_a08_r11_r1.so','policy/tri_a08_r11_r2.so').replace("flags=[f'-DA08_SALE_FLOOR_DP", "flags=[f'-DA08_ANIMAL_COLLECTION_LABOR={a.animal_collection_labor}',f'-DA08_SALE_FLOOR_DP").replace("'revision':'TRI_A08_r11_r1'","'revision':'TRI_A08_r11_r2','animal_collection_labor':a.animal_collection_labor").replace("'parent_native_sha256':'3b95f2c633242e31329adffae2438440a76b5eb6b11ece4897fe69f5710bb9cb'","'parent_native_sha256':'bb2f248f689b7a75042c7bf4897700099f1f6d1980b1abc38442db0a1bbadc4c'").replace('A08_r11_from_d2b4dad66c24be494421cfbcb48942a10162e12d8d81ea660eae5a7aeaab40c3','TRI_A08_r11_r1_from_751c049fc9d765319264cddad8a5a4544059ec15101354d2fcc69e874ec670ac');f.write_text(s)
