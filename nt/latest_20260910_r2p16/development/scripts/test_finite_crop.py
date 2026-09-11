"""Native forecasts vs official unit actions and daily crop refresh."""
from pathlib import Path
import subprocess,itertools,json,time
import run_panel as panel
HERE=Path(__file__).resolve().parent;OUT=HERE/'candidate_r2p10';exe=OUT/'test_finite_crop'
def main():
    panel.init_worker();e=panel.ENGINE
    cmd=['g++-13','-std=c++20','-O2','-DR2_FINITE_FERTILIZER=1',str(OUT/'test_finite_crop.cpp'),str(OUT/'policy/executor/vendor/simulator.cpp'),'-o',str(exe)]
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
    child=subprocess.Popen([str(exe)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
    def native(k,birth,begin,finish,t,price,fprice,enable):
        row=[k,birth,begin,finish,t['yield_units'],t['consecutive_unwatered'],t['fertilized_until_day'],int(t['watered_today']),price,fprice,4,enable]
        child.stdin.write(' '.join(map(str,row))+'\n');child.stdin.flush()
        line=child.stdout.readline();assert line,'Native test child failed'
        a=list(map(float,line.split()));assert len(a)==34
        return a
    cases=queries=selected_fert=0;started=time.perf_counter()
    try:
        for k,name,ends,cap in [(0,'WHEAT',[2,3,4],6),(1,'CARROT',[2,3],4),(4,'MELON',[10,11,12],6)]:
            for birth,age_end,price,fprice,enable in itertools.product([0,11,24,27],ends,[50,240],[1,90,1000],[0,1]):
                finish=birth+age_end
                if finish>29:continue
                for age_begin in sorted({0,max(0,age_end-2),age_end}):
                    begin=birth+age_begin
                    for dry,wet,fert_active,quantity in itertools.product([0,1],[False,True],[False,True],[1,cap]):
                        farm=e._new_farm(10,10000);priv=e._new_private();farm['farmer']=[2,2];priv['inventories'][0]['FERTILIZER']=100
                        t=e._new_plant(name,birth,24);t.update(yield_units=quantity,consecutive_unwatered=dry,watered_today=wet,fertilized_until_day=begin+1 if fert_active else -1)
                        farm['tiles'][2][2]=t
                        initial=native(k,birth,begin,finish,t,price,fprice,enable);queries+=1;used=0;actual_f=[0]*30
                        for d in range(begin,finish+1):
                            t=farm['tiles'][2][2];assert isinstance(t,dict) and t['kind']=='PLANT'
                            z,w,*_=native(k,birth,d,finish,t,price,fprice,enable);queries+=1
                            if z:
                                before=priv['inventories'][0]['FERTILIZER'];e._apply_unit_action(farm,priv,0,['FERTILIZE'],10,d,24)
                                assert priv['inventories'][0]['FERTILIZER']==before-1
                                used+=1;actual_f[d]=-1
                            if w:e._apply_unit_action(farm,priv,0,['WATER'],10,d,24)
                            if d==finish:e._apply_unit_action(farm,priv,0,['HARVEST'],10,d,24)
                            else:e._daily_refresh_plants(farm,d,24)
                        harvested=priv['inventories'][0].get(name,0)
                        assert harvested==initial[2] and used==initial[3],(k,birth,begin,finish,price,fprice,dry,wet,fert_active,quantity,initial[:4],harvested,used)
                        assert actual_f==initial[4:],(k,birth,begin,finish,initial[4:],actual_f)
                        selected_fert+=used>0;cases+=1
        assert cases>1000 and selected_fert>0
        result=dict(status='PASS',official_crop_episodes=cases,native_forecast_queries=queries,cases_using_fertilizer=selected_fert,seconds=time.perf_counter()-started,official_sha256=panel.sha(panel.old.REF/'official/kaggriculture.py'),source_sha256={str(p.relative_to(OUT)):panel.sha(p) for p in [OUT/'test_finite_crop.cpp',OUT/'policy/triad.hpp',OUT/'policy/finite_fertilizer.hpp']},scope='Actual official per-unit WATER/FERTILIZE/HARVEST and daily refresh; fixed prices/deadlines; not proof of full farm affordability or optimality')
        panel.save(OUT/'UNIT_TESTS.json',result);print(json.dumps(result),flush=True)
    finally:
        child.stdin.close();child.wait(timeout=5)
if __name__=='__main__':main()
