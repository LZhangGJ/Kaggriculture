"""Run in WSL with the existing pybind11 environment. Never edits shared engine."""
from pathlib import Path
import concurrent.futures, hashlib, json, shutil, subprocess, sysconfig, time
import pybind11

ROOT=Path(__file__).resolve().parents[1]
NATIVE=ROOT/'native'

def main():
    build=NATIVE/'build';build.mkdir(exist_ok=True)
    previous=build/'build_receipt.json'
    if previous.exists():
        old=json.loads(previous.read_text());oldbin=build/Path(old['binary']).name
        if oldbin.exists():
            assert hashlib.sha256(oldbin.read_bytes()).hexdigest()==old['binary_sha256']
            archive=NATIVE/'build_history'/old['binary_sha256'];archive.mkdir(parents=True,exist_ok=True)
            for path in (previous,oldbin):
                # Rebuilding identical sources can produce the same binary with
                # a different timing receipt. Preserve both, without overwrites.
                name=('build_receipt_'+hashlib.sha256(path.read_bytes()).hexdigest()+'.json') if path==previous else path.name
                target=archive/name
                if target.exists():assert target.read_bytes()==path.read_bytes()
                else:shutil.copy2(path,target)
    sources=[NATIVE/'vendor/simulator.cpp',NATIVE/'module.cpp',NATIVE/'fieldbook_adapter.cpp',NATIVE/'boatlee_v29.cpp',NATIVE/'kaito_v58.cpp',NATIVE/'lynn_v5.cpp',NATIVE/'three_day_adapter.cpp']
    inputs=[*sources,NATIVE/'policy.hpp',NATIVE/'cash_audit.hpp',NATIVE/'production_audit.hpp',NATIVE/'admission_audit.hpp',NATIVE/'vendor/simulator.hpp',NATIVE/'vendor/native_teammate.hpp',NATIVE/'vendor/native_teammate.cpp']
    inputs += [NATIVE/'fieldbook_adapter.hpp',*sorted((ROOT/'opponents/yhay81_six_day/output/sixday_r4_source').glob('*.hpp')),ROOT/'opponents/yhay81_six_day/output/sixday_r4_source/policy.cpp']
    inputs += [NATIVE/'boatlee_v29.hpp']
    inputs += [NATIVE/'investment_audit.hpp']
    inputs += [NATIVE/'investment_candidates.hpp',NATIVE/'investment_branch_audit.hpp']
    inputs += [NATIVE/'resource_handoff_audit.hpp']
    inputs += [NATIVE/'recoordination_inspection.hpp']
    inputs += [NATIVE/'resource_exchange.hpp']
    inputs += [NATIVE/'intraday_admission.hpp']
    inputs += [NATIVE/'joint_portfolio.hpp']
    inputs += [NATIVE/'observed_day_scenario.hpp']
    inputs += [NATIVE/'observation_view.hpp',NATIVE/'day_consequence.hpp',NATIVE/'crop_delivery.hpp']
    inputs += [NATIVE/'idle_task_handoff.hpp']
    inputs += [NATIVE/'live_execution_repairs.hpp']
    inputs += [NATIVE/'rotation_calendar.hpp',NATIVE/'rotation_cash.hpp']
    inputs += [NATIVE/'compile_choices.hpp']
    inputs += [NATIVE/'pack_memo.hpp',NATIVE/'regret_pack_incremental.inc']
    inputs += [NATIVE/'crop_delivery.hpp']
    inputs += [NATIVE/'service_recovery.hpp']
    inputs += [NATIVE/'kaito_v58.hpp']
    inputs += [NATIVE/'lynn_v5.hpp']
    inputs += [NATIVE/'three_day_adapter.hpp',ROOT/'opponents/yhay81_three_day/output/source/policy.cpp',ROOT/'opponents/yhay81_three_day/output/source/tape.inc',*sorted((ROOT/'opponents/yhay81_three_day/output/source/include').glob('*.hpp'))]
    sources += [NATIVE/'ecobot_v7_core.cpp',NATIVE/'ecobot_v7.cpp']
    inputs += [NATIVE/'ecobot_v7_core.cpp',NATIVE/'ecobot_v7.cpp',NATIVE/'ecobot_v7_core.hpp',NATIVE/'ecobot_v7.hpp',ROOT/'opponents/ecobot_v7/output/main.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    inc=['-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],'-I'+str(NATIVE)]
    commands=[]
    def compile_one(src):
        dst=build/(src.stem+'.o')
        cmd=['g++','-std=c++20','-O3','-DNDEBUG','-march=native','-fopenmp','-fPIC',*inc,'-c',str(src),'-o',str(dst)]
        if src.stem in ('boatlee_v29','kaito_v58','lynn_v5','ecobot_v7','ecobot_v7_core'):cmd.insert(1,'-ffp-contract=off')  # Match Python's separate rounded operations.
        if src.stem=='three_day_adapter':cmd.insert(1,'-I'+str(ROOT/'opponents/yhay81_three_day/output/source/include'))
        commands.append(cmd);subprocess.run(cmd,check=True);return str(dst)
    started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16,len(sources))) as pool: objects=list(pool.map(compile_one,sources))
    dest=build/('_dp7_native'+sysconfig.get_config_var('EXT_SUFFIX'))
    cmd=['g++','-shared','-fopenmp',*objects,'-o',str(dest)]
    commands.append(cmd);subprocess.run(cmd,check=True)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in inputs)
    receipt=dict(seconds=time.perf_counter()-started,source_hashes=hashes,commands=commands,binary=str(dest),binary_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (build/'build_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
    print(json.dumps(dict(seconds=receipt['seconds'],binary=str(dest),binary_sha256=receipt['binary_sha256'])),flush=True)

if __name__=='__main__':main()
