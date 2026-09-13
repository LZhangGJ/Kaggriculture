#!/usr/bin/env python3
"""Portable TRI_A08_r11_r2_fix1 builder (parent: exact TRI_A08_r11_r2). Offline candidate; formal acceptance not run."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, time
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--animal-collection-labor',type=int,choices=[0,1],default=1);p.add_argument('--sale-floor-dp',type=int,choices=[0,1],default=1);p.add_argument('--sale-schedule-dp',type=int,choices=[0,1],default=1);p.add_argument('--paid-continuation',type=int,choices=[0,1],default=1);p.add_argument('--complete-crop-stops',type=int,choices=[0,1],default=1);p.add_argument('--market-integral',type=int,choices=[0,1,2],default=1);p.add_argument('--sale-clock',type=int,choices=[0,1,2],default=2);p.add_argument('--cxx',default=os.environ.get('CXX','g++'));p.add_argument('--mode',type=int,choices=[0,1,2],default=2);p.add_argument('--realization',type=int,choices=[0,1,2],default=0);p.add_argument('--crop-portfolio',type=int,choices=[0,1],default=1);p.add_argument('--cash-ledger',type=int,choices=[0,1],default=0);p.add_argument('--prefix-days',type=int,choices=[1,3,5],default=5);p.add_argument('--out',type=Path);a=p.parse_args()
 if shutil.which(a.cxx) is None:raise SystemExit('C++20 compiler not found: '+a.cxx)
 out=(a.out or ROOT/'policy/tri_a08_r11_r2_fix1.so').resolve();out.parent.mkdir(parents=True,exist_ok=True)
 if out.is_relative_to(ROOT/'reference'):raise SystemExit('Refusing to overwrite an immutable ancestor')
 flags=[f'-DA08_ANIMAL_COLLECTION_LABOR={a.animal_collection_labor}',f'-DA08_SALE_FLOOR_DP={a.sale_floor_dp}',f'-DA08_SALE_SCHEDULE_DP={a.sale_schedule_dp}',f'-DA08_PAID_CONTINUATION={a.paid_continuation}','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off','-DR2_STARTUP_SUPPLY_MODE=2','-DR2_LOCAL_SALE_TIMING=1','-DR2_FINITE_FERTILIZER=1','-DR2_CROP_CLOCK_MODE=1','-DR2_OBSERVE_PUBLIC_TRADES=1',f'-DR2_MARKET_INTEGRAL={a.market_integral}',f'-DR2_SALE_CLOCK_MODE={a.sale_clock}',f'-DA08_COMPLETE_CROP_STOPS={a.complete_crop_stops}',f'-DA08_LAND_DP_MODE={a.mode}',f'-DA08_REALIZATION_MODE={a.realization}',f'-DA08_CROP_PORTFOLIO_MODE={a.crop_portfolio}',f'-DA08_CASH_LEDGER={a.cash_ledger}',f'-DA08_PREFIX_DAYS={a.prefix_days}','-fPIC','-shared','-Wl,-Bsymbolic']
 # Compile to a staging file. Never replace a known-good native if compilation
 # OR the exact-native action/value gate fails; rejected artifacts are retained.
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 stage=out.with_name(out.stem+'.building-'+stamp+'.so')
 logdir=ROOT/'build_runs'/stamp;logdir.mkdir(parents=True)
 cmd=[a.cxx,*flags,str(ROOT/'policy/bridge.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(stage)]
 (logdir/'compile.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 tick=time.perf_counter()
 with (logdir/'compile.stdout').open('w') as stdout,(logdir/'compile.stderr').open('w') as stderr:
  compiled=subprocess.run(cmd,stdout=stdout,stderr=stderr)
 (logdir/'compile.exit').write_text(str(compiled.returncode)+'\n')
 if compiled.returncode:raise SystemExit('Compilation failed; prior native preserved; see '+str(logdir))
 check_path=logdir/'native_choice_gate.json'
 check_command=[sys.executable,str(ROOT/'tests/native_choice_gate.py'),'--library',str(stage),'--enabled',str(a.animal_collection_labor),'--out',str(check_path)]
 (logdir/'native_choice_gate.command.json').write_text(json.dumps(check_command,indent=2)+'\n')
 with (logdir/'native_choice_gate.stdout').open('w') as stdout,(logdir/'native_choice_gate.stderr').open('w') as stderr:
  checked=subprocess.run(check_command,stdout=stdout,stderr=stderr,timeout=45)
 (logdir/'native_choice_gate.exit').write_text(str(checked.returncode)+'\n')
 if checked.returncode:raise SystemExit('Exact-native choice/value gate FAILED; prior native preserved; rejected staging library and logs retained: '+str(logdir))
 gate=json.loads(check_path.read_text())
 if gate.get('status')!='PASS' or gate.get('native_sha256')!=sha(stage):raise SystemExit('Gate/native identity mismatch; not promoted')
 os.replace(stage,out)

 files=sorted(x for x in (ROOT/'policy').rglob('*') if x.suffix in ['.hpp','.cpp','.inc','.py','.json'] and not x.name.endswith(('.BUILD.json','.CHECK.json')))
 files += [ROOT/'main.py',ROOT/'build.py'];receipt={'author':'A08','revision':'TRI_A08_r11_r2_fix1','animal_collection_labor':a.animal_collection_labor,'parent_native_sha256':'afca1d69fde38a0474546b70ecee38911477548dfde9aadc685c58a2a0b88a2f','sale_floor_dp':a.sale_floor_dp,'sale_schedule_dp':a.sale_schedule_dp,'paid_continuation':a.paid_continuation,'recovery_base':'TRI_A08_r11_r2_from_2f1557875d550000ae492c3814c84c78d5402e99b3a754308dc3907aa32b2178','complete_crop_stops':a.complete_crop_stops,'market_integral':a.market_integral,'sale_clock':a.sale_clock,'cash_ledger':a.cash_ledger,'prefix_days':a.prefix_days,'land_mode':a.mode,'realization_mode':a.realization,'crop_portfolio_mode':a.crop_portfolio,'status':'COMPILED; trajectory evaluation recorded separately','compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'command':cmd,'flags':flags,'build_log_directory':str(logdir),'native_choice_gate':gate,'native_choice_gate_command':check_command,'validation_sources':{'tests/native_choice_gate.py':sha(ROOT/'tests/native_choice_gate.py')},'seconds':time.perf_counter()-tick,'binary_sha256':sha(out),'sources':{str(f.relative_to(ROOT)):sha(f) for f in files}}
 out.with_suffix('.BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
