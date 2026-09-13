#!/usr/bin/env python3
"""Portable TRI_A08_r11_r1 builder (parent: exact A08_r11). Offline candidate; formal acceptance not run."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, time
ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--sale-floor-dp',type=int,choices=[0,1],default=1);p.add_argument('--sale-schedule-dp',type=int,choices=[0,1],default=1);p.add_argument('--paid-continuation',type=int,choices=[0,1],default=1);p.add_argument('--complete-crop-stops',type=int,choices=[0,1],default=1);p.add_argument('--market-integral',type=int,choices=[0,1,2],default=1);p.add_argument('--sale-clock',type=int,choices=[0,1,2],default=2);p.add_argument('--cxx',default=os.environ.get('CXX','g++'));p.add_argument('--mode',type=int,choices=[0,1,2],default=2);p.add_argument('--realization',type=int,choices=[0,1,2],default=0);p.add_argument('--crop-portfolio',type=int,choices=[0,1],default=1);p.add_argument('--cash-ledger',type=int,choices=[0,1],default=0);p.add_argument('--prefix-days',type=int,choices=[1,3,5],default=5);p.add_argument('--out',type=Path);a=p.parse_args()
 if shutil.which(a.cxx) is None:raise SystemExit('C++20 compiler not found: '+a.cxx)
 out=(a.out or ROOT/'policy/tri_a08_r11_r1.so').resolve();out.parent.mkdir(parents=True,exist_ok=True)
 if out==ROOT/'reference/P16/policy/startupsupply2.so':raise SystemExit('Refusing to overwrite frozen P16')
 flags=[f'-DA08_SALE_FLOOR_DP={a.sale_floor_dp}',f'-DA08_SALE_SCHEDULE_DP={a.sale_schedule_dp}',f'-DA08_PAID_CONTINUATION={a.paid_continuation}','-std=c++20','-O3','-DNDEBUG','-march=x86-64','-ffp-contract=off','-DR2_STARTUP_SUPPLY_MODE=2','-DR2_LOCAL_SALE_TIMING=1','-DR2_FINITE_FERTILIZER=1','-DR2_CROP_CLOCK_MODE=1','-DR2_OBSERVE_PUBLIC_TRADES=1',f'-DR2_MARKET_INTEGRAL={a.market_integral}',f'-DR2_SALE_CLOCK_MODE={a.sale_clock}',f'-DA08_COMPLETE_CROP_STOPS={a.complete_crop_stops}',f'-DA08_LAND_DP_MODE={a.mode}',f'-DA08_REALIZATION_MODE={a.realization}',f'-DA08_CROP_PORTFOLIO_MODE={a.crop_portfolio}',f'-DA08_CASH_LEDGER={a.cash_ledger}',f'-DA08_PREFIX_DAYS={a.prefix_days}','-fPIC','-shared','-Wl,-Bsymbolic']
 cmd=[a.cxx,*flags,str(ROOT/'policy/bridge.cpp'),str(ROOT/'policy/executor/vendor/simulator.cpp'),'-o',str(out)]
 tick=time.perf_counter();subprocess.run(cmd,check=True)
 files=sorted(x for x in (ROOT/'policy').rglob('*') if x.suffix in ['.hpp','.cpp','.inc','.py','.json'] and not x.name.endswith('.BUILD.json'))
 files += [ROOT/'main.py',ROOT/'build.py'];receipt={'author':'A08','revision':'TRI_A08_r11_r1','parent_native_sha256':'3b95f2c633242e31329adffae2438440a76b5eb6b11ece4897fe69f5710bb9cb','sale_floor_dp':a.sale_floor_dp,'sale_schedule_dp':a.sale_schedule_dp,'paid_continuation':a.paid_continuation,'recovery_base':'A08_r11_from_d2b4dad66c24be494421cfbcb48942a10162e12d8d81ea660eae5a7aeaab40c3','complete_crop_stops':a.complete_crop_stops,'market_integral':a.market_integral,'sale_clock':a.sale_clock,'cash_ledger':a.cash_ledger,'prefix_days':a.prefix_days,'land_mode':a.mode,'realization_mode':a.realization,'crop_portfolio_mode':a.crop_portfolio,'status':'COMPILED; trajectory evaluation recorded separately','compiler':subprocess.check_output([a.cxx,'--version'],text=True).splitlines()[0],'command':cmd,'seconds':time.perf_counter()-tick,'binary_sha256':sha(out),'sources':{str(f.relative_to(ROOT)):sha(f) for f in files}}
 out.with_suffix('.BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
