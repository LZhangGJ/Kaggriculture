from pathlib import Path
import subprocess,sysconfig,time
R=Path(__file__).parent;N=R/'arena/experiments/daily_dp_v7_20260903/native';B=N/'build'
inc=Path(sysconfig.get_paths()['purelib'])/'torch/include'
try:
 import pybind11
 inc=pybind11.get_include()
except ImportError:pass
start=time.perf_counter()
cmd=['g++','-std=c++20','-O2','-DNDEBUG','-fPIC','-fopenmp','-shared','-Wl,-Bsymbolic','-I'+str(inc),'-I'+sysconfig.get_paths()['include'],'-I'+str(N),str(R/'src/panel.cpp')]+[str(B/(s+'.o')) for s in ['simulator','fieldbook_adapter','boatlee_v29','kaito_v58','lynn_v5','three_day_adapter','ecobot_v7_core','ecobot_v7']]+['-ldl','-o',str(R/('_triad_panel'+sysconfig.get_config_var('EXT_SUFFIX')))]
subprocess.run(cmd,check=True);print('panel seconds',time.perf_counter()-start)
