from pathlib import Path
import sys,subprocess,ctypes,json
import numpy as np
R=Path(__file__).resolve().parents[1];d=R/'models'/sys.argv[1]
s=R/'tests/model_predict.cpp';s.write_text('#include "../policy/learned_value.hpp"\nextern "C" double predict(const double*x,int n){return triad::learned::predict(std::vector<double>(x,x+n));}\n')
lib=R/'tests/model_predict.so';subprocess.run(['g++','-std=c++20','-O2','-fPIC','-shared',str(s),'-o',str(lib)],check=True)
f=ctypes.CDLL(str(lib)).predict;f.argtypes=[ctypes.POINTER(ctypes.c_double),ctypes.c_int];f.restype=ctypes.c_double
z=np.load(d/'prediction_fixture.npz');x=np.ascontiguousarray(z['X']);p=np.array([f(row.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),len(row)) for row in x]);err=float(np.abs(p-z['pred']).max());assert err<1e-10,err
(d/'export_parity.json').write_text(json.dumps(dict(status='PASS',samples=len(x),maximum_absolute_error=err),indent=2));print('model export PASS',err)
