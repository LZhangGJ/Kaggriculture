"""Expose the simulator already compiled inside frozen P16; do not rebuild it."""
from pathlib import Path
import json
import subprocess
import sysconfig
from pool_test import ROOT, WORKSPACE, BASELINE, digest

def main():
    panel = WORKSPACE / 'Kaggriculture/nt/latest_20260909_t2_r1_r2/arena/panel.cpp'
    text = panel.read_text()
    bridge = text[text.index('constexpr const char* ops[]'):text.index('dp7::View view(')]
    # Match official int(quantity), including Python floats and integer strings.
    bridge = bridge.replace('if(s.size()>2&&py::isinstance<py::int_>(s[2]))a.quantity=py::cast<int>(s[2]);',
                            'if(s.size()>2)a.quantity=py::cast<int>(py::module_::import("builtins").attr("int")(s[2]));')
    bridge = bridge.replace('o["player"]=seat;', 'o["remainingOverageTime"]=60;o["player"]=seat;')
    source = ROOT / 'native_bridge.cpp'
    source.write_text('#include "simulator.hpp"\n#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n'
                      '#include <algorithm>\nnamespace py=pybind11;\nusing namespace fastkag;\n' + bridge +
                      (ROOT / 'native_audit.inc').read_text())
    binary = BASELINE / 'policy/startupsupply2.so'
    output = ROOT / ('_pool_sim' + sysconfig.get_config_var('EXT_SUFFIX'))
    include = '/home/lzhang/kag/nt/latest_20260906_c3_f3_j7c3_search/.venv/lib/python3.12/site-packages/pybind11/include'
    cmd = ['g++-13','-std=c++20','-O3','-DNDEBUG','-fPIC','-shared','-Wl,-Bsymbolic',
           '-I'+include,'-I'+sysconfig.get_paths()['include'],
           '-I'+str(BASELINE / 'policy/executor/vendor'), str(source), str(binary), '-o',str(output)]
    subprocess.run(cmd, check=True)
    symbols = subprocess.check_output(['nm','-D','-C',str(output)],text=True)
    assert ' U fastkag::Simulator::step(' in symbols
    assert ' T fastkag::Simulator::step(' not in symbols
    receipt = dict(command=cmd, engine='Existing P16 compiled simulator, dynamically linked; no simulator rebuild',
                   hashes={str(p.relative_to(WORKSPACE)):digest(p) for p in
                           [Path(__file__),source,ROOT/'native_audit.inc',panel,binary,output,
                            BASELINE/'policy/executor/vendor/simulator.hpp',BASELINE/'policy/executor/vendor/simulator.cpp']},
                   dynamic_dependencies=subprocess.check_output(['ldd',str(output)],text=True),
                   compiled_engine_step_symbol='undefined in bridge, resolved from frozen P16 startupsupply2.so')
    (ROOT / 'NATIVE_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__ == '__main__':
    main()
