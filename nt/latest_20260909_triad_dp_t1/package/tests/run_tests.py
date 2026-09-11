from pathlib import Path
import subprocess,sys
R=Path(__file__).resolve().parents[1];out=R/'tests/test_mechanisms'
subprocess.run(['g++','-std=c++20','-O2',str(R/'tests/test_mechanisms.cpp'),str(R/'policy/executor/vendor/simulator.cpp'),'-o',str(out)],check=True)
subprocess.run([str(out)],check=True)
