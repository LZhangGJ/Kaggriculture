#!/usr/bin/env python3
from pathlib import Path
import argparse,subprocess,tempfile
p=argparse.ArgumentParser();p.add_argument('--cxx',default='g++');a=p.parse_args()
root=Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix='a06_unit_') as d:
    out=Path(d)/'sale_order'
    subprocess.run([a.cxx,'-std=c++20','-O2',str(root/'tests/test_sale_order.cpp'),'-o',str(out)],check=True)
    subprocess.run([str(out)],check=True)
