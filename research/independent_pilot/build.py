"""Build the isolated CPU extension; record the source and binary hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig
import torch

ROOT = Path(__file__).resolve().parent


def main():
    out = ROOT / ('native' + sysconfig.get_config_var('EXT_SUFFIX'))
    cmd = ['g++', '-O3', '-std=c++20', '-shared', '-fPIC', '-pthread',
           '-I' + sysconfig.get_paths()['include'],
           '-I' + str(Path(torch.__file__).parent / 'include'),
           str(ROOT / 'native.cpp'), '-o', str(out)]
    subprocess.run(cmd, check=True)
    receipt = {'command': cmd, 'compiler': subprocess.check_output(['g++', '--version'], text=True),
               'torch': torch.__version__, 'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in [ROOT / 'native.cpp', ROOT / 'planner.hpp', ROOT / 'actor.hpp', out]}}
    (ROOT / 'build_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
