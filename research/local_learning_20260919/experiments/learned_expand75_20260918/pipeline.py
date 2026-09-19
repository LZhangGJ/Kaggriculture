"""Collect training-only games, expand by labels, train, then evaluate a frozen candidate."""
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))


def save(value):
    temp = ROOT / 'PIPELINE.tmp'
    temp.write_text(json.dumps(dict(heartbeat_unix=time.time(), **value), indent=2), encoding='utf8')
    temp.replace(ROOT / 'PIPELINE.json')


def run(script, *arguments):
    subprocess.run([sys.executable, '-B', '-u', str(ROOT / script), *map(str, arguments)], cwd=ROOT, check=True)


def wait_for_collection():
    ticket_path = ROOT / 'WAIT_FOR.json'
    if not ticket_path.exists():
        return
    ticket = read('WAIT_FOR.json')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    api.GetProcessTimes.restype = wintypes.BOOL
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x100000 | 0x1000, False, ticket['pid'])
    if not handle:
        if ctypes.get_last_error() == 87:
            return
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        created, exited, kernel, user = [wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(handle, *[ctypes.byref(x) for x in (created, exited, kernel, user)]):
            raise ctypes.WinError(ctypes.get_last_error())
        creation = (created.dwHighDateTime << 32) | created.dwLowDateTime
        if creation != ticket['creation_filetime']:
            return
        while True:
            result = api.WaitForSingleObject(handle, 1000)
            if result == 0:
                break
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
            save(dict(status='WAITING_FOR_LIVE_COLLECTOR', dependency_pid=ticket['pid'],
                      dependency_creation_filetime=creation, dependency_confirmed_live=True))
    finally:
        api.CloseHandle(handle)


def main():
    try:
        # Never silently change executable code after the experiment was registered.
        for name, expected in read('CODE_HASHES.json').items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
        wait_for_collection()
        if not (ROOT / 'data/fresh_packed/READY.json').exists():
            save(dict(status='COLLECTING_TRAINING_DATA'))
            run('collect.py')
        if not (ROOT / 'run/DONE.json').exists():
            save(dict(status='TRAINING_EXPANDED_MARKET'))
            run('train.py')
        done = read('run/DONE.json')
        assert done['status'] == 'COMPLETE' and done['best_step'] > 0
        if not done['passes_regression_guard']:
            save(dict(status='VALIDATION_REGRESSION', training=done))
            return
        candidate = ROOT / 'run/best.pt'
        save(dict(status='DEVELOPMENT', training=done))
        run('evaluate.py', '--checkpoint', candidate, '--name', 'development', '--panel', 'development')
        development = read('evaluation/development/SUMMARY.json')
        if not development['meets_target']:
            save(dict(status='DEVELOPMENT_BELOW_TARGET', development=development, training=done))
            return
        target = ROOT / 'selected.pt'
        if target.exists():
            assert target.read_bytes() == candidate.read_bytes()
        else:
            shutil.copy2(candidate, target)
        save(dict(status='FINAL_EVALUATION', training=done, development=development))
        run('evaluate.py', '--checkpoint', target, '--name', 'fresh_final', '--panel', 'final')
        final = read('evaluation/fresh_final/SUMMARY.json')
        save(dict(status='TARGET_MET' if final['meets_target'] else 'FINAL_BELOW_TARGET',
                  final=final, training=done, development=development,
                  selected_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    except BaseException as exc:
        save(dict(status='FAILED', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
