"""chroot-v1 bot sandbox for hosts without docker (vast containers). source-v73, env PPO_ARENA_SANDBOX=chroot.

Same runtime as the docker sandbox: the pinned arena image (e6b2...) exported to PPO_CHROOT_RUNTIME (its
runtime-receipt.json must match .arena/config.json image). One jail per BotMux: `cp -al` of the runtime (hard links,
same filesystem as PPO_CHROOT_STATE), the mux worker + arena_obs at /opt/arena_gpu, bot zips at /bots/<gid>.zip, the
shared vendor runtime (student torch) at /opt/arena_gpu/vendor. Every mux worker runs as its own unprivileged UID with a
private 0700 /work/w<N>, a seccomp filter (no sockets/ptrace/mount/chroot/...), no_new_privs and rlimits, launched
by arena_chroot_launch.py (needs root: chroot, setuid, mknod). Differences from docker: no network namespace (sockets
blocked by seccomp instead), no cgroup memory cap (RLIMIT_AS is not used: torch bots reserve huge virtual memory)."""
import fcntl, json, os, shutil, stat, subprocess, sys, uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAUNCH = HERE / 'arena_chroot_launch.py'


def _uid(state):
    with open(state / 'uid-counter', 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0); uid = int(f.read() or '300000') + 1
        if uid >= 2000000000: uid = 300001
        f.seek(0); f.truncate(); f.write(str(uid)); f.flush()
    return uid


def check_runtime(cfg):
    '''Docker-free stand-in for tools.arena.sandbox.preflight: immutable digest + exported runtime receipt matches it.'''
    image = cfg.get('image') or ''
    if '@sha256:' not in image and not image.startswith('sha256:'):
        raise RuntimeError('Configure an immutable Docker image digest first')
    rt = Path(os.environ['PPO_CHROOT_RUNTIME'])
    rec = json.loads((rt.parent / 'runtime-receipt.json').read_text())
    if rec.get('image') != image or not (rt / 'usr/local/bin/python').exists():
        raise RuntimeError('chroot runtime is not the pinned arena image: %s vs %s' % (rec.get('image'), image))


class Jail:
    def __init__(self, vendor=None):
        self.runtime = Path(os.environ['PPO_CHROOT_RUNTIME'])
        self.state = Path(os.environ.get('PPO_CHROOT_STATE', str(self.runtime.parent / 'chroot-state')))
        self.state.mkdir(parents=True, exist_ok=True)
        rec = json.loads((self.runtime.parent / 'runtime-receipt.json').read_text())
        cfg = json.loads(Path(os.environ.get('PPO_ARENA_CONFIG', '/home/keith/kaggriculture-arena/.arena/config.json')).read_text())
        if rec.get('image') != cfg['image']:
            raise RuntimeError('chroot runtime is not the pinned arena image: %s vs %s' % (rec.get('image'), cfg['image']))
        self.path = self.state / ('jail-%s' % uuid.uuid4().hex[:12])
        subprocess.run(['cp', '-al', str(self.runtime), str(self.path)], check=True)
        for name, mode in (('work', 0o711), ('tmp', 0o1777), ('bots', 0o755), ('opt/arena_gpu', 0o755)):
            p = self.path / name; p.mkdir(parents=True, exist_ok=True); p.chmod(mode)
        (self.path / 'dev').mkdir(exist_ok=True)
        for name, minor in (('null', 3), ('zero', 5), ('random', 8), ('urandom', 9)):
            p = self.path / 'dev' / name
            if not p.exists(): os.mknod(p, stat.S_IFCHR | 0o666, os.makedev(1, minor))
            os.chmod(p, 0o666)
        for f in ('arena_mux_worker.py', 'arena_obs.py'):
            dst = self.path / 'opt/arena_gpu' / f
            if dst.exists(): dst.unlink()           # never write through a hard link into the shared runtime
            shutil.copyfile(HERE / f, dst); dst.chmod(0o644)
        if vendor:
            dst = self.path / 'opt/arena_gpu/vendor'
            r = subprocess.run(['cp', '-al', str(vendor), str(dst)])
            if r.returncode:  # cross-device: a partial hard-link tree exists; cp -a into it would nest vendor/<name>
                shutil.rmtree(dst, ignore_errors=True); subprocess.run(['cp', '-a', str(vendor), str(dst)], check=True)
        self.workers = []

    def add_zip(self, gid, src):
        dst = self.path / 'bots' / ('%d.zip' % gid)
        shutil.copyfile(src, dst); dst.chmod(0o644)
        return '/bots/%d.zip' % gid

    def command(self, w, mem_mb=0):
        uid = _uid(self.state)
        wd = self.path / 'work' / ('w%d' % w); wd.mkdir(); os.chown(wd, uid, uid); wd.chmod(0o700)
        self.workers.append(uid)
        return ['/usr/bin/python3', str(LAUNCH), str(self.path), str(uid), '/work/w%d' % w,
                '/usr/local/bin/python', '/opt/arena_gpu/arena_mux_worker.py', '/work/w%d' % w], '/work/w%d' % w

    def cleanup(self):
        shutil.rmtree(self.path, ignore_errors=True)
