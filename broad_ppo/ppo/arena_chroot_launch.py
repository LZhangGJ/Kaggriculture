"""Root launcher for one chroot mux worker (source-v73 chroot-v1; derived from the arena's chroot_launch.py).
argv: jail uid workdir command...   Differences from the arena launcher: no CPU pinning, no RLIMIT_AS (torch bots),
larger NPROC/NOFILE/FSIZE for a worker hosting several games; stdin/stdout stay the frame pipes."""
import ctypes, errno, os, resource, sys

root, uid, workdir, *command = sys.argv[1:]
uid = int(uid)
lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
lib.seccomp_init.argtypes = [ctypes.c_uint32]; lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x7fff0000)
assert ctx
for name in ('socket', 'socketpair', 'ptrace', 'process_vm_readv', 'process_vm_writev',
             'mount', 'umount2', 'pivot_root', 'chroot', 'setns', 'unshare', 'bpf',
             'keyctl', 'add_key', 'request_key', 'open_by_handle_at', 'io_uring_setup',
             'setsid', 'setpgid', 'reboot', 'kexec_load', 'init_module', 'finit_module'):
    nr = lib.seccomp_syscall_resolve_name(name.encode())
    if nr >= 0:
        assert lib.seccomp_rule_add(ctx, 0x50000 | errno.EPERM, nr, 0) == 0
os.chroot(root)
os.chdir(workdir)
os.setgroups([])
os.setgid(uid)
os.setuid(uid)
resource.setrlimit(resource.RLIMIT_NPROC, (512, 512))
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))
resource.setrlimit(resource.RLIMIT_FSIZE, (8192 * 1024 ** 2, 8192 * 1024 ** 2))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
libc = ctypes.CDLL(None, use_errno=True)
assert libc.prctl(38, 1, 0, 0, 0) == 0          # PR_SET_NO_NEW_PRIVS
assert lib.seccomp_load(ctx) == 0
os.environ.clear()
os.environ.update(PATH='/usr/local/bin:/usr/bin:/bin', HOME='/tmp', PYTHONUNBUFFERED='1',
                  OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.execv(command[0], command)
