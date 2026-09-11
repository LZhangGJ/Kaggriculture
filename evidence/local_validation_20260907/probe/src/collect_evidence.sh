#!/usr/bin/env bash
# Collect small receipts from the WSL build/validation into the repo (agent.md section 8 rules:
# JSON > 256 KB gzipped with pre-compression SHA-256 recorded; no .npz / .so / build objects).
set -e
B="$HOME/kag/nt/latest_20260906_c3_f3_j7c3_search"
K="$HOME/kag/nt/latest_20260906_keep2_rl_1000"
D="/mnt/d/Kaggriculture-keep2-rl-1000/evidence/local_validation_20260907"
PY="$B/.venv/bin/python"
rm -rf "$D"; mkdir -p "$D/build" "$D/smoke" "$D/portable" "$D/evaluate" "$D/probe/src"

cp "$B/build/BUILD_RECEIPT.json" "$D/build/base_BUILD_RECEIPT.json"
cp "$K/build/BUILD_RECEIPT.json" "$D/build/keep2_BUILD_RECEIPT.json"
cp "$B/runs/smoke_first/PORTABLE_ACCEPTANCE.json" "$D/smoke/PORTABLE_ACCEPTANCE.json"
# Regenerated keep2 receipt is byte-identical to the committed one; record hash only.
sha256sum "$K/PORTABLE_ACCEPTANCE.json" | awk '{print $1}' > "$D/portable/regenerated_PORTABLE_ACCEPTANCE.sha256"
sha256sum /mnt/d/Kaggriculture-keep2-rl-1000/nt/latest_20260906_keep2_rl_1000/PORTABLE_ACCEPTANCE.json | awk '{print $1}' > "$D/portable/committed_PORTABLE_ACCEPTANCE.sha256"
cp "$K/runs/keep/summary.json" "$D/evaluate/keep_summary.json"
cp "$K/runs/keep/games.json" "$D/evaluate/keep_games.json"
cp "$K/runs/aux1000_greedy/summary.json" "$D/evaluate/aux_r0_step1000_greedy_summary.json"
cp "$K/runs/aux1000_greedy/games.json" "$D/evaluate/aux_r0_step1000_greedy_games.json"
cp "$B/runs/deltav_probe_100/REPORT.json" "$D/probe/REPORT.json"
for n in keep_f3 keep_patched probe_mode4; do cp "$B/runs/deltav_probe_100/${n}_games.json" "$D/probe/${n}_games.json"; done
cp "$B/src/f3_deltav_policy.cpp" "$D/probe/src/"
cp "$B/deltav_probe.py" "$D/probe/src/"
cp /tmp/probe_chain.sh "$D/probe/src/probe_chain.sh"
cp /tmp/validate_chain.sh "$D/probe/src/validate_chain.sh"
cp "$0" "$D/probe/src/collect_evidence.sh" 2>/dev/null || true

# Manifest with pre-compression hashes, then gzip any JSON over 256 KB.
"$PY" - "$D" "$B" "$K" <<'EOF'
import hashlib, json, os, sys, gzip, subprocess, platform
D, B, K = sys.argv[1:4]
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''): h.update(c)
    return h.hexdigest()
files = {}
for root, _, names in os.walk(D):
    for n in sorted(names):
        p = os.path.join(root, n); rel = os.path.relpath(p, D).replace(os.sep, '/')
        if rel == 'MANIFEST.json': continue
        size = os.path.getsize(p); entry = dict(bytes=size, sha256=sha(p))
        if n.endswith('.json') and size > 256 * 1024:
            with open(p, 'rb') as f: data = f.read()
            with gzip.open(p + '.gz', 'wb', compresslevel=9) as g: g.write(data)
            os.remove(p); entry['stored_as'] = rel + '.gz'; entry['gz_bytes'] = os.path.getsize(p + '.gz'); entry['transform'] = 'gzip_lossless'
        files[rel] = entry
env = dict(
    distro=open('/etc/os-release').read().split('PRETTY_NAME=')[1].split('\n')[0].strip('"'),
    kernel=platform.release(), cores=os.cpu_count(),
    gxx=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
    python=sys.version.split()[0],
    numpy=__import__('numpy').__version__, pybind11=__import__('pybind11').__version__,
    libraries={os.path.basename(p): sha(p) for p in [f'{B}/build/f3.so', f'{B}/build/f3_deltav.so', f'{K}/build/keep2.so']
               + [os.path.join(f'{B}/build', n) for n in os.listdir(f'{B}/build') if n.startswith('_economic_rl_native')]},
)
json.dump(dict(date='2026-09-07', environment=env, files=files), open(os.path.join(D, 'MANIFEST.json'), 'w'), indent=2)
print(json.dumps(env, indent=2))
EOF
echo "=== evidence tree ==="
find "$D" -type f -printf '%8s  %P\n' | sort -k2
echo COLLECT_DONE
