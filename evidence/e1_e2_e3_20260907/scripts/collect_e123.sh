#!/usr/bin/env bash
# Copy E1/E2/E3 receipts into the repo evidence dir (agent.md section 8 rules).
# Re-runnable: copies whatever exists, gzips JSON > 256 KB, rewrites MANIFEST.
# Never touches decisions.npz or dev_*/games.json (large; stay in WSL).
set -e
SP="/mnt/c/Users/DHU_Z/AppData/Local/Temp/claude/d--Kaggriculture-keep2-rl-1000/748d2719-6a93-4579-8d30-be90143373d3/scratchpad"
K="$HOME/kag"
D="/mnt/d/Kaggriculture-keep2-rl-1000/evidence/e1_e2_e3_20260907"
PY="$K/nt/latest_20260906_c3_f3_j7c3_search/.venv/bin/python"
mkdir -p "$D/scripts" "$D/E1" "$D/E2" "$D/E3"

# scripts (LF-normalised copies that actually ran)
cp "$K/e1_scripts/e1_train.py" "$K/e2_scripts/e2_train.py" "$K/e3_scripts/e3_train.py" "$D/scripts/"
cp /tmp/queue_e2_e3.sh "$D/scripts/queue_e2_e3.sh" 2>/dev/null || true
tr -d '\r' < "$SP/e123_stats.py" > "$D/scripts/e123_stats.py"
cp "$SP/e123_stats.json" "$D/E123_STATS.json"
cp "$0" "$D/scripts/collect_e123.sh" 2>/dev/null || true
cp /tmp/e1_full.log "$D/E1/train.log" 2>/dev/null || true
cp /tmp/queue_e2_e3.log "$D/queue_e2_e3.log" 2>/dev/null || true

copy_run () {  # $1 = source run dir, $2 = dest dir
    mkdir -p "$2"
    for f in COMPLETE.json PROGRESS.json PROTOCOL.json selection.json; do [ -f "$1/$f" ] && cp "$1/$f" "$2/"; done
    for sub in final_test final_heldout final_full_panel; do
        if [ -d "$1/$sub" ]; then mkdir -p "$2/$sub"; cp "$1/$sub/summary.json" "$1/$sub/games.json" "$2/$sub/" 2>/dev/null || true; fi
    done
}

# E1
cp "$K/e1_training/aux_SUMMARY_SO_FAR.json" "$D/E1/SUMMARY.json"
mkdir -p "$D/E1/keep_on_e1_panel"; cp "$K/e1_training/keep_on_e1_panel/summary.json" "$K/e1_training/keep_on_e1_panel/games.json" "$D/E1/keep_on_e1_panel/"
for i in 0 1 2 3 4 5; do copy_run "$K/e1_training/aux_e$i" "$D/E1/aux_e$i"; done
# E2
for k in three_day g003 kaito; do
    mkdir -p "$D/E2/keep_$k"; cp "$K/e2_training/keep_$k/summary.json" "$K/e2_training/keep_$k/games.json" "$D/E2/keep_$k/"
    for r in 0 1; do copy_run "$K/e2_training/e2_${k}_r$r" "$D/E2/e2_${k}_r$r"; done
done
# E3 (whatever exists)
[ -f "$K/e3_training/E3_SUMMARY_SO_FAR.json" ] && cp "$K/e3_training/E3_SUMMARY_SO_FAR.json" "$D/E3/SUMMARY.json"
for i in 0 1 2; do [ -d "$K/e3_training/aux_paired_e$i" ] && copy_run "$K/e3_training/aux_paired_e$i" "$D/E3/aux_paired_e$i"; done

# gzip large JSON, write manifest with pre-compression hashes
"$PY" - "$D" <<'EOF'
import hashlib, json, os, sys, gzip, subprocess, platform
D = sys.argv[1]
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
        if n.endswith('.json.gz'):                       # already compressed on a previous run
            files[rel[:-3]] = dict(stored_as=rel, gz_bytes=os.path.getsize(p), transform='gzip_lossless', **files.get(rel[:-3], {}))
            continue
        size = os.path.getsize(p); entry = dict(bytes=size, sha256=sha(p))
        if n.endswith('.json') and size > 256 * 1024:
            data = open(p, 'rb').read()
            with gzip.open(p + '.gz', 'wb', compresslevel=9) as g: g.write(data)
            os.remove(p); entry.update(stored_as=rel + '.gz', gz_bytes=os.path.getsize(p + '.gz'), transform='gzip_lossless')
        files[rel] = entry
env = dict(distro=open('/etc/os-release').read().split('PRETTY_NAME=')[1].split('\n')[0].strip('"'),
           kernel=platform.release(), cores=os.cpu_count(),
           gxx=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
           python=sys.version.split()[0], torch=__import__('torch').__version__,
           gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip(),
           numpy=__import__('numpy').__version__)
json.dump(dict(date='2026-09-07', environment=env, files=files), open(os.path.join(D, 'MANIFEST.json'), 'w'), indent=1)
print(json.dumps(env, indent=2))
EOF
echo "=== evidence tree ==="
find "$D" -type f -printf '%8s  %P\n' | sort -k2 | head -80
du -sh "$D"
echo COLLECT_E123_DONE
