"""Read-only, streaming audit of recoverable seeds. Python standard library only."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import time

TEXT = {'.json', '.jsonl', '.log', '.py', '.md', '.toml', '.txt', '.csv',
        '.sh', '.ps1', '.yaml', '.yml', '.cpp', '.hpp', '.h', '.inc', '.patch', ''}
KEY = re.compile(rb'''["']?\b([A-Za-z_]*seed[A-Za-z_0-9]*)["']?\s*[:=]\s*(\[[\d\s,+'"-]{1,65000}\]|-?\d+)''', re.I)
CLI = re.compile(rb'''--seed(?:-start|_start)?["'\s,:=]+(-?\d+)''', re.I)
SEED_LIST_CLI = re.compile(rb'''--seeds?["'\s,:=]+((?:\d+["'\s,]+){1,512})''', re.I)
FILENAME = re.compile(r'(?:seed[-_]?|trace-.*-)(\d+)(?:[-_.]|$)', re.I)
INTEGER = re.compile(rb'(?<![\w.])-?\d+(?![\w.])')
DOMAIN = 2**31
CHUNK = 4 * 1024 * 1024
OVERLAP = 65536


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(CHUNK): h.update(block)
    return h.hexdigest()


def constant(node, env):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, str)):
        return node.value
    if isinstance(node, ast.Name): return env[node.id]
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [constant(x, env) for x in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -constant(node.operand, env)
    if isinstance(node, ast.BinOp):
        a, b = constant(node.left, env), constant(node.right, env)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
        if isinstance(node.op, ast.Mult) and isinstance(a, int) and isinstance(b, int): return a*b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        args = [constant(x, env) for x in node.args]
        if node.func.id == 'range' and all(type(x) is int for x in args):
            r = range(*args)
            if len(r) <= 100000: return list(r)
        if node.func.id in ('list', 'tuple', 'set') and len(args) == 1: return list(args[0])
        if node.func.id == 'int' and len(args) == 1: return int(args[0])
    raise ValueError('Not a bounded constant expression')


def audit(repo, out):
    repo, out = repo.resolve(), out.resolve()
    if out == repo or repo in out.parents: raise ValueError('Audit output must be outside source')
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    paths = sorted(p for p in repo.rglob('*') if p.is_file() and
                   '.git' not in p.relative_to(repo).parts and '__pycache__' not in p.parts and p.suffix != '.pyc')
    # Freeze byte limits first. Files may keep growing in the active campaign.
    inventory = [(p, p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    exclusions, rows, gaps = set(), [], []
    method_counts = Counter()
    total = 0
    for index, (p, size, mtime) in enumerate(inventory):
        rel = p.relative_to(repo).as_posix()
        found, methods = set(), Counter()
        def add(values, method):
            for x in values:
                if type(x) is int and 0 <= x < 2**64:
                    found.add(x); methods[method] += 1
        for match in FILENAME.finditer(p.name): add([int(match[1])], 'filename')
        rec = dict(path=rel, size_at_inventory=size, mtime_ns_at_inventory=mtime)
        compressed = p.name.endswith('.gz')
        if p.suffix not in TEXT and not compressed:
            rec['status'] = 'binary_not_parsed'
            rec['sha256'] = sha(p)
            gaps.append(dict(path=rel, reason='Binary content not parsed; adjacent manifests/text scanned'))
        else:
            digest = hashlib.sha256(); read_bytes = 0
            # Gzip files are small in this corpus. Freeze compressed bytes, then stream expansion.
            if compressed:
                with p.open('rb') as f: data = f.read(size)
                digest.update(data); read_bytes = len(data)
                stream = gzip.GzipFile(fileobj=io.BytesIO(data))
            else:
                stream = p.open('rb')
            tail = b''; small = bytearray(); expanded = 0
            try:
                while True:
                    n = CHUNK if compressed else min(CHUNK, size-read_bytes)
                    if n <= 0: break
                    block = stream.read(n)
                    if not block: break
                    if not compressed: digest.update(block); read_bytes += len(block)
                    expanded += len(block)
                    if size < 8*1024*1024 and not compressed: small.extend(block)
                    text = tail+block
                    for match in KEY.finditer(text):
                        key = match[1].decode().lower()
                        vals = [int(x) for x in INTEGER.findall(match[2])]
                        add(vals, 'seed_key_literal')
                        if 'range' in key and len(vals) in (2, 3):
                            try:
                                rr = range(*vals)
                                if len(rr) <= 100000: add(rr, 'seed_range_inclusive_conservative'); add([vals[1]], 'seed_range_endpoint')
                            except ValueError: pass
                    for match in CLI.finditer(text): add([int(match[1])], 'cli_seed_or_start')
                    # Treat --seeds numeric arguments as exclusions too; some are counts.
                    for match in SEED_LIST_CLI.finditer(text): add([int(x) for x in INTEGER.findall(match[1])], 'cli_seed_list_or_count')
                    tail = text[-OVERLAP:]
                rec.update(status='scanned', sha256=digest.hexdigest(), bytes_hashed=read_bytes, expanded_bytes=expanded)
            except (OSError, EOFError) as e:
                rec.update(status='read_error', error=str(e)); gaps.append(dict(path=rel, reason=str(e)))
            finally: stream.close()
            if small and p.suffix == '.py':
                try:
                    tree = ast.parse(small.decode('utf-8-sig')); env = {}
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Assign, ast.AnnAssign)):
                            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                            try: value = constant(node.value, env)
                            except (ValueError, KeyError, TypeError, OverflowError): continue
                            for target in targets:
                                if isinstance(target, ast.Name):
                                    env[target.id] = value
                                    if 'seed' in target.id.lower(): add(value if isinstance(value, list) else [value], 'python_seed_constant')
                        if isinstance(node, ast.Call):
                            # Constant ranges are a conservative superset of source seed ranges.
                            if isinstance(node.func, ast.Name) and node.func.id == 'range':
                                try: add(constant(node, env), 'python_constant_range_conservative')
                                except (ValueError, KeyError, TypeError, OverflowError): pass
                            for kw in node.keywords:
                                if kw.arg and 'seed' in kw.arg.lower():
                                    try:
                                        v = constant(kw.value, env); add(v if isinstance(v, list) else [v], 'python_seed_keyword')
                                    except (ValueError, KeyError, TypeError, OverflowError): pass
                except (SyntaxError, UnicodeError) as e: gaps.append(dict(path=rel, reason='Python AST parse: '+str(e)))
            if small and p.suffix == '.csv':
                try:
                    for row in csv.DictReader(io.StringIO(small.decode('utf-8-sig'))):
                        for key, value in row.items():
                            if key and 'seed' in key.lower():
                                try: add([int(value)], 'csv_seed_column')
                                except (TypeError, ValueError): pass
                except (UnicodeError, csv.Error) as e: gaps.append(dict(path=rel, reason=str(e)))
        now = p.stat()
        rec['changed_during_audit'] = now.st_size != size or now.st_mtime_ns != mtime
        rec.update(recovered_seeds=sorted(found), methods=dict(methods))
        exclusions.update(found); method_counts.update(methods); rows.append(rec); total += size
        if index % 250 == 0: print(json.dumps(dict(files=index+1, total_files=len(paths), bytes_seen=total, unique_exclusions=len(exclusions))), flush=True)
    dump(out/'exclusions.json', dict(schema_version=1, source_root=str(repo), seeds=sorted(exclusions),
          purpose='Conservative exclusion superset; includes seed literals, counts and constant ranges'))
    with (out/'files.jsonl').open('w', encoding='utf-8') as f:
        for rec in rows: f.write(json.dumps(rec, sort_keys=True)+'\n')
    current_paths = {p.relative_to(repo).as_posix() for p in repo.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    summary = dict(source_root=str(repo), inventory_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(started)),
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), seconds=time.time()-started,
        inventory_files=len(rows), scanned_files=sum(r['status']=='scanned' for r in rows), bytes_at_inventory=total,
        parsed_expanded_bytes=sum(r.get('expanded_bytes',0) for r in rows), files_with_seed_evidence=sum(bool(r['recovered_seeds']) for r in rows),
        unique_exclusions=len(exclusions), exclusions_in_sampling_domain=sum(s<DOMAIN for s in exclusions),
        method_match_counts=dict(method_counts), exclusions_sha256=sha(out/'exclusions.json'),
        file_inventory_sha256=sha(out/'files.jsonl'),
        changed_files=[r['path'] for r in rows if r['changed_during_audit']],
        new_files_after_inventory=sorted(current_paths-{r['path'] for r in rows}), gaps=gaps,
        limitations=['Only this supplied source tree; no Git history, sibling checkouts, remote training host, deleted logs, or external replay stores.',
                    'Read-only byte-prefix inventory, not an atomic campaign snapshot; changed and new files need a release-time re-audit.',
                    'Dynamic seeds absent from saved literals, rows, manifests or filenames cannot be recovered.',
                    'Binary arrays/models are not decoded; adjacent text is scanned. Constant ranges and seed counts over-exclude conservatively.',
                    'Seed arrays longer than 65,000 bytes or nonliteral nested expressions may be missed unless emitted as individual row seeds.'])
    dump(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('gaps','method_match_counts','limitations','changed_files','new_files_after_inventory')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); audit(a.repo,a.out)
