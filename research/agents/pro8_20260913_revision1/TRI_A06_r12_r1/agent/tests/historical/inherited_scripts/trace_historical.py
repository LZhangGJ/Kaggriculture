from pathlib import Path
import importlib.util,ctypes,gzip,json,hashlib
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();C=Path(__file__).resolve().parents[1];H=C/'validation/history';D=args.out.resolve();D.mkdir(parents=True,exist_ok=False);s=importlib.util.spec_from_file_location('tracecodec',C/'policy/agent.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
cfg=json.loads((C/'policy/config.json').read_text());a=m.Agent(cfg|{'a06_r11_recovery':0},C/'policy/a06.so')
f=a.lib.td_r11_conditional_trace;f.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.c_int,ctypes.c_int];f.restype=ctypes.c_char_p
r=json.loads(gzip.decompress((H/'author_A05_r4_2610020000_seat0.replay.json.gz').read_bytes()))
for t in range(219):assert a(r['steps'][t][0]['observation'])==r['actions'][t][0]
o=r['steps'][219][0]['observation'];packed=m._pack(o);results={}
for enabled in (0,1):
 text=f(a.handle,packed,len(packed),enabled,21).decode();assert not text.startswith('ERROR'),text;j=json.loads(text);results[str(enabled)]=j
 (D/f'A05_COMMON219_CONDITIONAL_{enabled}.json').write_text(json.dumps(j,indent=2))
print(json.dumps({k:{'frames':len(v['frames']),'endcash':v['endpoint_cash'],'rival':v['endpoint_rival_cash'],'requests':sum(any(a[0]==18 for a in z['market']) for z in v['frames']),'confirmed':max(z['confirmed_recovery_workers'] for z in v['frames']),'committed':max(z['committed_routes'] for z in v['frames'])}for k,v in results.items()}))
# Restore source audit only: evaluating conditional clones never mutates the original context.
assert a(o)==r['actions'][219][0];a.close()
(D/'CONDITIONAL_TRACE_META.json').write_text(json.dumps({'native_sha256':hashlib.sha256((C/'policy/a06.so').read_bytes()).hexdigest(),'original_prefix_matches':219,'conditional_transitions':42,'new_games':0,'no_actual_suffix_used':True,'original_context_unchanged':True},indent=2))
