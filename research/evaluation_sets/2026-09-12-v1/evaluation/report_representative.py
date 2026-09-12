"""Read-only representative-panel report while other frozen panels continue."""
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from analyze_panels import ORDER,LABELS,KEY,interval

HERE=Path(__file__).resolve().parent
PANELS=HERE.parent/'outputs/seed_sets_v1'
OUT=HERE.parent/'outputs/representative_results';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_bytes((json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
roster=json.loads((HERE/'roster.json').read_bytes())
manifest=json.loads((HERE/'runs/development-v1/MANIFEST.json').read_bytes())
assert sha(HERE/'roster.json')==manifest['roster_sha256']
release=json.loads((HERE/'HOLDOUT_RELEASE.json').read_bytes())
assert sha(HERE/'analyze_panels.py')==release['analysis_code_sha256']
for f,h in roster['runtime_files'].items():assert sha(HERE/'runtime'/f)==h,f
seeds=json.loads((PANELS/'manifests/representative.json').read_bytes())['seeds']
opponents=[o['id'] for o in json.loads((PANELS/'opponents.json').read_bytes())['opponents']]
assert len(seeds)==256 and len(opponents)==16
shape=(6,256,16,2);wins=np.zeros(shape,dtype=np.int8);ties=np.zeros(shape,dtype=np.int8);cash=np.zeros(shape);margin=np.zeros(shape)
seen=np.zeros(shape,dtype=bool);diagnostics=np.zeros(shape,dtype=bool)
source=HERE/'runs/development-v1/rows.jsonl';size=source.stat().st_size
with source.open('rb') as f:prefix=f.read(size)
lines=prefix.splitlines(keepends=True);representative=[];ignored_tail=0
for line in lines:
    if not line.endswith(b'\n'):ignored_tail+=1;continue
    r=json.loads(line)
    if r['panel']!='representative':continue
    assert r['valid'] and not r.get('runtime_error') and r['steps']==719
    assert r['candidate_seat']==1-r['opponent_seat']
    assert r['win']==(r['own_cash']>r['opponent_cash']) and r['tie']==(r['own_cash']==r['opponent_cash'])
    assert r['margin']==r['own_cash']-r['opponent_cash']
    assert len(r['action_hash'])==64
    at=(ORDER.index(r['candidate_id']),seeds.index(r['seed']),opponents.index(r['opponent']),r['candidate_seat'])
    assert not seen[at],'Duplicate representative cell'
    seen[at]=True;wins[at]=r['win'];ties[at]=r['tie'];cash[at]=r['own_cash'];margin[at]=r['margin']
    diagnostics[at]=bool(r.get('diagnostic_error'));representative.append(line)
assert seen.all() and len(representative)==49152
raw=b''.join(representative)
(OUT/'rows.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
rng=np.random.default_rng(int.from_bytes(hashlib.sha256((KEY+':representative').encode()).digest(),'big'))
draws=rng.integers(0,256,size=(4000,256));counts=np.zeros((4000,256),dtype=np.int16)
np.add.at(counts,(np.repeat(np.arange(4000),256),draws.ravel()),1)
def group(mask):
    total=int(mask.sum());den=mask.sum(axis=(1,2));bootstrap_den=counts@den;result={};boots={}
    for c,cid in enumerate(ORDER):
        w=(wins[c]*mask).sum(axis=(1,2));nwin=int(w.sum());ndraw=int((ties[c]*mask).sum())
        boots[cid]=(counts@w)/bootstrap_den
        result[cid]=dict(wins=nwin,draws=ndraw,losses=total-nwin-ndraw,games=total,win_rate=nwin/total,
                        ci95=interval(boots[cid]),mean_cash=float((cash[c]*mask).sum()/total),mean_margin=float((margin[c]*mask).sum()/total))
    for cid in ORDER:
        result[cid]['paired_differences']={baseline:dict(delta=result[cid]['win_rate']-result[baseline]['win_rate'],
                 ci95=interval(boots[cid]-boots[baseline])) for baseline in ORDER[:2]}
    return result
full=np.ones((256,16,2),dtype=bool)
overall=group(full);by_opponent={};by_seat={};by_opponent_seat={}
for oi,opponent in enumerate(opponents):
    mask=np.zeros_like(full);mask[:,oi,:]=True;by_opponent[opponent]=group(mask);by_opponent_seat[opponent]={}
    for seat in (0,1):
        part=mask.copy();part[:,:,1-seat]=False;by_opponent_seat[opponent][str(seat)]=group(part)
for seat in (0,1):
    mask=full.copy();mask[:,:,1-seat]=False;by_seat[str(seat)]=group(mask)
for cid,r in overall.items():
    assert sum(v[cid]['wins'] for v in by_opponent.values())==r['wins']
    assert sum(v[cid]['wins'] for v in by_seat.values())==r['wins']
report=dict(status='representative_complete_other_panels_pending',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
     scheduled=49152,completed=49152,invalid=0,missing=0,duplicate_cells=0,candidate_labels=LABELS,
     overall=overall,by_opponent=by_opponent,by_seat=by_seat,by_opponent_and_seat=by_opponent_seat,
     diagnostic_parse_errors={cid:int(diagnostics[c].sum()) for c,cid in enumerate(ORDER)},
     roster_sha256=sha(HERE/'roster.json'),representative_manifest_sha256=sha(PANELS/'manifests/representative.json'),
     representative_rows_sha256=hashlib.sha256(raw).hexdigest(),raw_gzip_sha256=sha(OUT/'rows.jsonl.gz'),
     source_prefix_bytes=size,source_prefix_sha256=hashlib.sha256(prefix).hexdigest(),ignored_incomplete_live_tail=ignored_tail,
     report_script_sha256=sha(Path(__file__)),frozen_analysis_sha256=sha(HERE/'analyze_panels.py'),
     bootstrap=dict(key=KEY+':representative',resamples=4000,unit='seed; all 32 opponent-seat cells together',algorithm='NumPy PCG64',index_matrix_sha256=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()),
     qualification='Completed representative panel only. No stress or holdout outcome enters this report; no frozen policy or selection changes. Per-opponent comparisons are exploratory.')
dump(OUT/'RESULTS.json',report)
text='# Representative results\n\nAll 49,152 representative games are complete: 8,192 per candidate, using 256 seeds, 16 opponents and both seats. Zero invalid, missing or duplicate cells. Stress and holdout are still pending.\n\n'
text+='| Candidate | Wins | Draws | Losses | Strict win rate | 95% seed-cluster interval |\n|---|---:|---:|---:|---:|---:|\n'
ranking=sorted(ORDER,key=lambda cid:overall[cid]['wins'],reverse=True)
for cid in ranking:
    r=overall[cid];lo,hi=r['ci95'];text+=f"| {LABELS[cid]} | {r['wins']:,} | {r['draws']:,} | {r['losses']:,} | {r['win_rate']:.2%} | {lo:.2%}–{hi:.2%} |\n"
text+='\nDraws contribute zero wins. Intervals use the frozen 4,000-draw bootstrap and keep all games for a seed together. These are local CPU results under the official 1.32.7 rules, not Kaggle leaderboard scores.\n\n'
text+='[Win rates by opponent](BY_OPPONENT.md) · [Opponent-by-seat tables](BY_OPPONENT_AND_SEAT.md) · [Exact counts, intervals and paired differences](RESULTS.json).\n\n'
text+='The v37 feed-reserve policy uses a public route-based parent; its score does not establish the team’s Three-Layer architecture goal. All six candidates remain frozen for the rest of this comparison. Optional terminal-suffix debug JSON errors are recorded separately; action and terminal checks passed.\n'
(OUT/'RESULTS.md').write_bytes(text.encode())
def opponent_table(byseat=False):
    txt='# Representative win rates by opponent'+(' and candidate seat' if byseat else '')+'\n\nCells show strict win rate and wins/games. Draws count as zero wins. All 256 representative seeds are complete; exact draws, losses and intervals are in RESULTS.json.\n'
    for seat in (0,1) if byseat else (None,):
        txt+='\n'+('## Candidate seat '+str(seat)+'\n\n' if byseat else '')
        txt+='| Opponent | '+' | '.join(LABELS[cid] for cid in ORDER)+' |\n|---|'+'---:|'*6+'\n'
        for o in opponents:
            rs=by_opponent_seat[o][str(seat)] if byseat else by_opponent[o]
            txt+='| '+o+' | '+' | '.join(f"{rs[cid]['win_rate']:.2%} ({rs[cid]['wins']}/{rs[cid]['games']})" for cid in ORDER)+' |\n'
    return txt
(OUT/'BY_OPPONENT.md').write_bytes(opponent_table().encode())
(OUT/'BY_OPPONENT_AND_SEAT.md').write_bytes(opponent_table(True).encode())
print(json.dumps(dict(overall={LABELS[cid]:overall[cid] for cid in ranking},valid_games=49152,diagnostic_parse_errors=report['diagnostic_parse_errors']),indent=2))
