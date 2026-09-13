"""Validate every scheduled cell and publish matched seed-cluster statistics."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
KEY='kaggriculture-seed-contract-v1-bootstrap'
LABELS={'ppo-495ebc48210f14ae':'AFS R2','ppo-c90da6b7b3ea061d':'Day9',
        'ppo-bf0aa03b41428d97':'Day3 v1','ppo-9980111aa0f1e7f6':'Day3 v2',
        'team-terminal-suffix-v1':'Terminal suffix','team-v37-feed-reserve-v1':'v37 feed reserve'}
LABELS.update({'pro8-a08-r11': 'A08 r11', 'pro8-a06-r6': 'A06 r6', 'pro8-a06-r12': 'A06 r12 calendar'})
ORDER=list(LABELS)
PRODUCTS=('WHEAT','CARROT','TOMATO','STRAWBERRY','MELON','EGG','MILK','WOOL','FERTILIZER')
SHOPS={'BAKERY':('EGG','WHEAT'),'PIZZA_SHOP':('MILK','TOMATO','WHEAT'),
 'BRUNCH_SPOT':('EGG','WHEAT','STRAWBERRY'),'YARN_STORE':('WOOL',),
 'ICE_CREAM_SHOP':('STRAWBERRY','MILK','WHEAT'),'PET_CAFE':('CARROT',),
 'SMOOTHIE_SHOP':('STRAWBERRY','MILK'),'FARMERS_MARKET':('WHEAT','CARROT','TOMATO','STRAWBERRY')}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while block:=f.read(4*1024**2):h.update(block)
    return h.hexdigest()
def dump(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf8')
def interval(values):
    valid=values[np.isfinite(values)]
    return list(map(float,np.quantile(valid,[.025,.975],method='linear'))) if valid.size else None
def pct(x):return f'{100*x:.2f}%'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--panels',type=Path,required=True)
    p.add_argument('--roster',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True)
    p.add_argument('--holdout',type=Path)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    roster=json.loads(args.roster.read_bytes());roster_hash=sha(args.roster)
    assert set(ORDER)=={r['id'] for r in roster['candidates']}
    threshold=json.loads((args.roster.parent/'condition_thresholds.json').read_bytes())['thresholds']
    assert sha(args.roster.parent/'condition_thresholds.json')==roster['condition_thresholds_sha256']
    opponents=[r['id'] for r in json.loads((args.panels/'opponents.json').read_bytes())['opponents']]
    names=['representative','stress']+(['holdout'] if args.holdout else [])
    datasets={};receipts={}
    for name in names:
        path=args.panels/('sealed' if name=='holdout' else 'manifests')/(name+'.json')
        seeds=json.loads(path.read_bytes())['seeds'];n=len(seeds);shape=(len(ORDER),n,16,2)
        datasets[name]=dict(seeds=seeds,seedindex={s:i for i,s in enumerate(seeds)},wins=np.zeros(shape,dtype=np.int8),
             ties=np.zeros(shape,dtype=np.int8),margin=np.zeros(shape),cash=np.zeros(shape),seen=np.zeros(shape,dtype=bool),
             floor=np.zeros(shape+(9,),dtype=np.int16),inventory=np.zeros(shape+(9,)),demand=np.zeros(shape+(9,)),
             diagnostics=np.zeros(shape,dtype=bool),latency=np.zeros(shape),seconds=np.zeros(shape))
    for run in [args.development]+([args.holdout] if args.holdout else []):
        status=json.loads((run/'STATUS.json').read_bytes());manifest=json.loads((run/'MANIFEST.json').read_bytes())
        assert status['complete'] and status['invalid']==0 and status['missing']==0
        assert status['roster_sha256']==roster_hash and manifest['roster_sha256']==roster_hash
        receipts[str(run.name)]=dict(rows_sha256=sha(run/'rows.jsonl'),manifest=manifest,status=status)
        with (run/'rows.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line);d=datasets[row['panel']]
                assert row['valid'] and row['steps']==719 and not row.get('runtime_error')
                assert row['candidate_seat']==1-row['opponent_seat']
                at=(ORDER.index(row['candidate_id']),d['seedindex'][row['seed']],opponents.index(row['opponent']),row['candidate_seat'])
                assert not d['seen'][at],f'Duplicate cell: {at}'
                d['seen'][at]=True
                assert row['win']==(row['own_cash']>row['opponent_cash'])
                assert row['tie']==(row['own_cash']==row['opponent_cash'])
                assert row['margin']==row['own_cash']-row['opponent_cash']
                for field,source in [('wins','win'),('ties','tie'),('margin','margin'),('cash','own_cash'),('latency','latency_max'),('seconds','seconds')]:d[field][at]=row[source]
                d['diagnostics'][at]=bool(row.get('diagnostic_error'))
                economy=row['economy'];assert economy['samples']==180
                assert [u['step'] for u in economy['unlock_events']]==[72,144,216,288,360,432,504,576]
                assert [u['shop'] for u in economy['unlock_events']]==row['terminal_shops']
                for i,item in enumerate(PRODUCTS):
                    m=economy['market'][item];assert 0<=m['floor_samples']<=180
                    d['floor'][at+(i,)]=m['floor_samples'];d['inventory'][at+(i,)]=m['inventory_sum']/180
                    d['demand'][at+(i,)]=sum((180-u['step']//4)*(2 if len(SHOPS[u['shop']])==1 else 1) for u in economy['unlock_events'] if item in SHOPS[u['shop']])
    all_results={};condition_results={};actual_results={}
    for name,d in datasets.items():
        assert d['seen'].all(),f'Missing cells in {name}'
        n=len(d['seeds']);rng=np.random.default_rng(int.from_bytes(hashlib.sha256((KEY+':'+name).encode()).digest(),'big'))
        draws=rng.integers(0,n,size=(4000,n))
        # A common matrix of seed multiplicities preserves matched resampling.
        counts=np.zeros((4000,n),dtype=np.int16)
        np.add.at(counts,(np.repeat(np.arange(4000),n),draws.ravel()),1)
        draws_hash=hashlib.sha256(draws.astype('<i8').tobytes()).hexdigest()
        def summarize(mask,paired=True):
            # mask has (seed, opponent, candidate seat); a shared reference group.
            denominator=mask.sum(axis=(1,2));total=int(denominator.sum());support=int((denominator>0).sum())
            if total==0:return dict(games=0,distinct_seeds=0,candidates={})
            bootstrap_den=counts@denominator
            by_candidate={};bootstrap_rates={}
            for c,cid in enumerate(ORDER):
                seed_wins=(d['wins'][c]*mask).sum(axis=(1,2));wins=int(seed_wins.sum());ties=int((d['ties'][c]*mask).sum())
                with np.errstate(divide='ignore',invalid='ignore'):rates=(counts@seed_wins)/bootstrap_den
                bootstrap_rates[cid]=rates
                by_candidate[cid]=dict(games=total,wins=wins,draws=ties,losses=total-wins-ties,win_rate=wins/total,
                   win_rate_ci95=interval(rates),mean_cash=float((d['cash'][c]*mask).sum()/total),
                   mean_margin=float((d['margin'][c]*mask).sum()/total))
            if paired:
                for cid in ORDER:
                    by_candidate[cid]['paired_differences']={}
                    for baseline in ORDER[:2]:
                        by_candidate[cid]['paired_differences'][baseline]=dict(
                          win_rate_difference=by_candidate[cid]['win_rate']-by_candidate[baseline]['win_rate'],
                          ci95=interval(bootstrap_rates[cid]-bootstrap_rates[baseline]))
            return dict(games=total,distinct_seeds=support,small_seed_group=support<20,candidates=by_candidate)
        full=np.ones((n,16,2),dtype=bool)
        result=dict(overall=summarize(full),by_opponent={},by_seat={},by_opponent_and_seat={},
                    bootstrap=dict(resamples=4000,algorithm='numpy PCG64',key=KEY+':'+name,index_matrix_sha256=draws_hash),
                    diagnostics={})
        for i,opponent in enumerate(opponents):
            mask=full.copy();mask[:]=False;mask[:,i,:]=True
            result['by_opponent'][opponent]=summarize(mask)
            result['by_opponent_and_seat'][opponent]={}
            for seat in (0,1):
                seatmask=mask.copy();seatmask[:,:,1-seat]=False
                result['by_opponent_and_seat'][opponent][str(seat)]=summarize(seatmask)
        for seat in (0,1):
            mask=full.copy();mask[:,:,1-seat]=False;result['by_seat'][str(seat)]=summarize(mask)
        for c,cid in enumerate(ORDER):
            result['diagnostics'][cid]=dict(optional_debug_parse_errors=int(d['diagnostics'][c].sum()),
                maximum_measured_policy_latency_seconds=float(d['latency'][c].max()),
                mean_game_seconds=float(d['seconds'][c].mean()))
        all_results[name]=result
        feature_path=args.panels/'features'/(name+'.jsonl')
        if name=='holdout' and not feature_path.exists():
            sys.path.insert(0,str(args.panels/'tools'));from economy_features import characterize
            feature_rows=[characterize(s) for s in d['seeds']]
            # Holdout was already released and evaluated; this is post-run analysis.
            target=args.out/'holdout_features.jsonl'
            target.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in feature_rows),encoding='utf8')
        else:feature_rows=[json.loads(line) for line in feature_path.read_text().splitlines()]
        feature_map={r['seed']:r for r in feature_rows};feature_rows=[feature_map[s] for s in d['seeds']]
        masks={}
        for shop in sorted(SHOPS):masks['reference_first_shop='+shop]=np.array([r['reference_first_shop']==shop for r in feature_rows])
        for field in ('ref_milk_minus_wool','ref_crop_minus_animal'):
            x=np.array([r['features'][field] for r in feature_rows])
            masks[field+'=negative']=x<0;masks[field+'=zero']=x==0;masks[field+'=positive']=x>0
        for field,cut in [('ref_shop_variety',4),('ref_max_duplicate',2)]:
            x=np.array([r['features'][field] for r in feature_rows])
            masks[field+'<='+str(cut)]=x<=cut;masks[field+'>'+str(cut)]=x>cut
        for field,t in threshold.items():
            x=np.array([r['features'][field] for r in feature_rows]);low=x<=t['q25'];high=(x>=t['q75'])&~low
            masks[field+'=low']=low;masks[field+'=middle']=~(low|high);masks[field+'=high']=high
        condition_results[name]={group:summarize(np.broadcast_to(mask[:,None,None],(n,16,2))) for group,mask in masks.items()}
        actual_results[name]={}
        for c,cid in enumerate(ORDER):
            actual_results[name][cid]={}
            for item_index,item in enumerate(PRODUCTS):
                floor=d['floor'][c,...,item_index];stock=d['inventory'][c,...,item_index];demand=d['demand'][c,...,item_index]
                groups={}
                for label,mask in [('zero',floor==0),('below_one_quarter',(floor>0)&(floor<45)),('at_least_one_quarter',floor>=45)]:
                    total=int(mask.sum());support=int(mask.any(axis=(1,2)).sum())
                    den=mask.sum(axis=(1,2));wins=(d['wins'][c]*mask).sum(axis=(1,2))
                    with np.errstate(divide='ignore',invalid='ignore'):rates=(counts@wins)/(counts@den)
                    groups[label]=dict(games=total,distinct_seeds=support,wins=int(wins.sum()),
                        win_rate=float(wins.sum()/total) if total else None,ci95=interval(rates),
                        causal_comparison=False)
                actual_results[name][cid][item]=dict(mean_sampled_floor_fraction=float(floor.mean()/180),
                    mean_sampled_market_inventory=float(stock.mean()),mean_shop_demand_units=float(demand.mean()),
                    floor_exposure_groups=groups)
    result=dict(schema_version=1,created_unix=time.time(),roster_sha256=roster_hash,
       analysis_sha256=sha(Path(__file__)),numpy_version=np.__version__,python=sys.version,
       candidate_labels=LABELS,panels=all_results,run_receipts=receipts,
       status='complete_all_three_panels' if len(names)==3 else 'complete_development_holdout_pending')
    dump(args.out/'RESULTS.json',result);dump(args.out/'REFERENCE_CONDITIONS.json',condition_results)
    dump(args.out/'REALIZED_MARKETS.json',actual_results)
    title='# Candidate results\n\n'
    if len(names)!=3:title+='Development panels are complete. Holdout results are pending.\n\n'
    title+='Strict win rates count draws as zero wins. Each seed uses all 16 pinned opponents in both seats. Rates below are local CPU results under the official 1.32.7 rules.\n\n'
    title+='| Candidate | '+' | '.join(name.title() for name in names)+' |\n|---|'+'---:|'*len(names)+'\n'
    for cid in ORDER:
        cells=[]
        for name in names:
            r=all_results[name]['overall']['candidates'][cid];lo,hi=r['win_rate_ci95']
            cells.append(f"{pct(r['win_rate'])} ({r['wins']:,}/{r['games']:,}); 95% CI {pct(lo)}–{pct(hi)}")
        title+='| '+LABELS[cid]+' | '+' | '.join(cells)+' |\n'
    title+='\n[Win rates by opponent](BY_OPPONENT.md) · [Opponent and seat breakdown](BY_OPPONENT_AND_SEAT.md) · [Exact counts, intervals and paired differences](RESULTS.json).\n\n'
    title+='Representative performance estimates the eligible seed distribution. Stress results describe selected extremes. The holdout provides one frozen comparison and is retired after use. All uncertainty intervals resample whole seeds. Repeated comparisons are exploratory; these results do not authorize a champion change or a Kaggle submission.\n\n'
    title+='[Reference-condition results](REFERENCE_CONDITIONS.json) retain the PASS/PASS controller qualification. [Realized market results](REALIZED_MARKETS.json) summarize action-dependent shops and market prices sampled every four turns; their groups are descriptive.\n'
    (args.out/'RESULTS.md').write_text(title,encoding='utf8')
    def tables(byseat=False):
        text='# Win rates by opponent'+(' and candidate seat' if byseat else '')+'\n\nEach cell shows strict win rate and wins/games. Exact draws, losses and seed-cluster intervals appear in RESULTS.json.\n'
        for name in names:
            for seat in (0,1) if byseat else (None,):
                text+='\n## '+name.title()+(' — candidate seat '+str(seat) if byseat else '')+'\n\n'
                text+='| Opponent | '+' | '.join(LABELS[cid] for cid in ORDER)+' |\n|---|'+'---:|'*len(ORDER)+'\n'
                for opponent in opponents:
                    group=all_results[name]['by_opponent_and_seat'][opponent][str(seat)] if byseat else all_results[name]['by_opponent'][opponent]
                    cells=[]
                    for cid in ORDER:
                        r=group['candidates'][cid];cells.append(f"{pct(r['win_rate'])} ({r['wins']}/{r['games']})")
                    text+='| '+opponent+' | '+' | '.join(cells)+' |\n'
        return text
    (args.out/'BY_OPPONENT.md').write_text(tables(),encoding='utf8')
    (args.out/'BY_OPPONENT_AND_SEAT.md').write_text(tables(True),encoding='utf8')
    print(json.dumps(dict(status=result['status'],games=sum(int(d['seen'].sum()) for d in datasets.values()),
       panels={name:{LABELS[cid]:r['win_rate'] for cid,r in result['panels'][name]['overall']['candidates'].items()} for name in names})))

if __name__=='__main__':main()
