"""Postprocess existing C++ receipts and retain selected configurations."""
from pathlib import Path
import hashlib,json,random,shutil,statistics
EXP=Path(__file__).resolve().parents[1]
review=json.loads((EXP/'receipts/s3_review_new50/results.json').read_text())
renew=json.loads((EXP/'receipts/s3_renewal_new50/results.json').read_text())
configs=json.loads((EXP/'s3_review_candidates.json').read_text())
out=EXP/'profiles';out.mkdir(exist_ok=False)
(out/'candidates.json').write_text(json.dumps(configs,indent=2),encoding='utf8')
bootstrap=random.Random(391403);summary=[]
for s in review['summaries']:
    label=s['label'];rows=[r for r in review['rows'] if r['label']==label and r['opponent']=='g001']
    expected={ (r['seed'],r['seat']):(r['cash'],r['opponent_cash'],r['win']) for r in rows}
    current={ (r['seed'],r['seat']):(r['cash'],r['opponent_cash'],r['win']) for r in renew['rows'] if r['label']==label+'_renew0' and r['opponent']=='g001'}
    assert expected==current,(label,'default-feature regression')
    clusters=[]
    for seed in sorted({r['seed'] for r in rows}):clusters.append(statistics.fmean(r['win'] for r in rows if r['seed']==seed))
    draws=sorted(statistics.fmean(bootstrap.choices(clusters,k=len(clusters))) for _ in range(10000))
    summary.append(dict(label=label,pass_stats=s['pass'],g001_stats=s['g001'],seed_cluster_bootstrap95=[draws[250],draws[9749]],current_binary_renew_off_cash_regression=True))
    (out/(label+'.json')).write_text(json.dumps(configs[label],indent=2),encoding='utf8')
native=EXP/'native';build=json.loads((native/'build/build_receipt.json').read_text())
for rel,h in build['source_hashes'].items():
    p=EXP/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
receipt=dict(stage='S3 development, NOT final acceptance',preferred_candidate='S3C03',original_development_seeds=[20261101,20261110],expanded_development_seeds=[20261401,20261450],formal_holdouts_used=False,build=build,summary=summary)
(out/'stage_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt['summary'],indent=2))
