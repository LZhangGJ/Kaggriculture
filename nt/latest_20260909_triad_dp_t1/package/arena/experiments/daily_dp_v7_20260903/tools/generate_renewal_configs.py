from pathlib import Path
import json
EXP=Path(__file__).resolve().parents[1]
base=json.loads((EXP/'s3_review_candidates.json').read_text());out={}
for label,p in base.items():
    for renew in (False,True):out[f'{label}_renew{int(renew)}']=dict(p,renew_ongoing=renew)
(EXP/'s3_renewal_configs.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(len(out))
