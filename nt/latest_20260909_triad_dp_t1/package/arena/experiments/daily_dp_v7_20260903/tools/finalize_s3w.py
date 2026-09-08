"""Record the no-promotion decision without discarding variants or evidence."""
from pathlib import Path
import hashlib,json
EXP=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt=json.loads((EXP/'receipts/s3w_execution_v1/acceptance.json').read_text())
    assert receipt['status']=='COMPLETE_DEVELOPMENT_NOT_GOAL_ACCEPTANCE'
    effects=json.loads((EXP/'receipts/s3w_factorial_effects_v1/summary.json').read_text())
    build=json.loads((EXP/'native/build/build_receipt.json').read_text());assert receipt['build']==build
    for rel,h in build['source_hashes'].items():assert sha(EXP/rel)==h
    freeze=json.loads((EXP/'profiles/s3w/freeze.json').read_text())
    for rel,h in freeze['source_hashes'].items():assert sha(EXP/rel)==h
    cfg=EXP/'profiles/s3w/configs.json';assert sha(cfg)==freeze['config_sha256']
    assert sha(next((EXP/'native/build').glob('_dp7_native*.so')))==build['binary_sha256']
    summary=json.loads((EXP/'receipts/s3w_ABM_summary_v1/summary.json').read_text())
    # Read gate contradictions explicitly; no surrogate metrics mark success.
    for label,opponents in summary['aggregate'].items():
        assert not (opponents['pass']['cash']>=180000 and all(v['win_rate']>=.9 for k,v in opponents.items() if k!='pass'))
    out=EXP/'profiles/s3w/recommended_development.json';assert not out.exists()
    out.write_text(json.dumps(dict(status='NO_PROMOTION_FOR_STRENGTH_TARGET',primary_label='old',
        source_configs='profiles/s3w/configs.json',config_sha256=sha(cfg),binary_sha256=build['binary_sha256'],
        retained_labels=['calendar','turnover','calendar_turnover'],final_acceptance=False,python_submission_accepted=False,
        final_holdout_used=False,reason='Capital-time ranking reduces broad strength. Local positive interaction does not satisfy PASS plus all opponents. Keep every variant; do not route by identity.',
        report='reports/S3W_INVESTMENT_TIMELINE_REVIEW_ZH.md'),indent=2),encoding='utf8')
    print(json.dumps(dict(status='S3W_CLOSED_NO_PROMOTION',binary_sha256=build['binary_sha256'],full_goal_complete=False)))
if __name__=='__main__':main()
