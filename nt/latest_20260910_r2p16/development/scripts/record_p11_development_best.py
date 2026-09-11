from pathlib import Path
import json
import run_panel as panel
from validate_crop_clock import paired
HERE=Path(__file__).resolve().parent

def main():
    folder=HERE/'crop_clock_screen100/cropclock1_sale1/cropclock1_sale1'
    result=json.loads((folder/'RESULTS.json').read_text());assert result['overall']['games']==2200 and result['overall']['errors']==0
    binary=HERE/'candidate_r2p11/policy/cropclock1_sale1.so';assert panel.sha(binary)=='6ce9e670163a7f6fcea7f449a152d70f758d0f0984eb60c22e7e81446d5610d1'
    path=HERE/'DEVELOPMENT_BEST.json';old=json.loads(path.read_text())
    if old['macro_win_rate']>result['overall']['r2_win_rate']:print('A stronger candidate already exists; not demoting.');return
    previous=HERE/'candidate_r2p9/DEVELOPMENT_BEST_SUPERSEDED_BY_P11.json'
    if old['name']!='R2P11 observed crop clock + P9 split sale':
        assert not previous.exists();panel.save(previous,old)
    rows=json.loads((folder/'rows.json').read_text());base=json.loads((HERE/'baseline_rows_all11.json').read_text());p9=json.loads((HERE/'local_sale_screen100/local_sale_1/local_sale_1/rows.json').read_text())
    stats,_=paired(base,rows);incremental,_=paired(p9,rows)
    best=dict(name='R2P11 observed crop clock + P9 split sale',status='BEST_DEVELOPMENT_ONLY_INDEPENDENT_CONFIRMATION_PENDING_NOT_90_PERCENT',binary=str(binary.relative_to(HERE)),binary_sha256=panel.sha(binary),build_receipt='candidate_r2p11/cropclock1_sale1.BUILD.json',config=str((folder/'CONFIG.json').relative_to(HERE)),protocol='crop_clock_screen100/cropclock1_sale1/PROTOCOL.json',results=str((folder/'RESULTS.json').relative_to(HERE)),games=2200,wins=result['overall']['r2_wins'],baseline_wins=result['base_wins'],rescued_losses=result['rescued'],lost_old_wins=result['lost_wins'],mean_cash_delta=result['mean_cash_delta'],mean_margin_delta=result['mean_margin_delta'],macro_win_rate=result['overall']['r2_win_rate'],development_seeds=[2609110000,2609110099],paired_original=stats,paired_previous_p9=incremental,independent_confirmation=dict(status='PENDING',plan='candidate_r2p11/INDEPENDENT_CONFIRMATION_PLAN_ZH.md',seeds=[2609123000,2609123031]),previous_best_snapshot=str(previous.relative_to(HERE)),final_holdout_used=False,goal_remains_active=True,released_original_unchanged=True)
    panel.save(path,best);print(json.dumps(dict(wins=best['wins'],rate=best['macro_win_rate'],paired_original=stats,paired_p9=incremental)),flush=True)
if __name__=='__main__':main()
