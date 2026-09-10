from pathlib import Path
import json
import run_panel as panel
HERE=Path(__file__).resolve().parent
def main():
    folder=HERE/'startup_supply_screen100/startupsupply2/startupsupply2'
    result=json.loads((folder/'RESULTS.json').read_text())
    assert result['overall']['games']==2200 and result['overall']['errors']==0
    binary=HERE/'candidate_r2p16/policy/startupsupply2.so';receipt=json.loads((HERE/'candidate_r2p16/startupsupply2.BUILD.json').read_text())
    assert panel.sha(binary)==receipt['binary_sha256']=='4d7ef55bb79e4b312445a67628dd15609421e8c9afff21e8192a9b1b055f875b'
    for p,h in receipt['sources'].items():assert panel.sha(binary.parent/p)==h
    assert panel.sha(panel.old.R2/'agent.so')=='877f196113692722edc8a5b3e30d2d3dd1d4700b0772c13f1e3f2ad7a9ac0ef2'
    file=HERE/'DEVELOPMENT_BEST.json';old=json.loads(file.read_text());name='R2P16 P12 + initial rival investment prior'
    if old['name']==name:print('Already preserved; confirmation status not overwritten');return
    assert old['macro_win_rate']<result['overall']['r2_win_rate']
    snapshot=HERE/'candidate_r2p11/DEVELOPMENT_BEST_SUPERSEDED_BY_P16.json'
    if snapshot.exists():assert json.loads(snapshot.read_text())==old
    else:panel.save(snapshot,old)
    full=json.loads((HERE/'candidate_r2p16/FULL_DEVELOPMENT.json').read_text())
    best=dict(name=name,status='BEST_DEVELOPMENT_ONLY_NEW32_PENDING_NOT_90_PERCENT',
        binary=str(binary.relative_to(HERE)),binary_sha256=panel.sha(binary),build_receipt='candidate_r2p16/startupsupply2.BUILD.json',
        config=str((folder/'CONFIG.json').relative_to(HERE)),results=str((folder/'RESULTS.json').relative_to(HERE)),
        games=2200,wins=result['overall']['r2_wins'],baseline_wins=result['base_wins'],
        rescued_losses=result['rescued'],lost_old_wins=result['lost_wins'],
        mean_cash_delta=result['mean_cash_delta'],mean_margin_delta=result['mean_margin_delta'],
        macro_win_rate=result['overall']['r2_win_rate'],development_seeds=[2609110000,2609110099],
        paired_original=full['paired']['original_to_p16'],paired_parent_p12=full['paired']['p12_to_p16'],
        independent_confirmation=dict(status='RUNNING',protocol='startup_supply_validation32/PROTOCOL.json',seeds=[2609124000,2609124031]),
        previous_best_snapshot=str(snapshot.relative_to(HERE)),final_holdout_used=False,goal_remains_active=True,released_original_unchanged=True)
    panel.save(file,best);print(json.dumps(dict(name=name,wins=best['wins'],rate=best['macro_win_rate'])),flush=True)
if __name__=='__main__':main()
