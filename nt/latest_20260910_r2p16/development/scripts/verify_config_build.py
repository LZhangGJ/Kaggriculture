"""Check score-3 trajectories against original GCC14 release with same config."""
from pathlib import Path
import json
import run_panel as panel

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def main():
    target=HERE/'candidate_r2p2/CONFIG3_BUILD_EQUIVALENCE.json'
    assert not target.exists()
    panel.init_worker()
    pool={x['id']:x for x in json.loads((HERE/'pool_all11.json').read_text())}
    rows=json.loads((HERE/'competitive3_100/externality3/rows.json').read_text())
    by={(r['opponent'],r['seed'],r['opponent_seat']):r for r in rows}
    config=json.loads((panel.old.R2/'config.json').read_text());config['competition']=3
    fixtures=[('thomas_955_v2',2609110028,0),('aurax_reactive_v1',2609110014,0),
              ('shop0909',2609110042,1),('nagatakengo_v70',2609110085,1)]
    checks=[]
    for name,seed,seat in fixtures:
        candidate=by[(name,seed,seat)]
        original=panel.game((name,str(ROOT/pool[name]['working']),seed,seat,str(HERE/'candidate_r2p2/original_config3_check'),config,None))
        assert not original['runtime_error'],original
        same=all(original[k]==candidate[k] for k in ['joint_action_sha256','r2_cash','opponent_cash','steps'])
        checks.append(dict(opponent=name,seed=seed,opponent_seat=seat,pass_=same,
                           original=original,candidate=candidate))
        assert same,checks[-1]
    panel.save(target,dict(status='PASS',fixtures=len(checks),actions=len(checks)*719,checks=checks,
                           source_binary_sha256=panel.sha(panel.old.R2/'agent.so'),
                           candidate_binary_sha256=panel.sha(HERE/'candidate_r2p2/policy/r2p2.so')))
    print(json.dumps(dict(status='PASS',fixtures=len(checks),actions=len(checks)*719)),flush=True)


if __name__=='__main__':main()
