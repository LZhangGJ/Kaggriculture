"""Read-only audit of frozen sources, full-game records and on-policy datasets."""
from runtime import *

def main():
    assert read(P/'FINAL_RESULTS.json')['status']=='COMPLETE'
    original=read(P/'SOURCE_FREEZE.json')
    altered=[f for f,h in original.items() if not Path(f).is_file() or digest(f)!=h]
    assert not altered, altered
    for gate in ('G1_RECEIPT.json','G2_RECEIPT.json'):
        receipt=read(P/gate);assert receipt['status']=='PASS'
        for f,h in receipt.get('hashes',{}).items():assert digest(f)==h, f
    totals=dict(training_games=0,training_decisions=0,effective_actor_records=0,evaluation_games=0)
    rows=[]
    for rep in (0,1):
        for arm in ('c3auto','f3','c3j7'):
            folder=P/'training'/f'{arm}_r{rep}'
            protocol=read(folder/'protocol.json')
            assert all(digest(f)==h for f,h in protocol['hashes'].items())
            selections=read(folder/'selection.json')
            selected=max(selections,key=lambda x:tuple(x['score']))
            assert selected['step']==read(folder/'FINAL.json')['best_step']
            for step in range(1,21):
                root=folder/f'rollout_{step:03}'
                games=read(root/'games.json');s=read(root/'summary.json')
                expected={(seed,NAMES[opp],seat) for seed,seat,opp in jobs(61000000+rep*100000+(step-1)*16,16)}
                actual={(r['seed'],r['opponent'],r['seat']) for r in games}
                assert len(games)==len(actual)==224 and actual==expected
                assert s['mode']==2 and s['status']=='PASS'
                assert all(r['steps']==719 and not r['error'] for r in games)
                if arm!='c3j7':assert all(r['reference_calls']==0 for r in games)
                with np.load(root/'decisions.npz') as d:
                    n=len(d['choice']);assert n==224*30
                    for k in ('global','candidates','probability','logp','value','reward'):
                        assert np.isfinite(d[k]).all(), (arm,rep,step,k)
                    mask=d['mask'].astype(bool);choice=d['choice'].astype(int)
                    assert mask[np.arange(n),choice].all()
                    assert np.allclose(d['probability'].sum(1),1,atol=2e-6)
                    assert (d['probability'][~mask]==0).all()
                    assert np.allclose(np.log(d['probability'][np.arange(n),choice]),d['logp'],atol=2e-6)
                    for idx,g in enumerate(games):
                        positions=np.flatnonzero(d['game']==idx)
                        assert len(positions)==30 and np.all(np.diff(positions)==1)
                        assert np.array_equal(d['day'][positions],np.arange(30))
                        rewards=d['reward'][positions]
                        assert (rewards[:-1]==0).all()
                        expected_reward=np.sign(g['margin'])+.1*np.tanh(g['margin']/50000.)
                        assert np.isclose(rewards[-1],expected_reward,atol=2e-6)
                    totals['training_decisions']+=n
                    totals['effective_actor_records']+=int((mask.sum(1)>1).sum())
                totals['training_games']+=len(games)
            for evaluation in folder.glob('*/games.json'):
                name=evaluation.parent.name
                if name.startswith('rollout_'):continue
                games=read(evaluation)
                start,count=(62000000,16) if name.startswith('dev_') else (63000000,100)
                expected={(seed,NAMES[opp],seat) for seed,seat,opp in jobs(start,count)}
                actual={(r['seed'],r['opponent'],r['seat'])for r in games}
                assert len(games)==len(actual)==count*14 and actual==expected
                assert all(r['steps']==719 and not r['error'] for r in games)
                totals['evaluation_games']+=len(games)
            rows.append(dict(arm=arm,run=rep,selected_step=selected['step'],status='PASS'))
    for f in (P/'baselines').glob('*/games.json'):
        games=read(f);assert len(games)==1400
        assert all(r['steps']==719 and not r['error'] for r in games)
        totals['evaluation_games']+=len(games)
    assert totals['training_games']==26880 and totals['training_decisions']==806400
    assert totals['evaluation_games']==read(P/'FINAL_RESULTS.json')['evaluation_games']
    save(P/'FINAL_AUDIT.json',dict(status='PASS',original_files_unchanged=len(original),totals=totals,runs=rows))
    print(read(P/'FINAL_AUDIT.json'))

if __name__=='__main__':main()
