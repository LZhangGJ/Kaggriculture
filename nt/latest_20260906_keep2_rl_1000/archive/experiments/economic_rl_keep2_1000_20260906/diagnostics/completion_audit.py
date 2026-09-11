"""Goal-level evidence closure. Run only after all seven stages and holdout finish."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf8'))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def verify_manifest(directory):
    m=read(directory/'MANIFEST.json')
    for relative,entry in m['files'].items():
        p=directory/relative
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],str(p)
    assert len(m['files'])==m['count']
    return m['count']

def main():
    state=read(ROOT/'CURRENT_STATUS.json')
    assert state['status']=='COMPLETE' and state['end_round']==1000
    assert read(ROOT/'diagnostics'/'SCHEDULE_ACCEPTANCE.json')['status']=='PASS'
    milestones=read(ROOT/'MILESTONES.json')
    assert [x['round']for x in milestones]==list(range(400,1001,100))
    models=('control_r0','control_r1','aux_r0','aux_r1');stages={};total=0
    expected_library=None
    for end in range(400,1001,100):
        p=ROOT/'stages'/f'round{end:04}';start=end-100
        files=verify_manifest(p);a=read(p/'FINAL_AUDIT.json');g0=read(p/'G0_ACCEPTANCE.json')
        assert a['status']==g0['status']=='PASS'
        assert a['new_training_games']==89600 and a['total_training_games']==end*896
        assert a['final_games']==29400 and a['optimizer_continued_exactly']
        assert a['training_dev_monitor_holdout_disjoint'] and a['selection_before_final'] and a['old_files_unchanged']
        cfg=read(p/'PROTOCOL.json')
        assert cfg['start_round']==start and cfg['end_round']==end and cfg['keep_bonus']==2
        assert cfg['games_per_round']==224 and cfg['repeats']==2
        assert cfg['development_seed_start']==69800000 and cfg['development_seeds']==32
        assert cfg['final_seed_start']==70300000 and cfg['final_seeds']==100
        for name in models:
            receipt=read(p/'training'/name/'COMPLETE.json')
            assert receipt['status']=='PASS' and receipt['total_training_games']==end*224
            assert [x['step']for x in receipt['history']]==list(range(start+1,end+1))
            assert receipt['final_optimizer']['steps']==[end*56]
            assert receipt['keep_bonus']==2 and receipt['aux_weight']==(.1 if name.startswith('aux')else 0.)
            assert g0['models'][name]['restored_exact'] and g0['models'][name]['next_update_exact']
            assert g0['models'][name]['before']['steps']==[start*56]
            assert set(a['cpp_torch_onpolicy'][name])=={str(start+1),str(start+50),str(end)}
            assert all(max(v.values())<2e-5 for v in a['cpp_torch_onpolicy'][name].values())
            selection=read(p/'training'/name/'selection.json')
            assert len(selection)==end//20+1 and max(selection,key=lambda x:tuple(x['score']))==receipt['best']
        for d in (p/'final').iterdir():
            prov=read(d/'provenance.json')
            expected_library=expected_library or prov['library_sha']
            assert prov['library_sha']==expected_library and prov['sample']==9921
        for f in ('SUMMARY_ZH.md','ACCEPTANCE_ZH.md','FINAL_RESULTS.json','FINAL_SELECTION_FROZEN.json'):
            assert (p/f).is_file() and (p/f).stat().st_size>0
        stages[str(end)]=dict(status='PASS',manifest_files=files,new_training_games=89600,monitor_games=29400)
        total+=a['new_training_games'];print('STAGE_EVIDENCE_VERIFIED',end,flush=True)
    assert total==627200
    h=ROOT/'independent1000';hc=read(h/'ACCEPTANCE.json');hfiles=verify_manifest(h)
    assert hc['status']=='PASS' and hc['games']==40600 and hc['configs']==29
    assert hc['seed_start']==71100000 and hc['seeds']==100
    assert hc['selection_before_holdout'] and hc['holdout_not_used_for_training_or_selection']
    assert sum(x['games']for x in hc['checks'].values())==40600
    independent=read(h/'SELECTION_FROZEN.json')
    assert independent['seed_start']==71100000 and independent['sample']==9931
    for model,digest in independent['hashes'].items():assert sha(Path(model))==digest
    for relative,entry in read(ROOT/'MANIFEST_INDEX.json')['files'].items():
        p=ROOT/relative;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
    assert (ROOT/'FINAL_SUMMARY_ZH.md').stat().st_size>0
    old={}
    for end in (100,200,300):
        p=ROOT.parent/f'economic_rl_keep2_{end}_20260906';old[str(end)]=verify_manifest(p)
        print('ORIGINAL_ARTIFACTS_PRESERVED',end,flush=True)
    result=dict(status='PASS',new_training_games=627200,cumulative_training_games=896000,monitor_games=205800,
        independent_games=40600,independent_manifest_files=hfiles,stages=stages,original_artifacts_preserved=old,
        note='Computer-side completion only. The assistant must additionally verify delivery of all seven milestone reports in the conversation before completing the goal.')
    (ROOT/'diagnostics'/'COMPLETION_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('GOAL_EVIDENCE_PASS',flush=True)

if __name__=='__main__':main()
