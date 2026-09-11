from common300 import *
from selftest100 import parity


def main():
    assert read(SOURCE/'FINAL_AUDIT.json')['status']=='PASS'
    verified=check_source_manifest();configure_torch();pool=runtime.make_pool()
    out=P/'preflight';out.mkdir(exist_ok=False);receipts={}
    for rep in (0,1):
        for arm in ('control','aux'):
            name=f'{arm}_r{rep}';src=SOURCE/'training'/name/'step200.pt'
            m,opt,state=restore(src);assert state['step']==200
            before=optimizer_receipt(opt);assert before['steps']==[11200]
            assert before['momentum_l1']>0 and before['variance_l1']>0
            m.export(out/f'{name}_restored.bin')
            assert digest(out/f'{name}_restored.bin')==digest(src.with_suffix('.bin'))
            r=rollout(pool,src.with_suffix('.bin'),2,70200000+rep,1,70900+rep)
            errors=parity(m,r);store_result(out/name,r)
            twin,opt2,_=restore(src)
            stats=update_aux(m,opt,r,'cuda',201+rep*1000,.1 if arm=='aux'else 0.)
            update_aux(twin,opt2,r,'cuda',201+rep*1000,.1 if arm=='aux'else 0.)
            for k,v in m.state_dict().items():assert torch.equal(v,twin.state_dict()[k]),k
            s1=opt.state_dict();s2=opt2.state_dict()
            assert s1['param_groups']==s2['param_groups']
            for k,state1 in s1['state'].items():
                for field,v in state1.items():assert torch.equal(v,s2['state'][k][field]),field
            after=optimizer_receipt(opt);assert after['steps']==[11204]
            m.export(out/f'{name}_updated.bin')
            assert digest(out/f'{name}_updated.bin')!=digest(src.with_suffix('.bin'))
            receipts[name]=dict(source=str(src),source_sha=digest(src),bin_sha=digest(src.with_suffix('.bin')),
                before=before,after_smoke_update=after,cpp_torch=errors,restored_exact=True,next_update_exact=True,update=stats)
    save(P/'G0_ACCEPTANCE.json',dict(status='PASS',source_manifest_files_verified=verified,models=receipts,
         preflight_games=56,preflight_seeds=[70200000,70200001],native_kernel_recompiled=False))
    print('CONTINUATION_G0_PASS',flush=True)


if __name__=='__main__':main()
