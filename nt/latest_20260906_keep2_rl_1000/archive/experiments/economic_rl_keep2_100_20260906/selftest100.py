from run100_common import *
from bias2_model import Bias2Model,update_aux,outcomes
import torch

def parity(model,r):
    with torch.no_grad():
        d,v=model(torch.as_tensor(r['global'],device='cuda'),torch.as_tensor(r['candidates'],device='cuda'),torch.as_tensor(r['mask'],device='cuda'))
        lp=d.log_prob(torch.as_tensor(r['choice'],device='cuda').long())
    e=dict(prob=float(np.max(abs(d.probs.cpu().numpy()-r['probability']))),value=float(np.max(abs(v.cpu().numpy()-r['value']))),
           logp=float(np.max(abs(lp.cpu().numpy()-r['logp']))),ratio=float(np.max(abs(np.exp(lp.cpu().numpy()-r['logp'])-1))))
    assert max(e.values())<2e-5,e
    return e

def main():
    assert read(P/'BUILD_RECEIPT.json')['status']=='PASS';configure_torch()
    out=P/'checks';out.mkdir(exist_ok=False);torch.manual_seed(701);m=Bias2Model().cuda();m.export(out/'initial.bin')
    assert digest(out/'initial.bin')==digest(OLD/'training/f3_r0/step000.bin')
    pool=runtime.make_pool()
    # Validate complete greedy behavior against the independently tested bias=2 wrapper.
    checkpoint=AUX/'training/aux_r0/step020.bin'
    previous=rollout(pool,checkpoint,101,70010000,1,trace=True,prior=True)
    current=rollout(pool,checkpoint,1,70010000,1,trace=True)
    assert previous['traces']==current['traces']
    for k in ('global','candidates','mask','choice','probability','reward','value'):assert np.array_equal(previous[k],current[k]),k
    r=rollout(pool,out/'initial.bin',2,70010000,1,911,trace=True)
    r1=rollout(pool,out/'initial.bin',2,70010000,1,911,threads=1,trace=True)
    assert r['traces']==r1['traces'];before=parity(m,r);store_result(out/'before',r)
    count=r['mask'].sum(1);expected=np.exp(2)/(np.exp(2)+count-1)
    assert np.max(abs(expected-r['probability'][:,0]))<1e-6
    y,mask=outcomes(r)
    for hi,h in enumerate((1,3,7)):assert np.array_equal(mask[:,hi,0],r['day']+h<=29)
    stats=[]
    for weight in (0.,.1):
        trial=Bias2Model().cuda();trial.load_state_dict(m.state_dict());opt=torch.optim.Adam(trial.parameters(),lr=3e-4)
        metric=update_aux(trial,opt,r,'cuda',701,weight)
        trial.export(out/f'update_{weight}.bin')
        assert digest(out/f'update_{weight}.bin')!=digest(out/'initial.bin')
        after=rollout(pool,out/f'update_{weight}.bin',2,70010000,1,911)
        errs=parity(trial,after)
        # Save and restore optimizer too; the next update must be bit-identical.
        torch.save(dict(model=trial.state_dict(),optimizer=opt.state_dict()),out/f'optimizer_{weight}.pt')
        resumed=Bias2Model().cuda();opt2=torch.optim.Adam(resumed.parameters(),lr=3e-4)
        saved=torch.load(out/f'optimizer_{weight}.pt',weights_only=False)
        resumed.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer'])
        update_aux(trial,opt,after,'cuda',702,weight);update_aux(resumed,opt2,after,'cuda',702,weight)
        for k,v in trial.state_dict().items():assert torch.equal(v,resumed.state_dict()[k]),k
        stats.append(dict(aux_weight=weight,update=metric,after_parity=errs,optimizer_resume_exact=True))
    receipt=dict(status='PASS',initial_weight_identical=True,old_bias2_greedy_full_trace_identical=True,
                 sampled_threads_1_16_full_trace_identical=True,initial_cpp_torch=before,updates=stats,
                 keep2_cpp_and_torch=True,initial_keep_probability_mean=float(r['probability'][count>1,0].mean()),
                 initial_nonkeep=int((r['choice']!=0).sum()),games=84)
    save(P/'G0_ACCEPTANCE.json',receipt)
    (P/'G0_ACCEPTANCE_ZH.md').write_text('# KEEP=2训练工程验收\n\n通过：初始权重一致；训练与推理两边的KEEP均为2；C++/Torch probability、value、logp和更新前ratio一致；随机采样1/16线程动作一致；与此前推理bonus2完整动作一致；真实更新改变模型；模型及优化器恢复后的下一次更新逐参数一致。\n\n高层选择mask和完整719步结束通过，不冒充新增全部原子动作的官方Python复验。\n',encoding='utf8')
    print('G0_PASS',receipt,flush=True)

if __name__=='__main__':main()
