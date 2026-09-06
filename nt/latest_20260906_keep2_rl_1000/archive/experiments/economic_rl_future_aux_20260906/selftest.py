from common import *
from aux_model import AuxModel, outcomes, update_aux
from model import Model, update, tensor_batch
import torch

def main():
    configure_torch();out=P/'checks';out.mkdir(exist_ok=False)
    torch.manual_seed(701);m=AuxModel().cuda();m.export(out/'initial.bin')
    assert digest(out/'initial.bin')==digest(OLD/'training/f3_r0/step000.bin')
    pool=runtime.make_pool();r=rollout(pool,out/'initial.bin',2,69400000,1,711,trace=True)
    r1=rollout(pool,out/'initial.bin',2,69400000,1,711,threads=1,trace=True)
    assert r['traces']==r1['traces']
    store_result(out/'before',r)
    b=tensor_batch(r,'cuda')
    with torch.no_grad():dist,val=m(b['global'],b['candidates'],b['mask'])
    errs=dict(prob=float(np.max(abs(dist.probs.cpu().numpy()-r['probability']))),value=float(np.max(abs(val.cpu().numpy()-r['value']))))
    assert max(errs.values())<2e-5,errs
    y,mask=outcomes(r)
    for hi,h in enumerate((1,3,7)):
        assert (mask[:,hi,0]==(r['day']+h<=29)).all()
        for i in np.flatnonzero(mask[:,hi,0]):
            assert r['game'][i]==r['game'][i+h]
            assert abs(y[i,hi,0]-(r['global'][i+h,2]-r['global'][i,2]))<1e-7
    changed=dict(r);changed['global']=r['global'].copy();changed['global'][-30:,2]+=1
    y2,_=outcomes(changed);assert np.array_equal(y[:-30],y2[:-30])
    # Shape/offset fixture independently makes each named field distinctive.
    fixture=dict(r);fixture['global']=np.zeros_like(r['global'])
    for i in range(len(r['day'])):
        day=r['day'][i];fixture['global'][i,2]=day*.1;fixture['global'][i,4]=day*.2
        fixture['global'][i,72:81]=day*.01;fixture['global'][i,82]=day*.3;fixture['global'][i,83]=day*.4
    fy,fm=outcomes(fixture)
    for hi,h in enumerate((1,3,7)):
        assert np.allclose(fy[fm[:,hi,0],hi],np.array([.1,.2,.09,.3,.4])*h,atol=2e-6)
    # A zero coefficient reproduces the historical PPO update exactly.
    base=Model().cuda();base.load_state_dict({k:v for k,v in m.state_dict().items()if not k.startswith('future.')})
    control=AuxModel().cuda();control.load_state_dict(m.state_dict())
    update(base,torch.optim.Adam(base.parameters(),lr=3e-4),r,'cuda',101)
    update_aux(control,torch.optim.Adam(control.parameters(),lr=3e-4),r,'cuda',101,0)
    for k,v in base.state_dict().items():assert torch.equal(v,control.state_dict()[k]),k
    base.export(out/'old_update.bin');control.export(out/'control_update.bin')
    assert digest(out/'old_update.bin')==digest(out/'control_update.bin')
    m.zero_grad();pred=m.forecast(b['global'],b['candidates'],b['choice'])
    loss=(pred[torch.as_tensor(mask,device='cuda')]-torch.as_tensor(y,device='cuda')[torch.as_tensor(mask,device='cuda')]).square().mean()
    loss.backward();grad=float(m.actor[0].weight.grad.abs().sum());assert grad>0
    aux=AuxModel().cuda();aux.load_state_dict(m.state_dict())
    stats=update_aux(aux,torch.optim.Adam(aux.parameters(),lr=3e-4),r,'cuda',101,.1)
    assert not torch.equal(aux.actor[0].weight,control.actor[0].weight)
    aux.export(out/'aux_update.bin')
    after=rollout(pool,out/'aux_update.bin',2,69400000,1,711)
    with torch.no_grad():
        dd,vv=aux(torch.as_tensor(after['global'],device='cuda'),torch.as_tensor(after['candidates'],device='cuda'),torch.as_tensor(after['mask'],device='cuda'))
    posterr=float(np.max(abs(dd.probs.cpu().numpy()-after['probability'])));assert posterr<2e-5
    store_result(out/'after',after)
    save(P/'G0_ACCEPTANCE.json',dict(status='PASS',torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
         base_parameters=70402,auxiliary_parameters=1935,cpp_torch_error=errs,postupdate_probability_error=posterr,
         zero_aux_exact_old_update=True,full_trace_threads_1_16_exact=True,labels_time_and_episode_checked=True,
         independent_field_fixture=True,actor_aux_gradient=grad,aux_stats=stats,simulated_games=42))
    (P/'G0_ACCEPTANCE_ZH.md').write_text('# G0 工程验收\n\n通过：修账本F3主体不变；原始权重一致；零辅助权重与旧PPO更新逐参数相同；辅助梯度到达actor；C++/GPU推理一致；1/16线程完整动作一致；1/3/7天标签不跨局；字段偏移独立夹具通过。\n\n未来头预测归一化状态变化（未来减当前），等价于在当前状态基础上预测未来；待收产量不是累计生产/售出。输出头仅训练保留，线上仍70,402参数。\n',encoding='utf8')
    print('G0_PASS',errs,'post',posterr,flush=True)

if __name__=='__main__':main()
