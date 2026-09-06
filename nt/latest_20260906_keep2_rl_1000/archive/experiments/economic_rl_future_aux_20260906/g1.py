"""Paired terminal candidate labels, grouped holdout, supervised learnability."""
from common import *
import argparse
from collections import defaultdict
sys.path.insert(0, str(AUDIT))
import run_audit as audit
import torch
from torch import nn

OUT = P/'g1'
BINS = [(0,2),(3,6),(7,10),(11,14),(15,18),(19,22),(23,26),(27,28)]

def generate():
    OUT.mkdir(exist_ok=False)
    # Freeze only dependencies used in this stage; no mutations to prior work.
    paths = [P/'g1.py', P/'common.py', P/'PLAN_ZH.md', AUDIT/'run_audit.py', AUDIT/'build/audit.so',
             FIX/'build/f3.so', OLD/'rl_common.hpp', OLD/'runtime.py']
    frozen = {str(f):digest(f) for f in paths}
    save(OUT/'PROTOCOL.json', dict(hashes=frozen, train=[68000000,16], dev=[68100000,8], test=[68200000,16],
         old_audit_usage='training only, including previously called new seeds', tail=0,
         source_policy=str(P.parent/'economic_rl_f3_ledger_rerun_20260906/training/f3_r0/step020.bin')))
    pool = audit.native.RLPool(audit.data())
    model = str(P.parent/'economic_rl_f3_ledger_rerun_20260906/training/f3_r0/step020.bin')
    allstates=[]; results={}; rng=np.random.default_rng(2060906); tic=time.perf_counter()
    for split, start, count in [('train',68000000,16),('dev',68100000,8),('test',68200000,16)]:
        source=dict(id=split, model=model)
        jobs=[dict(seed=s,seat=p,opponent=o,mode=2,rng=(8811+audit.GOLDEN*(i+1))%(1<<64))
              for i,(s,p,o) in enumerate(runtime.jobs(start,count))]
        base=audit.call(pool,source,jobs)
        store_result(OUT/f'base_{split}',base)
        pending=[]; states=[]
        for gi, job in enumerate(jobs):
            for lo,hi in BINS:
                day=int(rng.integers(lo,hi+1)); ix=gi*30+day; sid=len(allstates)+len(states)
                state=dict(id=sid,split=split,seed=job['seed'],seat=job['seat'],opponent=NAMES[job['opponent']],day=day,ix=ix)
                state['available']=np.flatnonzero(base['mask'][ix]).tolist();states.append(state)
                for action in state['available']:
                    pending.append(dict(job,mode=1000+day*10+action,state_id=sid,day=day,choice=action,base_ix=ix))
        save(OUT/f'states_{split}.json',states)
        rows=[]
        for offset in range(0,len(pending),224):
            chunk=pending[offset:offset+224]; r=audit.call(pool,source,chunk)
            for gi,(job,row) in enumerate(zip(chunk,r['rows'])):
                k=gi*30+job['day']; ref=job['base_ix']
                for name in ('global','candidates','mask','probability'):
                    assert np.array_equal(base[name][ref],r[name][k]), (split, job['state_id'],name)
                assert r['choice'][k]==job['choice']
                assert (r['choice'][k+1:gi*30+30]==0).all()
                rows.append(dict(state_id=job['state_id'],choice=job['choice'],margin=row['margin'],win=int(row['win']),
                                 cash=row['cash'],reward=float(r['reward'][gi*30+29])))
            save(OUT/'PROGRESS.json',dict(stage='labels',split=split,completed=len(rows),total=len(pending),seconds=time.perf_counter()-tic))
            print('LABEL',split,len(rows),'/',len(pending),round(time.perf_counter()-tic,1),'s',flush=True)
        save(OUT/f'branches_{split}.json',rows);allstates.extend(states);results[split]=rows
    assert all(digest(f)==h for f,h in frozen.items())
    save(OUT/'DATA_COMPLETE.json',dict(status='PASS',states=len(allstates),branch_games=sum(map(len,results.values())),
         source_games=560,seconds=time.perf_counter()-tic,prefix_features_identical=True))

def dataset():
    X=[];C=[];M=[];W=[];D=[];R=[];S=[];seeds=[];opps=[]
    def add(state, x,c,m, rows, split):
        keep=rows[0];w=np.zeros(10,np.float32);d=w.copy();r=w.copy()
        assert set(rows)==set(np.flatnonzero(m))
        for a,row in rows.items():
            w[a]=int(row['win'])-int(keep['win']);d[a]=(row['margin']-keep['margin'])/50000.
            r[a]=row['reward']-keep['reward']
        X.append(x);C.append(c);M.append(m);W.append(w);D.append(d);R.append(r);S.append(split)
        seeds.append(state['seed']);opps.append(state['opponent'])
    for split in ('train','dev','test'):
        with np.load(OUT/f'base_{split}/decisions.npz') as archive:
            base={k:archive[k] for k in ('global','candidates','mask')}
        by=defaultdict(dict)
        for row in read(OUT/f'branches_{split}.json'):by[row['state_id']][row['choice']]=row
        for state in read(OUT/f'states_{split}.json'):
            i=state['ix'];add(state,base['global'][i],base['candidates'][i],base['mask'][i],by[state['id']],split)
    sources={}
    for s in read(AUDIT/'PROTOCOL.json')['sources']:
        with np.load(AUDIT/'baselines'/s['id']/'decisions.npz') as archive:
            sources[s['id']]={k:archive[k] for k in ('global','candidates','mask')}
    by=defaultdict(dict)
    for f in sorted((AUDIT/'branches').glob('*.json')):
        for row in read(f):
            if row['tail']==0:by[row['state_id']][row['choice']]=row
    for state in read(AUDIT/'STATES.json'):
        b=sources[state['source']];i=state['game']*30+state['day']
        add(state,b['global'][i],b['candidates'][i],b['mask'][i],by[state['id']],'train')
    out={k:np.asarray(v) for k,v in dict(x=X,c=C,mask=M,win=W,margin=D,reward=R,split=S,seed=seeds,opponent=opps).items()}
    for a,b in [('train','dev'),('train','test'),('dev','test')]:
        assert not set(out['seed'][out['split']==a]) & set(out['seed'][out['split']==b])
    assert np.isfinite(out['x']).all() and np.isfinite(out['c']).all()
    np.savez_compressed(OUT/'dataset.npz',**out)
    return out

class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(160,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,2))
    def forward(self,x,c):
        v=self.net(torch.cat((x[:,None,:].expand(-1,10,-1),c),-1))
        # Score incremental win probability first, bounded margin tie-break second.
        return v, v[:,:,0]+.1*torch.tanh(v[:,:,1])

def metrics(score,data,indices):
    pairs=[];hit=[];margin=[];reward=[];wins=[];randmargin=[];grp=[]
    for ii,i in enumerate(indices):
        aa=np.flatnonzero(data['mask'][i]);q=data['reward'][i];s=score[ii]
        a=int(aa[np.argmax(s[aa])]); margin.append(float(data['margin'][i,a]*50000));reward.append(float(q[a]));wins.append(float(data['win'][i,a]))
        randmargin.append(float(data['margin'][i,aa].mean()*50000));grp.append(data['seed'][i])
        pp=[]
        for aj in aa:
            for ak in aa:
                if aj>=ak or abs(q[aj]-q[ak])<1e-6:continue
                pp.append(.5 if abs(s[aj]-s[ak])<1e-9 else float((s[aj]-s[ak])*(q[aj]-q[ak])>0))
        if pp:pairs.append((np.mean(pp),data['seed'][i]))
        hit.append(float(abs(q[a]-q[aa].max())<1e-6))
    return dict(nodes=len(indices),pairwise=bootstrap([p[0]for p in pairs],[p[1]for p in pairs]),best_hit=float(np.mean(hit)),
                margin_vs_keep=bootstrap(margin,grp),reward_vs_keep=bootstrap(reward,grp),win_vs_keep=bootstrap(wins,grp),
                margin_vs_uniform=bootstrap(np.array(margin)-randmargin,grp))

def learn():
    torch=configure_torch();data=dataset();device='cuda'
    b={k:torch.as_tensor(data[k],device=device) for k in ('x','c','mask','win','margin','reward')}
    ix={s:np.flatnonzero(data['split']==s) for s in ('train','dev','test')}
    outputs=[]
    for rep in (0,1):
        torch.manual_seed(2201+rep);m=ValueModel().to(device);opt=torch.optim.Adam(m.parameters(),lr=3e-4)
        log=[];best=None;rng=np.random.default_rng(321+rep)
        for epoch in range(1,81):
            losses=[]
            for subset in np.array_split(rng.permutation(ix['train']),max(1,int(np.ceil(len(ix['train'])/256)))):
                j=torch.as_tensor(subset,device=device);v,score=m(b['x'][j],b['c'][j]);mask=b['mask'][j].bool();non=mask.clone();non[:,0]=False
                if not non.any():continue
                fit=(v[:,:,0]-b['win'][j]).square()[non].mean()+.25*nn.functional.smooth_l1_loss(v[:,:,1][non],b['margin'][j][non])
                dif=b['reward'][j,:,None]-b['reward'][j,None,:];pm=mask[:,:,None]&mask[:,None,:]&(dif.abs()>1e-6)
                ranking=nn.functional.softplus(-torch.sign(dif)*(score[:,:,None]-score[:,None,:]))[pm].mean()
                anchor=v[:,0,:].square().mean();loss=fit+.25*ranking+.1*anchor
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(loss.item())
            if epoch in (1,5,10,20,40,60,80):
                with torch.no_grad():_,s=m(b['x'][ix['dev']],b['c'][ix['dev']])
                met=metrics(s.cpu().numpy(),data,ix['dev']);key=(met['pairwise']['mean'],-epoch)
                log.append(dict(epoch=epoch,loss=float(np.mean(losses)),development=met))
                if best is None or key>best[0]:
                    best=(key,epoch);torch.save(m.state_dict(),OUT/f'value_r{rep}.pt')
                print('G1_FIT',rep,epoch,'dev_pairwise',met['pairwise']['mean'],flush=True)
        m.load_state_dict(torch.load(OUT/f'value_r{rep}.pt',weights_only=True));entry=dict(rep=rep,best_epoch=best[1],history=log,metrics={})
        for split in ('train','dev','test'):
            with torch.no_grad():_,s=m(b['x'][ix[split]],b['c'][ix[split]])
            entry['metrics'][split]=metrics(s.cpu().numpy(),data,ix[split])
        outputs.append(entry)
    save(OUT/'RESULTS.json',dict(status='COMPLETE',runs=outputs,parameters=sum(p.numel()for p in m.parameters()),
         split_nodes={k:len(v)for k,v in ix.items()},conditional_continuation='F3 KEEP tail',direct_online_evidence=False))
    lines=['# G1 候选好坏学习验收','', '工程通过。下表为独立种子、同状态干预一次后交回F3的离线结果，不是完整模型对战胜率。', '',
           '|重复|留出排序正确率|95% seed区间|选择相对KEEP平均分差|开发选定epoch|','|---|---:|---|---:|---:|']
    for r in outputs:
        t=r['metrics']['test'];lines.append(f"|{r['rep']}|{t['pairwise']['mean']:.2%}|{t['pairwise']['ci95']}|{t['margin_vs_keep']['mean']:.0f}|{r['best_epoch']}|")
    lines+=['','训练/开发/留出按完整seed隔离；旧审计数据仅训练。候选结果是特定随机实现与后续F3策略下的条件价值。',
            '第一步不预热第二步PPO模型，以免把预训练与辅助损失效果混为一谈。详细证据：g1/RESULTS.json。']
    (P/'G1_ACCEPTANCE_ZH.md').write_text('\n'.join(lines),encoding='utf8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('generate','learn'));args=parser.parse_args()
    (generate if args.mode=='generate' else learn)()
