"""Exact compact worker features from the batched provisional state."""
import torch
from bc_runtime import ITEMS
from worker_phase import engine
from ppo.fast_features import QuantityFeatures
from ppo.worker_state import KIND,ITEM,BIRTH,YIELD,WATER,CARE,FERT,NEGLECT,AVAILABLE,LIFE,BONUS


class WorkerFeatures:
    def __init__(self,state,lean=False):
        # v40: lean=True evaluates tile vectors only at the five cells __call__ reads (same elementwise math, bitwise equal)
        self.state=state;device=state.device;self.lean=lean
        self.offsets=torch.tensor([(0,0),(0,-1),(0,1),(1,0),(-1,0)],device=device)
        cells=torch.tensor([(x,y) for y in range(10) for x in range(10)],device=device)
        self.distance=(cells[:,None]-cells[None]).abs().sum(-1)
        rules=[]
        for name in ITEMS:
            r=engine().CROPS.get(name) or engine().ANIMALS.get(name)
            rules.append([r.get('seed',r.get('cost',0)),r['first_yield_day'],r.get('max_yield',r.get('max_held',0)),r.get('ongoing',name in engine().ANIMALS)] if r else [0,0,0,0])
        self.rules=torch.tensor(rules,device=device,dtype=torch.float64)
        self.item_index=torch.arange(12,device=device)
        self.animal_order=torch.tensor([10,11,9],device=device)

    def tile_vectors(self,cells=None):
        s=self.state;t=s.tiles;day=s.day[:,None];step=s.seed['step'].reshape(s.n,1)
        if cells is not None:t=t[s.rows[:,None],cells]
        out=torch.zeros(s.n,t.shape[1],24,device=s.device)
        kind=t[:,:,KIND];item=t[:,:,ITEM]
        out[:,:,0]=kind==1;out[:,:,1]=kind==0;out[:,:,2]=kind==2
        out[:,:,3:8]=item[:,:,None]==torch.arange(5,device=s.device)
        out[:,:,8]=kind==5;out[:,:,9]=kind==4
        out[:,:,10:13]=item[:,:,None]==self.animal_order
        out[:,:,13]=QuantityFeatures.scaled(t[:,:,YIELD].double())
        out[:,:,14]=(day-t[:,:,BIRTH]).double()/30
        out[:,:,15]=t[:,:,WATER];out[:,:,16]=t[:,:,CARE]
        out[:,:,17]=(t[:,:,FERT]-day+1).clamp_min(0).double()/4
        out[:,:,18]=t[:,:,NEGLECT].double()/10;out[:,:,21]=t[:,:,AVAILABLE]
        out[:,:,22]=(t[:,:,LIFE]-step).clamp_min(0).double()/720
        out[:,:,23]=t[:,:,BONUS].double()/10
        return out

    def __call__(self,worker,counts):
        s=self.state;n=s.n;r=s.rows;scale=QuantityFeatures.scaled
        worker=worker if isinstance(worker,torch.Tensor) else torch.full((n,),worker,device=s.device,dtype=torch.long)
        pos=s.positions[r,worker];x,y=pos.unbind(-1);cell=y*10+x
        inv=s.inventory[r,worker];room=(100-s.shed.sum(-1)).clamp_min(0)
        total=s.seed['count'].reshape(n);remaining=total-worker-1
        day=s.day;hour=s.seed['hour'].reshape(n);step=s.seed['step'].reshape(n)
        initial=s.seed['seeds']
        destinations=pos[:,None]+self.offsets[None];bounds=((destinations>=0)&(destinations<10)).all(-1)
        destinations=torch.where(bounds[:,:,None],destinations,pos[:,None]);cells=destinations[:,:,1]*10+destinations[:,:,0]
        if self.lean:tv5=self.tile_vectors(cells);tv_cell=tv5[:,0]  # cells[:,0]==cell (zero offset, always in bounds)
        else:tv=self.tile_vectors();tv_cell=tv[r,cell];tv5=tv[r[:,None],cells]
        led=torch.zeros(n,128,device=s.device)
        led[:,:8]=torch.stack((x.double()/9,y.double()/9,worker==0,
            worker.double()/32,total.double()/32,
            remaining.double()/32,s.access[cell],scale(room.double())),-1)
        led[:,8:20]=scale(inv.double());led[:,20:32]=scale(s.shed.double());led[:,32:37]=scale(s.seeds.double())
        led[:,37:61]=tv_cell;led[:,61:66]=scale(initial.double());led[:,66:71]=scale(counts.double())
        led[:,71:76]=scale((initial-counts).double());led[:,76:81]=counts>initial
        led[:,81:86]=counts+remaining[:,None]<=initial
        led[:,86:91]=(counts<=initial)&(counts+remaining[:,None]>initial)
        led[:,91]=1;led[:,92]=(719-step).double()/719;led[:,93]=day.double()/30
        led[:,94]=hour.double()/24;led[:,95]=scale(s.seed['money'].reshape(n).double())
        t=s.tiles;kind=t[:,:,KIND];it=t[:,:,ITEM];crop=(it>=0)&(it<5);animal=it>=9
        maturity=((t[:,:,BIRTH]+s.rules[it.clamp_min(0),0]-day[:,None])*24-hour[:,None]).clamp_min(0)
        masks=torch.stack((s.access[None].expand(n,-1),(t[:,:,YIELD]>0)&(animal|(crop&(maturity==0))),
            crop&(t[:,:,WATER]==0),animal&(t[:,:,WATER]==0),animal&(t[:,:,CARE]==0),
            animal&(t[:,:,AVAILABLE]>0),crop,kind==0,(kind>=2)&~animal,(kind==4)&~animal,(kind==5)&~animal),-1)
        distances=self.distance[cells][:,:,:,None].masked_fill(~masks[:,None],100).min(2).values
        known=masks.any(1);distances=torch.where(known[:,None],distances,0)
        nav=torch.stack((distances.double()/18,(distances[:,:1]-distances).double()/18,
                         known[:,None].expand(-1,5,-1).double()),-1).float()
        item=torch.zeros(n,12,8,device=s.device)
        item[:,:,0]=scale(inv.double());item[:,:,1]=scale(s.shed.double())
        item[:,:5,2]=scale(s.seeds.double());item[:,:,3]=scale(s.seed['prices'].double())
        item[:,:,4]=scale(self.rules[:,0]);item[:,:,5]=self.rules[:,1]/30
        item[:,:,6]=scale(self.rules[:,2]);item[:,:,7]=self.rules[:,3]
        compact=dict(ledger=led,tiles=tv5,cells=cells,bounds=bounds,nav=nav,
            items=item,worker=worker,plant_initial=initial.float(),plant_count=counts.float())
        action_item=s.action_item.clamp_min(0)
        current_kind=kind[r,cell];current_item=it[r,cell]
        animal_place=(s.action_item>=9)[None] & (torch.arange(44,device=s.device)[None]<32)
        structure=torch.where(s.action_item==9,4,5)
        slot=animal_place&(current_kind[:,None]==structure)&(current_item[:,None]<0)
        need=s.access[cell,None]&(torch.arange(44,device=s.device)[None]>=20)&~slot
        stats=torch.stack((inv[:,action_item],s.shed[:,action_item],room[:,None].expand(-1,44)),-1).double()
        stats[:,:,:2]*=(s.action_item>=0)[None,:,None]
        return compact,need,stats
