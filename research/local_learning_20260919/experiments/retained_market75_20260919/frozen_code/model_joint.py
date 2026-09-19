"""Joint market-plan classification learned solely from training action skeletons."""
import copy
import torch
from torch import nn
from student import Policy as BasePolicy,Codec,ROOT,UNIT_TYPES,MARKET_TYPES

class Policy(BasePolicy):
    def __init__(self,patterns,**config):
        super().__init__(**config)
        width=config.get("width",128)
        self.register_buffer("patterns",torch.tensor(patterns,dtype=torch.long))
        powers=torch.tensor([len(MARKET_TYPES)**i for i in reversed(range(11))],dtype=torch.long)
        self.register_buffer("pattern_powers",powers)
        codes=(self.patterns*powers).sum(1)
        assert (codes[1:]>codes[:-1]).all()
        self.register_buffer("pattern_codes",codes)
        self.pattern_head=nn.Sequential(nn.Linear(11*width,2*width),nn.GELU(),nn.Linear(2*width,len(patterns)))
        self.pattern_embedding=nn.Embedding(len(patterns)+1,width)
        nn.init.zeros_(self.pattern_embedding.weight)
        self.market_quantity_head=copy.deepcopy(self.quantity_head)
    def pattern_ids(self,types):
        code=(types.long()*self.pattern_powers).sum(1)
        idx=torch.searchsorted(self.pattern_codes,code)
        found=(idx<len(self.patterns))&(self.pattern_codes[idx.clamp(max=len(self.patterns)-1)]==code)
        return torch.where(found,idx,len(self.patterns))
    def encode(self,b):
        board=b["board"].float().permute(0,1,3,4,2).reshape(-1,200,16)
        bt=self.kind(board[:,:,0].long())+self.board_value(board[:,:,1:])+self.position.weight[None]
        t=b["step"].long()
        context=self.global_value(b["global_features"].float())+self.time(t)+self.day((t//24).clamp(max=29))
        prevu=b["previous_u"].long();prevm=b["previous_m"].long();units=b["units"].float()
        xy=(units[:,:,:2]*9).round().long().clamp(0,9)
        local=bt.gather(1,(xy[:,:,1]*10+xy[:,:,0])[:,:,None].expand(-1,-1,bt.shape[-1]))
        ut=self.unit_value(units)+self.unit_index.weight[None]+local
        ut=ut+self.previous_u(prevu[:,:,0])+self.previous_q(prevu[:,:,1])
        mt=self.market_index.weight[None]+self.previous_m(prevm[:,:,0])+self.previous_q(prevm[:,:,1])
        tokens=torch.cat([context[:,None],bt,ut,mt],dim=1)+context[:,None]
        unit_mask=torch.arange(32,device=t.device)[None]>=b["unit_count"][:,None]
        mask=torch.cat([torch.zeros((len(t),201),dtype=torch.bool,device=t.device),unit_mask,
                        torch.zeros((len(t),11),dtype=torch.bool,device=t.device)],dim=1)
        h=self.encoder(tokens,src_key_padding_mask=mask)
        return h[:,201:233],h[:,233:]
    def forward(self,b,targets=True):
        uh,mh=self.encode(b)
        ul=self.unit_head(uh).clone();ul[:,:,0]=-1e4
        pl=self.pattern_head(mh.flatten(1))
        auxiliary=self.market_head(mh).clone();auxiliary[:,:,0]=-1e4
        if targets:
            uc=b["unit_y"][:,:,0].long();mc=b["market_y"][:,:,0].long();pid=self.pattern_ids(mc)
        else:
            uc=ul.argmax(-1);pid=pl.argmax(-1);mc=self.patterns[pid]
        uq=self.quantity_head(uh+self.unit_type(uc))
        mq=self.market_quantity_head(mh+self.market_type(mc)+self.pattern_embedding(pid)[:,None]).clone()
        mq[:,:,0]=-1e4
        return ul,pl,uq,mq,auxiliary
    @torch.no_grad()
    def predict(self,b):
        ul,pl,uq,mq,_=self(b,targets=False)
        ut=ul.argmax(-1);mt=self.patterns[pl.argmax(-1)]
        u=torch.stack([ut,torch.where(self.uq[ut],uq.argmax(-1),0)],dim=-1)
        m=torch.stack([mt,torch.where(self.mq[mt],mq.argmax(-1),0)],dim=-1)
        u.masked_fill_((torch.arange(32,device=u.device)[None]>=b["unit_count"][:,None])[:,:,None],0)
        return u,m

def loss_and_counts(model,b,result):
    ul,pl,uq,mq,aux=result
    u=b["unit_y"].long();m=b["market_y"].long();um=u[:,:,0]!=0;mm=m[:,:,0]!=0
    uqm=um&model.uq[u[:,:,0]];mqm=mm&model.mq[m[:,:,0]]
    pid=model.pattern_ids(m[:,:,0]);known=pid<len(model.patterns)
    ce=nn.functional.cross_entropy
    loss=ce(ul[um],u[:,:,0][um])+ce(pl[known],pid[known])+.2*ce(aux[mm],m[:,:,0][mm])
    loss=loss+(ce(uq[uqm],u[:,:,1][uqm]) if uqm.any() else uq.sum()*0)
    loss=loss+(ce(mq[mqm],m[:,:,1][mqm]) if mqm.any() else mq.sum()*0)
    with torch.no_grad():
        pred=model.patterns[pl.argmax(-1)]
        ue=(ul.argmax(-1)==u[:,:,0])&(~uqm|(uq.argmax(-1)==u[:,:,1]))
        me=(pred==m[:,:,0])&(~mqm|(mq.argmax(-1)==m[:,:,1]))
        counts=torch.stack([ue[um].sum(),um.sum(),me[mm].sum(),mm.sum(),
                           ((ue|~um).all(1)&(me|~mm).all(1)).sum(),torch.tensor(len(u),device=u.device)])
    return loss,counts

def load_checkpoint(path,device="cpu"):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=Policy(checkpoint["patterns"],**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"]);model.eval()
    return model,Codec(checkpoint["quantities"]),checkpoint
