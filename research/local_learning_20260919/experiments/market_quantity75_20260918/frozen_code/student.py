"""Learned single-teacher policy. No teacher import, opponent ID or random seed input."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from features import (CROPS,ITEMS,PRODUCTS,ANIMALS,UNIT_OPS,MARKET_OPS,
                      ITEM_OPS,QUANTITY_OPS,OPS,OP,decode_command,encode_player,encode_board)
ROOT=Path(__file__).resolve().parents[1]
UNIT_TYPES=[("PAD",None),("NOOP",None)]+[(x,None) for x in UNIT_OPS if x not in ITEM_OPS]
UNIT_TYPES += [("PLANT",x) for x in CROPS]+[(op,x) for op in ("PICKUP","PLACE") for x in ITEMS]
MARKET_TYPES=[("PAD",None),("EOS",None),("NOOP",None),("HIRE",None),("BUY_LAND",None)]
MARKET_TYPES += [("BUY_SEED",x) for x in CROPS]+[("BUY_PRODUCT",x) for x in PRODUCTS]
MARKET_TYPES += [("BUY_ANIMAL",x) for x in ANIMALS]+[("SELL",x) for x in PRODUCTS]

class Codec:
    def __init__(self, quantities):
        self.quantities=quantities;self.qmap={x:i for i,x in enumerate(quantities)}
        self.umap={tuple(x):i for i,x in enumerate(UNIT_TYPES)}
        self.mmap={tuple(x):i for i,x in enumerate(MARKET_TYPES)}
        self.uq=np.array([x[0] in QUANTITY_OPS for x in UNIT_TYPES])
        self.mq=np.array([x[0] in QUANTITY_OPS for x in MARKET_TYPES])
    def command(self,raw,market=False):
        if not raw:return (self.mmap if market else self.umap)[("NOOP",None)],0
        op=raw[0];key=(op,raw[1] if op in ITEM_OPS else None)
        return (self.mmap if market else self.umap)[key],self.qmap[raw[2] if len(raw)>2 else None]
    def encode(self,tokens,n):
        u=np.zeros((32,2),np.int16);m=np.zeros((11,2),np.int16)
        for i in range(n):u[i]=self.command(decode_command(tokens[i]))
        j=0
        for c in tokens[n:]:
            if int(c[0]) in (0,OP["EOS"]):break
            assert j<10
            m[j]=self.command(decode_command(c),True);j+=1
        m[j,0]=1
        return u,m
    def raw(self,pair,market=False):
        types=MARKET_TYPES if market else UNIT_TYPES
        op,item=types[int(pair[0])]
        if op=="NOOP":return []
        assert op not in ("PAD","EOS")
        cmd=[op]
        if item is not None:cmd.append(item)
        if op in QUANTITY_OPS and self.quantities[int(pair[1])] is not None:
            cmd.append(self.quantities[int(pair[1])])
        return cmd
    def decode(self,u,m,n):
        commands=[self.raw(c) for c in u[:n]]
        market=[]
        for c in m[:10]:
            if int(c[0]) in (0,1):break
            market.append(self.raw(c,True))
        return {"farmer":commands[0],"hands":commands[1:],"market":market}

class Policy(nn.Module):
    def __init__(self,quantity_count,width=128,layers=3,heads=4,dropout=0.05):
        super().__init__()
        self.config=dict(quantity_count=quantity_count,width=width,layers=layers,heads=heads,dropout=dropout)
        self.kind=nn.Embedding(13,width)
        self.board_value=nn.Linear(15,width)
        self.position=nn.Embedding(200,width)
        self.unit_index=nn.Embedding(32,width)
        self.market_index=nn.Embedding(11,width)
        self.time=nn.Embedding(720,width)
        self.day=nn.Embedding(30,width)
        self.global_value=nn.Sequential(nn.Linear(64,width),nn.GELU(),nn.Linear(width,width))
        self.unit_value=nn.Linear(16,width)
        self.previous_u=nn.Embedding(len(UNIT_TYPES),width,padding_idx=0)
        self.previous_m=nn.Embedding(len(MARKET_TYPES),width,padding_idx=0)
        self.previous_q=nn.Embedding(quantity_count,width,padding_idx=0)
        layer=nn.TransformerEncoderLayer(width,heads,4*width,dropout,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,layers,norm=nn.LayerNorm(width),enable_nested_tensor=False)
        self.unit_head=nn.Linear(width,len(UNIT_TYPES))
        self.market_head=nn.Linear(width,len(MARKET_TYPES))
        self.unit_type=nn.Embedding(len(UNIT_TYPES),width)
        self.market_type=nn.Embedding(len(MARKET_TYPES),width)
        self.quantity_head=nn.Sequential(nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,quantity_count))
        self.register_buffer("uq",torch.tensor([x[0] in QUANTITY_OPS for x in UNIT_TYPES]))
        self.register_buffer("mq",torch.tensor([x[0] in QUANTITY_OPS for x in MARKET_TYPES]))
    def forward(self,b,targets=True):
        board=b["board"].float().permute(0,1,3,4,2).reshape(-1,200,16)
        bt=self.kind(board[:,:,0].long())+self.board_value(board[:,:,1:])
        bt=bt+self.position.weight[None]
        t=b["step"].long()
        context=self.global_value(b["global_features"].float())+self.time(t)+self.day((t//24).clamp(max=29))
        prevu=b["previous_u"].long();prevm=b["previous_m"].long()
        units=b["units"].float()
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
        uh,mh=h[:,201:233],h[:,233:]
        ul,ml=self.unit_head(uh),self.market_head(mh)
        # PAD cannot be emitted; extra positions are ignored using actual unit count/EOS.
        ul=ul.clone();ml=ml.clone();ul[:,:,0]=-1e4;ml[:,:,0]=-1e4
        up=ul.argmax(-1);mp=ml.argmax(-1)
        uc=b["unit_y"][:,:,0].long() if targets else up
        mc=b["market_y"][:,:,0].long() if targets else mp
        uq=self.quantity_head(uh+self.unit_type(uc))
        mq=self.quantity_head(mh+self.market_type(mc))
        # All market quantity operations require an integer; None is only valid for units.
        mq=mq.clone();mq[:,:,0]=-1e4
        return ul,ml,uq,mq
    @torch.no_grad()
    def predict(self,b):
        ul,ml,uq,mq=self(b,targets=False)
        ut,mt=ul.argmax(-1),ml.argmax(-1)
        u=torch.stack([ut,torch.where(self.uq[ut],uq.argmax(-1),0)],dim=-1)
        m=torch.stack([mt,torch.where(self.mq[mt],mq.argmax(-1),0)],dim=-1)
        # Previous commands are exactly those that were actually returned.
        u.masked_fill_((torch.arange(32,device=u.device)[None]>=b["unit_count"][:,None])[:,:,None],0)
        m[:,10,0]=1; m[:,10,1]=0
        mt=m[:,:,0]
        after=((mt==1).cumsum(1)-(mt==1).long())>0
        m.masked_fill_(after[:,:,None],0)
        return u,m

def loss_and_counts(model,b,result):
    ul,ml,uq,mq=result;u=b["unit_y"].long();m=b["market_y"].long()
    um=u[:,:,0]!=0;mm=m[:,:,0]!=0
    uqm=um&model.uq[u[:,:,0]];mqm=mm&model.mq[m[:,:,0]]
    ce=nn.functional.cross_entropy
    loss=ce(ul[um],u[:,:,0][um])+ce(ml[mm],m[:,:,0][mm])
    loss=loss+(ce(uq[uqm],u[:,:,1][uqm]) if uqm.any() else uq.sum()*0)
    loss=loss+(ce(mq[mqm],m[:,:,1][mqm]) if mqm.any() else mq.sum()*0)
    with torch.no_grad():
        ue=(ul.argmax(-1)==u[:,:,0])&(~uqm|(uq.argmax(-1)==u[:,:,1]))
        me=(ml.argmax(-1)==m[:,:,0])&(~mqm|(mq.argmax(-1)==m[:,:,1]))
        metrics=torch.stack([ue[um].sum(),um.sum(),me[mm].sum(),mm.sum(),
                             ((ue|~um).all(1)&(me|~mm).all(1)).sum(),
                             torch.tensor(len(u),device=u.device)])
    return loss,metrics

def observation_batch(obs,previous_u,previous_m):
    g,u,n=encode_player(obs);seat=obs["player"];step=obs["step"]
    return dict(board=np.stack([encode_board(obs["farms"][p],obs["day"],step) for p in (seat,1-seat)]),
                global_features=g,units=u,unit_count=np.int16(n),step=np.int16(step),
                previous_u=previous_u,previous_m=previous_m)

def load_checkpoint(path,device="cpu"):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=Policy(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"]);model.eval()
    return model,Codec(checkpoint["quantities"]),checkpoint

if __name__=="__main__":
    codec=Codec([None]+list(range(1001)))
    for market,commands in [(False,[[],["PASS"],["PLANT","WHEAT"],["PICKUP","WOOL"],["PLACE","COW",1]]),
                            (True,[[],["HIRE"],["BUY_LAND"],["BUY_SEED","CARROT",1000],["SELL","MILK",17]])]:
        for command in commands:assert codec.raw(codec.command(command,market),market)==command
    print("STUDENT_CODEC_CHECK PASS")
