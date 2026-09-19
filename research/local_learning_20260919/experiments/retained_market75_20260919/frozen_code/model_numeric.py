"""BC policy with exact integer features in addition to existing visible-state features."""
import torch
from torch import nn
from student import Policy as BasePolicy,Codec,ROOT,loss_and_counts
class IntegerEncoder(nn.Module):
    def __init__(self,fields,width):
        super().__init__()
        self.embedding=nn.Embedding(256,8)
        self.projection=nn.Linear(fields*4*9,width)
        self.register_buffer("shifts",torch.tensor([0,8,16,24]))
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
    def forward(self,values):
        byte=(values.long()[...,None]>>self.shifts)&255
        # Both the category identity and the ordered byte value are retained.
        value=torch.cat([self.embedding(byte),byte.float()[...,None]/256.],dim=-1)
        return self.projection(value.flatten(-3))
class Policy(BasePolicy):
    def __init__(self,**config):
        super().__init__(**config)
        width=config.get("width",128)
        self.global_exact_encoder=IntegerEncoder(49,width)
        self.unit_exact_encoder=IntegerEncoder(12,width)
    def forward(self,b,targets=True):
        board=b["board"].float().permute(0,1,3,4,2).reshape(-1,200,16)
        bt=self.kind(board[:,:,0].long())+self.board_value(board[:,:,1:])
        bt=bt+self.position.weight[None]
        t=b["step"].long()
        context=self.global_value(b["global_features"].float())+self.time(t)+self.day((t//24).clamp(max=29))
        context=context+self.global_exact_encoder(b["global_exact"])
        prevu=b["previous_u"].long();prevm=b["previous_m"].long()
        units=b["units"].float()
        xy=(units[:,:,:2]*9).round().long().clamp(0,9)
        local=bt.gather(1,(xy[:,:,1]*10+xy[:,:,0])[:,:,None].expand(-1,-1,bt.shape[-1]))
        ut=self.unit_value(units)+self.unit_index.weight[None]+local
        ut=ut+self.unit_exact_encoder(b["unit_exact"])
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

def load_checkpoint(path,device="cpu"):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=Policy(**checkpoint["config"]).to(device);model.load_state_dict(checkpoint["model"]);model.eval()
    return model,Codec(checkpoint["quantities"]),checkpoint
