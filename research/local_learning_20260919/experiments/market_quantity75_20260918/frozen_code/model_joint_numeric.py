"""Joint market policy with zero-initialized exact-integer observation paths."""
import torch
from model_joint import Policy as JointPolicy,loss_and_counts
from model_numeric import IntegerEncoder
from student import Codec,ROOT
class Policy(JointPolicy):
    def __init__(self,patterns,**config):
        super().__init__(patterns,**config)
        width=config.get("width",128)
        self.global_exact_encoder=IntegerEncoder(49,width)
        self.unit_exact_encoder=IntegerEncoder(12,width)
    def encode(self,b):
        board=b["board"].float().permute(0,1,3,4,2).reshape(-1,200,16)
        bt=self.kind(board[:,:,0].long())+self.board_value(board[:,:,1:])+self.position.weight[None]
        t=b["step"].long()
        context=self.global_value(b["global_features"].float())+self.time(t)+self.day((t//24).clamp(max=29))
        context=context+self.global_exact_encoder(b["global_exact"])
        prevu=b["previous_u"].long();prevm=b["previous_m"].long();units=b["units"].float()
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
        return h[:,201:233],h[:,233:]

def load_checkpoint(path,device="cpu"):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    model=Policy(checkpoint["patterns"],**checkpoint["config"]).to(device)
    state=dict(checkpoint["model"])
    market_keys={"market_"+k for k in state if k.startswith("quantity_head.")}
    missing=market_keys-set(state)
    assert not missing or missing==market_keys,"Partially missing market quantity head"
    if missing:
        state.update({"market_"+k:v for k,v in list(state.items()) if k.startswith("quantity_head.")})
    model.load_state_dict(state);model.eval()
    return model,Codec(checkpoint["quantities"]),checkpoint
