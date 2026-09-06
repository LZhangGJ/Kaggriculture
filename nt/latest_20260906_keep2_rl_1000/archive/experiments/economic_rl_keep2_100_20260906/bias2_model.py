from run100_common import *
import torch
from aux_model import AuxModel, update_aux, outcomes, forecast_metrics

class Bias2Model(AuxModel):
    def forward(self,x,c,mask):
        z=torch.cat((x[:,None,:].expand(-1,10,-1),c),dim=-1)
        logits=self.actor(z).squeeze(-1)
        prior=torch.zeros_like(logits);prior[:,0]=2.
        logits=(logits+prior).masked_fill(~mask.bool(),-1e30)
        return torch.distributions.Categorical(logits=logits),self.critic(x).squeeze(-1)
