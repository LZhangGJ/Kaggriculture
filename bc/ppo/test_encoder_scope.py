import os,unittest
from unittest.mock import patch
import torch
from ppo.sampler import BatchedExactSampler

class TinyEncoder(torch.nn.Module):
    def encode(self,x,a,c):
        h=x['value']+a;critic=x['value']+c
        return h,critic,critic[:,:3]*2,{'market':h[:,None]*3,'cells':h[:,None]*4,'workers':h[:,None]*5}

@unittest.skipUnless(torch.cuda.is_available(),'CUDA graph ownership regression')
class EncoderScopeTest(unittest.TestCase):
    @torch.no_grad()
    def test_reference_cannot_overwrite_suspended_opponent(self):
        device=os.environ.get('PARITY_DEVICE','cuda:0');torch.cuda.set_device(device)
        model=TinyEncoder().to(device)
        sampler=BatchedExactSampler({'bc':model,'learner':model},model,[1],[1],device)
        state=(torch.zeros(2,256,device=device),torch.zeros(2,256,device=device))
        a={'value':torch.ones(2,256,device=device)};b={'value':torch.full((2,256),9.,device=device)}
        with patch.dict(os.environ,{'PPO_GRAPH_ENCODER':'1'}):
            opponent=sampler.encode(('policy','bc'),model,a,state)
            sampler.encode(('reference','learner'),model,b,state)
        eager=model.encode(a,*state)
        for actual,expected in zip(opponent[:3],eager[:3]):torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for key in eager[3]:torch.testing.assert_close(opponent[3][key],eager[3][key],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
