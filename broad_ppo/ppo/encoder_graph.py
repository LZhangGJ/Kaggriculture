"""Fixed-shape recurrent observation encoding for collection."""
import torch

class EncoderGraph:
    def __init__(self, model, observations, state):
        self.x={k:v.clone() for k,v in observations.items()}
        self.state=tuple(v.clone() for v in state)
        rng=torch.cuda.get_rng_state(state[0].device)
        stream=torch.cuda.Stream(device=state[0].device)
        stream.wait_stream(torch.cuda.current_stream(state[0].device))
        with torch.cuda.stream(stream):
            for _ in range(3):model.encode(self.x,*self.state)
        torch.cuda.current_stream(state[0].device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,capture_error_mode='thread_local'):self.output=model.encode(self.x,*self.state)
        torch.cuda.set_rng_state(rng,state[0].device)

    def __call__(self, observations, state):
        for k,v in observations.items():self.x[k].copy_(v)
        for target,value in zip(self.state,state):target.copy_(value)
        self.graph.replay()
        actor,critic,logits,context=self.output
        # Per-seat recurrence must outlive the next replay of this graph.
        return actor.clone(),critic.clone(),logits,context
