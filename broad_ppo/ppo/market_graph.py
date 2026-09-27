"""Capture the fixed ten-request market decoder with PyTorch CUDA graphs."""
import torch
import os
from ppo.fast_heads import score_event
from ppo.gpu_market import MarketBatch


class MarketGraph:
    def __init__(self, model, quantities, actor, context, seed, post_ledger, post_farm, greedy, retain_trace=True, generator=None):
        self.actor = actor.clone()
        self.context = {'market': context['market'].clone(),
                        'cells': torch.zeros_like(context['cells'][:, :1]),
                        'workers': torch.zeros_like(context['workers'][:, :1])}
        self.seed = {k: v.clone() for k, v in seed.items()}
        self.post_ledger = post_ledger.clone()
        self.post_farm = post_farm.clone()
        state = MarketBatch(self.seed)
        self.state = state
        if os.environ.get('PPO_COMPILE_MARKET') == '1':
            for name in ('vector','candidates','qstats','apply'):
                setattr(state,name,torch.compile(getattr(state,name),fullgraph=True,dynamic=True,
                    options={'triton.cudagraphs':False}))
        # Constructor constants stay outside capture. Dynamic state resets inside it.
        score=score_event;quantity_head=model.market_head.quantities;advance_head=model.market_head.advance
        if os.environ.get('PPO_COMPILE_HEADS')=='1':
            options={'triton.cudagraphs':False}
            score=torch.compile(score,fullgraph=True,dynamic=False,options=options)
            quantity_head=torch.compile(quantity_head,fullgraph=True,dynamic=False,options=options)
        if os.environ.get('PPO_COMPILE_FEEDBACK')=='1':
            from ppo.compiled_feedback import compiled_feedback
            advance_head=compiled_feedback(model.market_head)
        def decode():
            for value in (state.count, state.hires, state.lands, state.spend,
                          state.income, state.buys, state.sells, state.seed_buys):
                value.zero_()
            actor = self.actor
            n = len(actor); rows = state.rows
            post = model.summarize_post_worker(self.post_ledger, self.post_farm)
            prefix = torch.zeros_like(actor)
            active = torch.ones(n, device=actor.device, dtype=torch.bool)
            density = torch.zeros(n, device=actor.device)
            factors = density.clone(); trace = []; wire = []
            for depth in range(10):
                ledger = state.vector(); candidates = state.candidates(ledger)
                lp, gates, commands, embedding = score(model, 1, actor, prefix,
                    self.context, candidates, ledger, post)
                index = (torch.where(gates[:, 0] >= gates[:, 1], 0, commands.argmax(-1) + 1)
                         if greedy else torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1))
                index = torch.where(active, index, 0)
                need = state.item[index] >= 0
                qf, qm = quantities(1, index, state.qstats(index))
                qlp = quantity_head(actor, prefix, embedding[rows, index], ledger, qf, qm)
                qi = qlp.argmax(-1) if greedy else torch.multinomial(qlp.exp(), 1, generator=generator).squeeze(1)
                quantity = torch.where(need, quantities.vocab[1][qi], 0)
                density += (lp[rows, index] + torch.where(need, qlp[rows, qi], 0)) * active
                factors += (1 + need.float()) * active
                state.apply(index, quantity, active)
                delta = state.vector() - ledger
                advance = active & (index != 0)
                next_prefix = advance_head(prefix, embedding[rows, index], quantity, delta)
                prefix = torch.where(advance[:, None], next_prefix, prefix)
                wire.append(torch.stack((index, quantity, qi, need.long(), active.long()), -1))
                if retain_trace:trace.append((ledger, candidates, delta, qf, qm))
                active = advance
            return density, factors, torch.stack(wire), trace

        rng = None if generator is not None else torch.cuda.get_rng_state(actor.device)
        stream = torch.cuda.Stream(device=actor.device)
        stream.wait_stream(torch.cuda.current_stream(actor.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                decode()
        torch.cuda.current_stream(actor.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        if generator is not None:
            self.graph.register_generator_state(generator)
        # thread_local: a concurrent collection thread's CUDA calls must not fail this capture.
        with torch.cuda.graph(self.graph, capture_error_mode='thread_local'):
            self.output = decode()
        if rng is not None:
            torch.cuda.set_rng_state(rng, actor.device)

    def __call__(self, actor, context, seed, post_ledger, post_farm):
        self.actor.copy_(actor)
        self.context['market'].copy_(context['market'])
        for name, value in seed.items():
            self.seed[name].copy_(value)
        self.post_ledger.copy_(post_ledger); self.post_farm.copy_(post_farm)
        self.graph.replay()
        return self.output
