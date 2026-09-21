"""Streaming adapter around the pinned official interpreter, without replay copies."""
from kaggle_environments.core import process_schema
from worker_phase import engine
from types import SimpleNamespace


def history_snapshot(obs):
    # PublicHistory.observe reads only these previous-frame fields.
    return dict(step=obs['step'],day=obs['day'],hour=obs['hour'],player=obs['player'],
                market=dict(prices=dict(obs['market']['prices']),inventory=dict(obs['market']['inventory'])),
                farms=[dict(money=f['money']) for f in obs['farms']])


def step(env,actions):
    if env.done: raise RuntimeError('Cannot step a terminal game')
    if len(actions)!=len(env.state): raise ValueError('Wrong number of seat actions')
    next_step=env.state[0].observation.step+1
    schema=env._Environment__state_schema.properties.action
    for state,action in zip(env.state,actions):
        err,data=process_schema(schema,action)
        state.action=None if err else data
        if err:state.status='INVALID'
    view=SimpleNamespace(configuration=env.configuration,info=env.info,done=False)
    env.state=engine().interpreter(env.state,view)
    env.state[0].observation.step=next_step
    for state in env.state:
        if next_step>=env.configuration.episodeSteps-1 and state.status in ('ACTIVE','INACTIVE'):
            state.status='DONE'
        if state.status in ('ERROR','INVALID','TIMEOUT'):state.reward=None
    return env.state
