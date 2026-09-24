"""Original match function, relocated to this portable package; no policy changes."""
from pathlib import Path
import contextlib,copy,hashlib,json,os,random,sys,time,traceback
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
HERE=Path(__file__).resolve().parent
ROOT=HERE
sys.path.insert(0,str(HERE/'referee'))
from policy_host import Policy,LocalGame,load_engine

def evaluate(job):
    candidate, own_entry, opponent, rival_entry, seed, seat, actor_seed = job
    row = dict(candidate=candidate, opponent=opponent, seed=seed, seat=seat, actor_seed=actor_seed, error=None)
    started = time.monotonic()
    try:
        import torch
        import numpy as np
        torch.set_num_threads(1)
        torch.manual_seed(actor_seed)
        np.random.seed(actor_seed)
        random.seed(actor_seed)
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            own = Policy(ROOT / own_entry, 'candidate_entry')
            rival = Policy(ROOT / rival_entry, 'teammate_entry')
            env = LocalGame(seed, load_engine())
            maxima = [0., 0.]
            noninitial = [0., 0.]
            action_digest = [hashlib.sha256(), hashlib.sha256()]
            while not env.done:
                assert env.configuration.seed is None
                actions = [None, None]
                for player, policy in ((seat, own), (1-seat, rival)):
                    tick = time.monotonic()
                    actions[player] = policy(env.observation(player), copy.deepcopy(env.configuration))
                    duration = time.monotonic()-tick
                    maxima[player] = max(maxima[player], duration)
                    if env.t > 0: noninitial[player] = max(noninitial[player], duration)
                    action_digest[player].update(json.dumps(actions[player], sort_keys=True, separators=(',',':')).encode())
                env.advance(actions)
            module = rival.fn.__globals__
            instance = module['_instances'][1-seat]
            student = getattr(instance, 'dynamic', instance)
            audit = student.student_summary()
            row['opponent_audit'] = {k:v for k,v in audit.items() if k != 'event_trace'}
            row['opponent_sample'] = student.sample
            assert student.sample is True, 'Submission sample behavior changed'
            assert audit['days']==17 and audit['fallbacks']==0 and audit['illegal']==0, 'Student silently fell back'
            assert audit['successful_steps']==list(range(288,673,24))
            assert env.t == 719
            cash = [farm['money'] for farm in env.state[0].observation.farms]
            margin = cash[seat]-cash[1-seat]
            row.update(steps=env.t, own_cash=cash[seat], opponent_cash=cash[1-seat], margin=margin,
                win=int(margin>0),tie=int(margin==0),max_own_action_s=maxima[seat],
                max_opponent_action_s=maxima[1-seat],max_own_noninitial_s=noninitial[seat],
                max_opponent_noninitial_s=noninitial[1-seat],
                own_actions_sha256=action_digest[seat].hexdigest(),opponent_actions_sha256=action_digest[1-seat].hexdigest())
    except Exception:
        row['error'] = traceback.format_exc()
    row['seconds'] = time.monotonic()-started
    return row
