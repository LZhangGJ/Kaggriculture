import contextlib, inspect, json, os, random, runpy, sys
class Obj(dict):
    def __getattr__(self, key):
        try: return self[key]
        except KeyError: raise AttributeError(key)
def obj(x):
    if isinstance(x, dict): return Obj({k: obj(v) for k,v in x.items()})
    if isinstance(x, list): return [obj(v) for v in x]
    return x
random.seed(int(os.environ.get('ARENA_AGENT_SEED', '0')))
with contextlib.redirect_stdout(sys.stderr):
    scope = runpy.run_path('main.py', run_name='arena_policy')
    policy = scope.get('agent') or scope.get('my_agent')
    if not callable(policy): raise RuntimeError('No agent function in main.py')
    sig = inspect.signature(policy)
    accepts_config = len(sig.parameters) >= 2
for line in sys.stdin:
    request = json.loads(line)
    with contextlib.redirect_stdout(sys.stderr):
        observation = obj(request['observation'])
        action = policy(observation, obj(request['configuration'])) if accepts_config else policy(observation)
    print(json.dumps(action, allow_nan=False), flush=True)
