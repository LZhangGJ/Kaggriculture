import sys
from pathlib import Path
import argparse
sys.path.insert(0,str(Path(__file__).parent/'source'))
sys.path.insert(1,str(Path(__file__).parent.parent/'arena-eval'))
import broad_controller
import dispatcher
import importlib.util
# Arena execution keeps its existing hash-pinned CPU runtime. New PPO/controller
# modules are not dependencies of official greedy inference.
eval_source=Path(__file__).parent.parent/'league-v2-stage/bc-review2/ppo/league_eval.py'
spec=importlib.util.spec_from_file_location('broad_frozen_eval',eval_source)
frozen_eval=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen_eval)
dispatcher.core=frozen_eval
def runtime_identity(arena):
    from ppo.league_runtime import digest,PROTOCOL
    from ppo.league import sha
    root=Path(__file__).parent
    return digest(dict(evaluation=frozen_eval.runtime_identity(arena),controller=sha(root/'broad_controller.py'),wrapper=sha(__file__),curriculum=sha(root/'source/ppo/broad_curriculum.py'),dispatcher=sha(dispatcher.__file__),protocol=PROTOCOL))
broad_controller.runtime_identity=runtime_identity
broad_controller.evaluate=dispatcher.evaluate
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['root','training','manifest','panel','arena']:p.add_argument('--'+name,required=True)
    broad_controller.Controller(**vars(p.parse_args())).run()
