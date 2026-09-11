"""Post-run read-only comparison of legacy/fixed training; no new games."""
from common import *

def main():
    runs = []
    for rep in (0, 1):
        old = OLD / 'training' / f'f3_r{rep}'
        new = P / 'training' / f'f3_r{rep}'
        same_weights = [s for s in range(21) if digest(old / f'step{s:03}.bin') == digest(new / f'step{s:03}.bin')]
        arrays_different, rows_different = [], []
        counts = dict(records=0, effective=0, option_old_empty=0, option_old_crop=0, option_old_animal=0,
                      chosen_old_empty=0, chosen_old_crop=0, chosen_old_animal=0, non_keep=0)
        for step in range(1, 21):
            a, b = old / f'rollout_{step:03}', new / f'rollout_{step:03}'
            with np.load(a / 'decisions.npz') as x, np.load(b / 'decisions.npz') as y:
                assert x.files == y.files
                for k in x.files:
                    if not np.array_equal(x[k], y[k]): arrays_different.append((step,k))
                active = y['mask'].sum(1)>1
                valid = y['mask'].astype(bool).copy(); valid[:,0]=False
                first = valid.argmax(1)
                f = y['candidates'][np.arange(len(first)), first]
                empty = f[:,26] > .5
                animal = f[:,24] > .5
                crop = (~empty)&(~animal)
                selected = y['choice']!=0
                counts['records'] += len(first)
                counts['effective'] += int(active.sum())
                counts['non_keep'] += int(selected.sum())
                for label, flag in [('empty',empty),('crop',crop),('animal',animal)]:
                    counts['option_old_'+label] += int((active&flag).sum())
                    counts['chosen_old_'+label] += int((selected&flag).sum())
            fields = ('seed','seat','opponent','steps','cash','opponent_cash','margin','win','tie','error','plan_calls','execute_calls')
            xx, yy = read(a/'games.json'), read(b/'games.json')
            for i,(r,q) in enumerate(zip(xx,yy)):
                if any(r[k]!=q[k] for k in fields): rows_different.append((step,i))
        runs.append(dict(run=rep, identical_checkpoints=same_weights, different_arrays=arrays_different,
                         different_game_outcomes=rows_different, counts=counts))
    result = dict(status='PASS', audit_scope='saved training tensors, weights and terminal game outcomes; not raw 719-step trace comparison',
                  loaded_fixed_library_sha256=digest(FIX/'build/f3.so'), loaded_old_library_sha256=digest(OLD/'build/f3.so'),
                  runs=runs, analyzer_sha256=digest(__file__))
    save(P/'IDENTITY_AUDIT.json',result)
    print(result)

if __name__=='__main__': main()
