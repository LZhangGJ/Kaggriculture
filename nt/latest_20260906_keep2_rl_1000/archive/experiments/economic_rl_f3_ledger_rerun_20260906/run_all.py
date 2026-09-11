from common import *
import subprocess

def main():
    if (P / 'PROTOCOL.json').exists():
        raise FileExistsError('Experiment already started; do not overwrite')
    hashes = frozen_inputs()
    save(P / 'PROTOCOL.json', dict(status='FROZEN', hashes=hashes,
        final_seed_start=FINAL_START, final_seed_count=100, training_games=8960,
        primary='development-selected greedy versus paired KEEP; both training repeats',
        secondary='step020 greedy and sampled; old weights on both kernels',
        bootstrap_unit='seed with seven opponents and both seats together'))
    (P / 'logs').mkdir(exist_ok=False)
    tic = time.perf_counter()
    for rep in (0, 1):
        with (P / 'logs' / f'train_r{rep}.log').open('w') as log:
            result = subprocess.run([sys.executable, str(P / 'train_fixed.py'), '--run', str(rep)],
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Training {rep} failed; inspect log')
        print('COMPLETE_TRAIN', rep, read(P / 'training' / f'f3_r{rep}' / 'best.json')['step'], flush=True)
    # All checkpoint selection is frozen before opening final results.
    check_hashes(hashes)
    pool = make_pool()
    base = P / 'final'
    evaluate(pool, base / 'keep_old', library='old')
    evaluate(pool, base / 'keep_fixed')
    evaluate(pool, base / 'initial_sample_fixed', OLD / 'training/f3_r0/step000.bin', mode=2)
    for rep in (0, 1):
        oldrun = OLD / 'training' / f'f3_r{rep}'
        newrun = P / 'training' / f'f3_r{rep}'
        best = read(newrun / 'best.json')['step']
        evaluate(pool, base / f'new_r{rep}_selected_greedy', newrun / f'step{best:03}.bin', mode=1)
        evaluate(pool, base / f'new_r{rep}_last_greedy', newrun / 'step020.bin', mode=1)
        evaluate(pool, base / f'new_r{rep}_last_sample', newrun / 'step020.bin', mode=2)
        evaluate(pool, base / f'old_r{rep}_last_greedy', oldrun / 'step020.bin', mode=1, library='old')
        evaluate(pool, base / f'old_r{rep}_last_sample', oldrun / 'step020.bin', mode=2, library='old')
        evaluate(pool, base / f'old_r{rep}_fixed_sample', oldrun / 'step020.bin', mode=2)
    check_hashes(hashes)
    save(P / 'RUN_COMPLETE.json', dict(status='PASS', seconds=time.perf_counter()-tic,
                                     training_games=8960, final_evaluation_games=21000))
    subprocess.run([sys.executable, str(P / 'report.py')], check=True)

if __name__ == '__main__':
    main()
