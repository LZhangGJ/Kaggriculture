"""Same PPO budget/seeds/model as original train.py; only fixed F3 dynamics."""
from common import *
import argparse
import torch
from model import Model, update

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    frozen = read(P / 'PROTOCOL.json')['hashes']
    check_hashes(frozen)
    out = P / 'training' / f'f3_r{args.run}'
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'protocol.json', dict(run=args.run, batches=20, games_per_batch=224,
        hashes=frozen, training_seed_start=61000000 + args.run * 100000,
        development_seed_start=62000000, final_seed_start=FINAL_START,
        library_path=str(FIX / 'build/f3.so')))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(701 + args.run)
    device = 'cuda'
    model = Model().to(device)
    assert sum(p.numel() for p in model.parameters()) == 70402
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    pool = make_pool()
    model.export(out / 'step000.bin')
    assert digest(out / 'step000.bin') == digest(OLD / 'training' / f'f3_r{args.run}' / 'step000.bin')
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), out / 'step000.pt')
    history, selection = [], []
    tic = time.perf_counter()
    def dev(step):
        s = evaluate(pool, out / f'dev_{step:03}', out / f'step{step:03}.bin',
                     mode=1, start=62000000, count=16, sample=9301)
        selection.append(dict(step=step, score=(s['overall']['win_rate'], s['overall']['mean_margin'], -step), summary=s))
        save(out / 'selection.json', selection)
    dev(0)
    for step in range(1, 21):
        r = rollout(pool, out / f'step{step-1:03}.bin', mode=2,
                    start=61000000 + args.run * 100000 + (step-1) * 16,
                    count=16, sample=70000 + args.run * 10000 + step)
        if step == 1:
            with torch.no_grad():
                d, v = model(torch.as_tensor(r['global'], device=device),
                             torch.as_tensor(r['candidates'], device=device),
                             torch.as_tensor(r['mask'], device=device))
                prob_error = float(np.max(np.abs(d.probs.cpu().numpy() - r['probability'])))
                value_error = float(np.max(np.abs(v.cpu().numpy() - r['value'])))
            assert prob_error < 2e-6 and value_error < 2e-6
            save(out / 'NUMERICAL_PREFLIGHT.json', dict(status='PASS', probability_max_error=prob_error,
                 value_max_error=value_error, initial_checkpoint_identical=True))
        store_result(out / f'rollout_{step:03}', r)
        torch.cuda.synchronize()
        t = time.perf_counter()
        stats = update(model, opt, r, device, step + args.run * 1000)
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - t
        model.export(out / f'step{step:03}.bin')
        torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step), out / f'step{step:03}.pt')
        record = dict(step=step, training_games=step * 224, rollout=summary(r), ppo=stats,
                      update_seconds=update_seconds, elapsed=time.perf_counter() - tic)
        history.append(record)
        save(out / 'progress.json', dict(status='RUNNING', run=args.run, history=history))
        print('TRAIN_FIXED', args.run, step, 'wins', summary(r)['overall']['win_rate'],
              'rollout_s', round(r['wall_seconds'], 2), 'ppo_s', round(update_seconds, 2), flush=True)
        if step in (5, 10, 20):
            dev(step)
    best = max(selection, key=lambda z: tuple(z['score']))
    save(out / 'best.json', best)
    # No final results are used to select checkpoints or alter training.
    check_hashes(frozen)
    save(out / 'TRAINING_COMPLETE.json', dict(status='PASS', best_step=best['step'],
         training_games=4480, seconds=time.perf_counter()-tic, history=history))
    save(out / 'progress.json', dict(status='COMPLETE', run=args.run, history=history))
    print('TRAINING_DONE', args.run, 'best', best['step'], flush=True)

if __name__ == '__main__':
    main()
