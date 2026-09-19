"""Learn new market patterns while rehearsing every previously introduced rare class."""
import hashlib
import json
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pattern_masks(codes, initial_codes, reference_codes):
    import numpy as np
    return ~np.isin(codes, initial_codes), ~np.isin(codes, reference_codes)


def main():
    from cpu_budget import CpuBudget
    guard = CpuBudget(ROOT / 'training_cpu')
    guard.wait()
    import numpy as np
    import torch
    from common import SOURCE, arrays, batch, canonical_unit_history, save, load_checkpoint, loss_and_counts
    from model_joint_numeric import Policy

    plan = json.loads((ROOT / 'PLAN.json').read_text(encoding='utf8'))
    cfg = plan['training']
    assert digest(ROOT / 'initial.pt') == plan['initial_sha256']
    for name, expected in plan['frozen_code'].items():
        assert digest(ROOT / 'frozen_code' / name) == expected
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.30)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    initial = torch.load(ROOT / 'initial.pt', map_location='cpu', weights_only=False)
    initial_state = dict(initial['model'])
    quantity_keys = {'market_'+k for k in initial_state if k.startswith('quantity_head.')}
    missing_quantity = quantity_keys-set(initial_state)
    assert not missing_quantity or missing_quantity == quantity_keys
    if missing_quantity:
        initial_state.update({'market_'+k: v for k, v in list(initial_state.items()) if k.startswith('quantity_head.')})
    assert digest(plan['rare_pattern_reference']) == plan['rare_pattern_reference_sha256']
    reference = torch.load(plan['rare_pattern_reference'], map_location='cpu', weights_only=False)
    reference_codes = reference['model']['pattern_codes'].numpy().copy()
    assert set(reference_codes) <= set(initial['model']['pattern_codes'].tolist())
    del reference
    # Sequential reads avoid cold random page faults; data and sampling are unchanged.
    print(json.dumps(dict(phase='WARMING_TRAINING_FILE_CACHE')), flush=True)
    for folder in (SOURCE / 'data/packed/train', SOURCE / 'data/numeric/train'):
        for path in sorted(folder.glob('*.npy')):
            with path.open('rb') as stream:
                while True:
                    guard.wait()
                    if not stream.read(4*1024*1024):
                        break
    groups = [arrays('train')]
    validation = arrays('validation')
    supplemental = [Path(x) for x in plan['supplemental_folders']] + [ROOT / 'data/fresh_packed']
    audit = json.loads((SOURCE / 'manifests/SUPPLEMENTAL_TRAINING_AUDIT.json').read_text(encoding='utf8'))
    assert audit['status'] == 'PASS' and audit['heldout_splits_disjoint'] and audit['seed_groups_disjoint']
    sources = [dict(name='original', samples=len(groups[0]['step']))]
    validation_seeds = {x['seed'] for x in json.loads((SOURCE / 'data/packed/validation/episodes.json').read_text(encoding='utf8'))}
    evaluation_seeds = {job[0] for panel in plan['development_panels'] for job in panel} | {job[0] for job in plan['final_jobs']}
    training_seeds = {x['seed'] for x in json.loads((SOURCE / 'data/packed/train/episodes.json').read_text(encoding='utf8'))}
    assert not training_seeds & (validation_seeds | evaluation_seeds)
    for folder in supplemental:
        ready = json.loads((folder / 'READY.json').read_text(encoding='utf8'))
        assert ready['status'] == 'PASS' and ready['quantities'] == initial['quantities'] == plan['quantities']
        assert ready['shared_numeric_features_sha256'] == digest(ROOT / 'frozen_code/numeric_features.py')
        assert digest(folder / 'episodes.json') == ready['episodes_sha256']
        for filename, expected in ready['arrays_sha256'].items():
            guard.wait()
            assert digest(folder / filename) == expected, filename
        episodes = json.loads((folder / 'episodes.json').read_text(encoding='utf8'))
        seeds = {x['seed'] for x in episodes}
        assert not seeds & (training_seeds | validation_seeds | evaluation_seeds)
        training_seeds.update(seeds)
        group = {p.stem: np.load(p, mmap_mode='r') for p in folder.glob('*.npy')}
        assert set(group) == set(groups[0]) and len({len(a) for a in group.values()}) == 1
        for key, array in group.items():
            assert array.shape[1:] == groups[0][key].shape[1:] and array.dtype == groups[0][key].dtype
        groups.append(group)
        sources.append(dict(name=folder.name, samples=len(group['step']), seeds=len(seeds),
                            ready_sha256=digest(folder / 'READY.json')))

    # The vocabulary grows only from training labels. Preserve every old class.
    powers = np.array([31**i for i in reversed(range(11))], dtype=np.int64)
    old_codes = initial['model']['pattern_codes'].numpy().astype(np.int64, copy=False)
    code_parts = [old_codes]
    rare, new_examples = [], []
    for group_id, group in enumerate(groups):
        for start in range(0, len(group['step']), 16384):
            guard.wait()
            codes = (group['market_y'][start:start+16384, :, 0].astype(np.int64)*powers).sum(1)
            code_parts.append(np.unique(codes))
            new_mask, rare_mask = pattern_masks(codes, old_codes, reference_codes)
            new_examples.extend((group_id, int(i)) for i in np.flatnonzero(new_mask)+start)
            rare.extend((group_id, int(i)) for i in np.flatnonzero(rare_mask)+start)
    codes = np.unique(np.concatenate(code_parts))
    patterns = (codes[:, None]//powers[None]) % 31
    assert len(patterns) > len(old_codes) and new_examples
    assert set(new_examples) <= set(rare) and len(rare) > len(new_examples)
    assert not set(evaluation_seeds) & training_seeds

    # Warm-start mapping reused from source code/train_joint_expanded.py.
    def initialize():
        model = Policy(patterns.tolist(), **initial['config']).cuda()
        expanded = {'patterns', 'pattern_codes', 'pattern_head.2.weight', 'pattern_head.2.bias', 'pattern_embedding.weight'}
        loaded = model.load_state_dict({k: v for k, v in initial_state.items() if k not in expanded}, strict=False)
        assert not loaded.unexpected_keys and set(loaded.missing_keys) == expanded
        mapped = torch.searchsorted(model.pattern_codes, initial['model']['pattern_codes'].cuda())
        assert torch.equal(model.patterns[mapped], initial['model']['patterns'].cuda())
        with torch.no_grad():
            model.pattern_head[2].weight.zero_()
            model.pattern_head[2].bias.fill_(float(initial['model']['pattern_head.2.bias'].min())-4)
            model.pattern_head[2].weight[mapped] = initial['model']['pattern_head.2.weight'].cuda()
            model.pattern_head[2].bias[mapped] = initial['model']['pattern_head.2.bias'].cuda()
            model.pattern_embedding.weight[mapped] = initial['model']['pattern_embedding.weight'][:-1].cuda()
            model.pattern_embedding.weight[-1] = initial['model']['pattern_embedding.weight'][-1].cuda()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(('pattern_head.', 'pattern_embedding.', 'market_quantity_head.')))
        model.eval()  # Frozen encoder remains deterministic; gradients still train the market heads.
        return model, mapped

    def mixed_batch():
        selections = [list(rng.integers(len(g['step']), size=int(n))) for g, n in
                      zip(groups, rng.multinomial(cfg['batch'], cfg['group_probabilities']))]
        # Rehearse all introduced classes, so previous rare patterns remain represented.
        for group_id, ids in enumerate(selections):
            for offset in range(len(ids)):
                if rng.random() < cfg['rare_pattern_probability']:
                    ids[offset] = None
        selected = [[i for i in ids if i is not None] for ids in selections]
        missing = cfg['batch']-sum(map(len, selected))
        for index in rng.integers(len(rare), size=missing):
            group_id, sample_id = rare[index]
            selected[group_id].append(sample_id)
        pieces = []
        for group, ids in zip(groups, selected):
            if not ids:
                continue
            ids = np.asarray(ids, dtype=np.int64)
            b = batch(group, ids, 'cuda')
            chosen = np.flatnonzero((group['step'][ids] > 0) &
                                   (rng.random(len(ids)) < cfg['history_denoising_probability']))
            if len(chosen):
                previous = batch(group, ids[chosen]-1, 'cuda')
                assert (previous['step']+1 == b['step'][chosen]).all()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    u, m = model.predict(previous)
                    u = canonical_unit_history(u, previous)
                b['previous_u'][chosen] = u.to(b['previous_u'].dtype)
                b['previous_m'][chosen] = m.to(b['previous_m'].dtype)
            pieces.append(b)
        return {key: torch.cat([b[key] for b in pieces]) for key in pieces[0]}

    def market_loss(net, b):
        _, logits, _, quantities, _ = net(b)
        market = b['market_y'].long()
        ids = net.pattern_ids(market[:, :, 0])
        assert (ids < len(patterns)).all()
        mask = net.mq[market[:, :, 0]] & (market[:, :, 0] != 0)
        loss = torch.nn.functional.cross_entropy(logits, ids)
        if mask.any():
            loss = loss + torch.nn.functional.cross_entropy(quantities[mask], market[:, :, 1][mask])
        return loss

    run = ROOT / 'run'
    run.mkdir(exist_ok=True)
    model, mapped = initialize()
    trainable = [p for p in model.parameters() if p.requires_grad]
    protocol = dict(plan=plan, sources=sources, old_patterns=len(old_codes), patterns=len(patterns),
                    new_class_training_examples=len(new_examples), rare_training_examples=len(rare),
                    retained_rare_examples=len(rare)-len(new_examples), rare_reference_patterns=len(reference_codes),
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in trainable), training_seeds=len(training_seeds),
                    vocabulary_source='Training labels only, never validation or evaluation',
                    frozen_unit_controller=True, teacher_runtime_fallback=False)
    save(run / 'TRAINING_PROTOCOL.json', protocol)
    save(run / 'PATTERNS.json', patterns.tolist())

    if not (run / 'CHECKS.json').exists():
        probe = batch(groups[0], np.array([0, 1, 22, 23, 24, 80, 200, 718]), 'cuda')
        base, _, _ = load_checkpoint(ROOT / 'initial.pt', 'cuda')
        with torch.no_grad():
            expected = base(probe)
            actual = list(model(probe))
            actual[1] = actual[1][:, mapped]
            delta = max((a-b).abs().max().item() for a, b in zip(expected, actual))
            assert delta < 1e-4
            predicted = model.predict(probe)
            assert all(torch.equal(a, b) for a, b in zip(predicted, base.predict(probe)))
            changed = dict(probe, unit_y=torch.zeros_like(probe['unit_y']), market_y=torch.zeros_like(probe['market_y']))
            assert all(torch.equal(a, b) for a, b in zip(predicted, model.predict(changed)))
        del base
        rare_probe = {key: torch.cat([batch(groups[g], np.array([i]), 'cuda')[key]
                     for g, i in new_examples[:16]]) for key in groups[0]}
        new_rows = torch.ones(len(patterns), dtype=torch.bool, device='cuda')
        new_rows[mapped] = False
        test_optimizer = torch.optim.AdamW(trainable, lr=1e-3)
        before = market_loss(model, rare_probe).item()
        gradient = quantity_gradient = 0.
        for step in range(30):
            guard.wait()
            test_optimizer.zero_grad(set_to_none=True)
            loss = market_loss(model, rare_probe)
            loss.backward()
            if step == 0:
                gradient = model.pattern_head[2].weight.grad[new_rows].norm().item()
                assert math.isfinite(gradient) and gradient > 0
                quantity_gradient = model.market_quantity_head[2].weight.grad.norm().item()
                assert math.isfinite(quantity_gradient) and quantity_gradient > 0
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            test_optimizer.step()
        after = market_loss(model, rare_probe).item()
        assert after < before*.8, (before, after)
        with torch.no_grad():
            assert torch.equal(model.predict(probe)[0], predicted[0]), 'Frozen unit controller changed'
        frozen_unchanged = all(torch.equal(v.cpu(), initial['model'][k]) for k, v in model.state_dict().items()
            if k not in {'patterns', 'pattern_codes'} and not k.startswith(('pattern_head.', 'pattern_embedding.', 'market_quantity_head.')))
        assert frozen_unchanged
        save(run / 'CHECKS.json', dict(status='PASS', initial_logit_max_error=delta,
            initial_action_parity=8, prediction_independent_of_labels=True, new_class_gradient=gradient,
            rare_training_loss_before=before, rare_training_loss_after=after, new_class_learning_steps=30,
            market_quantity_gradient=quantity_gradient, frozen_unit_and_backbone_unchanged=True, seed_splits_disjoint=True))
        del test_optimizer, model
        torch.manual_seed(cfg['seed'])
        model, mapped = initialize()
        trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=model.pattern_head.parameters(), base_lr=cfg['learning_rate']),
        dict(params=model.pattern_embedding.parameters(), base_lr=cfg['embedding_learning_rate']),
        dict(params=model.market_quantity_head.parameters(), base_lr=cfg['quantity_learning_rate'])], weight_decay=.001, fused=True)
    indices = np.load(SOURCE / 'runs/bc_single_teacher_001/validation_indices.npy')

    def evaluate():
        counts = np.zeros(6, np.int64)
        total = 0.
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, len(indices), cfg['batch']):
                guard.wait()
                b = batch(validation, indices[start:start+cfg['batch']], 'cuda')
                value, c = loss_and_counts(model, b, model(b))
                total += value.item()*len(b['step'])
                counts += c.cpu().numpy()
        metric = dict(loss=total/len(indices), full_action_exact=float(counts[4]/counts[5]),
                      unit_exact=float(counts[0]/counts[1]), market_exact=float(counts[2]/counts[3]))
        assert all(math.isfinite(x) for x in metric.values())
        return metric

    start_step = 0
    selection = dict(best_step=0, best_metric=None, bad_checks=0)
    if (run / 'latest.pt').exists():
        ck = torch.load(run / 'latest.pt', map_location='cuda', weights_only=False)
        assert ck['training_protocol'] == protocol
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        selection = ck['selection']
        start_step = ck['expansion_step']
        rng.bit_generator.state = ck['numpy_rng']
        torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in ck['cuda_rng']])
        baseline = ck['baseline']
    else:
        baseline = evaluate()
        save(run / 'INITIAL_VALIDATION.json', baseline)
        print(json.dumps(dict(phase='INITIAL_VALIDATION', patterns=len(patterns), **baseline)), flush=True)

    def checkpoint(step, best):
        ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=initial['config'],
                  quantities=initial['quantities'], patterns=patterns.tolist(),
                  global_step=int(initial['global_step'])+step, expansion_step=step,
                  training_protocol=protocol, protocol=initial['protocol'], baseline=baseline, selection=selection,
                  numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        for name in (['latest.pt', 'best.pt'] if best else ['latest.pt']):
            temp = run / (name+'.tmp')
            torch.save(ck, temp)
            temp.replace(run / name)

    started = time.monotonic()
    completed = start_step
    accumulated = 0.
    for step in range(start_step+1, cfg['steps']+1):
        if selection['bad_checks'] >= cfg['patience']:
            break
        guard.wait()
        b = mixed_batch()
        schedule = min(1., step/200)*(.2+.8*.5*(1+math.cos(math.pi*step/cfg['steps'])))
        for group in optimizer.param_groups:
            group['lr'] = group['base_lr']*schedule
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = market_loss(model, b)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(torch.nn.utils.clip_grad_norm_(trainable, 1.))
        optimizer.step()
        accumulated += loss.detach().item()
        completed = step
        if step % 25 == 0:
            status = dict(status='TRAINING', pid=os.getpid(), step=step, maximum_steps=cfg['steps'],
                patterns=len(patterns), best_step=selection['best_step'], bad_checks=selection['bad_checks'],
                mean_market_loss=accumulated/(step-start_step), seconds=time.monotonic()-started,
                heartbeat_unix=time.time(), cpu_wait_seconds=guard.wait_seconds)
            save(run / 'STATUS.json', status)
            print(json.dumps(status), flush=True)
        if step % cfg['validate_every'] == 0 or step == cfg['steps']:
            metric = evaluate()
            improved = selection['best_metric'] is None or metric['loss'] < selection['best_metric']['loss']-cfg['minimum_improvement']
            if improved:
                selection = dict(best_step=step, best_metric=metric, bad_checks=0)
            else:
                selection['bad_checks'] += 1
            checkpoint(step, improved)
            with (run / 'metrics.jsonl').open('a', encoding='utf8') as stream:
                stream.write(json.dumps(dict(step=step, improved=improved, **metric))+'\n')
            print(json.dumps(dict(phase='VALIDATION', step=step, improved=improved, **metric)), flush=True)
    assert selection['best_step'] > 0
    best = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, initial['model'][key]) for key, value in best['model'].items()
        if key not in {'patterns', 'pattern_codes'} and not key.startswith(('pattern_head.', 'pattern_embedding.', 'market_quantity_head.')))
    done = dict(status='COMPLETE', steps=completed, best_step=selection['best_step'], patterns=len(patterns),
        old_patterns=len(old_codes), baseline=baseline, selection=selection,
        passes_regression_guard=selection['best_metric']['full_action_exact'] >= baseline['full_action_exact']-cfg['maximum_action_accuracy_drop'],
        frozen_unit_and_backbone_unchanged=True, heartbeat_unix=time.time())
    save(run / 'DONE.json', done)
    save(run / 'STATUS.json', done)
    print(json.dumps(done), flush=True)


if __name__ == '__main__':
    main()
