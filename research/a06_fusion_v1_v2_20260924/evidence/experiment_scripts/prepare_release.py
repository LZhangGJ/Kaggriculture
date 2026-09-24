"""Package the exact evaluated agent, evidence and portable replay entry."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    result = json.loads((HERE / 'ACCEPTANCE.json').read_text())
    serial = json.loads((HERE / 'SERIAL_VERIFICATION.json').read_text())
    assert serial['games'] == serial['exact_cash_and_result_matches']
    freeze = json.loads((HERE / 'FINAL_FREEZE.json').read_text())
    source = HERE / 'candidates' / freeze['candidate']
    run_id=freeze.get('run','holdout_v1')
    version=run_id.split('_')[-1].upper()
    config=json.loads((source/'policy/config.json').read_text())
    overlay=json.loads((source/'overlay.json').read_text())
    out = HERE / 'release' / ('A06_DualPanel_Fusion_'+version)
    for name, digest in freeze['files'].items():
        data = (source / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, name
        dest = out / 'agent' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    evidence = out / 'evidence'
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ('ACCEPTANCE.json', 'ACCEPTANCE_ZH.md', 'ACCEPTANCE_ZH.html', 'ACCEPTANCE_EN.md', 'ACCEPTANCE_EN.html', 'PER_OPPONENT.csv', 'FINAL_FREEZE.json', 'SOURCE_AUDIT.json',
                 'SOURCE_POOL.json', 'SEEDS.json', 'SERIAL_VERIFICATION.json', 'DEVELOPMENT_LOG_ZH.md', 'PLAN_ZH.md',
                 'OPENING_AUDIT.json', 'TERMINAL_PARITY.json', 'FAILURE_REVIEW.json', 'FAILURE_REVIEW_ZH.md'):
        shutil.copyfile(HERE / name, evidence / name)
    diagnostics = evidence / 'diagnostics'
    diagnostics.mkdir(exist_ok=True)
    for path in (HERE / 'diagnostics').glob('heldout_*_median_loss.json'):
        shutil.copyfile(path, diagnostics / path.name)
    for name in ('PROTOCOL.json', 'SUMMARY.json'):
        shutil.copyfile(HERE / 'runs' / run_id / name, evidence / name)
    run_names=['screen_v1','development_v1','screen_terminal6']
    for optional in ('development_v2','development_v2_followup','development_v2_dose','development_v2_refine','development_v2_full'):
        if (HERE/'runs'/optional/'SUMMARY.json').is_file():run_names.append(optional)
    if run_id!='holdout_v1':run_names.append('holdout_v1')
    run_names.append(run_id)
    for run in run_names:
        dest = 'holdout_games.jsonl.gz' if run == run_id else run + '_games.jsonl.gz'
        with gzip.open(evidence / dest, 'wb') as f:
            f.write((HERE / 'runs' / run / 'games.jsonl').read_bytes())
    for name in ('V2_PLAN_ZH.md','V2_RESULTS_ZH.md','SEEDS_V2.json','FINAL_FREEZE_V2.json','CROSS_RUN_REPRODUCTION.json',
                 'V2_PROTOCOL_AUDIT.json','V2_DEVELOPMENT_COMPARISON.json','OPENING_ASSET_COMPARISON.json','CANDIDATES.json',
                 'FINAL_REPORT_ZH.md','FINAL_REPORT_ZH.html','FINAL_REPORT_EN.md','FINAL_REPORT_EN.html','FINAL_REVIEW.json'):
        if (HERE/name).is_file():shutil.copyfile(HERE/name,evidence/name)
    if run_id!='holdout_v1':
        shutil.copytree(HERE/'versions/v1',evidence/'versions/v1',dirs_exist_ok=True)
        old=json.loads((HERE/'versions/v1/FINAL_FREEZE.json').read_text())
        for name,digest in old['files'].items():
            data=(HERE/'candidates'/old['candidate']/name).read_bytes()
            assert hashlib.sha256(data).hexdigest()==digest
            dest=out/'agent_v1'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        shutil.copyfile(HERE/'runs/holdout_v1/PROTOCOL.json',evidence/'versions/v1/PROTOCOL.json')
    if (HERE/'PAIRED_EXTERNAL.json').is_file():
        for name in ('PAIRED_EXTERNAL.json','PAIRED_EXTERNAL_ZH.md'):
            shutil.copyfile(HERE/name,evidence/name)
        with gzip.open(evidence/'paired_external_v1_on_v2_games.jsonl.gz','wb') as f:
            f.write((HERE/'runs/paired_external_v1_on_v2/games.jsonl').read_bytes())
        shutil.copyfile(HERE/'runs/paired_external_v1_on_v2/PROTOCOL.json',evidence/'PAIRED_PROTOCOL.json')
    host = ROOT / 'dp/AFS_R2_DP_Fusion_R3_Experimental_Delivery_ver b/Kaggriculture_Fusion_R2_20260916/verification'
    referee = out / 'referee'
    referee.mkdir(exist_ok=True)
    shutil.copyfile(host / 'policy_host.py', referee / 'policy_host.py')
    for name in ('cpu_runtime.py', 'official/kaggriculture.py', 'official/kaggriculture.json', 'official/seed_utils.py', 'official/LICENSE'):
        dest = referee / 'referee' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(host / 'referee' / name, dest)
    shutil.copyfile(HERE / 'reproduce_release.py', out / 'reproduce_one.py')
    scripts = evidence / 'experiment_scripts'
    scripts.mkdir(exist_ok=True)
    for path in HERE.glob('*.py'):
        shutil.copyfile(path, scripts / path.name)
    inside, outside = (result['groups'][x] for x in ('internal', 'external'))
    text = f'''# A06 Dual Panel Fusion {version}

冻结候选：`{freeze['candidate']}`。正式验收：**{result['status']}**。

| 面板 | 严格胜率 | 胜 / 局数 |
|---|---:|---:|
| 内部原 13 改良版 | {inside['strict_win_rate']:.2%} | {inside['wins']} / {inside['games']} |
| 公开 10 条冻结入口 | {outside['strict_win_rate']:.2%} | {outside['wins']} / {outside['games']} |

50 个未见种子、双座位；严格胜率只计胜局。逐对手、置信区间和去重口径见 `evidence/ACCEPTANCE_ZH.html`。不是逐对手 90% 保证，也不是 Kaggle 沙箱或线上天梯认证。外战池为 2026-09-23/24 冻结快照。

## 改了什么

1. 保留 Cashflow 的产出回款估值：田间正产出按一半当日、一半次日估值；不是读取真实未来。
2. 保留其根据对手公开成熟产出进行出售的逻辑。
3. 加入 Liquidity 开局小麦买 {overlay['opening_liquidity']} / 卖 {overlay['opening_liquidity']} 意图，有现金、仓储与订单数约束。
4. 关闭日内新增项目准入/采购；日初经济规划、既有任务动态执行仍保留。
5. 雇工上限为 {config['max_hands']}。

完整参数变化见 `evidence/SOURCE_AUDIT.json`。当前 `animal_bias={config['animal_bias']}`、`batch_delivery={config['batch_delivery']}`。

不使用 ML/RL、测试种子、对手名称/文件或对手私有状态。`SOURCE_AUDIT.json` 证明 64 个文件与 Cashflow 原包一致；原生 C++ 库未修改。

## 运行

入口：`agent/main.py` 的 `agent(observation, configuration)`。独立本地场次可使用 `create_agent()` 并在结束时 `close()`。
包含的 `.so` 适用于兼容 Linux x86-64 / WSL，不是 Windows DLL。沿用父版构建，需 GLIBC_2.32、GLIBCXX_3.4.31 或兼容运行库。`agent/build.py` 可从 C++20 源码重编译；改编译器后必须重新核验动作和现金。

串行核验 {serial['games']} 局，现金及胜负全部与正式并发测试一致；本机最大单次调用 {serial['max_serial_action_s']:.3f} 秒，不代表 Kaggle 平台认证。

## 复现某一局

对手沿用已共享仓库的 `research/a06_dynamic_variants_public10_20260924`，不重复装入本轻量包。仓库分支 `research/a06-dynamic-variants-public10-20260924`，提交 `c00bcb4c5ed0cb43d750ad5a684e9f1ae318f80a`。

```text
python reproduce_one.py --pool /path/to/research/a06_dynamic_variants_public10_20260924 --opponent internal/r14_cashflow --seat 0
```

脚本核验对手文件哈希，使用本包官方 Python 规则；默认复跑第一个验收种子并自动比对双方终局现金。外部示例 `--opponent external/n69_hosen42`。完整逐局回执保存在 `evidence/holdout_games.jsonl.gz`。

`agent/README.md`、`agent/reports/`、`agent/RELEASE_MANIFEST.json` 是原 Cashflow 的历史文件，保留用于来源审计，不是本融合版成绩或新清单。本包以此 README、`evidence/FINAL_FREEZE.json`、`evidence/ACCEPTANCE.json` 和根目录 `MANIFEST.json` 为准。
'''
    (out / 'README_ZH.md').write_text(text, encoding='utf-8')
    english=f'''# A06 Dual Panel Fusion {version}

Candidate: `{freeze['candidate']}`. Acceptance: **{result['status']}**.

- Internal frozen 13: {inside['wins']}/{inside['games']} strict wins ({inside['strict_win_rate']:.2%}).
- External frozen 10: {outside['wins']}/{outside['games']} strict wins ({outside['strict_win_rate']:.2%}).
- 50 new seeds, both seats; draws are not wins. These are panel averages, not per-opponent guarantees or Kaggle sandbox certification.

Open `evidence/ACCEPTANCE_EN.html` for the English report, or `evidence/ACCEPTANCE_ZH.html` for Chinese. `evidence/ACCEPTANCE_EN.md` includes the changes, remaining weaknesses, exact reproduction command and runtime requirements.

Entry: `agent/main.py`, callable `agent(observation, configuration)` or local `create_agent()` with explicit `close()`.

Use compatible Linux x86-64 / WSL. The inherited `.so` needs compatible GLIBC_2.32 and GLIBCXX_3.4.31 libraries; C++20 source and `agent/build.py` are included for rebuilding. `MANIFEST.json` hashes all package files. Parent README/reports inside `agent/` are historical, not the current fusion result.

```text
python reproduce_one.py --pool /path/to/research/a06_dynamic_variants_public10_20260924 --opponent internal/r14_cashflow --seat 0
```

The opponents are in the previously shared branch `research/a06-dynamic-variants-public10-20260924`, commit `c00bcb4c5ed0cb43d750ad5a684e9f1ae318f80a`. The script verifies source hashes and exact recorded cash. Runtime decisions use no test seed, opponent code/identity or hidden rival state; there is no ML/RL.
'''
    (out/'README_EN.md').write_text(english,encoding='utf-8')
    if (HERE/'FINAL_REVIEW.json').is_file():
        decision=json.loads((HERE/'FINAL_REVIEW.json').read_text())
        if decision['retained_baseline']=='cf_liq_h12_nointraday':
            zh_note='> **结论：双 90% 目标未达成。保留 agent_v1/main.py 作为基线；agent/main.py 为未晋级的第二轮对照版本。先阅读 evidence/FINAL_REPORT_ZH.html。**\n\n'
            en_note='> **The dual 90% target was not met. Retain agent_v1/main.py as the baseline; agent/main.py is the unpromoted V2 comparison candidate. Start with evidence/FINAL_REPORT_EN.html.**\n\n'
            (out/'README_ZH.md').write_text(zh_note+(out/'README_ZH.md').read_text(encoding='utf-8'),encoding='utf-8')
            (out/'README_EN.md').write_text(en_note+(out/'README_EN.md').read_text(encoding='utf-8'),encoding='utf-8')
    if run_id!='holdout_v1':
        with (out/'README_ZH.md').open('a',encoding='utf-8') as f:
            f.write('\n本包也保留第一版源码和运行库 `agent_v1/main.py`，对应 `evidence/versions/v1` 与第一批验收回执。复现脚本加 `--candidate-version v1` 可切到第一版。新旧版同种子外战对照见 `evidence/PAIRED_EXTERNAL_ZH.md`，不要把两批不同种子的百分比直接相减来归因。\n')
        with (out/'README_EN.md').open('a',encoding='utf-8') as f:
            f.write('\nThe frozen V1 source/runtime is also included at `agent_v1/main.py`, with its original acceptance evidence under `evidence/versions/v1`. Use `--candidate-version v1` in reproduce_one.py. See `evidence/PAIRED_EXTERNAL.json` for the matched-seed external comparison; raw percentages from different seed panels are not a paired effect estimate.\n')
    manifest = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'
                and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = out.parent / ('A06_DualPanel_Fusion_'+version+'.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted([*manifest, 'MANIFEST.json']):
            z.write(out / name, out.name + '/' + name)
    with zipfile.ZipFile(archive) as z:
        for name, digest in manifest.items():
            assert hashlib.sha256(z.read(out.name + '/' + name)).hexdigest() == digest
    receipt = dict(path=str(archive), bytes=archive.stat().st_size, files=len(manifest)+1,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), contents_verified=True)
    (HERE / 'RELEASE_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
