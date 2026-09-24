# 工作入口

先读 `README.md`、`HANDOFF_ZH.md`，以及本轮的分析与诊断记录 `docs/ECONOMIC_MODEL_ZH.md`。
本工程唯一主线是：高手 replay 离线提取/聚类 → 路线互打 → 147 维状态上的浅树切换 replay 路线 →
中期接管无开局模板的 JointAFS R1。

## 当前三地 RL 续训快照 v682（`handoff/rl-student-economic-v682-20260924`）

- 这是本机三地连续训练**完整 v682 轮后停止点**的隔离 GitHub 分支；下轮从 v683 开始，不是 v678 或先前 Kaggle v463。冻结 `models/student-v682/{actor.pt,actor.bin,manifest.json,metrics.json}`，不含 rollout NPZ/BC mmap/原始 replay/本机 aarch64 `.so`。v682 `.pt` 含模型与 `rl_optimizer` 28 个 AdamW 状态，维度 `2233/3145/374`、scale=3；`.bin` 是同 checkpoint 的 C++ 权重。SHA-256：`.pt` `2c7fd47490de9b4b8d6fc0cc1a47e7a2b4b95f1c8851cb8e923c9164ec7f9c4b`、`.bin` `8979d2f529a2ec1db43a49139a7459806a36fe89826b6cf71a88e70dfa66cf0d`、manifest `25e12c5bafc1f2aca509dfc8dfd1d6d6678042b94d06d8294cc8724c0591dc87`。v682 metrics `PASS`，`1370/1536` 是**v681 输入策略**的同池 rollout，不是 v682 独立评测；其 `checkpoint_out_sha256` 验证 v682 `.pt`，`.bin` 已从该 `.pt` 独立重新导出并匹配 SHA。
- 行为契约：G275/旧浅树 + funded replay 到 step288，之后 17 个日初逐格**采样**，`student_intraday=0`；训练池 Thomas/Meta/Fieldcraft 各 1/3。原生训练所用 student bridge SHA `ae666af83c9639ca54160e35019e8823365dfc807ebe40025d1ef0d5d6c182de`，funded JobBatch SHA `019d24a32b157b4f63e07fabb1f11b00ed8a1f82b48148eb0d9f6dac31e4c79e`。本分支源码含小型对手资产；换架构必须重建全部 `.so`，并核对首次 fresh rollout 的 `native_job_inputs_sha256`、old-policy replay、非法/fallback，而不能复制本机二进制或仅凭 checkpoint 可加载就称逐步 exact。不要启用四地 lifecycle/shop gate/旧 `sale_dp=1`；交易博弈 DP 未接入本快照，也未获 Kaggle 上传许可。
- **源码复建边界**：由本分支干净源码重新编译 funded JobBatch 的 SHA 不等于上述本机训练二进制；在 v678、五手 × 8 seed × 双座的 80 局中，重建版与本机版的每局终局现金、actor hash、719 帧状态一致，但 `prefix_action_hash` 80/80 不同。逐步对拍定位到 step20 重建版多发一笔未成交的 `BUY_SEED WHEAT 1`，故**不能声称前缀动作严格复现**；更早 d897 源码复建也未消除该差异。v682 `.pt`/`.bin` 本身精确同轮、可从 v683 新 seed 做 on-policy 续训，但这属于有明确前缀实现差异的续训分叉，不能把新主机的训练轨迹冒称原二进制逐步 exact。首次训练先看前缀订单是否在新 seed 变成真实成交，并保留独立评测。
- 评测独立于训练：先前 v619 在五个 C++ 强手 ×128 seed×双座的 dev 采样评测为 `1123/1280`，不可冒充 v682 的成绩。下例 seed `3300200000` 须与队友自己的训练/保留盲测段再次核对互斥；本机停止后的训练不会自动同步。

```bash
(cd fast_kaggriculture && python setup.py build_ext --inplace)
bash experiments/native_student_rollout/build_student_bridge.sh work/agent-student-actor-owned-v3.so
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/thomas_2945_cpp/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/metav4_2965/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/fieldcraft_2887/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/salemali7_2900/build.sh
PYTHON_BIN="$(command -v python)" BUILD_DIR=experiments/native_student_rollout/build-economic-funded bash experiments/native_student_rollout/build.sh
PYTHONPATH=. python -u experiments/run_student_rl_continuous.py \
  --start-round 683 --checkpoint models/student-v682/actor.pt \
  --weights models/student-v682/actor.bin --manifest models/student-v682/manifest.json \
  --binary work/agent-student-actor-owned-v3.so \
  --native-job-module-dir experiments/native_student_rollout/build-economic-funded \
  --economic-input-v1 --no-day-state-baseline --student-intraday 0 \
  --opponents thomas_2945_cpp,metav4_2965,fieldcraft_2887 \
  --seed-start 3300200000 --native-job-threads 32 --device cpu \
  --runtime-dir work/continuous-rl-economic-v682-handoff --tag economic-v1-v682
```

CPU 主机用 `--device cpu`；有 Ascend NPU 时改本机设备号和线程数。Python 3.11、PyTorch、NumPy、scikit-learn、pybind11、GCC/C++20、OpenSSL 开发库为必要环境。原 manifest 中旧 BC 路径不被原生 PPO 读取。

## 2026-09-24 当前研究与 Kaggle 提交

- 已按用户要求从近期同池 RL rollout 的已记录胜数直接选模型、未追加胜率测试：v464 rollout `1377/1536` 由 **v463 `checkpoint_in`** 产生。打包三地 G275/旧浅树 + funded replay、step288 后 17 天逐格采样、`intraday=0`；原样包 `submissions/student-v463-funded-kaggle.tar.gz`（SHA-256 `7a8fcabeb04096e330bacc3e6d6acb42a9cb3b93567c0bfb1b4cb1de05092445`）已上传 Kaggle，submission `56517329` 已 COMPLETE。初始公开分 `600.0` 不能代表匹配强度。训练胜数不是独立盲测；此包也没有替换普通 `agent/main.py`。
- 四地 D011/v4 生命周期首轮完整 1536 局虽完成，但 `0/1536`、均分差 `−47.3k`，四地链已暂停，三地连续训练仍运行。新动作不是主缺口：同 48 局三地 v474 `42/48,+6.4k`，四地旧生命周期 `0/48,−40.9k`，四地 v4 关闭新动作 `0/48,−43.1k`。D011 接管后 NN 很快填满 100 格；day15 的 14 株胡萝卜因漏浇变草。配置雇工上限 14，单局升到允许的 15 会消除这波草，却让 48 局终局均分差再降约 4.3k，不能当作完整修复。DSM 258 局中后期实际雇工中位 12、最高 14。细节与下步见 `HANDOFF_ZH.md` 首节。
- 交易 DP 子 agent 持续隔离研究。固定动作市场/日末守恒对官方已达 `1728/1728` tick；69 个固定销售状态中，当步现金与假想跨日清仓的双方钱差方向有 9 次翻转。可信 continuation value 仍缺，不启用旧 `sale_dp=1`，不改三地训练或此次 Kaggle 包。见 `docs/ECONOMIC_MODEL_ZH.md`。

## 当前交接：v306 采样 RL 与 Kaggle 提交（`handoff/rl-student-v306-kaggle-20260924`）

- 仓库 `https://github.com/LZhangGJ/Kaggriculture` 的本分支保留 v285 不动，新增 `models/student-v306/{actor.pt,actor.bin,manifest.json,metrics.json}`。v306 的 `.pt` 是可续训的模型与 AdamW 状态，`.bin` 是同轮 C++ 前向权重；两者 SHA-256 分别为 `14ff8eeced3fb1fe9414de70909ddba82271ce5fc73fa1e46676bc89a931d31f`、`ea877102df5342ec70de5c54dbe90b541566657749b4c4f87a6ab5b59e48ed08`。manifest 与 v285/v45 内容相同，不要求下载旧 BC 文件。
- `submissions/student-v306-kaggle.tar.gz`（SHA-256 `908235e386d73252505a7b3879079d07e835d1c0ec78533f13a69ea1f2c2bdaa`）是已提交 Kaggle 的**原包**：前 288 步 replay/浅树，后 17 个日初由逐格 v306 **采样**，`intraday=0`。内含 Jammy x86_64 R1 bridge、权重和冻结 tokenizer，可直接作为提交物；普通 `agent/main.py` 仍是非 NN replay→R1，不要误用旧 `scripts/pack_kaggle_submission.py` 来声称提交 v306。只上传这份 10.6 MB 原包和小模型，未上传 rollout NPZ、replay、缓存、本机 aarch64 `.so`。
- Kaggle 提交 `56511458` 已 `COMPLETE`：验证自我对战两侧均跑满 719 步，17 个 NN 决策日无 fallback/非法/日志错误；日初耗时约 `0.07–0.23s`，首步加载约 `19.4s` 从 60s overage 扣除。页面初值 `600.0` 时只有一局 validation、没有公开匹配局，**不是可比较的强度分**。后续看胜率必须等 public episodes。
- 提交端 2233/3145/374 输入与原生训练端已做定点 parity：token 数、连续值、类别与经济观察一致；Jammy x86_64 和本机 aarch64 对同一 warm 接管后 17 步 `td_observe` 输出逐项一致。源码重打包用 `PYTHONPATH=. python scripts/pack_student_v306_kaggle.py --so <Jammy x86_64 student bridge> --output build/student-v306-kaggle.tar.gz`；`experiments/student_v306_vendor/kaggrl/` 是随分支冻结、仅供 v306 提交入口使用的 tokenizer，不需要本机 starter 仓库，也不遮蔽其他项目的 `kaggrl`。重打包产物未必与已提交原包字节相同，不应冒充提交 ID 对应文件。
- 队友续训先按下方 v285 交接命令重建各原生扩展，再把训练命令中的 `--start-round` 改成 `307`、三个模型/manifest 路径改成 `models/student-v306/`、seed 段换成新的互斥段（例如 `3300000000`），保留 `--economic-input-v1 --no-day-state-baseline --student-intraday 0` 与采样三手池。不要从 Kaggle 的 tarball 解出 `.pt` 当作唯一训练入口；本目录中的 `.pt` 含 AdamW 续训状态。

## 历史交接：v285 采样对手与续训（`handoff/rl-student-v285-20260924`）

- 从 `https://github.com/LZhangGJ/Kaggriculture` 的**本分支**检出。冻结的正式 economic RL v285 在 `models/student-v285/`：`actor.pt`（模型与 AdamW 状态，SHA-256 `77db07d754fd3d79000e37b7742bb5a19199c7f2c55f51b21ea741968841e2a0`）、`actor.bin`（同轮 C++ 前向权重，SHA-256 `800227b1f399bdfe78d436dc75c422c4156e1c2991aa39edb813c0722ff54e04`）、`manifest.json`（固定训练契约，SHA-256 `25e12c5bafc1f2aca509dfc8dfd1d6d6678042b94d06d8294cc8724c0591dc87`）。manifest 与 v45 相同，旧本机 BC 路径不是原生 PPO 续训的数据依赖。v285 是训练链快照，不是盲测最优或 Kaggle 提交包；本机后续训练不会自动更新此分支。
- 运行行为必须是**前 288 步 replay/浅树，之后逐格 actor 采样，`student_intraday=0`**。不传 `--sample` 就是 argmax，不能拿来代表交给队友的对手。`agent/main.py` 目前仍运行非 NN 的 replay→R1；它不会自动加载 `actor.bin`。本分支提供可复现的原生 JobBatch 对局入口与续训入口，尚未提供可直接塞进任意外部 RL 框架或 Kaggle 的独立 bot 回调；那种接入仍需适配此执行/特征/随机数契约并核对动作。
- 在 Python 3.11、PyTorch、NumPy、scikit-learn、pybind11、GCC/C++20、OpenSSL 开发库环境，从仓库根目录重建原生扩展；x86_64 主机**不能**复制本机 aarch64 `.so`。只用下列三手已随仓库带的小型 C++ 对手资产即可复现当前训练池，不需原始 replay、历史 rollout NPZ、BC mmap 或实验缓存。
- 本机按本分支源码隔离重建的 student bridge SHA-256 为 `ae666af83c9639ca54160e35019e8823365dfc807ebe40025d1ef0d5d6c182de`，与 v285 训练所用二进制完全一致；重建的原生 JobBatch 用 v285 采样，五手 × 前 8 seed × 双座共 `80/80` 局的终局现金、actor hash 和前缀动作 hash 均与原评测一致。跨架构主机仍须在本机重建后复核，不应复制这两个 aarch64 `.so`。

```bash
(cd fast_kaggriculture && python setup.py build_ext --inplace)
bash experiments/native_student_rollout/build_student_bridge.sh work/agent-student-actor-owned-v3.so
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/thomas_2945_cpp/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/metav4_2965/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/fieldcraft_2887/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/salemali7_2900/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_student_rollout/build.sh
PYTHONPATH=. python experiments/eval_student_native_argmax.py \
  --weights models/student-v285/actor.bin \
  --binary work/agent-student-actor-owned-v3.so \
  --module-dir experiments/native_student_rollout/build \
  --opponents thomas_2945_cpp,metav4_2965,fieldcraft_2887 \
  --sample --student-intraday 0 --seed-start 3200000000 --seeds 256 \
  --threads 32 --output work/teammate-v285-sample.json
```

续训同一 v285 economic 训练链时用以下参数；Ascend 主机可把 `--device cpu` 改为本机 NPU，`--native-job-threads` 取实际可用核心数。`--economic-input-v1`、三手池和 `--student-intraday 0` 不可漏；`--shop-resource-v1` / `--shop-action-head-v1` 是另一个未晋级实验，不属于 v285。

```bash
PYTHONPATH=. python -u experiments/run_student_rl_continuous.py \
  --start-round 286 --checkpoint models/student-v285/actor.pt \
  --weights models/student-v285/actor.bin --manifest models/student-v285/manifest.json \
  --binary work/agent-student-actor-owned-v3.so \
  --native-job-module-dir experiments/native_student_rollout/build \
  --economic-input-v1 --no-day-state-baseline --student-intraday 0 \
  --opponents thomas_2945_cpp,metav4_2965,fieldcraft_2887 \
  --seed-start 3200010000 --native-job-threads 32 --device cpu \
  --runtime-dir work/continuous-rl-economic-v285-handoff --tag economic-v1-v285
```

## 历史交接：v45（`handoff/rl-student-v45-20260923`）

- 仓库：`https://github.com/LZhangGJ/Kaggriculture`；请从本交接分支检出，不要把它当成 `main` 的状态。
- 已训练到 v45 的逐格 actor 快照放在 `models/student-v45/`：`actor.pt`（模型和 AdamW 状态，7.3 MB）、`actor.bin`（同一轮的 C++ 前向权重，2.5 MB）及 `manifest.json`（checkpoint 的固定数据契约，约 10 MB）。v45 是训练链快照，不代表盲测最优，也尚未替换线上 R1。
- 本分支包含 v45 续训所需的 Python/C++ 源码及 Thomas、Meta、Fieldcraft 的小型原生对手资产。无需下载原始/日级 replay、历史逐局 rollout、BC mmap shards、实验缓存或生成资产时用过的公开脚本；`manifest.json` 在原生 PPO 中用于校验 checkpoint 身份，不读取其中指向本机的旧 BC 文件路径。
- 本机现成的 Python/C++ `.so` 是 aarch64 构建物。队友若用 x86_64，必须在自己的 Python 环境重建原生扩展和 R1，不能直接复用这些 `.so`；`actor.pt` 与 `actor.bin` 本身是跨主机的数据文件。
- 在仓库根目录，以 Python 3.11、PyTorch、NumPy、scikit-learn、pybind11、GCC/C++20 和 OpenSSL 开发库准备环境；Ascend 主机可装匹配版本的 `torch-npu`，其他主机用 `--device cpu`。各 `build.sh` 用 `PYTHON_BIN` 指向当前 Python，不依赖 `/root/miniforge3`。

```bash
(cd fast_kaggriculture && python setup.py build_ext --inplace)
bash experiments/native_student_rollout/build_student_bridge.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/thomas_2945_cpp/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/metav4_2965/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/fieldcraft_2887/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_opponents/salemali7_2900/build.sh
PYTHON_BIN="$(command -v python)" bash experiments/native_student_rollout/build.sh
PYTHONPATH=. python -u experiments/run_student_rl_continuous.py \
  --start-round 46 --checkpoint models/student-v45/actor.pt \
  --weights models/student-v45/actor.bin \
  --manifest models/student-v45/manifest.json \
  --binary work/agent-student-actor-owned-v3.so \
  --seed-start 2640000000 --native-job-threads 32 --device cpu
```

这段仅供复现旧 v45 交接，不是当前 v285 续训命令；当前训练链仍在本机后台推进，后续轮次不会自动推送到交接分支。

## 2026-09-24 v261 本机旧对照（非本分支交接快照）

- 本次全 C++ 对手旧评测锁定**正式训练链 v261**；其小型快照仅保留在本机 `models/student-v261/{actor.pt,actor.bin,manifest.json}`，不上传本分支。bin SHA-256 为 `e9afc1f2c423d7752b62b43f54a635876a68388da9ece7a555b5fd22fff33d17`。不能把 v261 结果冒称后续版本。v261 为 637,127 参数、`scale=3`，输入宽度 `2233/3145/374`，没有逐格 9 维商店率或 shop-gate 实验头。
- 当前 `agent/main.py` 仍是 replay→R1；`actor.bin` 不是可独立运行的 Kaggle 对手。v261 的实际对局入口是 `experiments/eval_student_native_argmax.py` 的原生 JobBatch：前 288 步 replay/浅树，后续每天由逐格 student 决策，`student_intraday=0`。**交给队友当训练对手时用 `--sample` 对应的逐格采样策略，不用 argmax**；`--sample` 只切换 student 动作抽样，前 288 步与执行器契约不变。队友须复用这套原生执行/特征契约或制作通过逐步动作与终局 parity 的采样对手适配器；不要把旧 Python `StudentActionEventAgent` 或单个权重文件直接当成 v261 完整 bot。
- v261 的本机快照仅供旧对照，**本分支交接的是上面的 v285**；不要推原始 replay、历史 rollout NPZ、评测轨迹及缓存。现有 `models/student-v45/` 仍只是旧交接快照，x86_64 主机须重建 `.so`，不能复制本机 aarch64 构建物。
- `--shop-action-head-v1` 是独立、默认关闭的研究分支，**不是**本节正式 v261。它从 v254 零漂移分叉并通过原生 old-policy 回放，尚未通过互斥 seed 闭环收益验证；不得误交给队友作为正式对手。
- 冻结 v261 的原生 argmax 评测：seed `3000060000..3000060127`，每手 128 seed × 双座（256 局）。Thomas `204/256=79.7%`、Meta `209/256=81.6%`、Fieldcraft `210/256=82.0%`、Soil `206/256=80.5%`、SaleMali `256/256=100%`；五手共 `1085/1280=84.8%`。逐局结果在本机 `work/student-v1/eval-economic-v261-allcpp-argmax-s128a.json`，它是 dev 评测，不是盲测或正式七强 R1 验收。另测 replay 壳 G397 `58/64=90.6%`（32 seed × 双座），不算第六个独立公开对手。当前源码重编译后，同一前 8 seed 的 80 局完整逐步动作与终局和评测所用旧模块完全一致。
- 后续常规 student 评测运行 `experiments/eval_student_native_argmax.py`，默认只存逐局比分与权重/JobBatch 模块哈希；只有建 BC 数据或抽样执行审计时才加 `--save-trajectories`。上述 1,280 局是切换默认值前启动的旧评测，因此耗时 `1481s`，其中大量时间花在 719 帧逐局重放与 JSONL.gz；不可把这个总时长当成原生 rollout 耗时，也不可据此推断正式 RL 训练会因关闭评测归档而提速。
- 同一 v261、同一 128 seed × 双座 × 五手，`--sample --threads 192` 且不存轨迹的结果在 `work/student-v1/eval-economic-v261-allcpp-sample-s128-scoreonly.json`：Thomas `203/256=79.3%`、Meta `205/256=80.1%`、Fieldcraft `212/256=82.8%`、Soil `207/256=80.9%`、SaleMali `256/256=100%`，合计 `1083/1280=84.6%`。与 argmax 逐局配对为输转赢 43、赢转输 45；原生评测用时 `17.75s`（约 `72.1` 局/秒），没有生成逐帧档案。该吞吐含同时运行的正式 RL 负载，不等于正式训练吞吐。
- 正式 economic 链 v285 也只跑采样：同一 seed、五手、双座、192 线程，只存比分 `work/student-v1/eval-economic-v285-allcpp-sample-s128-scoreonly.json`。Thomas `206/256=80.5%`、Meta `195/256=76.2%`、Fieldcraft `213/256=83.2%`、Soil `198/256=77.3%`、SaleMali `256/256=100%`；合计 `1068/1280=83.4%`，比 v261 采样少 15 胜。同一环境 seed 的逐局胜负转移为输转赢 34、赢转输 49；采样的 policy RNG 不应默认视为两版本完全耦合。五手平均钱差均略升但胜数下降，因此不能用钱差代替胜率。这不是盲测晋级结论；本分支明确交接的是 v285 而非 v261。
- **训练/评测胜率口径已核实**：v286 训练指标的 `1287/1536=83.8%` 是更新前 **v285** 在新 256 seed × Thomas/Meta/Fieldcraft × 双座上的 rollout，不是 v286 更新后的回测；五手 dev 中这三手的 v285 采样为 `614/768=79.9%`。把评测换成 v286 rollout 的同一环境 seed 与三手、但仍用评测默认的逐局编号采样 seed，得到 `1294/1536=84.2%`；再把保存的 v286 rollout 每局 `policy_seed` 原样回放，`1536/1536` 局我方与对手终局现金均逐元一致、胜局严格 `1287/1536`，无错误。权重、执行二进制和 JobBatch 模块 SHA 与训练记录一致。因此当前没有发现“评测误用 argmax/错误执行器”的证据；约 4pp 的三强差距主要来自不同环境 seed 段，另有采样 RNG 差异。固定 dev 和训练流不能互相冒充泛化成绩。
- 按同一 v285 权重、三手、采样、192 线程另取三个互斥的 256-seed 段，不写轨迹：`3000070000` 为 `1273/1536=82.88%`，`3000071000` 为 `1245/1536=81.05%`，`3000072000` 为 `1308/1536=85.16%`；合计 `3826/4608=83.03%`。逐手合计 Thomas `1291/1536=84.05%`、Meta `1252/1536=81.51%`、Fieldcraft `1283/1536=83.53%`。逐局结果在 `work/student-v1/eval-economic-v285-triad-sample-s256-seed{3000070000,3000071000,3000072000}.json`，三个段各有 256 个独立环境 seed，双座不算独立 seed。这进一步显示此前单个 128-seed dev 段的 `79.9%` 不代表稳定总体水平；这些反复查看的段也都应视作 dev，不冒称永久盲测。

## 硬约束

- 不得把 `experiments/results` 中带原生 opening-template 的旧 Triad 扫描当作 R1 结论。
- 不使用 Nash 选开局。
- 正式公开对手验证只用 `opponents/` 的 7 个强对手，同 seed 双座、多 seed。
- replay 只离线使用。线上 `agent/main.py` 只读取导出的路线、浅树和当前公开观察。
- 先复用 `scripts/` 的成熟聚类、原生对打和鲁棒树训练，不另写简化替代品。

## 模型研究方法论（强约束）

本节约束后续所有数学模型、规划器和执行器研究，优先级高于历史参数 A/B 经验。

1. **理论先于现象。** 先从官方规则写出目标、状态转移、现金/库存/市场守恒和信息边界，再检查实现；
   轨迹只负责证伪、确认触发域和估计量级。不能从败局相关性反推原因，也不能要求一个理论错误必须先与
   败局相关才允许修。
2. **分清四个层次。** 每一项都明确标记为：精确规则/不变量、主动模型假设、计算近似、经验补偿参数。
   `competition=2`、`supply=.85`、`replant=.5` 等有效参数可能在补偿别处的错误，不能反过来当作理论真值。
3. **先做影响量级筛选。** 深挖前先估算
   `误差上界 × 实际触发覆盖率 × 决策敏感度`。数值误差只有在候选间的差分足以跨过 score gap、改变
   `argmax` 或约束可行性时才可能改变计划；同时报告绝对误差和候选差分，不能只报前者。
4. **内部不变量优先验收。** 核心模型修复先用最小子系统诊断：逐项跟踪预测流、现金、库存、约束、
   tail value、候选 score 和排序。能用保存状态或只跑到 checkpoint 验证时，不跑完整对局；能在固定
   portfolio 上隔离估值差时，不把重新规划、执行和未来随机性混进同一个结果。
5. **正确性与当前胜率分开。** 修正一个模型错误后端到端性能下降，只能说明旧参数/旧误差之间存在补偿，
   不能证明修正错误，也不能证明旧近似更正确。应沿依赖顺序重新校准下游假设和参数；不得为了保留当前
   胜率而恢复已确认违反规则或守恒的不变量。
6. **端到端验证是最后一道部署门槛，不是理论裁判。** 顺序固定为：理论推导 → 内部不变量 → 量级与
   排序影响 → 下游重校准 → 固定状态 suffix → 互斥 seed 的七强完整 warm 链。最终是否部署仍以正式
   胜率口径决定，但端到端负结果不能抹掉已经证明的模型缺陷。
7. **保持可归因。** 理论修复先以隔离构建或诊断开关验证，不同时改无关模块；记录修复前后的内部量、
   哪些候选换序以及换序贡献。只有内部机制成立后才扩大实验。

下文“已实测封死”限制的是在旧模型上继续盲扫同类参数或直接部署旧候选，不禁止修复其上游理论错误后，
按依赖顺序重新校准受影响的参数。旧模型下的负 A/B 不能永久否决一个后来已由规则证明正确的不变量。

当前 R1 首要结构问题已进一步定位为 **`score()` 的静态 portfolio tail 不是线上每日重规划策略的
continuation value**。在 8 个固定 handoff 的 6 个非零 outer 反事实上，现有一日静态 score 对真实
终局方向只对 `3/6`；用同一模型真正逐日闭环规划到终局后方向对 `5/6`，聚合预测由错误的 `+919`
改为 `−2,053`，接近真实 `−2,800`。1/3/5/10 日截断均不稳定，只有终局 horizon 在这批状态上恢复
主要方向。因此优先修的是 Bellman/控制时域一致性，不再扩首日 beam 或把长期 `model.value` 搜得更精确。

对手跨日库存/成交响应是其后的重要状态缺失，而不是已经单独解释排序的结论。公开账本在 Thomas seed100
的 day24 给出 WOOL 隐库存区间 `[0,33]`，随后候选使市场库存/价格分叉到 `−51/+195`，对手多赚
`$5,659`；但以同一 forecast flow 做“立即卖 vs 跨日最优清仓”的库存 best-response 诊断，对相关候选
几乎共同扣约 `$6,994`，winner 不变。这证明库存时序对绝对价值很大，却也证明仅加一个库存点估计不够；
剩余误差需要公开历史上的多情景 `rival stock belief + conditional liquidation`，并允许未来重规划与
价格反馈联动。

一次 top-4 终局闭环重排虽令内部 winner 在 `7/8` 状态变化，固定 suffix 仅从原始 outer 的 `−2,800`
改善到 `−1,939`，但该 A/B **不能准入**：baseline 与 outer 各自按短分截 top-4，候选集不嵌套，outer
会漏掉 baseline 经闭环重排后的 normal winner；两例甚至选择了闭环分更低的 outer。下一次必须让 outer
候选集显式包含 baseline winner，再做固定状态比较。所有新逻辑仍是默认关闭的诊断，正式策略未部署。

该协议现已修正为：baseline 评短分前 4 个 normal，outer 评同一组 normal 加短分前 4 个 outer，8/8
满足扩展集合的预测最优值不下降。修正后的固定 suffix 中，闭环 normal 重排相对旧 normal 合计改善
`+9,774` margin；加入 outer 后相对旧 normal 仍改善 `+9,344`，说明闭环 continuation 是有效抓手。
但 extra-outer 相对同一闭环 normal 仍为 `−430`，其预测却为 `+7,970`，4 个非零选择只对 `2/4` 方向；
Thomas 两例分别误判 `+364→−5,697`、`+4,016→−1,586`。这暴露了第二层瓶颈：终局闭环消除静态
tail 错误后，单条 open-loop 对手供给仍会被候选搜索利用。不能直接部署 top-K；下一步在闭环 scorer 内
加入公开信息可生成的对手响应多情景，并用嵌套候选集继续验收。

## 当前部署语义（2026-09-21）

- 开局固定 `G275`；浅树在 step 144 / 168 切换，最多一次。
- **接管延迟 2 天**，即固定到 step 288（实测 288 为单峰最优：264 差 52 胜、312 差 29 胜）。
- `target_fallbacks` = `{G114: G275, G019: G195}`。G019 是按叶归因找出的坏叶（条件胜率 69.7%）。
- G275 的 **cp144 节点已换成用真实对手重训的 depth-3 树**（896 个「状态 / 5 候选目标 / 结局」单元）；
  原树是在自对弈矩阵（对手 = 245 条库内路线）上训练的，标签域与线上不符。

改动明细与配对 A/B 证据见 `README.md`「当前部署语义」。

## 验收口径（容易踩的坑）

`experiments/run_strong_ab.py` 的 `both_seats` 几乎不提供独立信息：本对局对称且双方确定，
**约 87% 的 (对手, seed) 配对中 seat0 与 seat1 的 margin 逐元相同**。所有「N/896」的有效样本
约为名义值的 57%。历史上一批 8–16 seed 块的 accept/reject 决策都建立在这之上。

- 用 `experiments/eval_seed_paired.py`：按 (对手, seed) 配对 + Wilson 区间。
- **seed 段互斥**，不要重复使用同一段。
- 判定「每个对手 ≥80%」需要约 683–1537 seed（现在只有 512）。`engine fast` 跑 64 seed × 7 手 × 双座约 5 分钟。

常规胜率评测**不强制保存 719 帧 JSON**：至少保留每局 seed、座位、对手、策略/权重与执行模块哈希、
终局现金/胜负及汇总；有需要时保存紧凑动作轨迹。完整公共 observation、双方 private/action 的
`jsonl.gz` 只在明确建设 BC corpus、做执行审计或需要逐帧复现的抽样对局时显式采集。
`experiments/eval_student_native_argmax.py` 默认只写逐局比分，`--save-trajectories` 才做慢速完整归档；
`experiments/run_strong_ab.py` 的既有轨迹默认行为是该脚本自身配置，不再是所有评测的强制协议。
forced-candidate 诊断轨迹不可无标记并入专家数据；训练/验证仍须按 seed 与对手分组切分，避免泄漏。

中盘学习路线见 `docs/LEARNED_MIDGAME_PLAN_ZH.md`。硬边界：RL/NN 只作用于 step288 后；慢 R1/终局
闭环只作离线 teacher，线上 student 不再每局跑完整 DP。大规模自博弈、反事实 suffix、特征提取和小模型
推理尽量放 C++，Python/NPU 只负责训练与调度。**学习主线只训练自回归逐格 actor**：按稳定格序直接输出
`SKIP / 5 crop / 3 animal`，每步应用精确规则 mask，并把已选前缀的预算、库存和 30 天资源日历传给下一格。
不得退回“R1 先生成整盘 candidate、NN 只做 candidate-ranking”的替代任务。critic 只能作为逐格 actor 的
value/baseline 或完整序列辅助头，不能代替逐格决策；最终动作仍须经过 executor 可行性检查。

现有非 NN 动态 R1 是冻结 baseline：学习代码必须走独立、默认关闭的实验入口，禁止覆盖
`policy/r1/agent.so`、`policy/r1/config.json` 和 `agent/main.py`。learned off 时须做逐步 action parity；
student 发生 OOD、NaN 或 feasibility 失败时必须回退原 R1。模型首版预算 `8万–18万` 参数，CPU 优先；
NPU 只在千万级样本/大规模 ensemble 时使用，不能用来替代 C++ teacher rollout 加速。

学习输入遵守 `docs/MIDGAME_STUDENT_FEATURE_AUDIT_ZH.md`：`192/32` 是 encoder latent 宽度，绝非
已完成的 raw 字段；必须明确输入剩余时域、完整公开棋盘/市场/我方 private、公开历史 belief、内部承诺
和虚拟前缀后的全剩余期资源日历。11 项资源占位和无 mask 的数据不得开训。原始 JSON 仅作档案，训练
只读一次转换并校验过的 mmap binary shards；网络下载仍单线程，下载后的独立文件解析才允许并行。

历史 candidate-critic 的四个公开供给情景是**无概率的 support 向量**，只保留作旧 scorer 诊断，不进入
当前 actor 训练或线上 forward；不得把它们等权平均成“期望”，也不得借这些资产恢复 candidate-ranking
旁路。逐格策略改进若使用真实 suffix，所有 future replicas 仍须按原 checkpoint 成组聚合。

轨迹恢复的有效域也要 fail closed：`observe_external` 只足以 warm 首次 step288 的公开 ledger/SaleClock/
CropClock，此时 R1 commitment 为空是合法状态。任何 step>288 teacher 样本必须从 step288 起实际驱动
同一 R1，逐步核对 action，并在采样点核对 candidate key、book/joint/restore_day；不能从公开盘面猜回
内部承诺。BC corpus 的策略指纹必须覆盖 Python 入口、R1 `.so`/配置、deployment、route policy/tapes
及实际加载的执行模块；单独 `sha256(agent/main.py)` 不是可复现的 policy identity。

step288 后 RL 的 terminal reward 以胜负为主，钱差只能作小且有界的 shaping；现金/资产等中间量不得
直接累加，除非写成终局为零、可望远镜的 potential difference。BC/critic 轨迹可多 epoch 复用；PPO
轨迹必须保存原 policy version、逐 slot mask/log-prob，且只作少量 on-policy epoch，旧轨迹没有明确
importance correction 时不得用于 policy-gradient。所有候选、future replicas、双座按原 checkpoint
聚合并以 `(seed, opponent family)` 整组切分；seed/RNG/对手身份永不进入 forward，保留永久盲测 seed 段。
当前 reward `sign(margin)+0.1*tanh(margin/10000)` 在 v16–v21 的分解中，对所有非零胜负
advantage 都没有翻转符号；tanh 占平均绝对 advantage 约 `4.9%–5.6%`，占平方信号仅
`0.03%–0.05%`。它在 `55%–64%` 胜负完全相同的 seed block 中提供微小 tie-break 信号，
不是当前方差或泛化瓶颈，不据盲段一次“分差升、胜数平”盲改 reward。

逐格 actor 在一天内自回归生成的**整条计划是 PPO action unit**：当天 joint log-ratio 必须等于各个
actionable slot log-ratio 之和，并只对这个 joint ratio 做一次 clip；forced slot 只推进 hidden，不进入
likelihood。终局 advantage 对每个 actionable day bundle 应用一次，**不得再除以当天 slot 数**，否则会
把期望终局收益改成依赖动作长度的另一目标。正式 batch 对同一环境 seed 跑全部「对手 × 双座」，当前局的
baseline 使用同 seed 的其他独立 policy-RNG 轨迹做 leave-one-out；它不读取当前局 reward/action，故只作
无偏 control variate，seed/对手身份仍不得进入网络输入。v11/v12 的两段 1,536 局审计中，该 baseline 将
advantage 方差分别降至旧 `(opponent family, seat)` LOO 的 `65.1% / 54.3%`；更复杂的 two-way 修正没有
额外收益。训练游戏数必须整除 `2 × 对手数`，且每个 seed 恰好覆盖完整的「对手 × 双座」块。每轮必须
连续恢复 AdamW state，并在完整 rollout 上重放行为策略、报告 day-chain KL 后才保存更新。

原生 rollout 只有在三层证据同时成立后才可替换训练入口：任意新 seed 不依赖预生成 cache；固定 cache
逐步比对前288 replay/浅树动作、step288 packed/context、全部 actor event、后缀动作与终局 exact；输出
训练所需的 context/observation/tokens/resources/legal/action/old-logprob 连续数组并通过 CPU/NPU old-policy
replay gate。重复少数 cache 得到的吞吐只算性能 microbenchmark，不算 RL 可用性。

原生 rollout 持久化直接保存 `ppo_arrays()` 的未压缩 NPZ 与最小 JSON metadata，不得先膨胀成几十万
Python event dict 再 `torch.save`。resume 时重建视图并重新执行 policy/binary/manifest、对手 artifact、
native actor weights、JobBatch module、逐局 identity 与 `route=-1` 门；NPZ 不是绕过 old-policy replay gate
的理由。B8 fresh/resume 已做到逐 tensor update exact，保存 12.99MB 仅 `0.0365s`。

RL 对手池按源码家族去重：经 Python↔C++ 逐步 action parity 的最新强公开代表作为主力，
student 历史冻结快照用于 self-play。慢动态 R1 **只做离线 BC teacher**，不得进入 RL rollout、对手池、
reward 计算或 policy-gradient；若事后给 student 状态补 R1 标签，也只能作为下一轮 BC 数据。不得因多个脚本共享
Ahmed/MetaV4 chassis 就将它们当成独立分布重复加权，也不得等所有对手 C++ 化后才开始首训。

截至 2026-09-23，原生 JobBatch train-head 为 v22，dev-best 仍为 v20。v22 首次按源码 family 用
Thomas/Fieldcraft 各 1/2，并启用 whole-seed 6-fold day-state cross-fit control variate；1,536 局
rollout `34.08s`（`45.07 games/s`），非法与 fallback 均为 0。critic 将 advantage 方差降至
`79.66%`、slot-count 加权后 `77.24%`，六折与 17 个决策日全部改善；端到端 rollout、NPU 更新和审计
共 `143.61s`。old-logprob max/mean/KL 为 `1.416e-4 / 2.206e-7 / 8.11e-13`。
学习率在 v19 起由 `2e-5` 降为 `1.8e-5`；v22 day-chain KL 为 `0.01091`，48 个 minibatch
裁剪前 gradient norm 中位数 `68.50`、p95 `100.55`，100% 超过 `max_grad_norm=1`。`target_kl` 是 post-epoch soft
early-stop，在 `epochs=1` 时不回滚 checkpoint。

v15 的 Thomas/Meta/G397 各1/3 完整拆解显示：G397 占 `39.8%` bundle KL 但只贡献 `18.8%`
clipped-surrogate gain，与两强手的 same-seed reward 相关系数仅 `0.016/-0.083`；删掉它后
advantage 方差从 `0.918` 降至 `0.680`。因此 G397 留作 retention eval，不再等权1/3。
Salemali JobBatch code4 已在新 seed 双座与权威 Python 源做到 `11,504/11,504` 动作和 `16/16`
终局 exact；但历史 student 对它 `509/512` 胜，也只作 canary。独立 family Fieldcraft 的 JobBatch
code5 已对 standalone C++ 做到 `5,752/5,752` 动作、`8/8` 终局与 identity exact，正式进入 v22 对手池。
反复查看过的固定 1,024 局集合只能称 dev：v9/v10/v11/v12/v13/v14/v15/v16/v17/v18/
v19/v20 胜局依次为 `542/554/548/559/566/554/577/560/576/598/593/613`。v20 相对
v15 为 `+36` 胜、margin `+490.9`、McNemar `p=.00275`、seed-cluster 胜率 CI
`[+1.37pp,+5.66pp]`。但预留 milestone blind `2631600000..2631600511` 的一次性配对复核仅为
v20/v15 `553/552`（净 `+1`）、margin `+197.7`，Thomas `-3`胜、Meta `+4`胜；固定 dev 的
胜率提升没有跨 seed 复现。因此 v20 保留为 dev-best，v21 按“非灾难性震荡不回滚训练链”作
train-head，不把 dev 改善当作泛化证据。
旧 blind 段已退役；`2633000000..2633000511` 为新的未查看 milestone blind 段，禁止训练。

## 性能研究的边界（已实测封死，别再走）

只评估真实 warm 链（replay 浅树切换 → delay=2 温接管 R1）。以下方向已逐一实测，收益为 0 或负：

| 方向 | 结果 |
|---|---|
| R1 配置杆（21+ 项，含叠加） | 均分差最高 +830，胜率 0（翻转一局需 +5,017） |
| 换 opening（7 条） | −66 ~ −367 |
| 编辑 tape（作物 / 预算 / 无耦合） | −454 ~ −566 |
| 叶回退（G024 / G275 / G316 方向） | −10 ~ −289 |
| 二次切换、更早切换点 cp72 | 胜率 0 / 明显更差 |
| 候选路线前瞻特征（48 步、120 步逐日） | +1.2pp / −3.9pp（后者更差） |
| 树重训扩样本（896 → 2,240 单元）、换模型类别 | 饱和 / 全部差于 depth-2 |
| 移植对手的 sale-lead 抢卖层 | +2 胜（≈0） |

**oracle（逐状态最优目标）比可达高 6–8pp，但用现有 147 维特征 + 任何模型类别都取不到 —— 瓶颈是特征信息量，不是数据量或模型容量。**

## Kaggle 提交

```bash
PYTHONPATH=. python scripts/pack_kaggle_submission.py \
  --so <x86_64 且 Jammy 兼容的 policy/r1/agent.so> \
  --output build/kaggriculture_submission.py
```

两个必须注意的点（否则线上直接得零分）：

1. **架构**：本地 `policy/r1/agent.so` 是 aarch64，Kaggle 是 x86_64。
2. **glibc**：用 Noble 的 `x86_64-linux-gnu-g++` 编出的产物要求 GLIBC 2.36/2.38，Jammy 只有 2.35。
   必须用 Jammy sysroot 工具链（`kaggriculture-t2-ideas-v1/submissions/toolchain-jammy`，GCC 11.2），
   加 `-static-libstdc++ -static-libgcc -march=x86-64 -ffp-contract=off -Wl,-Bsymbolic`，
   产物最高需求 GLIBC 2.34、无 libstdc++ 依赖。

改完编译器/宏后**必须做跨架构 parity**：同一批真实对局观测分别喂 aarch64（本地）与 x86_64（qemu + Jammy sysroot），
逐步比对 `td_observe` 输出。本会话已验证 300 步 0 mismatch。
