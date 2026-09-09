# T2R1 / T2R2：源码、Public成品、实测与交接

这是2026-09-09已经提交Public并通过官方验证的两个版本。**R1指T2-R1 ServiceAligned，不是上一个T1编译修复版。**

| 版本 | Public提交 | 改动 | 七对手平均胜率 | 对710胜率 |
|---|---|---|---:|---:|
| R1 | [56114689](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114689) | 动物生命周期估值与实际维护对齐 | 1291/1400，92.21% | 145/200，72.5% |
| R2 | [56114767](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114767) | R1＋混合货物一次卸货、同一步共享仓容预留 | 1285/1400，91.79% | 148/200，74.0% |

两版均为规则自主经营＋动态C++执行，不是固定高手路线回放，也没有训练中的RL模型。R2不是已证明全面强于R1：七对手净少6胜，对710多3胜，二者配对区间都包含0。G003、Boatlee、710仍有明显短板。

## 1. 直接找文件

| 要做什么 | R1 | R2 |
|---|---|---|
| 原始完整策略源码 | [agents/R1/policy](agents/R1/policy/) | [agents/R2/policy](agents/R2/policy/) |
| 核心经营规划 | [triad.hpp](agents/R1/policy/triad.hpp) | [triad.hpp](agents/R2/policy/triad.hpp) |
| 底层执行器 | [executor/policy.hpp](agents/R1/policy/executor/policy.hpp) | [executor/policy.hpp](agents/R2/policy/executor/policy.hpp) |
| 默认配置 | [config.json](agents/R1/policy/config.json) | [config.json](agents/R2/policy/config.json) |
| 实際上传包 | [submission.tar.gz](public/R1/submission.tar.gz) | [submission.tar.gz](public/R2/submission.tar.gz) |
| 可直接运行的Public入口 | [main.py](public/R1/main.py) | [main.py](public/R2/main.py) |
| 官方回执和身份验收 | [RELEASE_ACCEPTANCE.json](public/R1/RELEASE_ACCEPTANCE.json) | [RELEASE_ACCEPTANCE.json](public/R2/RELEASE_ACCEPTANCE.json) |
| 作者解释改了什么 | [原报告](agents/R1/SOURCE_REPORT_ZH.md) | [原报告](agents/R2/SOURCE_REPORT_ZH.md) |
| 编译行为检查 | [check_binary.py](agents/R1/rebuild_checks/check_binary.py) | [check_binary.py](agents/R2/rebuild_checks/check_binary.py) |

**推荐默认加载public目录的已提交兼容binary。** `agents/*/policy/agent.so`是原始受测binary，`public/*/t2r*_runtime/agent.so`是通用x86-64、GCC11、静态C++运行库提交binary。哈希不同但已逐局完整复现相同1600局的现金、胜负、动作哈希、操作计数、overflow；源码、配置和包装器未改变。两套都保留，防止混淆来源。

ABI参数数量：R1为37项，R2为38项。必须成套使用wrapper、config和binary，不能只换一个`.so`。本地`run.py`有数量及哈希检查。

## 2. 一分钟开始使用

工作环境：Linux x86-64/WSL2。不需要GPU或Kaggle凭据。原生Windows不能直接加载Linux`.so`。

先从仓库根目录进入本目录，Python3.12可使用附带的已验证C++面板：

```bash
cd nt/latest_20260909_t2_r1_r2
python3 verify_package.py
python3 run.py --version R1 --suite seven --reference --count 1 --threads 4 --tag r1_reference
python3 run.py --version R2 --suite seven --reference --count 1 --threads 4 --tag r2_reference
python3 run.py --version R1 --suite 710 --reference --count 1 --threads 4 --tag r1_710_reference
python3 run.py --version R2 --suite 710 --reference --count 1 --threads 4 --tag r2_710_reference
```

交接时已在迁移后的Git目录实际执行以上四组：32局全部与原记录一致，零差异；证据为`evidence/relocation_*.json`。这只是搬迁回归，不能当作新增32局独立强度样本。

用新种子评测（以下为示例种子，不保证未被其他人用过）：

```bash
python3 run.py --version R1 --suite seven --start 2609110000 --count 100 --threads 16 --tag r1_new100
python3 run.py --version R2 --suite seven --start 2609110000 --count 100 --threads 16 --tag r2_new100
```

每个seed自动交换座位。`seven`每seed14局；`710`每seed2局。对手均实时读取真实状态并决策，不是冻结动作续播。每个`--tag`必须新建，程序拒绝覆盖结果。输出在`runs/<tag>`，包含全部逐局结果，不删除败局。

最多16线程。这里只提供评测入口，不会自动上传或改Kaggle状态。

## 3. 换Python版本或Linux工具链

附带`arena/_triad_panel.cpython-312-x86_64-linux-gnu.so`只适配Linux Python3.12，不能跨Python ABI。若无法导入，需要重编译面板：

```bash
python3 -m pip install pybind11
python3 build_panel.py --jobs 4
```

需要g++、Python开发头文件。面板编译只复用同一仓库的：

`../latest_20260909_triad_dp_t1/package/arena/experiments/daily_dp_v7_20260903/native`

这是共享规则引擎和对手源码，不是使用T1我方策略。交接时与本次实测依赖逐文件核对一致，清单为`SOURCE_PROVENANCE.json`。重建会产生本地`build/`缓存，不应提交Git。重建后必须重跑上节的`--reference`测试，不能只看编译退出码。

如果只想重编译R1/R2策略，使用各自原发布构建器：

```bash
python3 agents/R1/build_policy.py --cxx g++ --out build/r1_rebuilt.so --workers 4
python3 agents/R2/build_policy.py --cxx g++ --out build/r2_rebuilt.so --workers 4
```

构建器通过5条/8条完整官方轨迹检查后才发布目标库，失败不覆盖旧库。代码保留了此前`animal_path`的`noipa/noinline`保护。上述`--out`不会替换已提交版本，默认`run.py`仍运行原Public成品；需要实验新策略时应复制到新的实验入口，不修改本交接包来追分。

`verify_package.py`按发布哈希检查。如果主动重编译或改代码后不再匹配，这是预期的来源变化，不能重新生成清单掩盖修改。

## 4. 本地实测结果，不混淆作者小面板

七对手：相同100种子`2609099000..2609099099`、双座位，每版1400局；710另用100种子`2609100000..2609100099`、双座位，每版200局。

| 对手 | 原T2 | R1 | R2 |
|---|---:|---:|---:|
| G001 | 96.0% | 99.5% | 100.0% |
| G003 | 81.5% | 83.0% | 81.5% |
| Boatlee V29 | 83.0% | 87.0% | 88.5% |
| Kaito V58 | 83.5% | 95.5% | 90.5% |
| Lynn V5 | 87.5% | 91.0% | 91.0% |
| Six-Day | 85.0% | 92.5% | 93.5% |
| Three-Day | 92.0% | 97.0% | 97.5% |
| 710 | 51.5% | 72.5% | 74.0% |

全部原始逐局记录在`evidence/seven/*_rows.json.gz`与`evidence/710/*_rows.json.gz`，为JSON无损gzip压缩。各含T2对照、R1、R2；完整汇总、协议和配对CSV同目录。gzip文件用`json.loads(gzip.decompress(path.read_bytes()))`读取。

正式报告在`reports/`；作者原报告在`agents/*/SOURCE_REPORT_ZH.md`，其中112/224局属于更早的开发数据，不能当成本次1400局，也不能与其重复累计。

## 5. Public取证

2026-09-09日本时间14:46确认两版均COMPLETE、错误信息为空：

- R1：[Validation107025900](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114689&episodeId=107025900)。
- R2：[Validation107026853](https://www.kaggle.com/competitions/kaggriculture/submissions?submissionId=56114767&episodeId=107026853)。

每份录像720帧、双方DONE，双方1438个动作与上传包一致。`public/*/CLOUD_ACTION_IDENTITY.json`及Validation回执保存证据。包中不含约60MB的两份完整录像，文件哈希和官方Episode ID仍保留，可另行获取。

每版上传前另做1600局兼容构建回归、6局官方文件入口检查，最大本地调用82.959/80.182ms，无默认1秒超时。这不是对线上所有未来局面的保证。600.0仅是取证时的初始Public分，不是稳定天梯评级。

## 6. 完整性和范围

- `MANIFEST_SHA256.json`：本次所有交付文件的字节清单；`.gitattributes`禁用文本换行转换，防止历史哈希被Git改变。
- `SOURCE_PROVENANCE.json`：原始目录、源码和binary/提交包身份、共享C++依赖。
- `HANDOFF_ACCEPTANCE.json`：搬迁后实际回归范围。
- 没有大缓存、对象文件、工具链、账户凭据或大量Replay；小型官方轨迹夹具保留，便于重新编译验收。
- 原证据JSON/报告中的旧绝对路径作为历史记录保留，某些作者历史实验未纳入本精简交接；**可运行入口及可访问路径以本README为准**。
