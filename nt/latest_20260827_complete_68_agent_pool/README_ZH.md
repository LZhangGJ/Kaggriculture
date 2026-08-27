# Kaggriculture 完整68-Agent池增量包

冻结日期：2026-08-27

状态：**PASS，补齐此前队友 Git 中缺失的12个可运行 Agent资产，并加入统一68池清单。**

## 1. 为什么这是增量包

此前队友 Git 已分阶段保存：

- `latest_20260821_arena_fusion_fc2a/`：历史28个严格JAX Agent；
- `latest_20260823_fc24b/`：FC24B冻结基线及依赖；
- `latest_20260825_front40_public_v48/`：旧Rank40复原主体、Kaito V48和公共运行时。

为保持旧快照和其 SHA256 清单不变，本目录不修改上述冻结包，而是只增加缺失文件。将四个 `workspace/` 按日期顺序覆盖到同一个工作区后，构成完整68-Agent池。

## 2. 本次补齐的12个 Agent

### 旧 Rank40 验证型：9个

- `rank14_kaan_diniz_validation_v1`
- `rank18_mandgeee_validation_v1`
- `rank19_xiaowenhao404_validation_v1`
- `rank20_u_validation_v1`
- `rank22_victor_tufa_validation_v1`
- `rank26_efe_validation_v1`
- `rank27_atakan_validation_v1`
- `rank33_unknownrobi_validation_v1`
- `rank35_webmaking_validation_v1`

这9个 Agent 的路线库和报告此前已经进入2026-08-25快照，本包补齐缺失的最终 `validation_firstshop` 路由表。

### 当前 Top20 Replay 重建：3个

- `top20_20260825_tetsuya_current_firstshop_v1`
- `top20_20260825_hasegawa_new77_firstshop_v1`
- `top20_20260825_crop_dusta_current_unrestricted_v1`

每个补齐：

- JAX路线库；
- 公开状态路由表；
- 对FC24B的独立双座位验收回执。

这些是依据公开 Replay 构造的公开状态策略代表，不代表恢复了作者闭源源码。

## 3. 完整68池构成

| 类别 | 数量 |
|---|---:|
| FC24B冻结基线 | 1 |
| 历史公开严格JAX池 | 28 |
| 旧Rank40保留重建 | 35 |
| Kaito V48 | 1 |
| 当前Top20新增重建 | 3 |
| 合计 | 68 |

权威机器清单：

```text
workspace/experiments/front40_fusion_v1/configs/complete_validation_pool_v1.json
```

人读总清单：

```text
workspace/docs/LOCAL_AVAILABLE_AGENT_INVENTORY_20260825_ZH.md
```

队友 Git 中68项逐个定位索引：

```text
AGENT_INDEX_68_ZH.md
```

## 4. 组装方法

四个快照是按日期递增的覆盖层。建议创建一个新的空工作目录，不要直接修改冻结快照：

```text
latest_20260821_arena_fusion_fc2a/workspace
  -> latest_20260823_fc24b/workspace
  -> latest_20260825_front40_public_v48/workspace
  -> latest_20260827_complete_68_agent_pool/workspace
```

PowerShell示例：

```powershell
$NtRoot = 'E:\ai_coding\kaggle\kaggriculture\.publish\LZhangGJ_Kaggriculture\nt'
$Target = Join-Path $NtRoot '_assembled_complete_68_workspace'

New-Item -ItemType Directory -Path $Target -Force | Out-Null

foreach ($Layer in @(
    'latest_20260821_arena_fusion_fc2a',
    'latest_20260823_fc24b',
    'latest_20260825_front40_public_v48',
    'latest_20260827_complete_68_agent_pool'
)) {
    Copy-Item `
        -Path (Join-Path $NtRoot "$Layer\workspace\*") `
        -Destination $Target `
        -Recurse `
        -Force
}
```

组装后首先读取：

```text
_assembled_complete_68_workspace/experiments/front40_fusion_v1/configs/complete_validation_pool_v1.json
```

不同运行时的常用入口：

- 历史28 Agent：`experiments/expert_business_agent_v2/tools/run_all_exact_jax_round_robin.py`
- Trace Agent 对历史池：`experiments/front40_fusion_v1/tools/run_trace_backbones_vs_old28_batched.py`
- Trace Agent 对 Trace Agent：`experiments/front40_fusion_v1/tools/run_trace_map_vs_trace_map.py`
- Trace Agent 对 Kaito V48：`experiments/front40_fusion_v1/tools/run_trace_map_vs_kaito_v48.py`

`complete_validation_pool_v1.json` 是跨运行时总清单，不应假设一个旧 Arena 命令能自动调度全部68个实现类型；应根据 `implementation_kind` 使用对应运行入口。

## 5. 验收边界

- 本地权威清单计数为68；
- 本次缺失12个的全部源文件存在；
- 新增文件没有超过GitHub 100 MiB单文件限制；
- 四层合并后，权威清单中所有显式配置、路线库、路由表和验收回执均有对应文件；
- 68仍是跨运行时行为完全去重前的配置总数，不代表68种数学意义上互不相同的策略；
- `trace_*` 和 `gold_proxy_*` 必须保留代理/重建标识。

## 6. 完整性

`MANIFEST_SHA256.tsv` 记录本增量包内除自身外全部文件的相对路径、字节数和SHA256。旧快照使用各自独立的 manifest 验收。
