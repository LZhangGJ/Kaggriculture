# 最新公开 8 方案源码冻结与迁移分流报告

日期：2026-08-20
官方裁判：`kaggle-environments==1.32.7`
状态：**源码冻结 PASS；JAX 全量迁移尚未完成**

## 1. 冻结范围

来源目录：

`public_notebooks/recent_latest_20260820_192648`

冻结目录：

`references/public_latest8_20260820`

采用纯静态提取，不执行 Notebook 中的任意代码。若 Notebook 自带
`main.py` 哈希，则必须与提取结果完全一致；所有提取结果必须通过
Python 语法编译。

机器可读清单：

`references/public_latest8_20260820/manifest.json`

## 2. 源码结果

| 页面身份 | `main.py` SHA256 | 字节 | 初步迁移分类 |
|---|---|---:|---|
| Deniz V111 | `f029fa0cb66a9eb509afbe44e3f59b800332d0419db91607183410e4089c4d19` | 18,946 | RC5 内核复用候选 |
| Boatlee V20 | `8ac34abce129cf5c9456776c90edf7d2233b3a280bbdcf7622628825ef3669a0` | 148,200 | 与既有 Exact5 源码完全相同 |
| Kunal 2026 V1 | `8ac34abce129cf5c9456776c90edf7d2233b3a280bbdcf7622628825ef3669a0` | 148,200 | 与 Boatlee V20 完全相同 |
| Ray Rank Agent | `6c709f6d3ce6cf221a9495de7e716fcd1b660e3bbc8ee5679b63233d0265a812` | 154,027 | 与既有 Exact5 源码完全相同 |
| Kaito V36 | `47ebf29039463dc0eb803ccf38d5a6f0c130d2b49f3698b20c53f495c1062dc8` | 46,395 | 新固定混合路线与稀疏反馈外壳 |
| X562 | `e4980a8548baa1f3e3ae51a92b1759fdeaaf8ff0ca517c1e8094ce5c015dc915` | 84,331 | 新 feasible-window / seed-trim 逻辑 |
| Tetsutani 最新版 | `c26402b67a0d04a46348353069645b1a49c3cb3df6df69d7fa35d8adbbdbeae4` | 148,034 | 相对旧 Exact5 为新源码 |
| Flex 最新版 | `9bdfbafb6755067182d88ce594fd46fb1d712713ffd6931e83d5d50e84bc6fb2` | 21,330 | Notebook 声明为 byte-exact public v25 artifact |

8 个页面实际只有 7 个唯一 `main.py`。Kunal 与 Boatlee 不是两套不同
Agent，而是完全相同的源码；二者应共享同一个 JAX 实现，但保留两个
页面身份的验收记录。

## 3. 官方 Python 整局烟雾测试

测试口径：

- 固定独立种子 `2082001`；
- 每个页面身份双座位；
- 对手为静止对手；
- 每局完整 720 帧；
- 共 16 局。

结果：16/16 局均为双方 `DONE`，无 `ERROR/INVALID`。

| 页面身份 | seat0 现金 | seat1 现金 |
|---|---:|---:|
| Deniz V111 | 144,286 | 172,525 |
| Boatlee V20 | 191,595 | 191,595 |
| Kunal 2026 V1 | 191,595 | 191,595 |
| Ray Rank Agent | 191,835 | 191,835 |
| Kaito V36 | 166,872 | 169,371 |
| X562 | 192,601 | 192,601 |
| Tetsutani 最新版 | 191,595 | 191,595 |
| Flex 最新版 | 157,338 | 163,263 |

机器可读收据：

`experiments/expert_business_agent_v2/receipts/latest_public8_official_smoke_v1.json`

这些现金只证明源码能够在官方环境完整执行，不是 Public 分数预测，也
不能用于跨日期判断新旧版本强弱。

## 4. JAX 迁移分流

迁移按以下顺序执行：

1. Boatlee V20、Kunal、Ray：源码与已验收 Exact5 完全相同，直接重跑
   最新页面身份的严格 parity，不重复写控制器。
2. Deniz V111：验证其 RC5 源码是否可直接复用现有
   `public_rc5_weed_player_action_v1`。
3. Flex 最新版：验证其公开 v25 artifact 是否可直接复用
   `public_v25_player_action_v1`。
4. Tetsutani 最新版：对旧 Exact5 Tetsutani 动态外壳做源码差异审计，
   只补新增分支。
5. Kaito V36：提取新 route、clone 检测、稀疏反馈和市场配置，建立新
   carry 与动作函数。
6. X562：实现 feasible-window、一次性种子裁剪和异常恢复。

## 5. 精确性边界

当前只完成源码冻结和官方 Python 烟雾测试。除源码完全相同的已有
Exact5 身份外，其余方案尚不能标为 `EXACT_PARITY_ACCEPTED`。

每个新身份最终必须通过：

- 双座位；
- 独立随机事件；
- 719 个决策步；
- 所有单位动作和 10 个市场槽逐字段一致；
- 每帧公开/私有状态一致；
- 终局现金与奖励一致；
- 批量 reset 无跨局状态污染。

只复现固定路线、但没有复现动态路由、杂草恢复、市场顺序或容量保护的
实现，只能标记为部分迁移。
