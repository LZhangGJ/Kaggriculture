# FC0：基线冻结与验收口径

日期：2026-08-21  
状态：PASS（基线已冻结，尚未产生新候选）

## 冻结证据

- 严格 JAX 全池回执：`experiments/expert_business_agent_v2/receipts/all_exact_jax_round_robin_n28_seed530001_n50x2_v1.json`
- 回执 SHA256：`F4A0E02C4333B69416C949398EEA9C99E82B4456F6B00CF05987EABBB102BB02`
- 对手池：28 个去重后的完整 Agent。
- 历史基线：378 组、每组 100 局、50 个事件种子、双方换座，共 37,800 局。
- 官方规则：Kaggriculture 1.32.7。
- 完整性：全部终局完成；hand cap、market loop cap、price LUT 越界均为 0。

## 当前基座

`rayk_k320_adaptive_rank1`：

- 全池排名第 1；
- 总得分率 86.7%；
- 2,342 胜、0 平、358 负；
- 平均现金 94,565；
- 平均分差 +9,356。

K320 对 27 个对手中，只有 14 个已经达到至少 90%；仍有 13 个低于目标。

| 对手 | K320 得分率 |
|---|---:|
| gold_proxy_rank14_recursion | 24% |
| public_g04_soil_rain | 64% |
| flexonafft_v59_multi_route | 65% |
| public_g01_boatlee_v16 | 74% |
| gold_proxy_rank12_ai_b2b67_saas | 77% |
| public_g02_rc5_c166 | 80% |
| local_prt_v6 | 80% |
| deniz_v111_8c4s_latest | 80% |
| kaito_v36_latest | 80% |
| x562_latest | 86% |
| gold_proxy_rank07_junichiro_morita | 88% |
| gold_proxy_rank19_manu_nicholas_jacob | 88% |
| kaito_v27_midgame_reset | 89% |

## “直接选现有最强拳”是否足够

不够。历史矩阵显示：

- 对 `rank14_recursion`，现有最佳是 `x562_latest`，也只有 72%；
- 对 `local_prt_v6`，现有最佳仍是 K320，只有 80%；
- 对 `x562_latest`，现有最佳仍是 K320，只有 86%；
- 对 `rank07` 和 `rank19`，现有最佳仍是 K320，只有 88%；
- 对 `rank12`，现有最佳只有 77%。

因此，即使能够完美识别对手并从现有 Agent 中进行神谕路由，也无法完成全对手 90%。正式工作必须创造新的规则能力，而不是只做身份分类或现有路线拼接。

## 指标与数据边界

### 优化主指标

```text
min_opponent_score_rate
```

次级指标依次为：

1. 达到 90% 的对手数量；
2. 全池平均得分率；
3. 最坏 5 个对手平均得分率；
4. 平均分差和 P10 分差；
5. 硬错误、在线 CPU 延迟和 JAX 吞吐。

### 事件种子

- 历史基线：530001--530050，只用于比较，不参与新规则选择；
- 开发筛选：540001 起；
- 独立验证：550001 起；
- 最终冻结验收：560001 起。

### 真实性边界

- `gold_proxy_*` 是逐步一致性验收通过的模仿 Agent，不等于原金牌源码；
- 本地固定对手池 90% 不等于 Public 90%；
- 在线条件不得使用对手身份、隐藏状态或未来信息；
- 规则只有在新种子和双方换座下继续提升，才视为有效。

## 第一批研究对象

1. `rank14_recursion`：最大短板，先解释 24% 的结构性原因；
2. `Soil Rain` 与 `Flexonafft`：验证是否属于同一市场/经营碰撞家族；
3. `rank12`、`local_prt_v6`、`x562`：现有路线神谕也不足 90%，需要新能力；
4. 对 K320 原本 95%--100% 的强对局建立回归门，防止修一个弱点却破坏优势面。

