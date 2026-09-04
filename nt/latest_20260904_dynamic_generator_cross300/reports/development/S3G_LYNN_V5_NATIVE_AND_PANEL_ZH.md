# S3G：Lynn V5 完整 C++ 对手接入与开发面板

日期：2026-09-03。结论：迁移验收通过；自主 v7 最终目标未通过。

## 范围与实现

冻结 `opponents/lynn_v5/output/generated_submission/` 全包，不能只取325字节入口。17个Python/配置源文件哈希由 `receipts/native_lynn_v5_static_export.json` 保存。原始包不修改。

原版有效链为：Kenjo原始动作 → 牛羊需求替换 → 杂草恢复 → 同回合仓库投影、SELL额度和顺序、融资过滤 → 动物末回合清仓 → 潜在牧场激活 → 配送修正 → 追加羊分流 → 全产品清仓。

C++分别保存旧配送、修正配送、分流配送状态。某层取消后上层继续执行是原版真实行为，不合并为一个布尔开关。去掉的是Python全局函数临时替换机制，不是策略分支；每局状态独立。原版报价/仓库估算中的近似也保留，不以“修复对手”改变基准实力。我方经济策略源码及官方规则未改。

## 验收证据

| 检查 | 结果/位置 |
|---|---|
| 官方1.32.7完整逐步对照 | 10局×719步，6层动作、全部行为记忆、双方官方状态差异均0；`receipts/lynn_original_official_parity_v2/` |
| 分支覆盖 | 4种牛羊选择结构、追加配送/分流、杂草恢复；10局录像已保存 |
| 临界条件夹具 | 291组、904次调用，含价格/现金/容量阈值、暂缓信号、同一步重复和时间回退，差异0；`receipts/lynn_edge_branches_v1/` |
| 隔离 | 100局串行+1,000局16线程复跑，另6条完整轨迹hash对照，差异0；`receipts/lynn_context_isolation_v2/` |
| 原机制回归 | 546项通过；`receipts/lynn_unchanged_policy_mechanisms_v1/` |
| 旧对手回归 | Boatlee、Kaito、Fieldbook在新构建重新逐步核对和1,000局隔离；原面板4,800条记录完全不变 |
| 总验收 | `receipts/lynn_integration_acceptance_v1/acceptance.json` |

有限夹具不是所有合法状态的数学等价证明；官方对照使用冻结解释器本地主持器，不冒充最终完整Kaggle超时/schema验收。1,000局重复只验证隔离，不当作1,000个独立样本。

构建SHA256：`ed2aede8ebe3548b6858bea4d89d893d751784ae158e633bfb2b6ecc9e30a2b7`。
Lynn数据SHA256：`ba6bf9da8ed87c18d122ee0c93f05a19126c88b2755203523839aa3c61f09aab`。

## 对 Lynn 的开发成绩

冻结同一对照配置。A为20261401–20261450，B为20261501–20261550，均双座位各100局；这些均是既有开发集。

| 我方 | A胜场 | B胜场 | 合计胜率 | 我方平均现金 | 对手平均现金 | 平均分差 |
|---|---:|---:|---:|---:|---:|---:|
| L3_base / S3C03 | 17/100 | 19/100 | 18% | 96,483.16 | 109,233.49 | -12,750.33 |
| L3_return 暂定物流修正 | 19/100 | 26/100 | 22.5% | 96,994.59 | 108,664.69 | -11,670.11 |

按seed聚类的近似95%区间分别为10.56%–25.44%、14.33%–30.67%，不是排行榜保证。没有新参数晋级，最终holdout未使用。

吞吐：16线程1,000局重复0.931秒，约1,075局/秒。这里是一整局719步、我方完整规划器对Lynn、纯C++循环；不包含编译、磁盘录像和官方Python复验，不是所有对手或推理环境的通用速度承诺。

## 当前池与后续

6/8真实对手已接入：G001、G003、Boatlee V29、Kaito V58、Lynn V5、Six-Day Fieldbook。公开EcoBot V7、用户新加Three-Day尚待接入。Three-Day完整下载及哈希核对见 `receipts/three_day_source_acquisition_v1.json`，不替换Fieldbook。

继续补齐两者，然后执行跨对手通用生产/融资/维护消融；不退回只对G001选参数，不把对手的专用路线日期/身份条件接入自主规划器。最终两个硬门不变。

## 运行

用已有WSL Python执行，实验根目录下：

```text
python tools/build_native.py
python tools/check_lynn_native.py --seeds 20261401,20261403,20261404,20261410,20261414 --seats 0,1 --out receipts/NEW_LYNN_PARITY
python tools/test_lynn_branches.py --out receipts/NEW_LYNN_BRANCHES
python tools/test_lynn_isolation.py --out receipts/NEW_LYNN_ISOLATION
python tools/run_opponent_panel.py --opponents pass,g001,g003,yhay81_six_day,boatlee_v29,kaito_v58,lynn_v5 --configs profiles/s3c/hauling_configs.json --labels L3_base,L3_return --seed 20261401 --count 50 --repeat 2 --threads 16 --out receipts/NEW_PANEL
```

输出目录必须新建；旧失败/旧构建/旧收据保留。面板会核对注册的当前构建验收，若重编译改变binary hash，需重新核验对应对手，不能只改哈希绕过。
