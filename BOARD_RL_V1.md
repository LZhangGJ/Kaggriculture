# 双棋盘 PyTorch Agent V1

本版本在 `agent/triton-inventory-market` 的现有 PyTorch Agent 上增加结构化双棋盘编码，继续复用：

- `VectorFastEnv` 本地并行环境；
- 现有单位/市场动作词表；
- 合法动作 mask；
- scripted-agent 行为克隆流程；
- CUDA/Triton 环境实现。

## 输入

- `boards`: `[B, 2, C, 10, 10]`，按“自己、对手”排序；
- `global_features`: 时间、座位、双方现金、土地、市场、商店顺序和自己的私有库存；
- `unit_features`: `[B, 2, 17, F]`，包含双方公开位置和自己的私有携带库存；
- `unit_mask`: 当前存在的单位。

对手私有 shed、seeds 和单位 inventory 不会进入模型；未来随机事件和模拟器内部状态也不会进入模型。

## 网络

两个农场使用同一个 CNN，随后融合：

```text
self, opponent, self-opponent, abs(self-opponent)
+ 双方单位集合编码
+ 全局经济/市场编码
```

输出仍与原 PyTorch Agent 兼容：

- 每个单位一个动作分布；
- 每回合一个市场动作分布；
- 一个 value 预测。

第一版暂不改变原动作空间，因此还不能在同一回合生成多条市场订单。后续应升级为候选完整 Action 或自回归市场订单。

## 本地训练

可以先用现有高分 Agent 做行为克隆预热：

```bash
pip install -e ".[gpu,dev]"
pytest tests/test_board_policy.py -q
python scripts/train_board_bc.py \
  --envs 128 \
  --updates 2000 \
  --teacher-a starter \
  --teacher-b starter \
  --device cuda
```

输出默认为 `artifacts/board_bc_v1.pt`。把两个 teacher 参数换成可由
`resolve_agent` 加载的高分 Agent 文件或 `module:callable`。

随后运行真正的自博弈 A2C：

```bash
python scripts/train_board_rl.py \
  --envs 32 \
  --updates 500 \
  --rollout-steps 64 \
  --init-checkpoint artifacts/board_bc_v1.pt \
  --device cuda
```

若不传 `--init-checkpoint`，则从随机权重开始。双方共享同一个策略网络，
奖励是 actor 视角下 `log(自己资金) - log(对手资金)` 的逐步变化；它是反对称的，
并在 `gamma=1` 时望远镜求和到最终相对财富。终局另加胜负奖励，GAE 用于长时序信用分配。
默认输出为 `artifacts/board_a2c_v1.pt`。

V1 的 RL 重点是验证完整数据链和可学习性。后续建议加入历史对手池、整局 Action
候选头或自回归市场订单，再升级 PPO 并在固定种子联赛上做门控。
