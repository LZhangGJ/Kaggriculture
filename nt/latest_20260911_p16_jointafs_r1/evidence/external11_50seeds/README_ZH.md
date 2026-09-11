# P16 JointAFS R1：11 对手全新种子复验

本实验只验证 `P16 JointAFS R1`。协议严格复用上一轮四版本复验：

- 官方规则：1.32.7；
- 对手：同一组 11 个实时 Python Agent；
- 种子：`2610100000..2610100049`，共 50 个；
- 双座位：每个对手 100 局；
- 总计：1,100 局；
- 我方默认自主开局，不输入 seed、对手身份或未来 Replay；
- 16 个进程。

父版 `T3R1 TakeoverMerged R1` 在该协议下的 1,100 局结果已经完成，直接按
`对手 + seed + 座位`配对复用，不重复消耗时间。

运行：

```bash
cd /mnt/e/ai_coding/kaggle/kaggriculture
/home/mitubant/vllm-env/bin/python experiments/p16_jointafs_r1_newseeds_public11_100_20260910/run.py
```

输出：`PROTOCOL.json`、`PROGRESS.json`、`rows.json`、`paired.json`、`RESULTS.json`。
