# A06 R12 Rule-R18

2026-09-24，规则/DP 版本，无 ML/RL 训练模型。固定十对手各 16 局：115/160 胜（71.875%），未达到 80%。

保留整个目录。入口为 `main.py:agent(observation, configuration)`；原始源码加载器的最后 callable 为 `submission_entry`。
运行需要 Linux x86-64/WSL、Python 3 及兼容的 C++ 运行库。预编译 `policy/a06.so` 已随包提供。
不是 Windows 原生 DLL，也不是 macOS/ARM 预编译库。

需要本地重编译时：

```bash
python3 build.py --unit
```

编译需要 g++ 的 C++20 支持。Agent 的 Python 层只用标准库。不要单独移动 main.py。
测试使用本地 C++ 模拟器并作独立规则一致性核对，没有官方线上/限时器认证。
本面板用于开发，不是独立保留集，不能当作天梯胜率保证。

最终配置：竞争权重 1.5，资本幂/折扣 0，scenario=0，盘中竞争契约=3，雇工上限 12，day 0–20 的规划用工机会成本 6，day 21 起为 4，批量交付模式 2，SELL 排序模式 3。

`learned_value.hpp` 是禁用存根，不含原先训练树。当前版本不使用实验性的公开开局分支。
完整报告、原始对手及复现脚本见同名 handoff 包。`RELEASE_MANIFEST.json` 标识当前经过测试的运行文件。
