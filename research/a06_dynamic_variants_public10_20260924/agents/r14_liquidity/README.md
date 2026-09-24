# A06 R14 Liquidity

R12 → R13 → R14，纯规则／动态规划，无 ML／RL。运行入口为本目录 `main.py` 的 `agent`；必须保持 `policy/` 与 `main.py` 的相对目录关系。

已附 Linux x86-64 原生库 `policy/a06.so`。Python 包装仅使用标准库；不需要 GPU。非 Linux x86-64 系统不应直接加载该 `.so`；Windows 可在 WSL 内重编译。

```bash
python3 build.py --unit
```

需要支持 C++20 的 g++。本次实际编译器为 Debian g++ 14.2.0。编译参数、源码散列和原生库散列在 `COMPILER_FLAGS.json`、`policy/a06.BUILD.json`。

```python
from main import agent
action = agent(observation, configuration)
```

每个 Agent 实例对应一个对局；`main.agent` 隔离两个 seat，并在新的 step 0 重置对应状态。`main.create_agent()` 可生成显式独立实例。

有效改动只有 `opening_liquidity: 10`：标准初始状态下，在 R13 原有市场订单之前增加同回合买小麦 10、卖小麦 10；现有单位动作、原订单相对次序以及后续动态规划保持不变。资金、仓储和订单槽不足时限制数量或不发起，不假定市场订单必然成交。设 `opening_liquidity: 0` 可关闭该改动。

这是冻结附件模拟器上的运行包，不含 Kaggle 线上验证承诺。完整测试结果与边界见完整包 `REPORT_ZH.md`。
