# R15C：78/160，07对手0/16

这是基于用户包内 A06 R12 修改的纯规则实验 Agent，**未达成每个对手80%或90%胜率，不是已验收的替换版**。完整对照、失败记录和两版取舍见配套证据包。

入口：main.py；同时需要 policy/ 下匹配的源码、config.json 和 a06.so。本二进制在当前 Linux x86-64 环境运行验证，未在 Kaggle 线上提交验收。环境不匹配时使用 C++20 编译器重建：

```bash
python build.py --unit
```

已有二进制可先检查：`python tests/run_units.py`。该检查仅覆盖ABI、配置和上下文，不替代完整比赛测试。

RUNTIME_MANIFEST.json 记录实际运行配置；原 a06.BUILD.json 是历史编译收据，运行配置可能与其中的旧快照不同。不使用ML/RL；learned-value权重已删除，负scenario拒绝。
