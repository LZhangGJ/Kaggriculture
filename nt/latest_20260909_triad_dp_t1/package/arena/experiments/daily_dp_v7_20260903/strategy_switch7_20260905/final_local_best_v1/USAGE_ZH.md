# 最终本地Agent使用说明

在WSL中将当前目录加入`sys.path`后：

```python
from final_local_best_v1.local_agent import Agent
agent = Agent()
agent.reset()
action = agent(observation)
```

必须逐局`reset()`。该入口依赖当前工作区中的冻结打包器与`agent.so`，不是可直接上传Kaggle的单文件包。
