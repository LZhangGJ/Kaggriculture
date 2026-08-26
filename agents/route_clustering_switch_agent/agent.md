# Agent maintenance guide

本文约束在 `agents/route_clustering_switch_agent/` 内工作的代码 Agent。目标是让修改
保持可复现、可提交，并且不误伤 Kaggriculture 仓库的其他目录。

## 工作边界

- 默认只读写本目录及其子目录；不要修改父目录、兄弟 Agent 或仓库级配置。
- 若任务确实需要跨目录变更，先说明具体文件、原因和影响，获得用户明确授权后再做。
- 工作区可能已有用户修改。先检查本目录的状态，保留无关改动，不覆盖或回滚它们。
- Replay、生成矩阵、搜索缓存和本机构建产物必须写到源码树外的 `OUTPUT_ROOT`。

## 先识别修改层级

- `main.py`：生成的单文件提交，不作为日常手工编辑入口。
- `runtime/`：已导出的可读提交快照，包含最终路线库、动作带和策略。
- `src/meta_agent/`：特征、策略控制器和执行器的开发源码。
- `scripts/`：离线数据处理、搜索、评估和导出逻辑。
- `fast_kaggriculture/`：C++20/pybind11 仿真器；其独立说明见本目录内 README。
- `docs/`：方法说明与图表，不是运行时依赖。

修改行为逻辑时，优先修改 `src/`、`scripts/` 或原生源码，再用
`export_teammate_meta_submission.py` 和 `export_single_file_submission.py` 生成成品。
不要直接修改压缩后的 `main.py`。`runtime/` 是导出快照，和 `src/` 的同名文件不一定
完全相同；不要进行未经验证的双向覆盖。

## 关键不变量

- 当前 Agent 是基于路线搜索的浅决策树策略，不依赖 PPO、PyTorch 或 `.pt` 权重。
- 当前在线策略从 G001 开局，在 144、168、216 checkpoint 判断，最多切换一次。
- 搜索和评估必须使用 `runtime/teammate_base.py` 的路线干预接口。
- 特征 schema 为 `semantic_route_switch_v1`；改变 schema 时必须同步训练、导出和运行时。
- 原实验的 275/175/11/63 等数量来自特定数据，不是换数据后仍成立的常量。
- `OUTPUT_ROOT` 应位于源码树外；不要把大体积生成物加入普通 Git 历史。

如有意改变这些不变量，应同时更新 README、运行时 manifest、相关测试和方法文档，
并明确说明兼容性影响。

## 推荐工作流程

1. 阅读 `README.md`，再按任务查看方法文档、脚本或原生模块。
2. 用 `git status --short -- agents/route_clustering_switch_agent` 检查已有改动。
3. 只修改完成任务所需的最小文件集；不要顺手清理无关文件。
4. 先运行与改动最相关的测试，再执行下面的基础验证。
5. 若重新导出提交，对比 `runtime/MANIFEST.json`、生成日志和最终文件，并记录输入来源。
6. 交付时列出修改文件、验证命令以及未执行的高成本步骤。

## 验证清单

文档或轻量 Python 修改至少运行：

```bash
python -I main.py
python -I runtime/main.py
PYTHONPATH=src pytest -q tests
```

涉及原生仿真器时，另在源码树外构建并运行其测试：

```bash
bash scripts/build_native.sh /path/to/output-root
pytest -q fast_kaggriculture/tests
```

涉及训练、路线数量、checkpoint 或收益结论时，应使用独立 `OUTPUT_ROOT` 运行对应流水线
阶段，并以新生成的摘要为准。完整流水线成本很高；未运行时必须明确写明，不得把静态
检查描述成实验复现。

## 数据与文档约定

- Replay 清单遵循 `configs/manifest.example.json`，Replay 路径相对清单文件解析。
- 配置文件遵循 `configs/pipeline.env.example`；本地 `configs/pipeline.env` 不提交。
- 来源、恢复过程或可信边界变化时更新 `SOURCE_PROVENANCE.md`。
- 删除或外置大型产物时更新 `OMITTED_ARTIFACTS.md`，保留原因、大小和可用哈希。
- README 面向首次使用者，优先保持入口、验证、复现步骤和结果边界准确；深入公式留在
  `docs/`，原生实现细节留在 `fast_kaggriculture/README.md`。
