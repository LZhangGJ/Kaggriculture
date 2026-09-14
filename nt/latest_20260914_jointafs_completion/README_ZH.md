# JointAFS Completion：已提交x86版本

Kaggle提交 **56225552**，2026-09-14，状态Complete、提交错误为空。提交名称：NT JointAFS Completion - immediate sale + exact memo (portable x86)。evidence/kaggle_submission.json中的600.0是提交时分数快照，不是最终排名。

相对队友高分R1，保留联合动物/饲料/继作宏观规划：在雇工估计与任务编译中启用完成式regret分工，原方案完整则保留；替代分工必须全部完成才采用。关闭延迟销售策略。另通过插入计算不变量外提、必定被拒绝的部分构造早停、复用既有完整输入packmemo做等价加速。没有引入旧autonomousPlanner框架。

## 文件和运行

- source/policy：完整42文件源码与配置，逐文件哈希见MANIFEST.json。
- submission.tar.gz：与Kaggle实际上传包逐字节相同，内含main.py、afs_runtime包和x86 agent.so。没有ARM库、QEMU或工具链依赖。
- main.py：提交包入口的源码副本；使用时应先解包完整tar，不能单独复制main.py。
- build.py：在Linux x86-64/GCC11环境重建，输出rebuilt_submission.tar.gz，不覆盖实际上传包。使用CXX环境变量指定编译器，例如CXX=g++-11 python3 build.py。实际提交由隔离Ubuntu22.04 GCC11交叉工具链构建，静态链接C++运行库，最高GLIBC需求2.34；重建后的实际运行/跨架构行为仍需验证。
- verify.py：python3 verify.py核对源码、实际tar及x86 ELF身份。
- evidence：结果与提交回执；不包含训练数据、凭据或本地工具链。

运行原提交：mkdir agent && tar -xzf submission.tar.gz -C agent；在Linux x86-64环境将agent目录加入Python路径后调用main.agent(observation, configuration)。原wrapper支持双座隔离与新局重置。官方时间预算每步1秒，加每agent整局共享60秒加时；不是每步硬限2秒。

## 完整对战结果

每个对手128新seed双座=256局，各座位128局。八公开对手，前一默认completion版1838/2048（89.75%）→本版1917/2048（93.60%），提升3.86个百分点，配对seed簇95%区间[+1.90,+5.86]。公开对照不是原R1，g001少赢2局，其余七个增胜。

对原高分R1（56146577）233胜23负，91.02%；对原高分R2（56149565）210胜46负，82.03%。R1是已核实源码的ARM复现；R2直接用QEMU执行原上传x86库，不使用后续workflow修复版，且不因模拟速度判对手输。R2原C++源码身份仍未知，原可执行身份已核实。R1/R2采用不同新seed，不作为彼此同seed强弱比较。

等价加速704个旧样本回归终局一致，其中512原生动作哈希一致，不另算新胜率样本。实际提交x86包与已测ARM版4局2876动作全部一致，旧GLIBC系统库加载检查通过。我方R2对打累计加时最大0.844秒/60秒，本机样本较宽裕，不等同Kaggle托管硬件计时认证。
