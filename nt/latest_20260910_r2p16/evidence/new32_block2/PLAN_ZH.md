# 用户追加：再测一批全新种子

三个版本保持冻结：原发布R2、P12、P16。三者使用相同的新32种子2609125000–2609125031；每个种子对11个原程序实时对手、双座位，共704局/版，2112局总计。启动前检索本实验既有PROTOCOL未发现此seed段；最终预留2609130000–0099不使用。

不修改策略、配置、对手或规则；只让已有验证脚本支持独立输出目录与种子起点。每版二进制哈希与上一批相同，协议启动前落盘。一次只启动一个16-worker面板，避免争抢影响时延测量。

报告本批胜负平、逐对手胜率、现金/分差与配对救回/丢旧胜；并合并上一批三者共同测过的2609124000–4031，形成公平的64种子比较。不要拿P12独有的更早32seed直接对比P16。

边界：这是追加的确认实验，不因某批更好就只报好的一批；不自动晋升或提交。先前已重复开发筛选，最终90%目标仍须另外完成。

```text
wsl.exe -d Ubuntu-24.04 --cd /mnt/e/ai_coding/kaggle/kaggriculture .venv_wsl_cpp/bin/python experiments/r2_public11_upgrade_20260910/validate_startup_supply.py --seed-start 2609125000 --output experiments/r2_public11_upgrade_20260910/startup_supply_validation32_b2
```
