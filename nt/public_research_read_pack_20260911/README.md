# Kaggriculture 中英轻量阅读包 / Bilingual reading pack

[下载阅读包 / Download ZIP](KAGGRICULTURE_READ_PACK_ZH_EN_20260911.zip) · **3.44 MB**

解压后打开 **START_HERE.html**，选择中文或英文。两份报告均内嵌 514 份逐项审阅、27 个重点来源、16 张待验证实验卡及 405 条历史待办处置。中文路径：全量索引 → 搜索 → 完整审阅记录与本地来源。

Extract the ZIP and open **START_HERE.html** to choose Chinese or English. Both reports embed 514 individual reviews, 27 selected sources, 16 proposed experiment cards, and 405 followup dispositions. In English, use Full index → search → Full review and local source references.

本包不含原始源码附件、模型、Replay 或完整研究档案。原始附件链接需要另传对应材料；阅读逐项笔记不需要这些附件。英文历史审阅采用已标注的本地机译，核心分析和实验卡已单独校对。所有改进建议仍是 PROPOSED_NOT_RUN；页面交互验收仍受原 URL 策略限制，不声称已通过。

Raw source attachments, models, replays, and the full research archive are excluded. Local attachment links require a separate scoped evidence pack; the embedded reviews do not. English archival reviews use disclosed local machine translation; the core synthesis and experiment cards were checked separately. All ideas remain PROPOSED_NOT_RUN. The original browser interaction limitation remains explicit.

## 校验 / Verification

`python verify_package.py` checks the ZIP's CRC, every payload SHA256, both embedded review counts, and the untested status of all 16 ideas. It does not run a browser or an Agent. The reports are byte-identical to their accepted local originals.

- ZIP SHA256: `d3e108545100fceec0897f231bf74b8da23df118612e0de79c46d2bef8873071`
- [Payload manifest](PACKAGE_MANIFEST.json)
- [Package acceptance](PACKAGE_ACCEPTANCE.json)

## 仓库体积说明 / Archive rationale

用户明确要求将两种语言一起打包并推送，供中美队友直接下载。这份 3.44 MB ZIP 是本目录唯一的报告载荷；Git 中不再重复提交解压后的 HTML、巨型 JSON 或任何模型缓存。此处将必要的小型交接包及哈希清单作为仓库上传规范的有说明例外。

The team explicitly requested one small bilingual download in Git. This 3.44 MB ZIP is the sole report payload: the unpacked HTML and large duplicate JSON are not also committed. This documents the necessary small handoff-archive exception to the repository's upload policy. No models, build outputs, or replay corpora are added.
