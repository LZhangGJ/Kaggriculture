# Public pool refresh: September 14, 2026

The requested 16 notebooks contain 15 distinct packages. One of those packages also matches the existing V41 entry. The resulting roster has 22 agents: six team agents and 16 public implementations.

Deduplication compares every package file plus the run command, build command, and resource settings. It ignores ZIP timestamps. Shared strategy ancestry alone does not qualify as a duplicate.

| Notebook | Arena entry |
|---|---|
| premaananda108/rule-agent-ecobot-v7-arena-analytics | Ecobot V7 |
| guruprasaathas111/kaggriculture-master-engine-v3 | Alias of existing V41 |
| aurax7/kaggriculture-shop-router-reactive-v5 | Reactive V5 |
| arsgorynich/kaggriculture-v40-challenger | V40 Challenger |
| alperen5252525/kaggriculture-metacounter-r1-scored-agent | MetaCounter R1, retained |
| tetsutani/market-smart-farming-kaggriculture | Alias of Reactive V5 |
| aurax7/kaggriculture-shop-router-reactive-v4 | Reactive V4, retained |
| ahmedberatozer/kaggriculture-v38-smarter-feed-stronger-margins | V38 |
| pilkwang/kaggriculture-structured-economic-policy | Structured Economic Policy |
| lynnsakurai/farming-score-v4-a-better-shop | Farming Score V4 |
| ahmedberatozer/more-yield-smarter-labor | More Yield, Smarter Labor |
| ahmedberatozer/kaggriculture-v25-new-production-routes-with-rea | V25 |
| ahmedberatozer/kaggriculture-v23-adaptive-routes-smart-sales | V23 |
| ahmedberatozer/kaggriculture-v36-guarded-four-turn-sales | V36 |
| flexonafft/kaggriculture-most-powerfull-route | Most Powerfull Route |
| kunaldesale2408/kaggriculture-2026-v1 | Kaggriculture 2026 V1 |

V41 and Utils V1 remain as the strongest existing public references in the cumulative BT snapshot checked at 16:23 UTC. Their fitted strengths were 1.490 and 1.194, respectively, with more than 9,200 games each. These are internal results, not Kaggle scores.

Retired public entries: evgendvorkin/Kaggriculture, renjistarfall/Best Agent, Shop Router 0913, and Adaptive Land Allocator. The first two have identical packages. Their internal results trailed the retained references; Land Allocator's results also include execution problems, so they do not establish its strategy's strength.

Retired entries remain visible with their Elo frozen at retirement. Their sources no longer refresh. Previously frozen rounds and tournaments finish unchanged; new rounds use the new roster.

All 15 distinct requested packages passed sandbox build checks and one complete game in each seat against AFS R2: 30 terminal games, each reaching 720 observations. These checks establish runtime compatibility, not comparative strength. V38, V25, and V23 were extracted from notebook source without executing notebook cells, and their published source hashes were checked.

The two duplicate package SHA-256 values are:

- V41 / Master Engine V3: `67cbfe1ab2d131a9de6e6a0cb0f9c2bc49470af113ec6e82bb0703e5bcb2b5a4`
- Reactive V5 / Market-Smart: `0bc86bff40f7c2eb2420f7648a1becf628776a92b16960fa4852521e7b5e8e64`
