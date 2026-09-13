# English workflow

## 1. Use Codex as the central controller

Codex coordinates eight independent ChatGPT 6 Pro chats. It prepares the rules and source materials, assigns development tasks, collects deliverables, runs local matches, analyzes results, and schedules further iterations.

Each chat develops a complete strategy independently. Authors may focus on different areas, such as economic planning, crop selection, market trading, resource allocation, or execution scheduling, while sharing the same rules, interfaces, and evaluation method.

## 2. Establish a 200,000-cash economic baseline

Each author first tests its strategy across multiple seeds without an active opponent. The target is an average terminal cash balance of 200,000 per game.

Authors can use dynamic programming and rolling planning to improve investment, production, maintenance, harvesting, and sales. Performance must come from completed games, rather than predicted returns. Strategies below the baseline continue iterating until they reach it.

Each author round is limited to two hours. Tasks should account for the chat environment’s actual CPU, memory, and execution speed, leaving enough time for implementation, validation, and packaging. Deliverables include complete source code, a runnable strategy, build instructions, and actual test results.

## 3. Run local matches between the eight strategies and select the best two

Codex downloads and verifies the eight strategies, then runs a local round-robin evaluation. Candidates share the same seeds within a round and play both seats to reduce map and seat effects.

The two strongest strategies are selected using the complete match results, considering overall win rate, matchup performance, and consistency. In this project, the selected lineages are A08 and A06. These lineages can produce multiple subsequent versions, including A08 r11, A06 r6, and A06 calendar r12.

## 4. Evaluate the selected strategies against the fixed agent pool

The selected strategies then compete against 11 designated public opponents plus the original R2.

The current formal evaluation draws 64 unused seeds from the public representative-256 and stress-128 collections. All candidates in a round share the same seeds and play both seats against every opponent:

64 seeds × 12 opponents × 2 seats = 1,536 games per candidate.

Each strategy revision receives a fresh evaluation panel. Reports include overall and per-opponent win rates, seat differences, and mean terminal cash.

Competitive acceptance requires an overall win rate strictly above 85%, with no 200,000-cash threshold. On a 1,536-game panel, this requires at least 1,306 strict wins; draws do not count as wins.

## 5. Analyze critical losses and return feedback to the original authors

Codex selects informative cases, including large losses, narrow losses, and recurring weaknesses against particular opponent types. It also retains narrow wins to check for regressions.

Analysis covers economic decisions, rolling scheduling, and execution: whether investments realize their expected value, whether planned tasks are feasible, and whether the executor actually performs the intended actions.

Codex sends the necessary losing-game data, analysis, and concrete improvement tasks back to the corresponding ChatGPT 6 Pro author. The author delivers a revised strategy, which Codex verifies and evaluates on fresh seeds.

This cycle continues until the candidate exceeds an overall win rate of 85% against the designated pool. Formal versions are published to GitHub with source code, English descriptions, evaluation settings, and complete results.

## Main prompts

### Controller goal:

```text
/goal
Act as the central controller for Kaggriculture strategy iteration.

Coordinate eight independent ChatGPT 6 Pro chats to develop complete strategies. First, require each strategy to average 200,000 terminal cash across multiple seeds without an active opponent.

Download and verify the strategies, run local matches between them, select the best two, and evaluate them against our fixed agent pool: 11 designated public opponents plus the original R2.

Use fresh evaluation seeds after each strategy revision. Candidates in the same round share 64 seeds and play both seats, producing 1,536 games per candidate.

Analyze the results and return critical losses, evidence, and improvement tasks to the corresponding authors. Continuously improve economic strategy, rolling scheduling, and execution until the overall competitive win rate strictly exceeds 85%. There is no cash threshold during competitive acceptance.

Plan each author task around its actual CPU, memory, and execution speed, with a maximum of two hours per round. Preserve and publish source code, English descriptions, evaluation settings, and complete results.
```

### Initial author task:

```text
Independently develop a complete Kaggriculture strategy. Read the supplied rules, source code, and runtime materials, then implement changes, run tests, and deliver a downloadable package.

Target an average terminal cash balance of 200,000 across multiple seeds without an active opponent. Use dynamic programming and rolling planning where useful, and ensure the resulting plans are executed.

Your research focus is: <research direction>.

Inspect actual CPU, memory, and execution speed before choosing implementation and validation scope. This round runs from <start> to <deadline>, with a maximum duration of two hours.

Deliver complete source code, a runnable strategy, build instructions, an English description, and the actual seeds, per-game results, and timings. Report cash from completed games, not predicted returns.
```

### Competitive iteration task:

```text
Continue developing <strategy>, using <parent version and file identity>.

Codex has completed the local agent-pool evaluation:
<overall win rate, matchup results, evaluation settings, and cash diagnostics>.

The attachments contain critical losses and supporting evidence. This round focuses on:
<observed problem, possible cause, and proposed improvement to investigate>.

Inspect economic decisions, rolling scheduling, and execution. Identify and fix the problem, while checking that the change preserves existing strengths and narrow wins. Do not add seed-specific rules.

Complete this round within two hours. Deliver the revised source code, runtime files, English change description, and actual validation records.

Codex will evaluate the revision on fresh seeds against all 11 public opponents plus the original R2. The final target is an overall win rate strictly above 85%; cash is diagnostic.
```
