"""Render one GitHub page from validated statistics without changing the analysis."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

PANEL_ORDER = ('representative', 'stress', 'holdout')
MEANING = {
    'representative': 'These 256 seeds were drawn uniformly from the eligible default 31-bit seed domain before their economic conditions were measured. This panel estimates typical seed performance against the fixed opponent mix. It does not estimate the live Kaggle opponent distribution. Once inspected, it becomes development evidence.',
    'stress': 'These 128 seeds were selected from a separate 4,096-seed pool to cover extremes and contrasts in 26 economic coordinates, all eight reference first-shop types, and 12 contrasting cells. Selection used no candidate outcomes. This panel tests coverage of selected conditions; its average is not a population estimate, and extreme conditions need not make a policy lose more often. Reference shop labels use a PASS/PASS controller. Actual shops and prices depend on both players’ actions.',
    'holdout': 'These 256 independently drawn seeds excluded known campaign history, representative seeds and the full stress selection pool. The original six candidates, settings and analysis were frozen before their comparison, and every candidate received this panel. That was their one check on unseen seeds. The panel was then published and retired. The three Pro8 agents and unchanged v37 reuse these same seeds as a known benchmark; their results do not provide fresh holdout confirmation. The audit cannot recover deleted, unsaved or out-of-scope history. Future confirmation needs a new independent set.'}

def pct(value):
    return '—' if value is None else f'{value:.2%}'

def ci(values, points=False):
    if values is None:
        return '—'
    return f'{values[0]*100:+.2f} to {values[1]*100:+.2f} pp' if points else f'{pct(values[0])}–{pct(values[1])}'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--baseline-layout-preview', action='store_true', help='Use only the completed six-agent data to check layout; never publish this preview')
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    data = json.loads((args.results / 'RESULTS.json').read_bytes())
    assert data['status'] == 'complete_all_three_panels'
    assert set(data['panels']) == set(PANEL_ORDER)
    labels = data['candidate_labels']
    ids = ['ppo-495ebc48210f14ae', 'ppo-c90da6b7b3ea061d', 'ppo-bf0aa03b41428d97',
           'ppo-9980111aa0f1e7f6', 'team-terminal-suffix-v1', 'team-v37-feed-reserve-v1']
    new_ids = ['pro8-a08-r11', 'pro8-a06-r6', 'pro8-a06-r12', 'public-ahmed-v37-original']
    if not args.baseline_layout_preview:
        ids += new_ids
        assert args.validation, 'Final publication requires the completed join receipt'
        receipt = json.loads(args.validation.read_bytes())
        assert receipt['status'] == 'PASS' and receipt['total_unique_games'] == 204800
        assert receipt['results_sha256'] == hashlib.sha256((args.results / 'RESULTS.json').read_bytes()).hexdigest()
    assert set(ids) == set(labels)
    names = [name for name in PANEL_ORDER if name in data['panels']]
    conditions = json.loads((args.results / 'REFERENCE_CONDITIONS.json').read_bytes())
    markets = json.loads((args.results / 'REALIZED_MARKETS.json').read_bytes())
    highlighted = 0
    groups_rendered = 0

    def best_ids(rows):
        eligible = {cid: Fraction(row['wins'], row['games']) for cid, row in rows.items() if row['games']}
        if not eligible:
            return set()
        best = max(eligible.values())
        return {cid for cid, value in eligible.items() if value == best}

    def rate_cell(cid, rows, counts=False, interval=False):
        nonlocal highlighted
        if cid not in rows or not rows[cid]['games']:
            return '—'
        row = rows[cid]
        value = pct(row['win_rate'])
        if cid in best_ids(rows):
            value = '**' + value + '**'
            highlighted += 1
        if counts:
            value += f" ({row['wins']}/{row['games']})"
        if interval:
            value += '; ' + ci(row['win_rate_ci95'])
        return value

    def matrix(groups, first_column, support=False):
        nonlocal groups_rendered
        text = '| ' + first_column + (' | Seeds | Games/candidate' if support else '') + ' | ' + ' | '.join(labels[cid] for cid in ids) + ' |\n'
        text += '|---|' + ('---:|---:|' if support else '') + '---:|' * len(ids) + '\n'
        for name, group in groups.items():
            rows = group['candidates']
            groups_rendered += 1
            text += '| ' + name.replace('|', '\\|')
            if support:
                text += f" | {group['distinct_seeds']} | {group['games']}"
            text += ' | ' + ' | '.join(rate_cell(cid, rows, counts=not support) for cid in ids) + ' |\n'
        return text

    def detail_table(groups):
        text = f'Each group spans {len(ids)} candidate rows. Paired differences and bracketed 95% intervals use percentage points (pp).\n\n'
        text += '| Group | Candidate | Games | Wins | Draws | Losses | Win rate | 95% seed interval | Mean cash | Mean margin | vs R2 (pp) [95% CI] | vs Day9 (pp) [95% CI] |\n'
        text += '|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|\n'
        for name, group in groups.items():
            rows = group['candidates']
            for cid in ids:
                if cid not in rows:
                    continue
                row = rows[cid]
                differences = []
                for baseline in ids[:2]:
                    delta = row['paired_differences'][baseline]
                    lo, hi = delta['ci95']
                    differences.append(f"{delta['win_rate_difference']*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]")
                group_label = name if cid == ids[0] else ''
                text += f"| {group_label} | {labels[cid]} | {row['games']} | {row['wins']} | {row['draws']} | {row['losses']} | {rate_cell(cid, rows)} | {ci(row['win_rate_ci95'])} | {row['mean_cash']:,.2f} | {row['mean_margin']:+,.2f} | " + ' | '.join(differences) + ' |\n'
        return text

    def interpretation(name, panel):
        rows = panel['overall']['candidates']
        leaders = best_ids(rows)
        ordered_leaders = [cid for cid in ids if cid in leaders]
        text = 'The highest observed aggregate rate is ' + ' and '.join(f"{labels[cid]} at **{pct(rows[cid]['win_rate'])}**" for cid in ordered_leaders) + '. '
        if name == 'representative':
            text += 'This supports a claim about average performance on eligible random seeds against this fixed roster. '
        elif name == 'stress':
            text += 'This describes the selected economic conditions. A higher rate here than on representative seeds does not by itself prove greater robustness. '
        else:
            rep = data['panels']['representative']['overall']['candidates']
            prior = best_ids(rep)
            text += ('The observed leader agrees with the representative panel. ' if leaders == prior else 'The observed leader differs from the representative panel. ')
            text += 'For later additions this measures performance on the reused benchmark; it cannot establish generalization to a fresh holdout. '
        for cid in ordered_leaders:
            if cid == ids[1]:
                continue
            delta = rows[cid]['paired_differences'][ids[1]]
            low, high = delta['ci95']
            text += f"{labels[cid]} differs from Day9 by {delta['win_rate_difference']*100:+.2f} percentage points (95% paired interval {ci(delta['ci95'], True)}). "
            text += 'That interval includes zero, so a lead over Day9 remains uncertain. ' if low <= 0 <= high else 'That interval excludes zero for this comparison. '
        native = panel['by_opponent']['native:r2']['candidates']
        text += f"Against AFS R2 directly, v37 feed reserve wins {pct(native[ids[5]]['win_rate'])} and Day9 wins {pct(native[ids[1]]['win_rate'])}. "
        text += 'An aggregate ranking can therefore differ sharply from a specific matchup. '
        text = text.rstrip() + '\n\n'
        for cid in [cid for cid in new_ids if cid in rows]:
            delta = rows[cid]['paired_differences'][ids[1]]
            text += f"{labels[cid]} wins {pct(rows[cid]['win_rate'])} overall and {pct(native[cid]['win_rate'])} against AFS R2. Its paired difference from Day9 is {delta['win_rate_difference']*100:+.2f} pp (95% interval {ci(delta['ci95'], True)}). "
        return text.rstrip() + '\n\n'

    total = sum(data['panels'][name]['overall']['games'] * len(ids) for name in names)
    assert total == (122880 if args.baseline_layout_preview else 204800)
    text = '# Kaggriculture master leaderboard\n\n'
    if args.baseline_layout_preview:
        text += '**Local layout preview using completed baseline results only. Pro8 results are absent. Do not publish this preview.**\n\n'
    text += f'All **{total:,} games** in the panels below passed validation, with zero invalid, missing or duplicate cells. The packages use the same pinned official 1.32.7 game rules, 16 opponents and both seats.\n\n'
    if not args.baseline_layout_preview:
        text += 'This update adds **20,480 unchanged-v37 games** to the **184,320 games from nine agents**. Every earlier statistic was preserved and reconciled exactly. The unchanged policy retains public source SHA256 `94c1c2c05ae7cde8fca9ee3957b01112c1cbab82c7434aae6a5f24fa7485bc4c`. Earlier Pro8 evaluation added 61,440 games to the original six-agent comparison. The Pro8 cohort includes A08 r11, A06 r6 and A06 r12 calendar from [source commit d653253](https://github.com/LZhangGJ/Kaggriculture/tree/d65325334067cc6c2b616f346bc43231b1c11190/research/agents/pro8_20260913). **The holdout is retired: Pro8 and unchanged v37 use it as a known benchmark.**\n\n'
    text += '**Bold marks the highest observed win rate among candidates in that comparison. Exact ties are all bold.** This is a numerical ranking, not a claim of statistical certainty. Draws count as zero wins. Confidence intervals resample whole seeds; repeated subgroup comparisons are exploratory. Mean cash and margin add context but do not determine the winner.\n\n'
    text += 'The roster includes AFS R2 itself, so the R2 baseline includes self-play draws. Its aggregate gap to a challenger can differ from the gap against other opponents. Use the opponent breakdowns when judging likely gains against the live field.\n\n'
    text += 'Jump to: ' + ' · '.join(f'[{name.title()}](#{name})' for name in names) + ' · [Methods and evidence](#methods-and-evidence). Detailed tables expand on this same page.\n\n'
    text += '| Candidate | ' + ' | '.join(name.title() for name in names) + ' |\n|---|' + '---:|' * len(names) + '\n'
    for cid in ids:
        text += '| ' + labels[cid] + ' | ' + ' | '.join(rate_cell(cid, data['panels'][name]['overall']['candidates'], interval=True) for name in names) + ' |\n'
    text += '\nEach summary cell shows win rate and its 95% seed-cluster interval. The panels answer different questions; their games are not pooled into one score.\n\n'

    balance = json.loads((args.results / 'MATCHUP_BALANCE.json').read_bytes())
    text += '## Weakest opponent matchups\n\n'
    text += 'Each cell below is the lowest win rate against any of the 16 fixed opponents, combining both seats within that panel. Higher values mean a stronger worst matchup on this roster. **Bold marks the highest minimum in that panel.** Passing means every observed opponent rate exceeds 50% in every panel. These are not round-robin results among leaderboard candidates, and they do not guarantee success against unseen opponents.\n\n'
    minima = {name:max(item['panels'][name]['minimum_win_rate'] for item in balance['candidates'].values()) for name in names}
    text += '| Candidate | Representative minimum | Stress minimum | Holdout minimum | Every opponent >50% in all panels? |\n|---|---:|---:|---:|---|\n'
    for cid in ids:
        item = balance['candidates'][cid]
        cells = []
        for name in names:
            value = item['panels'][name]['minimum_win_rate']
            cell = pct(value)
            if value == minima[name]: cell = '**'+cell+'**'
            cells.append(cell)
        text += '| '+labels[cid]+' | '+' | '.join(cells)+' | '+('Yes' if item['all_panels_all_opponents_over_50'] else 'No')+' |\n'
    best_min = max(item['minimum_across_panels'] for item in balance['candidates'].values())
    winners = [item['label'] for item in balance['candidates'].values() if item['minimum_across_panels']==best_min]
    text += '\n'+', '.join(winners)+' has the highest minimum across all opponent-and-panel combinations: **'+pct(best_min)+'**. This describes matchup coverage; the overall tables measure a different property. [Exact minimum rates and opponents](master_data/MATCHUP_BALANCE.json).\n\n'
    feed = json.loads((args.results / 'FEED_RESERVE_COMPARISON.json').read_bytes())
    text += '## Effect of the feed-reserve change\n\n'
    text += 'This comparison uses the unchanged public v37 parent and the same source with the team feed-reserve guard. It matches seeds, opponents and seats. Positive differences favor the guard. Intervals resample whole seeds using the same 4,000 draws as the main analysis.\n\n'
    text += '| Panel | Unchanged v37 | Feed reserve | Feed reserve minus unchanged | 95% paired interval |\n|---|---:|---:|---:|---|\n'
    for name in names:
        row=feed['panels'][name]
        rates=[row['unchanged_win_rate'],row['feed_reserve_win_rate']]
        cells=[('**'+pct(v)+'**') if v==max(rates) else pct(v) for v in rates]
        text += '| '+name.title()+' | '+' | '.join(cells)+' | '+f"{row['difference']*100:+.2f} pp"+' | '+ci(row['ci95'],True)+' |\n'
    action_changes=sum(r['matched_game_differences']['action_hash'] for r in feed['panels'].values())
    win_changes=sum(r['matched_game_differences']['win'] for r in feed['panels'].values())
    tie_changes=sum(r['matched_game_differences']['tie'] for r in feed['panels'].values())
    text += f'\nAcross all 20,480 matched games, the guard changed {action_changes} action hashes, {win_changes} win indicators and {tie_changes} draw indicators. '
    if win_changes==0 and tie_changes==0:
        text += 'Every win, draw and loss stayed the same. The high win rates are already present in the unchanged public parent; this benchmark shows no win-rate gain from the guard.\n'
    text += '\nThe guard buys a small wheat shortfall when animals have already missed feeding and the existing route is about to collect wheat. All other parent source remains as supplied. This isolates the effect of this code change on the known benchmark; it does not identify the cause of every individual win or loss. [Exact paired results](master_data/FEED_RESERVE_COMPARISON.json).\n\n'

    for name in names:
        panel = data['panels'][name]
        text += '## ' + name.title() + '\n\n' + MEANING[name] + '\n\n'
        text += f"The panel contains {panel['overall']['distinct_seeds']} seeds and {panel['overall']['games']:,} games per candidate: all 16 opponents in both seats.\n\n"
        text += interpretation(name, panel)
        text += '### Overall results\n\n| Candidate | Wins | Draws | Losses | Win rate | 95% seed interval |\n|---|---:|---:|---:|---:|---|\n'
        rows = panel['overall']['candidates']
        for cid in ids:
            row = rows[cid]
            text += f"| {labels[cid]} | {row['wins']:,} | {row['draws']:,} | {row['losses']:,} | {rate_cell(cid, rows)} | {ci(row['win_rate_ci95'])} |\n"
        text += '\n'
        text += '### Win rates by opponent\n\n' + matrix(panel['by_opponent'], 'Opponent') + '\n'
        text += 'Opponent cells show win rate and wins/games. AFS R2’s row against itself includes draws; a strict self-play win rate below 50% can therefore be expected. Opponent names include shared strategy families.\n\n'
        text += '### Win rates by candidate seat\n\n' + matrix(panel['by_seat'], 'Candidate seat') + '\n'
        text += '<details>\n<summary>Opponent-by-seat win rates</summary>\n\n'
        for seat in ('0', '1'):
            groups = {opponent: seats[seat] for opponent, seats in panel['by_opponent_and_seat'].items()}
            text += '#### Candidate seat ' + seat + '\n\n' + matrix(groups, 'Opponent') + '\n'
        text += '</details>\n\n'
        all_groups = {'All': panel['overall']}
        all_groups.update({'opponent=' + k: v for k, v in panel['by_opponent'].items()})
        all_groups.update({'seat=' + k: v for k, v in panel['by_seat'].items()})
        all_groups.update({f'{opponent}; seat={seat}': value for opponent, seats in panel['by_opponent_and_seat'].items() for seat, value in seats.items()})
        text += '<details>\n<summary>Opponent and seat counts, cash, margins, intervals and paired comparisons</summary>\n\n'
        text += detail_table(all_groups) + '\n</details>\n\n'
        text += '<details>\n<summary>Economic condition win rates</summary>\n\n'
        text += 'These reference groups retain the PASS/PASS qualification. Low, middle and high cutoffs were frozen from representative quartiles. Groups with fewer than 20 seeds have limited support; zero-support cells show a dash.\n\n'
        text += matrix(conditions[name], 'Reference condition', support=True) + '\n</details>\n\n'
        text += 'The [exact economic condition data](master_data/REFERENCE_CONDITIONS.json) also includes counts, cash, margins, seed intervals and paired comparisons for every group.\n\n'
        text += '<details>\n<summary>Realized markets and price-floor groups</summary>\n\n'
        text += 'Markets were sampled before actions every four turns (180 samples/game). These groups depend on policy actions and are descriptive, so they do not receive a best-policy highlight. Demand counts do not measure realized sales.\n\n'
        text += '| Candidate | Product | Mean floor exposure | Mean market inventory | Mean shop demand units |\n|---|---|---:|---:|---:|\n'
        for cid in ids:
            for product, row in markets[name][cid].items():
                text += f"| {labels[cid]} | {product} | {pct(row['mean_sampled_floor_fraction'])} | {row['mean_sampled_market_inventory']:,.2f} | {row['mean_shop_demand_units']:,.2f} |\n"
        text += '\n| Candidate | Product | Floor exposure group | Seeds | Games | Wins | Win rate | 95% seed interval |\n|---|---|---|---:|---:|---:|---:|---|\n'
        for cid in ids:
            for product, row in markets[name][cid].items():
                for group, value in row['floor_exposure_groups'].items():
                    text += f"| {labels[cid]} | {product} | {group} | {value['distinct_seeds']} | {value['games']} | {value['wins']} | {pct(value['win_rate'])} | {ci(value['ci95'])} |\n"
        text += '\n</details>\n\n'
        text += '<details>\n<summary>Runtime and diagnostic counts</summary>\n\n'
        text += '| Candidate | Optional debug parse errors | Maximum measured policy latency (s) | Mean game time (s) |\n|---|---:|---:|---:|\n'
        for cid in ids:
            row = panel['diagnostics'][cid]
            text += f"| {labels[cid]} | {row['optional_debug_parse_errors']} | {row['maximum_measured_policy_latency_seconds']:.6f} | {row['mean_game_seconds']:.3f} |\n"
        text += '\nTiming reflects the recorded host and concurrent workload. The original six ran locally; Pro8 ran on local and WRX90 CPU workers. These timings are not a matched speed comparison or a Kaggle runtime guarantee. Optional terminal debug errors are separate from action and terminal validation.\n\n</details>\n\n'

    text += '## Methods and evidence\n\n'
    text += 'The same frozen candidate files and opponent versions ran on each panel. Each game completed 719 transitions. The analysis uses 4,000 common seed bootstrap draws per panel, preserving every opponent/seat game for a seed. Paired intervals compare matched seeds; subtracting separate interval endpoints would not produce a paired interval. Bold uses exact wins/games before rounding.\n\n'
    text += 'The Pro8 agents ran through their supplied create_agent entry, codec, configuration and native library. Both hosts passed all 384 verification cases before primary play, with exact agreement on actions, terminal cash, shops and market summaries: 1,536 full verification games and 1,105,920 official observation checks. These verification games are excluded from win rates. Frozen, disjoint assignments split the 61,440 new games across two hosts.\n\n'
    text += 'Unchanged v37 passed 128 verification cases on each host: 512 full verification games and 368,640 official observation checks, with zero mismatches. The same seed bootstrap and analysis rules apply to all cohorts. Hash-pinned packages make this comparison reproducible; later edits require a new row and run receipt. The public-parent v37 policy does not establish the team’s Three-Layer architecture goal. No result on this page changes the champion or submits to Kaggle.\n\n'
    text += '[Exact ten-candidate data](master_data/RESULTS.json) · [Reference conditions](master_data/REFERENCE_CONDITIONS.json) · [Realized markets](master_data/REALIZED_MARKETS.json) · [Join validation](master_data/JOIN_VALIDATION.json) · [Unchanged v37 protocol and reproduction](../extensions/v37_original_20260913/README.md) · [Pro8 protocol, raw games and reproduction](../extensions/pro8_20260913/README.md) · [Original six-candidate data](RESULTS.json) · [Original raw game records](raw/) · [Original run reproduction](../REPRODUCE_RESULTS.md) · [Set definitions](../README.md) · [Current analysis plan](../extensions/v37_original_20260913/tools/ANALYSIS_PLAN.md) · [Retired holdout receipt](../retired_holdout/RETIREMENT.json).\n'
    # Compact table source to fit GitHub's renderer; displayed values are unchanged.
    text = '\n'.join(line.replace(' |', '|').replace('| ', '|').replace(', +', ',+').replace(', -', ',-').replace('+', '')
                     if line.startswith('|') else line for line in text.split('\n'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(text.encode())
    receipt = dict(panels=names, games=total, source_results_sha256=hashlib.sha256((args.results / 'RESULTS.json').read_bytes()).hexdigest(),
                   rendered_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest(), rendered_bytes=args.out.stat().st_size,
                   highlighted_cells=highlighted, comparison_matrix_rows=groups_rendered,
                   best_rule='Exact wins/games within each matched comparison; all exact ties highlighted', baseline_layout_preview=args.baseline_layout_preview)
    args.out.with_suffix('.render.json').write_bytes((json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
