#include "receding_horizon.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>

namespace g001::robust {

Plan choose_first_option(
    const std::vector<ScenarioOutcome>& outcomes,
    int horizon_steps,
    double cvar_tail_fraction
) {
    if (horizon_steps < 24 || horizon_steps > 72 || outcomes.empty() ||
        !(cvar_tail_fraction > 0.0 && cvar_tail_fraction <= 1.0)) {
        throw std::invalid_argument("invalid receding-horizon input");
    }
    std::map<std::uint32_t, std::vector<const ScenarioOutcome*>> grouped;
    for (const auto& outcome : outcomes) grouped[outcome.option_id].push_back(&outcome);

    std::vector<OptionScore> scores;
    for (const auto& [option, scenarios] : grouped) {
        OptionScore score;
        score.option_id = option;
        score.worst_margin = std::numeric_limits<double>::infinity();
        score.worst_own_money = std::numeric_limits<double>::infinity();
        std::vector<double> margins;
        margins.reserve(scenarios.size());
        for (const auto* scenario : scenarios) {
            score.worst_purchase_failures = std::max(
                score.worst_purchase_failures, scenario->purchase_failures
            );
            score.worst_feed_failures = std::max(
                score.worst_feed_failures, scenario->feed_failures
            );
            score.worst_overflow_units = std::max(
                score.worst_overflow_units, scenario->overflow_units
            );
            const auto margin = scenario->own_money - scenario->opponent_money;
            margins.push_back(margin);
            score.worst_margin = std::min(score.worst_margin, margin);
            score.worst_own_money = std::min(score.worst_own_money, scenario->own_money);
        }
        std::sort(margins.begin(), margins.end());
        const auto tail = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(margins.size() * cvar_tail_fraction))
        );
        score.cvar_margin = 0;
        for (std::size_t i = 0; i < tail; ++i) score.cvar_margin += margins[i];
        score.cvar_margin /= tail;
        scores.push_back(score);
    }

    // Lexicographic robustness, with no arbitrary weighted sum: production and
    // survival constraints dominate adversarial margin; own money only breaks
    // equally robust margin outcomes and avoids mutually destructive choices.
    const auto better = [](const OptionScore& left, const OptionScore& right) {
        return std::tuple{
            left.worst_purchase_failures,
            left.worst_feed_failures,
            left.worst_overflow_units,
            -left.worst_margin,
            -left.cvar_margin,
            -left.worst_own_money,
            left.option_id
        } < std::tuple{
            right.worst_purchase_failures,
            right.worst_feed_failures,
            right.worst_overflow_units,
            -right.worst_margin,
            -right.cvar_margin,
            -right.worst_own_money,
            right.option_id
        };
    };
    const auto selected = std::min_element(scores.begin(), scores.end(), better);
    return {selected->option_id, horizon_steps, *selected};
}

}  // namespace g001::robust
