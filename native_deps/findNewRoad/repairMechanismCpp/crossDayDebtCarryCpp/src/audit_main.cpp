#include "cross_day_debt_carry.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace carry = g001::cross_day_debt_carry;

int main(int argc, char** argv) {
  try {
    if (argc > 1 && std::string(argv[1]) == "--focused") {
      const auto game = carry::evaluate_focused(
          CROSS_DAY_TAPES, CROSS_DAY_LIBRARY, 970017, 1);
      const bool carried_explicitly =
          game.source191_finish_disposition ==
              carry::FinishDisposition::Carried &&
          std::find(game.day8_new_debt_ids.begin(),
                    game.day8_new_debt_ids.end(), game.debt_id) !=
              game.day8_new_debt_ids.end();
      if (game.steps != 216 || game.day7_commits != 24 ||
          game.day8_commits != 24 || !game.day8_revalidated ||
          game.source191_physical_receipt || game.source191_completed ||
          !carried_explicitly || game.source191_emitted_step != -1 ||
          game.day8_receipt_checks != 24 || game.day8_receipt_failures != 0 ||
          game.day8_move_expected != game.day8_move_emitted ||
          game.day8_move_early != 0 || game.day8_move_duplicates != 0 ||
          game.day8_move_drops != 0 || game.commit_failures != 0 ||
          game.provider_state_leaks != 0) {
        for (std::size_t i = 0; i < game.day8_new_debt_ids.size(); ++i)
          std::cerr << "debt[" << i << "]=" << game.day8_new_debt_ids[i]
                    << " source=" << game.day8_new_debt_source_steps[i]
                    << '\n';
        throw std::runtime_error(
            "focused day7-day8 gate failed steps=" +
            std::to_string(game.steps) + " d7=" +
            std::to_string(game.day7_commits) + " d8=" +
            std::to_string(game.day8_commits) + " revalidated=" +
            std::to_string(game.day8_revalidated) + " start_pos=" +
            std::to_string(game.day8_start_x) + "," +
            std::to_string(game.day8_start_y) + " source191_step=" +
            std::to_string(game.source191_emitted_step) + " physical=" +
            std::to_string(game.source191_physical_receipt) + " completed=" +
            std::to_string(game.source191_completed) + " receipts=" +
            std::to_string(game.day8_receipt_checks) + "/" +
            std::to_string(game.day8_receipt_failures) + " moves=" +
            std::to_string(game.day8_move_emitted) + "/" +
            std::to_string(game.day8_move_expected) + " early=" +
            std::to_string(game.day8_move_early) + " dup=" +
            std::to_string(game.day8_move_duplicates) + " drops=" +
            std::to_string(game.day8_move_drops) + " debts=" +
            std::to_string(game.day8_new_debts) + " first_debt=" +
            std::to_string(game.day8_new_debt_ids.empty()
                               ? 0
                               : game.day8_new_debt_ids.front()) +
            " first_source=" +
            std::to_string(game.day8_new_debt_source_steps.empty()
                               ? -99
                               : game.day8_new_debt_source_steps.front()) +
            " last_debt=" +
            std::to_string(game.day8_new_debt_ids.empty()
                               ? 0
                               : game.day8_new_debt_ids.back()) +
            " last_source=" +
            std::to_string(game.day8_new_debt_source_steps.empty()
                               ? -99
                               : game.day8_new_debt_source_steps.back()));
      }
      std::cout << "cross_day_debt_carry_audit: FOCUSED PASS source191_step="
                << game.source191_emitted_step << " moves="
                << game.day8_move_emitted << '/' << game.day8_move_expected
                << '\n';
      return 0;
    }
    const std::string output = argc > 1
                                   ? argv[1]
                                   : "cross-day-debt-carry-seed970017.json";
    const auto report = carry::evaluate(CROSS_DAY_TAPES, CROSS_DAY_LIBRARY,
                                        970017, 1);
    const auto& game = report.game;
    const bool carried_explicitly =
        game.source191_finish_disposition ==
            carry::FinishDisposition::Carried &&
        std::find(game.day8_new_debt_ids.begin(), game.day8_new_debt_ids.end(),
                  game.debt_id) != game.day8_new_debt_ids.end();
    if (!report.default_off_parity || !report.stale_rejected ||
        !report.tamper_rejected || !report.wrong_tile_rejected ||
        !report.wrong_item_rejected ||
        !report.midnight_duplicate_rejected ||
        !report.no_terminal_oracle || game.steps != 719 ||
        game.day7_commits != 24 || game.day8_commits != 24 ||
        !game.day7_debt_authorized || game.debt_source_step != 191 ||
        !game.day8_revalidated || game.day8_revalidation_reject !=
                                      carry::RevalidationReject::None ||
        game.source191_physical_receipt || game.source191_completed ||
        !carried_explicitly || game.source191_emitted_step != -1 ||
        game.day8_receipt_checks != 24 || game.day8_receipt_failures != 0 ||
        game.day8_move_expected != game.day8_move_emitted ||
        game.day8_move_early != 0 || game.day8_move_duplicates != 0 ||
        game.day8_move_drops != 0 || game.commit_failures != 0 ||
        game.provider_state_leaks != 0)
      throw std::runtime_error("cross-day release gate failed");
    std::ofstream file(output, std::ios::trunc);
    if (!file) throw std::runtime_error("cannot write audit artifact");
    file << report.json() << '\n';
    std::cout << "cross_day_debt_carry_audit: PASS source191_step="
              << game.source191_emitted_step << " day8_debts="
              << game.day8_new_debts << " output=" << output << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "cross_day_debt_carry_audit: " << error.what() << '\n';
    return 1;
  }
}
