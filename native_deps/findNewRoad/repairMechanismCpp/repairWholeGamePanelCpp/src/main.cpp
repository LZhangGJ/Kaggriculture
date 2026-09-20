#include "repair_whole_game_panel.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef REPAIR_WHOLE_GAME_TAPES
#error REPAIR_WHOLE_GAME_TAPES must be defined
#endif
#ifndef REPAIR_WHOLE_GAME_LIBRARY
#error REPAIR_WHOLE_GAME_LIBRARY must be defined
#endif

int main(int argc, char** argv) {
  try {
    const std::string output =
        argc > 1 ? argv[1] : "repair-whole-game-panel-smoke.json";
    g001::repair_whole_game_panel::Options options;
    options.tapes = REPAIR_WHOLE_GAME_TAPES;
    options.library = REPAIR_WHOLE_GAME_LIBRARY;
    const auto report = g001::repair_whole_game_panel::evaluate(options);
    if (!report.exact_parity())
      throw std::runtime_error("disabled whole-game panel lost exact parity");
    std::ofstream file(output, std::ios::trunc);
    if (!file) throw std::runtime_error("cannot write whole-game panel JSON");
    file << report.json() << '\n';
    std::cout << "wrote " << output << " with " << report.games.size()
              << " paired games; no benefit claim\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "repair_whole_game_panel_cli: " << error.what() << '\n';
    return 1;
  }
}
