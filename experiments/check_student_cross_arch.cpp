// Replay packed public observations through either architecture's R1 bridge.
#include <dlfcn.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template <class T> T symbol(void* library, const char* name) {
  auto* result = dlsym(library, name);
  if (!result) throw std::runtime_error(dlerror());
  return reinterpret_cast<T>(result);
}

template <class T> void read(std::ifstream& input, T& value) {
  if (!input.read(reinterpret_cast<char*>(&value), sizeof(value)))
    throw std::runtime_error("truncated parity input");
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) throw std::runtime_error(dlerror());
    auto create = symbol<void* (*)(const double*, std::size_t)>(library, "td_new");
    auto destroy = symbol<void (*)(void*)>(library, "td_delete");
    auto external = symbol<int (*)(void*, const double*, std::size_t,
                                  const int32_t*, std::size_t)>(library, "td_observe_external");
    auto activate = symbol<int (*)(void*, const double*, std::size_t)>(library,
                                                                        "td_activate_external");
    auto observe = symbol<int (*)(void*, const double*, std::size_t,
                                 int32_t*, std::size_t)>(library, "td_observe");
    auto debug = symbol<const char* (*)(void*)>(library, "td_debug");
    std::ifstream input(argv[2], std::ios::binary);
    uint32_t width = 0, rows = 0;
    read(input, width);
    if (width > 128) throw std::runtime_error("settings width");
    std::vector<double> settings(width);
    if (!input.read(reinterpret_cast<char*>(settings.data()), width * sizeof(double)))
      throw std::runtime_error("settings payload");
    read(input, rows);
    void* handle = create(settings.data(), settings.size());
    if (!handle) throw std::runtime_error("td_new failed");
    bool activated = false;
    for (uint32_t step = 0; step < rows; ++step) {
      uint32_t count = 0, encoded = 0;
      read(input, count);
      if (count > 4000) throw std::runtime_error("observation width");
      std::vector<double> packed(count);
      if (!input.read(reinterpret_cast<char*>(packed.data()), count * sizeof(double)))
        throw std::runtime_error("observation payload");
      read(input, encoded);
      if (encoded > 500) throw std::runtime_error("action width");
      std::vector<int32_t> action(encoded);
      if (!input.read(reinterpret_cast<char*>(action.data()), encoded * sizeof(int32_t)))
        throw std::runtime_error("action payload");
      if (encoded) {
        if (external(handle, packed.data(), packed.size(), action.data(), action.size()))
          throw std::runtime_error(debug(handle));
      } else {
        if (!activated) {
          if (activate(handle, packed.data(), packed.size()))
            throw std::runtime_error(debug(handle));
          activated = true;
        }
        int32_t output[256]{};
        const int result = observe(handle, packed.data(), packed.size(), output, 256);
        if (result < 0) throw std::runtime_error(debug(handle));
        std::cout << step << ':';
        for (int i = 0; i < result; ++i) std::cout << ' ' << output[i];
        std::cout << '\n';
      }
    }
    destroy(handle);
    dlclose(library);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
