cmake_minimum_required(VERSION 3.18)
project(fast_kaggriculture LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(OpenMP REQUIRED)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent / 'include')"
  OUTPUT_VARIABLE TORCH_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(_fast_kaggriculture MODULE
  src/bindings.cpp
  src/simulator.cpp
  src/native_teammate.cpp)
target_compile_features(_fast_kaggriculture PRIVATE cxx_std_20)
target_compile_options(_fast_kaggriculture PRIVATE -O3 -DNDEBUG -march=native)
target_include_directories(_fast_kaggriculture PRIVATE ${TORCH_INCLUDE})
target_link_libraries(_fast_kaggriculture PRIVATE Python3::Module OpenMP::OpenMP_CXX)
set_target_properties(_fast_kaggriculture PROPERTIES PREFIX "" OUTPUT_NAME "_fast_kaggriculture")
if(DEFINED Python3_EXTENSION_SUFFIX)
  set_target_properties(_fast_kaggriculture PROPERTIES SUFFIX "${Python3_EXTENSION_SUFFIX}")
endif()
