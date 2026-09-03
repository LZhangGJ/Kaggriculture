# Licensed under the Apache License, Version 2.0.
import pybind11
from setuptools import Extension, setup

setup(
    name="fast-kaggriculture",
    version="0.1.0",
    packages=["fast_kaggriculture"],
    package_dir={"fast_kaggriculture": "python/fast_kaggriculture"},
    ext_modules=[Extension(
        "fast_kaggriculture._fast_kaggriculture",
        [
            "src/bindings.cpp",
            "src/simulator.cpp",
            "src/native_teammate.cpp",
            "src/native_adaptive.cpp",
            "src/adaptive_candidates.cpp",
        ],
        include_dirs=["src", pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-DNDEBUG", "-std=c++20", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )],
)
