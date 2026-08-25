# Licensed under the Apache License, Version 2.0.
from pathlib import Path
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
import sysconfig

ROOT = Path(__file__).resolve().parent
TORCH_INCLUDE = Path(sysconfig.get_paths()["purelib"]) / "torch" / "include"

setup(
    name="fast-kaggriculture",
    version="0.1.0",
    packages=["fast_kaggriculture"],
    package_dir={"fast_kaggriculture": "python/fast_kaggriculture"},
    ext_modules=[Extension(
        "fast_kaggriculture._fast_kaggriculture",
        [
            str(ROOT / "src/bindings.cpp"),
            str(ROOT / "src/simulator.cpp"),
            str(ROOT / "src/native_teammate.cpp"),
        ],
        include_dirs=[str(ROOT / "src"), str(TORCH_INCLUDE)],
        language="c++",
        extra_compile_args=["-O3", "-DNDEBUG", "-std=c++20", "-march=native", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )],
)
