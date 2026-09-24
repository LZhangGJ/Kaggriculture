#!/usr/bin/env python3
"""Compile the frozen Thomas source as a C++ extension in this directory."""

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUILD = HERE / "build"
SOURCE = ROOT / "opponents/thomas_2945/main.py"
SHADOW = BUILD / "thomas_2945_native.pyx"

# Cython 3.2.9 incorrectly hoists this constant generator's closure variables
# into module initialization.  Keep an equivalent explicit predicate in the
# generated-only shadow source; the frozen opponent remains untouched.
source = SOURCE.read_text(encoding="utf-8")
default_config = """(configuration.get('boardSize',10)==10 and
            configuration.get('turnsPerDay',24)==24 and
            configuration.get('shedCapacity',100)==100 and
            configuration.get('maxMarketOrdersPerTurn',10)==10)"""
default_config_with_hand_cost = f"({default_config} and configuration.get('farmHandCostMult',1)==1)"
patterns = {
    """all(configuration.get(k,v)==v for k,v in
            [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)])""": default_config,
    "all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)])": default_config,
    """any(configuration.get(k,v)!=v for k,v in
        (('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)))""": f"not {default_config}",
    "any(configuration.get(k,v)!=v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10)])": f"not {default_config}",
    "all(configuration.get(k,v)==v for k,v in [('boardSize',10),('turnsPerDay',24),('shedCapacity',100),('maxMarketOrdersPerTurn',10),('farmHandCostMult',1)])": default_config_with_hand_cost,
}
if sum(source.count(pattern) for pattern in patterns) != 7:
    raise SystemExit("frozen Thomas source no longer matches the audited Cython patch")
for pattern, replacement in patterns.items():
    source = source.replace(pattern, replacement)
BUILD.mkdir(exist_ok=True)
SHADOW.write_text(source, encoding="utf-8")

setup(
    name="thomas-2945-native",
    ext_modules=cythonize(
        [Extension(
            "thomas_2945_native",
            [str(SHADOW)],
            language="c++",
            extra_compile_args=["-O3", "-DNDEBUG"],
        )],
        build_dir=str(BUILD),
        compiler_directives={"language_level": 3},
    ),
)
