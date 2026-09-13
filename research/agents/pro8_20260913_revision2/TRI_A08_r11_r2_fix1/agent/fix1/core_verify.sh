#!/bin/bash
set -euo pipefail
cd /mnt/data/fix1_work
python run_measured.py fix1_final_gcc14_build 100 python candidate/build.py
python run_measured.py fix1_clang17_checks 150 python candidate/tests/run_checks.py --cxx clang++ --library /mnt/data/fix1_work/repro/fix1_clang17.so --sanitize
python run_measured.py paired_original_fix1 170 python candidate/tests/compare_saved_entries.py --left /mnt/data/fix1_work/original_r2/main.py --right /mnt/data/fix1_work/candidate/main.py --traces /mnt/data/fix1_work/candidate/evidence/own_traces --out /mnt/data/fix1_work/repro/paired_original_fix1 --require-exact
python run_measured.py paired_gcc14_clang17 170 python candidate/tests/compare_saved_entries.py --left /mnt/data/fix1_work/candidate/main.py --right /mnt/data/fix1_work/candidate/main.py --right-library /mnt/data/fix1_work/repro/fix1_clang17.so --traces /mnt/data/fix1_work/candidate/evidence/own_traces --out /mnt/data/fix1_work/repro/paired_gcc14_clang17 --require-exact
