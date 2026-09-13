#!/bin/bash
set -e
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python run_measured.py clean_offline_rebuild 90 python clean_rebuild/build.py
sha256sum candidate/policy/tri_a08_r11_r2_fix1.so clean_rebuild/policy/tri_a08_r11_r2_fix1.so > logs/clean_rebuild_hashes.txt
cmp candidate/policy/tri_a08_r11_r2_fix1.so clean_rebuild/policy/tri_a08_r11_r2_fix1.so
python run_measured.py build_gate_refusal 90 python test_build_transaction.py
python run_measured.py reduction_gcc14_matrix 160 python candidate/tests/repro_matrix.py --cxx g++ --out /mnt/data/fix1_work/repro/matrix_gcc14
python run_measured.py reduction_clang17_matrix 160 python candidate/tests/repro_matrix.py --cxx clang++ --out /mnt/data/fix1_work/repro/matrix_clang17
python run_measured.py reduction_gcc13_unavailable 10 python candidate/tests/repro_matrix.py --cxx g++-13 --out /mnt/data/fix1_work/repro/matrix_gcc13_unavailable || test "$?" = 2
