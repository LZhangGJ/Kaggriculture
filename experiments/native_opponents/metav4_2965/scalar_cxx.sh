#!/usr/bin/env bash
# Offline parity control: keep the default JobBatch build untouched.
exec g++ -DSTUDENT_SCALAR_LINEAR "$@"
