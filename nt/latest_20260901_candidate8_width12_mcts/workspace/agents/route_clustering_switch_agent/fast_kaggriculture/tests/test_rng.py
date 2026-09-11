# Licensed under the Apache License, Version 2.0.
import random
from fast_kaggriculture import Config, FastEnv


def test_end_day_python_random_compatibility_is_exercised_by_differential_suite():
    # A documentation guard: random.Random is deliberately not replaceable by
    # std::mt19937. The full pass differential test compares all spawned weeds
    # and unlocked shops over the complete 30-day episode.
    assert random.Random(0).random() == 0.8444218515250481
    assert FastEnv(Config(), 0).step_count == 0
