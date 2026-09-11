"""Compare M3.5 and M3.6C one-step graph construction before XLA compile."""

from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
for source in (
    PROJECT / "src",
    ROOT / "gpu_sim" / "src",
    ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from project_route_search_v2.lifecycle import initialize_project_controller_v2  # noqa: E402
from project_route_search_v2.m35_controller import (  # noqa: E402
    ensure_m35_projects_v2,
    m35_policy_step_v2,
)
from project_route_search_v2.m35_genome import default_m35_farm_genome_v2  # noqa: E402
from project_route_search_v2.m36_calendar import kawashigi_opening_calendar_v3  # noqa: E402
from project_route_search_v2.m36_controller import m36c_light_policy_step_v3  # noqa: E402


def summarize(closed):
    counter = Counter(eqn.primitive.name for eqn in closed.jaxpr.eqns)
    return {
        "top_level_equations": len(closed.jaxpr.eqns),
        "top_primitives": counter.most_common(20),
        "text_characters": len(str(closed)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-m36", action="store_true")
    parser.add_argument("--compile-m35", action="store_true")
    args = parser.parse_args()
    states = jax.vmap(reset)(jnp.asarray((17,), dtype=jnp.int32))
    template = default_m35_farm_genome_v2(1)
    calendar = kawashigi_opening_calendar_v3(1)
    controller = initialize_project_controller_v2(states, 0)
    controller = ensure_m35_projects_v2(states, controller, template, 0)
    tables = load_tables()
    rows = {}
    functions = {
        "m35": lambda s, c, g, cal: m35_policy_step_v2(s, c, g, 0, tables),
        "m36_light": lambda s, c, g, cal: m36c_light_policy_step_v3(
            s, c, cal, g, 0
        ),
    }
    for name, function in functions.items():
        started = time.perf_counter()
        closed = jax.make_jaxpr(function)(states, controller, template, calendar)
        traced = time.perf_counter() - started
        started = time.perf_counter()
        lowered = jax.jit(function).lower(states, controller, template, calendar)
        lowered_s = time.perf_counter() - started
        rows[name] = {
            "trace_seconds": traced,
            "lower_seconds": lowered_s,
            "compiler_ir_characters": len(str(lowered.compiler_ir())),
            **summarize(closed),
        }
        should_compile = (args.compile_m36 and name == "m36_light") or (
            args.compile_m35 and name == "m35"
        )
        if should_compile:
            started = time.perf_counter()
            executable = lowered.compile()
            rows[name]["compile_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            result = executable(states, controller, template, calendar)
            jax.block_until_ready(result)
            rows[name]["first_execute_seconds"] = time.perf_counter() - started
        print(name, json.dumps(rows[name]), flush=True)
    output = PROJECT / "receipts" / "m36c_jaxpr_profile_v1.json"
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
