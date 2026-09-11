import json
from pathlib import Path

notebook = Path("notebooks/Kaggriculture_AI_Agent_Analysis.ipynb")

with notebook.open("r", encoding="utf-8") as f:
    nb = json.load(f)

marker = "KAGGRICULTURE_AUTO_SIMULATION_DASHBOARD"

# Prevent duplicate insertion
if any(
    marker in "".join(cell.get("source", []))
    for cell in nb.get("cells", [])
):
    print("Dashboard code is already in the notebook.")
else:

    dashboard_code = r'''# KAGGRICULTURE_AUTO_SIMULATION_DASHBOARD
# ============================================================
# 🌾 AUTOMATIC SIMULATION TIME-SERIES DASHBOARD
# Detects simulation data automatically and visualizes available
# turn/day-based performance metrics.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().parent

def find_simulation_data(root):
    """Automatically detect CSV/JSON simulation datasets."""

    candidates = []

    for path in root.rglob("*"):

        if not path.is_file():
            continue

        # Ignore the virtual environment and cache folders
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue

        if path.suffix.lower() not in {".csv", ".json"}:
            continue

        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_json(path)

            if df.empty:
                continue

            columns = {
                str(c).strip().lower()
                for c in df.columns
            }

            time_columns = {
                "turn",
                "day",
                "step",
                "round",
                "iteration",
                "time",
                "date"
            }

            metric_columns = {
                "reward",
                "score",
                "profit",
                "cash",
                "money",
                "balance",
                "wealth",
                "income",
                "revenue",
                "cost",
                "utility"
            }

            has_time = bool(columns & time_columns)
            has_metric = bool(columns & metric_columns)

            if has_time and has_metric:
                candidates.append((path, df))

        except Exception:
            continue

    return candidates


def create_simulation_dashboard(simulation_files):
    """Create automatic time-series plots from detected simulation data."""

    if not simulation_files:

        print("🌾 Kaggriculture AI Agent — Simulation Dashboard")
        print("=" * 60)
        print("⚠️ No simulation dataset was detected yet.")
        print()
        print("The dashboard is ready.")
        print("It will automatically plot simulation results when")
        print("turn/day-based simulation data becomes available.")

        return

    print("🌾 Kaggriculture AI Agent — Simulation Dashboard")
    print("=" * 60)
    print(f"📊 Simulation datasets detected: {len(simulation_files)}")

    for path, df in simulation_files:

        print()
        print(f"📁 Dataset: {path.relative_to(PROJECT_ROOT)}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")

        # --------------------------------------------------------
        # Detect time/turn column
        # --------------------------------------------------------

        time_candidates = [
            "turn",
            "day",
            "step",
            "round",
            "iteration",
            "time",
            "t",
            "date"
        ]

        time_col = next(
            (
                col
                for col in df.columns
                if str(col).strip().lower() in time_candidates
            ),
            None
        )

        if time_col is None:
            print("⚠️ No turn/day/time column detected.")
            continue

        # --------------------------------------------------------
        # Detect numerical simulation metrics
        # --------------------------------------------------------

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        numeric_cols = [
            col
            for col in numeric_cols
            if col != time_col
        ]

        if not numeric_cols:
            print("⚠️ No numerical simulation metrics detected.")
            continue

        # Prioritize important game metrics
        preferred = [
            "reward",
            "score",
            "profit",
            "cash",
            "money",
            "balance",
            "wealth",
            "income",
            "revenue",
            "cost"
        ]

        selected = []

        for preferred_name in preferred:
            for col in numeric_cols:

                if (
                    str(col).strip().lower()
                    == preferred_name
                ):
                    if col not in selected:
                        selected.append(col)

        # Add remaining numerical metrics
        for col in numeric_cols:

            if col not in selected:
                selected.append(col)

        # Keep dashboard readable
        selected = selected[:6]

        # --------------------------------------------------------
        # Plot each detected metric
        # --------------------------------------------------------

        for metric in selected:

            plot_df = df[[time_col, metric]].dropna()

            if plot_df.empty:
                continue

            plot_df = plot_df.sort_values(time_col)

            plt.figure(figsize=(12, 5))

            plt.plot(
                plot_df[time_col],
                plot_df[metric],
                linewidth=2
            )

            plt.title(
                f"{str(metric).replace('_', ' ').title()} Over "
                f"{str(time_col).replace('_', ' ').title()}",
                fontsize=15,
                fontweight="bold"
            )

            plt.xlabel(
                str(time_col).replace("_", " ").title()
            )

            plt.ylabel(
                str(metric).replace("_", " ").title()
            )

            plt.grid(
                alpha=0.25,
                linestyle="--"
            )

            plt.tight_layout()
            plt.show()

        # --------------------------------------------------------
        # Summary statistics
        # --------------------------------------------------------

        print()
        print("📈 Simulation metric summary:")

        summary = df[selected].describe().T[
            ["mean", "std", "min", "max"]
        ]

        display(summary.round(2))


# Detect simulation results automatically
simulation_files = find_simulation_data(PROJECT_ROOT)

# Build the dashboard
create_simulation_dashboard(simulation_files)
'''

    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dashboard_code.splitlines(True)
    }

    # Find the best insertion point:
    # immediately after the last import/setup cell.
    insertion_index = 0

    for i, cell in enumerate(nb.get("cells", [])):

        source = "".join(cell.get("source", []))

        if (
            "import pandas" in source
            or "import numpy" in source
            or "import matplotlib" in source
            or "import seaborn" in source
        ):
            insertion_index = i + 1

    nb["cells"].insert(insertion_index, new_cell)

    with notebook.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print()
    print("✅ Simulation dashboard inserted automatically.")
    print(f"📓 Notebook: {notebook}")
    print(f"📍 Inserted after the import/setup section.")
