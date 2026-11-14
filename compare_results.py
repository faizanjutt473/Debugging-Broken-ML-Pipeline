"""
compare_results.py
------------------
Runs the broken and fixed pipelines and shows how reported performance changes.
Saves a CSV table and a bar chart in the results/ folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from broken_pipeline import run_broken_pipeline
from fixed_pipeline import run_fixed_pipeline

OUTPUT_DIR = Path("results")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    broken = run_broken_pipeline(verbose=False)
    fixed = run_fixed_pipeline(verbose=False)

    table = pd.DataFrame({"Broken": broken, "Fixed": fixed}).round(4)
    table["Change"] = (table["Fixed"] - table["Broken"]).round(4)
    table.to_csv(OUTPUT_DIR / "comparison.csv")

    print("\n" + "=" * 60 + "\nBROKEN vs FIXED\n" + "=" * 60)
    print(table.to_string())

    ax = table[["Broken", "Fixed"]].plot(kind="bar", figsize=(10, 5), rot=0, width=0.75)
    ax.set_title("Reported performance: Broken vs Fixed pipeline")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison.png", dpi=150)
    print(f"\nSaved: {OUTPUT_DIR / 'comparison.csv'} and {OUTPUT_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
