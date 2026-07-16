"""Projected energy chart from a committed HeteroCore execution plan.

The HeteroCore projected total is drawn as a stack of its cost-model
components and set beside the all-digital baseline from the same plan file.
All values are analytical cost-model projections, not silicon measurements.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG = "#0d1117"
FG = "#e6edf3"
GRID = "#30363d"
COMPONENTS = [
    ("peripheral_energy_pj", "Peripheral (DAC/ADC/control)", "#f778ba"),
    ("compute_energy_pj", "Compute", "#58a6ff"),
    ("memory_energy_pj", "Memory", "#7ee787"),
    ("interconnect_energy_pj", "Interconnect", "#d29922"),
]


def main(plan_path: str, out_path: str) -> None:
    data = json.loads(Path(plan_path).read_text())
    s = data["summary"]
    model = data["model"]["name"]
    projected_uj = s["estimated_energy_uj"]
    baseline_uj = s["digital_baseline_energy_uj"]
    reduction = s["projected_energy_reduction"] * 100.0
    bd = s["energy_breakdown_pj"]

    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # All-digital baseline bar.
    ax.bar([0], [baseline_uj], width=0.5, color="#8b949e", label="All-digital baseline")
    ax.text(0, baseline_uj, f"{baseline_uj:.1f} uJ", ha="center", va="bottom", color=FG, fontsize=10)

    # HeteroCore projected bar, stacked by component (pJ -> uJ).
    bottom = 0.0
    for key, label, color in COMPONENTS:
        val_uj = bd[key] / 1_000_000.0
        ax.bar([1], [val_uj], bottom=[bottom], width=0.5, color=color, label=label)
        bottom += val_uj
    ax.text(1, projected_uj, f"{projected_uj:.1f} uJ", ha="center", va="bottom", color=FG, fontsize=10)
    ax.text(1, projected_uj * 0.5, f"-{reduction:.1f}%", ha="center", va="center",
            color="#0d1117", fontsize=13, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["All-digital\nbaseline", "HeteroCore\nprojected"], color=FG)
    ax.set_ylabel("Projected energy per inference (uJ)", color=FG, fontsize=11)
    ax.set_title(f"HeteroCore projected energy: {model}\nAnalytical cost-model projection, not measured silicon",
                 color=FG, fontsize=13, pad=12)
    ax.tick_params(colors=FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, framealpha=1.0, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=BG)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "results/tiny_transformer.plan.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "assets/projected_energy.png"
    main(src, dst)
