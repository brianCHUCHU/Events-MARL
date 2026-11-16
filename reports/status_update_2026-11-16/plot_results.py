from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "analysis.json").read_text())
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})
COLORS = {"event": "#2456A6", "dense": "#DF8D27", "middle": "#1D8A67",
          "sparse": "#8957A1", "single": "#666666"}
GROUPS = {
    "pressure_plate": [("event", "Hand-coded event", "event"), ("fixed_38", "Fixed K=38", "dense"),
                       ("fixed_76", "Fixed K=76", "middle"), ("fixed_152", "Fixed K=152", "sparse"),
                       ("single", "Single query", "single")],
    "navigation": [("event", "Hand-coded event", "event"), ("fixed_25", "Fixed K=25", "dense"),
                   ("fixed_50", "Fixed K=50", "middle"), ("fixed_100", "Fixed K=100", "sparse"),
                   ("single", "Single query", "single")],
}


def finish(fig, name):
    fig.savefig(HERE / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


for steps, name, label in [
    ("simulator_training_environment_steps", "learning_curves", "Simulator environment steps (millions)"),
    ("training_environment_steps", "learning_curves_valid_steps", "Valid training environment steps (millions)"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.6), sharey=True, layout="constrained")
    for ax, (env, schedules) in zip(axes, GROUPS.items()):
        for schedule, title, key in schedules:
            r = DATA[f"{env}_{schedule}_seed42"]
            x = [p[steps] / 1e6 for p in r["curve"]]
            y = [p["completion_rate"] for p in r["curve"]]
            ax.plot(x, y, label=title, color=COLORS[key], linewidth=2.2 if key == "event" else 1.7,
                    linestyle="--" if key == "single" else "-", alpha=0.95)
            if max(y) > 0:
                peak = int(np.argmax(y))
                ax.scatter([x[peak]], [y[peak]], color=COLORS[key], s=32, zorder=5)
        ax.set_title("Pressure Plate" if env == "pressure_plate" else "Navigation", fontweight="bold")
        ax.set_xlabel(label)
        ax.set_ylim(-2, 104)
        ax.grid(axis="y", alpha=0.18)
        ax.legend(frameon=False, fontsize=9.5, loc="upper left" if env == "pressure_plate" else "lower right")
    axes[0].set_ylabel("Team success (%)")
    fig.suptitle("Query schedules | seed 42 | 1,280 fixed-seed evaluation episodes per checkpoint", fontsize=13)
    finish(fig, name)

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5), layout="constrained")
for schedule, title, color in [("event", "Hand-coded event", COLORS["event"]),
                                ("fixed_76", "Fixed K=76", COLORS["middle"])]:
    r = DATA[f"pressure_plate_{schedule}_seed42"]
    axes[0].plot([p["training_round"] for p in r["curve"]],
                 [p["completion_rate"] for p in r["curve"]], label=title, color=color, linewidth=2)
    axes[1].plot([p["training_round"] for p in r["curve"]],
                 [p["avg_reward"] for p in r["curve"]], label=title, color=color, linewidth=2)
axes[0].set_ylabel("Team success (%)")
axes[0].set_ylim(-2, 104)
axes[0].set_title("Late loss of simultaneous goal occupancy")
axes[1].set_ylabel("Mean episodic team return")
axes[1].set_title("Reward remains high as success collapses")
for ax in axes:
    ax.set_xlabel("Training round (25,600 simulator steps each)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", alpha=0.18)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("Pressure Plate: task success falls while shaped return stays high", fontsize=13)
finish(fig, "pressure_plate_degradation")

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
labels, never, detected = [], [], []
for schedule, title in [("event", "Event"), ("fixed_50", "K=50")]:
    for phase, title2 in [("peak", "Peak"), ("final", "Final")]:
        r = DATA[f"navigation_{schedule}_seed42"][phase]
        missed_after_detection = float(np.mean(r["detected_but_not_collected_percent"]))
        labels.append(title + "\n" + title2)
        detected.append(missed_after_detection)
        never.append(100 - r["goal_completion_percent"] - missed_after_detection)
ax.bar(labels, never, label="No recorded target detection", color="#DF8D27", width=.6)
ax.bar(labels, detected, bottom=never, label="Detected but not collected", color="#2456A6", width=.6)
for i, (a, b) in enumerate(zip(never, detected)):
    ax.text(i, a + b + .25, f"{a+b:.1f}%", ha="center")
ax.set_ylim(0, 13)
ax.set_ylabel("Uncollected individual goals (%)")
ax.set_title("Navigation: most misses occur before target detection", fontweight="bold")
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", alpha=.18)
finish(fig, "navigation_failure_breakdown")

print("Saved four PNG/PDF figure pairs.")
