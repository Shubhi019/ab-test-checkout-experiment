"""
LinkedIn thumbnail chart for the one-page checkout A/B test.

Reuses the simulation and stats from ab_test_checkout_experiment.py
(seed 42, same TRUTH, same sample size) so the figure matches the
script's printed numbers exactly.

Run:  python make_chart.py
Out:  results.png (1200×630)
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Importing runs the experiment once (seed 42). Suppress its console dump.
with contextlib.redirect_stdout(io.StringIO()):
    import ab_test_checkout_experiment as exp

OUT_PATH = Path(__file__).resolve().parent / "results.png"

# ---------------------------------------------------------------------------
# Data — same helpers / frame as the analysis script
# ---------------------------------------------------------------------------
overall = exp.two_proportion_test(*exp.totals(exp.df))

segments = {}
for device in ("desktop", "mobile"):
    sub = exp.df[exp.df["device"] == device]
    res = exp.two_proportion_test(*exp.totals(sub))
    share = sub["sessions"].sum() / exp.df["sessions"].sum()
    segments[device] = {**res, "share": share}

daily = []
for day in range(3, exp.RUNTIME_DAYS + 1):
    upto = exp.df[exp.df["day"] <= day]
    res = exp.two_proportion_test(*exp.totals(upto))
    daily.append({"day": day, "p_value": res["p_value"], "rel_lift": res["rel_lift"]})


def to_relative(res):
    """Convert absolute-difference CI to relative lift / relative CI."""
    cvr_c = res["cvr_control"]
    return {
        "rel_lift": res["rel_lift"],
        "ci_low": res["ci_low"] / cvr_c,
        "ci_high": res["ci_high"] / cvr_c,
        "p_value": res["p_value"],
    }


# Verify against known printed values before drawing
_checks = [
    ("Overall", overall, 0.03016, 0.03177, 0.0533, 0.14),
    ("Mobile", segments["mobile"], 0.02229, 0.02112, -0.0524, 0.31),
    ("Desktop", segments["desktop"], 0.04500, 0.05175, 0.1501, 0.003),
]
for name, res, c_exp, t_exp, lift_exp, p_exp in _checks:
    assert abs(res["cvr_control"] - c_exp) < 5e-5, name
    assert abs(res["cvr_treatment"] - t_exp) < 5e-5, name
    assert abs(res["rel_lift"] - lift_exp) < 5e-4, name
    assert abs(res["p_value"] - p_exp) < 0.01, name

day6 = next(d for d in daily if d["day"] == 6)
day14 = next(d for d in daily if d["day"] == 14)
assert abs(day6["p_value"] - 0.0018) < 5e-4
assert abs(day14["p_value"] - 0.14) < 0.01

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.edgecolor": "#b0b0b0",
    "axes.labelcolor": "#333333",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "axes.linewidth": 0.8,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.7,
})

fig, (ax_left, ax_right) = plt.subplots(
    1, 2, figsize=(12, 6.3), dpi=100,
    gridspec_kw={"wspace": 0.28, "left": 0.16, "right": 0.97,
                 "top": 0.82, "bottom": 0.14},
)

COLOURS = {
    "desktop": "#0d9488",   # teal
    "mobile": "#e05a33",    # red-orange
    "overall": "#6b7280",   # grey
}

# ---- LEFT: relative lift with 95% CI ---------------------------------------
rows = [
    ("Desktop (35% of traffic)", to_relative(segments["desktop"]), COLOURS["desktop"]),
    ("Mobile (65% of traffic)", to_relative(segments["mobile"]), COLOURS["mobile"]),
    ("Overall", to_relative(overall), COLOURS["overall"]),
]

# Tight cluster: Desktop / Mobile / Overall sit close together
ys = [0.50, 0.25, 0.00]  # top = Desktop; compact spacing
for y, (label, res, colour) in zip(ys, rows):
    x = res["rel_lift"]
    xerr = np.array([[x - res["ci_low"]], [res["ci_high"] - x]])
    ax_left.errorbar(
        x, y, xerr=xerr, fmt="o", color=colour, ecolor=colour,
        elinewidth=2.2, capsize=5, capthick=1.6, markersize=8,
        zorder=3,
    )
    p = res["p_value"]
    p_txt = f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.4f}"
    # Place annotation just past the right CI tip
    anchor = res["ci_high"] + 0.012
    ax_left.text(
        anchor, y, p_txt, va="center", ha="left",
        fontsize=9.5, color=colour, fontweight="500",
    )

ax_left.axvline(0, color="#9ca3af", linestyle="--", linewidth=1.1, zorder=1)
ax_left.set_yticks(ys)
ax_left.set_yticklabels([r[0] for r in rows], fontsize=10)
# Explicit limits: generous pad so the three rows stay a compact central band
ax_left.set_ylim(-0.65, 1.15)
ax_left.set_xlabel("Relative lift (95% CI)", fontsize=10)
ax_left.set_title(
    "The aggregate was hiding the finding",
    fontsize=13, fontweight="600", pad=12, loc="left",
)
ax_left.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax_left.set_xlim(-0.22, 0.42)
ax_left.xaxis.grid(True, linestyle="-", alpha=0.9)
ax_left.set_axisbelow(True)
ax_left.spines["top"].set_visible(False)
ax_left.spines["right"].set_visible(False)
ax_left.tick_params(axis="both", labelsize=10, length=0)
ax_left.spines["left"].set_visible(False)

# ---- RIGHT: cumulative daily p-values --------------------------------------
days = [d["day"] for d in daily]
pvals = [d["p_value"] for d in daily]

ax_right.plot(
    days, pvals, color="#374151", linewidth=2.0,
    marker="o", markersize=5.5, markerfacecolor="#ffffff",
    markeredgewidth=1.6, markeredgecolor="#374151", zorder=3,
)

# Shade / mark days where p < 0.05
sig_days = [d["day"] for d in daily if d["p_value"] < exp.ALPHA]
if sig_days:
    # Continuous band covering the significant stretch
    ax_right.axvspan(
        min(sig_days) - 0.45, max(sig_days) + 0.45,
        color="#fde68a", alpha=0.45, zorder=0, label="p < 0.05",
    )

ax_right.axhline(
    exp.ALPHA, color="#b45309", linestyle="--", linewidth=1.2, zorder=2,
)
ax_right.text(
    days[-1] + 0.15, exp.ALPHA + 0.006, "alpha = 0.05",
    fontsize=9, color="#b45309", ha="right", va="bottom",
)

# Day-6 minimum annotation
ax_right.annotate(
    "would have shipped here",
    xy=(6, day6["p_value"]),
    xytext=(6.4, 0.055),
    fontsize=9.5,
    color="#92400e",
    fontweight="500",
    arrowprops=dict(
        arrowstyle="->", color="#92400e", lw=1.1,
        connectionstyle="arc3,rad=0.15",
    ),
    ha="left",
)

# Day-14 endpoint — label centred on x=14, short vertical arrow to that point only
ax_right.annotate(
    "actual result",
    xy=(14, day14["p_value"]),
    xytext=(14, 0.198),
    fontsize=9.5,
    color="#374151",
    fontweight="500",
    arrowprops=dict(
        arrowstyle="->", color="#6b7280", lw=1.1,
        connectionstyle="arc3,rad=0.0",
        shrinkA=1, shrinkB=5,
    ),
    ha="center", va="bottom",
)

ax_right.set_xlabel("Day of experiment", fontsize=10)
ax_right.set_ylabel("p-value", fontsize=10)
ax_right.set_title(
    "Why you don't check daily",
    fontsize=13, fontweight="600", pad=12, loc="left",
)
ax_right.set_xlim(2.5, 14.6)
ax_right.set_ylim(-0.005, 0.22)
ax_right.set_xticks(days)
ax_right.yaxis.set_major_locator(mticker.FixedLocator([0.00, 0.05, 0.10, 0.15, 0.20]))
ax_right.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax_right.yaxis.grid(True, linestyle="-", alpha=0.9)
ax_right.set_axisbelow(True)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)
ax_right.tick_params(axis="both", labelsize=10)

fig.suptitle(
    "A/B Test: One-Page Checkout — 14 days, ~100k sessions",
    fontsize=15, fontweight="700", y=0.96, color="#111827",
)

fig.savefig(OUT_PATH, dpi=100, facecolor="white", edgecolor="none")
print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
print(
    f"Overall  control {overall['cvr_control']:.3%}  treatment "
    f"{overall['cvr_treatment']:.3%}  lift {overall['rel_lift']:+.2%}  "
    f"p={overall['p_value']:.2f}"
)
print(
    f"Mobile   control {segments['mobile']['cvr_control']:.3%}  treatment "
    f"{segments['mobile']['cvr_treatment']:.3%}  lift "
    f"{segments['mobile']['rel_lift']:+.2%}  "
    f"p={segments['mobile']['p_value']:.2f}"
)
print(
    f"Desktop  control {segments['desktop']['cvr_control']:.3%}  treatment "
    f"{segments['desktop']['cvr_treatment']:.3%}  lift "
    f"{segments['desktop']['rel_lift']:+.2%}  "
    f"p={segments['desktop']['p_value']:.3f}"
)
print(f"Day 6 p={day6['p_value']:.4f}  Day 14 p={day14['p_value']:.2f}")
