"""Regenerates weekly_grid_2024_2025_barplot.png and weekly_grid_crypto5_2025_barplot.png
for the MARTA PFE report's Results chapter (chapters/results.tex), Section
'subsec:rl-weekly-cadence'. Built only from numbers already reported and verified in
Tables 'tab:rl-weekly-grid', 'tab:rl-weekly-2025', and 'tab:rl-weekly-crypto5-2025' --
no external data file is read, so these plots cannot drift out of sync with the text
they illustrate.
"""
import matplotlib.pyplot as plt
import numpy as np

BG = "#fafaf8"
RED, ORANGE, GREEN, BLUE, GRAY, PURPLE = "#b83227", "#e8622c", "#0a7a1e", "#1f77d4", "#555555", "#3d2f8f"

# ---------------------------------------------------------------------------
# Plot A: BTC-only weekly grid, 2024 validation (pre-registered checkpoint,
# Table tab:rl-weekly-2025) vs 2025 test, with the 3-seed spread (Table
# tab:rl-weekly-grid) overlaid as context for the 2024 bar.
# ---------------------------------------------------------------------------
conditions = ["A\n(raw)", "B\n(raw+vol)", "C\n(raw+regime)", "D\n(raw+placebo)"]
delta_2024 = [0.31, 0.45, 0.43, 0.57]          # pre-registered checkpoint, Table tab:rl-weekly-2025
delta_2025 = [-0.31, -1.19, 1.29, 0.18]         # same checkpoint, Table tab:rl-weekly-2025
flips = [True, True, False, False]              # sign flip between 2024 and 2025

# all-seed deltas per condition, Table tab:rl-weekly-grid (seeds 0,1,2)
seed_deltas = {
    "A\n(raw)": [0.26, 0.31, 0.10],
    "B\n(raw+vol)": [0.20, 0.45, 1.42],
    "C\n(raw+regime)": [0.43, 0.40, 0.84],
    "D\n(raw+placebo)": [1.50, 0.57, 0.01],
}

fig, ax = plt.subplots(figsize=(10.5, 6.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x = np.arange(len(conditions))
w = 0.34
bars24 = ax.bar(x - w / 2, delta_2024, width=w, color=BLUE, zorder=3, label="2024 val $\\Delta$ (pre-registered checkpoint)")
bars25 = ax.bar(x + w / 2, delta_2025, width=w,
                 color=[RED if d < 0 else GREEN for d in delta_2025], zorder=3,
                 label="2025 test $\\Delta$ (same checkpoint)")

# overlay individual-seed 2024 deltas as context scatter around the 2024 bar
rng = np.random.default_rng(0)
for xi, cond in zip(x, conditions):
    jitter = rng.uniform(-0.06, 0.06, size=3)
    ax.scatter(xi - w / 2 + jitter, seed_deltas[cond], color="#222222", s=22, zorder=5,
               marker="D", alpha=0.8)

ax.axhline(0, color="#999999", linewidth=1.0, zorder=2)

for xi, flip in zip(x, flips):
    if flip:
        ax.annotate("sign flip", xy=(xi, 0), xytext=(xi, -1.55), ha="center", fontsize=8.3,
                    color=RED, style="italic",
                    arrowprops=dict(arrowstyle="-", color=RED, lw=0.8, ls=(0, (2, 2))))

ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=9.5)
ax.set_ylabel("$\\Delta$ Sharpe vs. weekly-aligned buy-hold", fontsize=10.5)
ax.set_title("Native Weekly-Cadence Grid, BTC-only: 2024 Validation vs. 2025 Test\n"
              "(diamonds: individual-seed 2024 $\\Delta$)",
              fontsize=12.3, fontweight="bold", pad=12)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(axis="y", color="#e5e5e0", linewidth=0.8, zorder=0)
ax.legend(loc="upper center", frameon=False, fontsize=9, ncol=1)
ax.set_ylim(-1.9, 3.0)

plt.tight_layout()
plt.savefig("weekly_grid_2024_2025_barplot.png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print("wrote weekly_grid_2024_2025_barplot.png")

# ---------------------------------------------------------------------------
# Plot B: crypto5 panel extension, 2025-only delta vs buy-hold, 5 assets x 4
# conditions (Table tab:rl-weekly-crypto5-2025).
# ---------------------------------------------------------------------------
assets = ["BTC", "ETH", "BNB", "LTC", "XRP"]
cond_labels = ["A (raw)", "B (raw+vol)", "C (raw+regime)", "D (raw+placebo)"]
cond_colors = [GRAY, BLUE, PURPLE, ORANGE]
data = {
    "BTC": [1.27, -0.46, 0.15, 0.60],
    "ETH": [-0.39, -0.43, -1.68, -0.35],
    "BNB": [-0.89, -1.18, 0.65, -1.30],
    "LTC": [0.55, 0.86, 0.22, 0.57],
    "XRP": [2.41, -0.08, 0.01, 1.18],
}

fig, ax = plt.subplots(figsize=(11.5, 6.0))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x = np.arange(len(assets))
w = 0.19
for i, (cl, cc) in enumerate(zip(cond_labels, cond_colors)):
    vals = [data[a][i] for a in assets]
    offset = (i - 1.5) * w
    ax.bar(x + offset, vals, width=w, color=cc, zorder=3, label=cl)

ax.axhline(0, color="#999999", linewidth=1.0, zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(assets, fontsize=10.5)
ax.set_ylabel("$\\Delta$ Sharpe vs. buy-hold (2025-only slice)", fontsize=10.5)
ax.set_title("Native Weekly-Cadence Grid, Crypto5 Panel: 2025 Out-of-Window Check\n"
              "(extracted from a single continuous 2024–2025 walk-forward pass; 0/20 cells significant)",
              fontsize=12.0, fontweight="bold", pad=12)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(axis="y", color="#e5e5e0", linewidth=0.8, zorder=0)
ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)

# highlight ETH: all four conditions negative
ax.annotate("all 4 conditions\nnegative", xy=(1, -1.68), xytext=(1.55, -1.95),
            fontsize=8.6, color=RED, style="italic", ha="left",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

ax.set_ylim(-2.2, 2.9)

plt.tight_layout()
plt.savefig("weekly_grid_crypto5_2025_barplot.png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print("wrote weekly_grid_crypto5_2025_barplot.png")
