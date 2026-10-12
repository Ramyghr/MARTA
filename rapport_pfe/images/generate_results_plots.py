"""Regenerates rl_ablation_barplot.png and regime_characterization_barplot.png for the
MARTA PFE report's Results chapter (chapters/results.tex). Both are built directly from
numbers already reported and verified in Tables 'tab:rl-ablation-crypto5' and
'tab:regime-characterization' -- no external data file is read, so these plots cannot
drift out of sync with the text they illustrate.
"""
import matplotlib.pyplot as plt
import numpy as np

BG = "#fafaf8"
RED, ORANGE, GREEN, BLUE, GRAY = "#b83227", "#e8622c", "#0a7a1e", "#1f77d4", "#555555"

# ---------------------------------------------------------------------------
# Plot 1: RL ablation Sharpe vs. buy-and-hold (Table tab:rl-ablation-crypto5)
# ---------------------------------------------------------------------------
buy_hold_sharpe = 1.498
configs = [
    "1: raw_regime\n(no fixes)",
    "2: raw\n(no fixes)",
    "3: + reward shaping\n+ min_hold=24",
    "4: (3) + BC\nwarm-start",
    "5: discrete action\n+ min_hold=168",
]
sharpe = [-2.717, -1.211, -0.515, 1.461, -0.034]
ci_lo = [-7.103, -5.351, -4.099, -0.140, -3.714]   # CI on delta-vs-buyhold
ci_hi = [-1.329, -0.044, -0.048, 0.070, 0.548]
verdict = ["worse", "worse", "worse", "no", "inconclusive"]
color_map = {"worse": RED, "no": GREEN, "inconclusive": ORANGE}
colors = [color_map[v] for v in verdict]
# convert delta CI back to absolute-Sharpe CI for the error bar
err_lo = [sharpe[i] - (buy_hold_sharpe + ci_lo[i]) for i in range(5)]
err_hi = [(buy_hold_sharpe + ci_hi[i]) - sharpe[i] for i in range(5)]

fig, ax = plt.subplots(figsize=(10.5, 5.8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

x = np.arange(len(configs))
bars = ax.bar(x, sharpe, color=colors, width=0.58, zorder=3,
               yerr=[err_lo, err_hi], capsize=4, ecolor="#333333", error_kw=dict(lw=1.2, zorder=4))
ax.axhline(buy_hold_sharpe, color=GRAY, linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
ax.text(len(configs) - 0.42, buy_hold_sharpe + 0.18, f"buy-hold reference (Sharpe {buy_hold_sharpe})",
        color=GRAY, fontsize=9, ha="right", style="italic")
ax.axhline(0, color="#bbbbbb", linewidth=1.0, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Sharpe ratio", fontsize=10.5)
ax.set_title("RL Ablation: Sharpe vs. Buy-and-Hold, Crypto-Only Panel\n(error bars: 95% block-bootstrap CI on the delta)",
              fontsize=12.5, fontweight="bold", pad=12)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.grid(axis="y", color="#e5e5e0", linewidth=0.8, zorder=0)

handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[v]) for v in ("worse", "no", "inconclusive")]
ax.legend(handles, ["significantly worse", "not significant", "inconclusive"],
          loc="lower left", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("rl_ablation_barplot.png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print("wrote rl_ablation_barplot.png")

# ---------------------------------------------------------------------------
# Plot 2: Per-regime characterization (Table tab:regime-characterization)
# ---------------------------------------------------------------------------
regimes = ["R0", "R1", "R2", "R3", "R4", "R5"]
labels = ["Very Low Vol\nRange-bound", "Moderate Vol\nTrending", "Low Vol\nRange-bound",
          "Low Vol\nRange-bound", "Elevated Vol\nTrending", "Extreme Vol\nTrending"]
reg_sharpe = [-1.04, 1.11, -0.22, -0.26, 1.39, -0.44]
ann_vol = [33.5, 67.0, 68.3, 46.9, 101.5, 189.8]
time_share = [15.2, 17.2, 18.8, 20.3, 19.4, 9.0]

fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
fig.patch.set_facecolor(BG)
ax1.set_facecolor(BG)

x = np.arange(len(regimes))
bar_colors = [GREEN if s > 0 else RED for s in reg_sharpe]
ax1.bar(x, reg_sharpe, color=bar_colors, width=0.5, zorder=3)
ax1.axhline(0, color="#999999", linewidth=1.0, zorder=2)
ax1.set_ylabel("Historical Sharpe (population avg., 2018--2025)", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels([f"{r}\n{l}" for r, l in zip(regimes, labels)], fontsize=8.6)
for spine in ("top",):
    ax1.spines[spine].set_visible(False)
ax1.grid(axis="y", color="#e5e5e0", linewidth=0.8, zorder=0)

ax2 = ax1.twinx()
ax2.plot(x, ann_vol, color=BLUE, marker="o", linewidth=2.0, zorder=4, label="Annualized vol (%)")
ax2.set_ylabel("Annualized volatility (%)", fontsize=10, color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE)
ax2.spines["top"].set_visible(False)

for xi, ts in zip(x, time_share):
    ax1.text(xi, min(reg_sharpe) - 0.28, f"{ts}% of time", ha="center", fontsize=7.8, color="#666666")

ax1.set_title("Per-Regime Characterization: Historical Sharpe and Volatility\n(six-state, five-asset, 22-column configuration)",
              fontsize=12.5, fontweight="bold", pad=12)
ax1.set_ylim(min(reg_sharpe) - 0.65, max(reg_sharpe) + 0.5)

plt.tight_layout()
plt.savefig("regime_characterization_barplot.png", dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)
print("wrote regime_characterization_barplot.png")
