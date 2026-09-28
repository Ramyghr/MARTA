"""Regenerates gantt_chart.png for the MARTA PFE report's development-phases table
(chapters/introduction.tex, Table 'tab:phases' / Figure 'fig:gantt'). Keep the phase
list here in sync with that table if either one changes.
"""
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PHASES = [
    ("1. Literature Review &\nArchitecture Design", dt.date(2026, 3, 16), dt.date(2026, 5, 10), "#1f77d4", "8 wk"),
    ("2. Data Pipeline &\nFeature Engineering", dt.date(2026, 5, 11), dt.date(2026, 5, 24), "#e8622c", "2 wk"),
    ("3. Regime Encoder\n(TFT + InfoNCE) — v1", dt.date(2026, 5, 25), dt.date(2026, 6, 14), "#1f9c7a", "3 wk"),
    ("4. Regime Detection\n(HMM)", dt.date(2026, 6, 15), dt.date(2026, 6, 28), "#f0a000", "2 wk"),
    ("5. Multi-Agent System", dt.date(2026, 6, 29), dt.date(2026, 7, 19), "#e06ba0", "3 wk"),
    ("6. RL Execution Agent &\nEvaluation Fixes", dt.date(2026, 7, 20), dt.date(2026, 8, 30), "#0a7a1e", "6 wk"),
    ("7. Dashboard &\nLive Integration", dt.date(2026, 7, 27), dt.date(2026, 9, 6), "#3d2f8f", "6 wk"),
    ("8. Encoder Diagnostic &\nRejected Retrains", dt.date(2026, 8, 10), dt.date(2026, 8, 30), "#b83227", "3 wk"),
    ("9. Raw-Feature Pivot &\nCrypto5 Migration", dt.date(2026, 8, 27), dt.date(2026, 9, 16), "#5a5a5a", "3 wk"),
]

fig, ax = plt.subplots(figsize=(11, 9.5))
fig.patch.set_facecolor("#fafaf8")
ax.set_facecolor("#fafaf8")

n = len(PHASES)
for i, (label, start, end, color, wk_label) in enumerate(PHASES):
    y = n - 1 - i
    duration = (end - start).days + 1
    ax.barh(y, duration, left=start, height=0.55, color=color, zorder=3)
    ax.text(
        mdates.date2num(end) + 2.5, y, wk_label,
        va="center", ha="left", fontsize=12, color="#222222",
    )

ax.set_yticks([n - 1 - i for i in range(n)])
ax.set_yticklabels([p[0] for p in PHASES], fontsize=12)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.set_xlim(dt.date(2026, 3, 8), dt.date(2026, 9, 24))

ax.grid(axis="x", color="#dddddd", linewidth=0.8, zorder=0)
for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color("#999999")
ax.tick_params(axis="both", length=0)

ax.set_title("MARTA Project Planning (Mar – Sep 2026)", fontsize=17, fontweight="bold", loc="left", pad=18)

plt.tight_layout()
plt.savefig("gantt_chart.png", dpi=150, facecolor=fig.get_facecolor())
print("wrote gantt_chart.png")
