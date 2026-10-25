"""Regenerates pipeline_diagram.png for the MARTA PFE report's five-layer architecture
teaser (chapters/introduction.tex, Section 'sec:proposed-solution', Figure 'fig:pipeline').
Keep the layer list here in sync with that section's enumerate list if either one changes.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

LAYERS = [
    ("1. Feature\nEngineering", "Engineered volatility,\nmomentum, trend\nindicators (5 assets)", "#1f77d4"),
    ("2. Regime\nDetection", "Gaussian HMM fit on\nthe engineered\nfeature space", "#f0a000"),
    ("3. Multi-Agent\nDeliberation", "3 LLM agents + contrarian\n+ GARCH risk; deterministic\nfund-manager consensus", "#e06ba0"),
    ("4. RL Execution\nAgent", "PPO position sizing\n(evaluated separately\nfrom the debate)", "#0a7a1e"),
    ("5. Monitoring\nDashboard", "Next.js interface,\nregime + reasoning +\nexecution, real time", "#3d2f8f"),
]

fig, ax = plt.subplots(figsize=(14, 5.2))
fig.patch.set_facecolor("#fafaf8")
ax.set_facecolor("#fafaf8")

n = len(LAYERS)
box_w, box_h = 2.3, 1.6
gap = 0.5
total_w = n * box_w + (n - 1) * gap
x0 = -total_w / 2
y_center = 0

centers = []
for i, (title, subtitle, color) in enumerate(LAYERS):
    x = x0 + i * (box_w + gap)
    centers.append(x + box_w / 2)
    box = FancyBboxPatch(
        (x, y_center - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.06,rounding_size=0.08",
        linewidth=0, facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + box_w / 2, y_center + 0.4, title, ha="center", va="center",
             fontsize=12.5, fontweight="bold", color="white", zorder=4)
    ax.text(x + box_w / 2, y_center - 0.35, subtitle, ha="center", va="center",
             fontsize=9, color="white", zorder=4, linespacing=1.6)

for i in range(n - 1):
    arrow = FancyArrowPatch(
        (centers[i] + box_w / 2, y_center), (centers[i + 1] - box_w / 2, y_center),
        arrowstyle="-|>", mutation_scale=18, linewidth=1.8, color="#444444", zorder=2,
    )
    ax.add_patch(arrow)

# Grouping brackets tying layers back to the two-part problem statement.
bracket_y = y_center + box_h / 2 + 0.35
label_y = bracket_y + 0.22


def draw_bracket(x_left, x_right, label):
    ax.plot([x_left, x_left, x_right, x_right],
             [bracket_y - 0.08, bracket_y, bracket_y, bracket_y - 0.08],
             color="#666666", linewidth=1.2, zorder=2)
    ax.text((x_left + x_right) / 2, label_y, label, ha="center", va="bottom",
             fontsize=9.5, fontweight="bold", color="#333333")


draw_bracket(x0 - 0.05, x0 + 2 * box_w + gap + 0.05, "Regime Representation Learning")
draw_bracket(x0 + 2 * (box_w + gap) - 0.05, x0 + 4 * box_w + 3 * gap + 0.05,
             "Regime-Conditioned Decision Making")
ax.text(centers[4], label_y, "Interpretability\n(supporting)", ha="center", va="bottom",
         fontsize=9.5, fontweight="bold", color="#333333", linespacing=1.3)

ax.set_xlim(x0 - 0.5, x0 + total_w + 0.5)
ax.set_ylim(-1.3, label_y + 0.8)
ax.axis("off")
ax.set_title("MARTA Five-Layer Pipeline", fontsize=16, fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig("pipeline_diagram.png", dpi=150, facecolor=fig.get_facecolor())
print("wrote pipeline_diagram.png")
