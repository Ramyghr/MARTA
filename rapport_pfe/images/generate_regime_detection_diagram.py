"""Regenerates regime_detection_diagram.png for the MARTA PFE report's regime-detection
figure (chapters/methodology.tex, Section 'subsec:regime-final-fit', Figure
'fig:regime-detection'). Shows the production raw-feature HMM pipeline: engineered
features -> BIC-guided state-count search -> best-of-5-seed Gaussian HMM fit -> regime
probability stream, with the seed-stability check as a side validation step.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.5, 6.5))
fig.patch.set_facecolor("#fafaf8")
ax.set_facecolor("#fafaf8")

BOX_KW = dict(boxstyle="round,pad=0.06,rounding_size=0.08", linewidth=0, zorder=3)
TITLE_KW = dict(ha="center", va="center", fontsize=10.5, fontweight="bold", color="white", zorder=4)
SUB_KW = dict(ha="center", va="center", fontsize=8.2, color="white", zorder=4, linespacing=1.5)


def box(x, y, w, h, title, subtitle, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, facecolor=color, **BOX_KW))
    if subtitle:
        ax.text(x, y + 0.18, title, **TITLE_KW)
        ax.text(x, y - 0.26, subtitle, **SUB_KW)
    else:
        ax.text(x, y, title, **TITLE_KW)
    return x, y, w, h


def arrow(p1, p2, color="#444444", style="-|>", lw=1.6, ls="solid"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15,
                                  linewidth=lw, color=color, zorder=2, linestyle=ls))


y_mid = 0.0
x_feat, x_bic, x_hmm, x_stream = 0.9, 4.0, 7.6, 11.3

box(x_feat, y_mid, 2.0, 1.3, "Engineered\nFeatures", "14 dims $\\times$ asset\n(60h window)", "#333333")
box(x_bic, y_mid, 2.6, 1.3, "BIC Model\nSelection", "$n \\in \\{2,\\ldots,8\\}$, diag cov\nflat-curve guard", "#e8622c")
box(x_hmm, y_mid, 2.6, 1.3, "Gaussian HMM", "$n=6$ states\nfull training split", "#1f77d4")
box(x_stream, y_mid, 2.4, 1.3, "Regime Stream", "label (MAP) +\nprobability vector", "#3d2f8f")

arrow((x_feat + 1.05, y_mid), (x_bic - 1.35, y_mid))
arrow((x_bic + 1.35, y_mid), (x_hmm - 1.35, y_mid))
arrow((x_hmm + 1.35, y_mid), (x_stream - 1.25, y_mid))

# --- side validation box: best-of-5-seed EM + ARI check ---
val_x, val_y = x_hmm, 2.6
box(val_x, val_y, 3.2, 1.2, "Best-of-5-Seed EM", "pairwise ARI / NMI\nacross seeds", "#0a7a1e")
arrow((val_x, val_y - 0.65), (x_hmm, y_mid + 0.68), color="#0a7a1e", lw=1.8, ls=(0, (4, 2)))
ax.text(val_x + 1.9, (val_y + y_mid) / 2 + 0.3, "keeps highest-LL fit", fontsize=8.0,
        color="#0a7a1e", rotation=-58, style="italic")

# --- downstream consumers ---
cons_x, cons_y = x_stream, -2.6
box(cons_x, cons_y, 3.6, 1.1, "Downstream Consumers", "Multi-Agent Layer (Ch.3.6) +\nRL Observation (Ch.3.7)", "#b83227")
arrow((x_stream, y_mid - 0.65), (cons_x, cons_y + 0.55), color="#b83227", lw=1.8)

ax.set_xlim(-0.5, 13.3)
ax.set_ylim(-3.6, 3.6)
ax.axis("off")
ax.set_title("Regime Detection Pipeline (Raw-Feature HMM)", fontsize=16, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("regime_detection_diagram.png", dpi=150, facecolor=fig.get_facecolor())
print("wrote regime_detection_diagram.png")
