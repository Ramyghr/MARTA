"""Regenerates multi_agent_diagram.png for the MARTA PFE report's multi-agent
deliberation figure (chapters/methodology.tex, Section 'subsec:debate-flow', Figure
'fig:multi-agent'). Shows the three-stage debate flow: Round 1 (three LLM directional
agents + the deterministic Risk agent in parallel), the conditional Contrarian check,
conditional Round 2 rebuttals, and the deterministic Fund Manager synthesis producing
the final decision.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(15.5, 8.0))
fig.patch.set_facecolor("#fafaf8")
ax.set_facecolor("#fafaf8")

BOX_KW = dict(boxstyle="round,pad=0.05,rounding_size=0.07", linewidth=0, zorder=3)
TITLE_KW = dict(ha="center", va="center", fontsize=9.5, fontweight="bold", color="white", zorder=4)
SUB_KW = dict(ha="center", va="center", fontsize=7.6, color="white", zorder=4, linespacing=1.4)


def box(x, y, w, h, title, subtitle, color, title_fs=9.5, sub_fs=7.6):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, facecolor=color, **BOX_KW))
    tkw = dict(TITLE_KW); tkw["fontsize"] = title_fs
    skw = dict(SUB_KW); skw["fontsize"] = sub_fs
    if subtitle:
        ax.text(x, y + 0.15, title, **tkw)
        ax.text(x, y - 0.22, subtitle, **skw)
    else:
        ax.text(x, y, title, **tkw)
    return x, y, w, h


def arrow(p1, p2, color="#444444", style="-|>", lw=1.5, ls="solid"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                  linewidth=lw, color=color, zorder=2, linestyle=ls))


# --- column x-positions ---
x_ctx, x_r1, x_contra, x_r2, x_fm, x_out = 0.8, 3.6, 6.7, 9.5, 12.3, 15.0

# --- Regime Context (input) ---
box(x_ctx, 0.0, 1.9, 1.5, "Regime\nContext", "label + prob\nvector (Ch.3.5)", "#333333")

# --- Round 1: 3 LLM agents + Risk agent ---
y_macro, y_tech, y_sent, y_risk = 2.55, 0.85, -0.85, -2.55
box(x_r1, y_macro, 2.5, 1.2, "MacroAgent", "cross-asset corr.\n+ contagion flag", "#1f77d4")
box(x_r1, y_tech, 2.5, 1.2, "TechnicalAgent", "momentum / vol\n60-candle window", "#1f77d4")
box(x_r1, y_sent, 2.5, 1.2, "SentimentAgent", "FinBERT + 24h\nrecency decay", "#1f77d4")
box(x_r1, y_risk, 2.5, 1.2, "RiskAgent", "GARCH(1,1) $\\to$\nVaR/CVaR (no LLM)", "#e8622c")

ax.text(x_r1, 3.55, "Round 1", fontsize=10, color="#555555", style="italic", ha="center")

for y in (y_macro, y_tech, y_sent, y_risk):
    arrow((x_ctx + 1.0, 0.0 if abs(y) < 0.1 else None) if False else (x_ctx + 0.95, 0.0), (x_r1 - 1.3, y))

# --- Contrarian (conditional) ---
box(x_contra, 0.0, 2.6, 1.6, "ContrarianAgent", "forced-opposite bias;\nactivates @ conf $\\geq$0.60",
    "#b83227")
for y in (y_macro, y_tech, y_sent):
    arrow((x_r1 + 1.3, y), (x_contra - 1.4, 0.35 if y > 0 else (0.0 if y == 0 else -0.35)),
          color="#999999", lw=1.0, ls=(0, (2, 2)))
ax.text(x_contra, 1.1, "(conditional)", fontsize=7.5, color="#b83227", style="italic", ha="center")

# --- Round 2 rebuttals (conditional) ---
box(x_r2, 0.0, 2.6, 1.6, "Round 2\nRebuttals", "opposition, groupthink,\nor Contrarian active",
    "#e8622c")
arrow((x_contra + 1.35, 0.0), (x_r2 - 1.35, 0.0))
ax.text(x_r2, 1.1, "(conditional)", fontsize=7.5, color="#e8622c", style="italic", ha="center")

# --- Fund Manager synthesis ---
box(x_fm, 0.0, 2.6, 1.8, "FundManager\nSynthesis", "weighted vote +\nregime conviction cap\n(deterministic)", "#3d2f8f")
arrow((x_r2 + 1.35, 0.0), (x_fm - 1.35, 0.0))
arrow((x_r1 + 1.3, y_risk), (x_fm - 1.3, -1.0), color="#999999", lw=1.2, ls=(0, (2, 2)))
ax.text((x_r1 + x_fm) / 2 + 0.4, (y_risk - 1.0) / 2 - 0.15, "risk limits", fontsize=7.3,
        color="#888888", style="italic")

# --- Output ---
box(x_out, 0.0, 2.0, 1.6, "Decision", "action, size,\nconviction", "#0a7a1e")
arrow((x_fm + 1.35, 0.0), (x_out - 1.05, 0.0), color="#0a7a1e", lw=2.0)

ax.set_xlim(-0.4, 16.2)
ax.set_ylim(-3.6, 4.1)
ax.axis("off")
ax.set_title("Multi-Agent Deliberation Flow", fontsize=16, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("multi_agent_diagram.png", dpi=150, facecolor=fig.get_facecolor())
print("wrote multi_agent_diagram.png")
