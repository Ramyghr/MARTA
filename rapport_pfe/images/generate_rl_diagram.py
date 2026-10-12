"""Regenerates rl_diagram.png for the MARTA PFE report's RL execution agent figure
(chapters/methodology.tex, Section 'subsec:rl-policy', Figure 'fig:rl-execution').
Shows the MDP loop: observation -> shared policy trunk -> Categorical/Beta heads ->
action -> environment step -> reward, looping back to the next observation.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14.5, 7.5))
fig.patch.set_facecolor("#fafaf8")
ax.set_facecolor("#fafaf8")

BOX_KW = dict(boxstyle="round,pad=0.06,rounding_size=0.08", linewidth=0, zorder=3)
TITLE_KW = dict(ha="center", va="center", fontsize=10, fontweight="bold", color="white", zorder=4)
SUB_KW = dict(ha="center", va="center", fontsize=7.8, color="white", zorder=4, linespacing=1.4)


def box(x, y, w, h, title, subtitle, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, facecolor=color, **BOX_KW))
    if subtitle:
        ax.text(x, y + 0.17, title, **TITLE_KW)
        ax.text(x, y - 0.24, subtitle, **SUB_KW)
    else:
        ax.text(x, y, title, **TITLE_KW)
    return x, y, w, h


def arrow(p1, p2, color="#444444", style="-|>", lw=1.6, ls="solid", connectionstyle=None):
    kw = dict(arrowstyle=style, mutation_scale=15, linewidth=lw, color=color, zorder=2, linestyle=ls)
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch(p1, p2, **kw))


y_mid = 0.0
x_obs, x_trunk, x_heads, x_act, x_env, x_rew = 0.9, 3.7, 6.7, 9.5, 11.9, 14.3

box(x_obs, y_mid, 2.0, 1.9, "Observation", "regime + consensus\n+ portfolio +\nasset one-hot", "#333333")
box(x_trunk, y_mid, 2.6, 1.3, "Policy Trunk", "shared MLP\n156$\\to$256$\\to$128$\\to$64", "#1f77d4")

box(x_heads, 1.1, 2.2, 1.1, "Categorical", "direction\n(long/short/hold)", "#3d2f8f", )
box(x_heads, -1.1, 2.2, 1.1, "Beta($\\alpha,\\beta$)", "position\nfraction $\\in(0,1)$", "#3d2f8f")

box(x_act, y_mid, 1.9, 1.5, "Action", "direction,\nposition size", "#e8622c")
box(x_env, y_mid, 2.0, 1.7, "Env Step", "MARTATrading\nEnv", "#e8622c")
box(x_rew, y_mid, 2.2, 2.1, "Reward", "$\\dfrac{\\Delta\\mathrm{PnL}}{\\mathrm{CVaR}} - \\lambda\\cdot\\mathrm{DD}$\n$-\\ \\mathrm{idle}$", "#0a7a1e")

arrow((x_obs + 1.05, y_mid), (x_trunk - 1.35, y_mid))
arrow((x_trunk + 1.35, 0.5), (x_heads - 1.15, 1.1))
arrow((x_trunk + 1.35, -0.5), (x_heads - 1.15, -1.1))
arrow((x_heads + 1.15, 1.1), (x_act - 1.0, 0.4))
arrow((x_heads + 1.15, -1.1), (x_act - 1.0, -0.4))
arrow((x_act + 1.0, y_mid), (x_env - 1.05, y_mid))
arrow((x_env + 1.05, y_mid), (x_rew - 1.15, y_mid), color="#0a7a1e", lw=1.8)

# --- feedback loop: reward/env -> next observation ---
arrow((x_rew, 1.1), (x_obs, 2.1), color="#999999", lw=1.4, ls=(0, (4, 2)),
      connectionstyle="arc3,rad=0.28")
ax.text((x_obs + x_rew) / 2, 2.75, "next observation $s_{t+1}$", fontsize=8.5,
        color="#888888", style="italic", ha="center")

ax.set_xlim(-0.5, 15.8)
ax.set_ylim(-2.8, 3.6)
ax.axis("off")
ax.set_title("RL Execution Agent: Observation $\\to$ Action $\\to$ Reward Loop", fontsize=16,
             fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("rl_diagram.png", dpi=150, facecolor=fig.get_facecolor())
print("wrote rl_diagram.png")
