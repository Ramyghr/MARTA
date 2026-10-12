"""Regenerates contrastive_diagram.png for the MARTA PFE report's contrastive-pretraining
figure (chapters/methodology.tex, Section 'subsec:encoder-training', Figure
'fig:contrastive'). Classic SimCLR-style two-branch diagram: the same input window is
augmented twice, passed through a shared-weight encoder + projection head, and the two
resulting embeddings are pulled together (positive pair) while embeddings of other windows
in the mini-batch are pushed apart (negatives) via the NT-Xent loss.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13.5, 7.5))
fig.patch.set_facecolor("#fafaf8")
ax.set_facecolor("#fafaf8")

BOX_KW = dict(boxstyle="round,pad=0.06,rounding_size=0.08", linewidth=0, zorder=3)
TITLE_KW = dict(ha="center", va="center", fontsize=10.5, fontweight="bold", color="white", zorder=4)
SUB_KW = dict(ha="center", va="center", fontsize=8.2, color="white", zorder=4, linespacing=1.5)


def box(x, y, w, h, title, subtitle, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, facecolor=color, **BOX_KW))
    if subtitle:
        ax.text(x, y + 0.16, title, **TITLE_KW)
        ax.text(x, y - 0.22, subtitle, **SUB_KW)
    else:
        ax.text(x, y, title, **TITLE_KW)
    return x, y, w, h


def arrow(p1, p2, color="#444444", style="-|>", lw=1.6, ls="solid"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=15,
                                  linewidth=lw, color=color, zorder=2, linestyle=ls))


# --- Input ---
x_in, y_mid = 0.4, 0.0
box(x_in, y_mid, 1.7, 1.0, "Input window", "60 h $\\times$ 14 features", "#333333")

# --- Two branches ---
y_top, y_bot = 1.65, -1.65
x_aug, x_enc, x_proj, x_loss = 3.0, 5.6, 8.2, 11.0

arrow((x_in + 0.85, 0.35), (x_aug - 1.0, y_top - 0.15))
arrow((x_in + 0.85, -0.35), (x_aug - 1.0, y_bot + 0.15))
ax.text(x_aug - 0.4, y_top + 0.75, "view A", fontsize=8.5, color="#555555", style="italic")
ax.text(x_aug - 0.4, y_bot - 0.75, "view B", fontsize=8.5, color="#555555", style="italic")

for y in (y_top, y_bot):
    box(x_aug, y, 2.1, 1.05, "Augmentation", "scale · jitter ·\nmask · time warp", "#e8622c")
    box(x_enc, y, 2.1, 1.05, "Encoder", "TFTRegimeEncoder\n(shared weights)", "#1f77d4")
    box(x_proj, y, 2.1, 1.05, "Projection head", "shared weights", "#3d2f8f")
    arrow((x_aug + 1.05, y), (x_enc - 1.05, y))
    arrow((x_enc + 1.05, y), (x_proj - 1.05, y))

# shared-weights connectors between the two branches
for x, label in ((x_enc, "shared"), (x_proj, "shared")):
    ax.add_patch(FancyArrowPatch((x, y_top - 0.55), (x, y_bot + 0.55), arrowstyle="<->",
                                  mutation_scale=10, linewidth=1.1, color="#999999",
                                  linestyle=(0, (3, 2)), zorder=1))
    ax.text(x + 0.22, 0, label, fontsize=7.3, color="#888888", rotation=90, va="center")

# --- z embeddings -> loss ---
loss_x, loss_y = x_loss, 0.0
arrow((x_proj + 1.05, y_top), (loss_x - 1.15, 0.5), color="#0a7a1e", lw=2.0)
arrow((x_proj + 1.05, y_bot), (loss_x - 1.15, -0.5), color="#0a7a1e", lw=2.0)
ax.text((x_proj + loss_x) / 2 + 0.1, y_top - 0.42, "$z_A$", fontsize=9, color="#0a7a1e", fontweight="bold")
ax.text((x_proj + loss_x) / 2 + 0.1, y_bot + 0.55, "$z_B$", fontsize=9, color="#0a7a1e", fontweight="bold")

box(loss_x, loss_y, 2.3, 1.7, "NT-Xent Loss", "pull $z_A$, $z_B$\ntogether (positive pair)", "#0a7a1e")

# --- negatives ---
neg_x, neg_y = x_loss, 3.1
box(neg_x, neg_y, 2.5, 1.0, "Other windows in\nmini-batch", "negatives", "#b83227")
arrow((neg_x, neg_y - 0.53), (loss_x, loss_y + 0.88), color="#b83227", lw=1.8, ls=(0, (4, 2)))
ax.text(neg_x + 1.55, (neg_y + loss_y) / 2 + 0.3, "push apart", fontsize=8.3,
        color="#b83227", rotation=-58, style="italic")

ax.set_xlim(-0.7, 13.0)
ax.set_ylim(-2.6, 4.0)
ax.axis("off")
ax.set_title("Contrastive Pretraining (NT-Xent / InfoNCE)", fontsize=16, fontweight="bold", pad=12)

plt.tight_layout()
plt.savefig("contrastive_diagram.png", dpi=150, facecolor=fig.get_facecolor())
print("wrote contrastive_diagram.png")
