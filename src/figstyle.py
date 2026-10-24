"""figstyle.py -- common style of the study figures.

Every plotting script imports this module, so all figures share fonts, sizes,
colours and line widths.

Figures are drawn at their printed width, 8.5 cm (single column) or 15 cm
(full width), with 8 pt text, following the figure guidelines of IOP
Publishing, so the sizes set here are the sizes in print.

Each colour stands for the same thing in every figure: blue for the switched
film and the probe, orange for the best static film and the absorption dip,
teal, violet and green for further films or settings, and black or grey for
limits and reference lines. Curves also differ in line style or marker, so
the figures stay readable in greyscale.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CM = 1.0 / 2.54
FULL = 15.0 * CM              # full-width figure, in
SINGLE = 8.5 * CM             # single-column figure, in

# ---- colours
BLUE = "#2a78d6"              # switched film, probe
ORANGE = "#eb6834"            # best static film, absorption dip
TEAL = "#1baf7a"              # further films or settings
VIOLET = "#4a3aa7"            # further films or settings
GREEN = "#008300"             # further films or settings
BLACK = "#0b0b0b"             # text, limits
DARKGREY = "#52514e"          # secondary text
GREY = "#898781"              # annotations, reference lines
AXIS = "#6f6e69"              # frame and ticks
GRID = "#e6e5df"              # grid
BAND = "#f1f0ec"              # light shading

# ---- line weights (pt)
LW = 1.3                      # data lines
LW_THIN = 0.7                 # reference lines
MS = 4.2                      # marker size (pt)


def setup():
    """Apply the style and return pyplot."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans",
        "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold",
        "mathtext.sf": "sans",
        "mathtext.cal": "sans",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlepad": 6,
        "axes.labelpad": 3,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": DARKGREY,
        "ytick.labelcolor": DARKGREY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": LW,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "round",
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        "legend.borderaxespad": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return plt


def grid(ax, axis="both"):
    """Light grid drawn under the data."""
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def top_legend(fig, handles, ncol=None, y=1.0, x=0.5):
    """One legend for the whole figure, centred above the panels."""
    return fig.legend(handles=handles, loc="lower center", ncol=ncol or len(handles),
                      bbox_to_anchor=(x, y), handlelength=2.4, columnspacing=1.8,
                      borderaxespad=0.0)


def title(ax, letter, text=""):
    """Panel label: bold letter, plain title, left-aligned."""
    ax.set_title(f"$\\bf{{({letter})}}$  {text}" if text else f"$\\bf{{({letter})}}$",
                 loc="left", color=BLACK)


def label_end(ax, x, y, text, dx=4, dy=0, color=DARKGREY, ha="left", va="center", **kw):
    """Label placed dx, dy points from a data point, usually the end of a line."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=8, color=color, **kw)


def save(fig, name, out, width=FULL):
    """Write out/<name>.pdf (vector, preferred by IOP) and .png (600 dpi).

    The figure is narrowed until its tight bounding box fits `width`, so the
    type keeps its point size at the printed width."""
    fig.canvas.draw()
    pad = 0.02
    for _ in range(6):
        box = fig.get_tightbbox(fig.canvas.get_renderer())
        over = box.width + 2 * pad - width
        if over <= 0.005:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w - over, h, forward=True)
        fig.canvas.draw()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=pad, facecolor="white")
