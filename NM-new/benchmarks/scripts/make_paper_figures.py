"""
Figures for the initialization x restart paper.

Design notes (the choices are deliberate, not defaults):

* Form. All three answer "how does X change with dimension", so all three are
  lines over n. Dimension is plotted on a log axis because the levels are
  2/5/10/20; on a linear axis the n=2..10 range -- where most of the action is --
  would be squashed against the left edge.

* Colour is categorical, keyed to *trigger identity*, and the map is global: a
  trigger is the same colour in every figure. Hues are the documented
  categorical slots, and the exact sets used were checked with the palette
  validator rather than eyeballed. `size` sits on violet rather than the next
  free slot because magenta against orange fails the normal-vision floor
  (delta-E 12.9 < 15).

* Two of the palette hues fall below 3:1 against a white surface, so the
  "relief" rule applies: every value plotted here also appears in the paper's
  tables, and the series are marker-differentiated as well as colour-coded.

* No dual axes anywhere. Figure 2 shows two quantities of different scale, so
  it is two panels, not two y-scales on one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[2].parent / "paper" / "figures"

DIMS = (2, 5, 10, 20)
INITS = [
    "pfeffer", "spendley_full", "spendley_10pct", "spendley_2pct",
    "uniform", "gaussian", "positive_basis",
]

#: Global entity -> colour map. Validated with scripts/validate_palette.js
#: (light mode, adjacent pairs): all checks pass for every subset drawn below.
COLOR = {
    "none":         "#2a78d6",  # blue
    "size":         "#4a3aa7",  # violet
    "flat":         "#eb6834",  # orange
    "gbnm":         "#1baf7a",  # aqua
    "conditioning": "#008300",  # green
    "progress":     "#eda100",  # yellow
}
MARKER = {
    "none": "o", "size": "s", "flat": "^",
    "gbnm": "D", "conditioning": "v", "progress": "P",
}
LABEL = {
    "none": "no restart", "size": "size", "flat": "flat",
    "gbnm": "GBNM (size $\\vee$ degeneracy)", "conditioning": "degeneracy",
    "progress": "Kelley (sufficient decrease)",
}
ORDER = ["flat", "size", "progress", "none", "gbnm", "conditioning"]

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.dpi": 150,
})


def construction_of(trigger):
    return "oriented" if trigger == "progress" else "gaussian_best"


def load():
    return json.loads((RESULTS / "bbob_factorial_merged.json").read_text())["records"]


def style_axes(ax, ylabel, xlabel="dimension $n$"):
    """Recessive grid and axes; the data should be the darkest thing on the page."""
    ax.set_xscale("log")
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def line(ax, xs, ys, key, **kw):
    return ax.plot(
        xs, ys,
        color=COLOR[key], marker=MARKER[key], markersize=5.5,
        linewidth=1.8, markeredgecolor="white", markeredgewidth=0.7,
        label=LABEL[key], **{"zorder": 3, **kw},
    )


def solved(R, trigger, construction, dim, tau=1e-7):
    rs = [
        r for r in R
        if r["trigger"] == trigger and r.get("construction") == construction
        and r["dim"] == dim
    ]
    return 100 * np.mean([r[f"solved_tau{tau:g}"] for r in rs]) if rs else np.nan


def restarts(R, trigger, construction, dim):
    rs = [
        r for r in R
        if r["trigger"] == trigger and r.get("construction") == construction
        and r["dim"] == dim
    ]
    return float(np.mean([r["restarts"] for r in rs])) if rs else np.nan


def init_spread(R, trigger, construction, dim):
    cells = defaultdict(dict)
    for r in R:
        if (
            r["trigger"] == trigger and r.get("construction") == construction
            and r["dim"] == dim
        ):
            cells[r["problem"]][r["init"]] = r["best"]
    rows = [[c[i] for i in INITS] for c in cells.values() if len(c) == len(INITS)]
    if not rows:
        return np.nan
    mr = np.apply_along_axis(stats.rankdata, 1, np.asarray(rows)).mean(0)
    return float(mr.max() - mr.min())


def degeneracy_share(R, dim):
    rs = [
        r for r in R
        if r["trigger"] == "gbnm" and r.get("construction") == "gaussian_best"
        and r["dim"] == dim
    ]
    d = sum(r["restarts_by_degeneracy"] for r in rs)
    s = sum(r["restarts_by_size"] for r in rs)
    return 100 * d / (d + s) if (d + s) else np.nan


# --------------------------------------------------------------------------
def fig_triggers(R):
    """Figure 1 -- the dimensional story: two triggers cross the baseline in
    opposite directions."""
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for key in ORDER:
        c = construction_of(key)
        ys = [solved(R, key, c, d) for d in DIMS]
        line(ax, DIMS, ys, key,
             linestyle="--" if key == "none" else "-",
             zorder=4 if key in ("none", "progress") else 3)

    # Both principled detectors cross the baseline at nearly the same dimension,
    # in opposite directions. That coincidence is the figure's point, so it gets
    # one annotation in free space rather than two on top of the data.
    ax.annotate(
        "both principled detectors cross\n"
        "the no-restart baseline near $n\\approx7$,\n"
        "in opposite directions",
        xy=(6.9, 35.5), xytext=(2.08, 3.5),
        fontsize=7.5, color=INK_2, ha="left", va="bottom",
        arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.7,
                        shrinkA=6, shrinkB=2,
                        connectionstyle="angle3,angleA=0,angleB=70"),
    )

    style_axes(ax, "problems solved at $\\tau=10^{-7}$ (%)")
    ax.set_ylim(0, 88)
    ax.legend(frameon=False, loc="upper right", labelcolor=INK_2,
              handlelength=2.2, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1-triggers.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_gbnm(R):
    """Figure 2 -- why the degeneracy trigger collapses. Two scales, so two
    panels; never two y-axes."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.9))

    shares = [degeneracy_share(R, d) for d in DIMS]
    a1.plot(DIMS, shares, color=COLOR["conditioning"], marker=MARKER["conditioning"],
            markersize=5.5, linewidth=1.8, markeredgecolor="white",
            markeredgewidth=0.7, zorder=3)
    for d, v in zip(DIMS, shares):
        a1.annotate(f"{v:.0f}%", (d, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7.5, color=INK_2)
    style_axes(a1, "degeneracy share of firings (%)")
    a1.set_ylim(45, 116)
    a1.set_title("Which half of GBNM fires", color=INK)

    for key in ("gbnm", "size"):
        line(a2, DIMS, [restarts(R, key, "gaussian_best", d) for d in DIMS], key)
    style_axes(a2, "restarts per run")
    a2.set_yscale("log")
    a2.set_title("What it costs", color=INK)
    a2.legend(frameon=False, loc="upper left", labelcolor=INK_2, handlelength=2.2)

    fig.tight_layout()
    fig.savefig(OUT / "fig2-gbnm.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_spread(R):
    """Figure 3 -- restarts absorb the initializer effect, less so as n grows."""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for key in ORDER:
        c = construction_of(key)
        line(ax, DIMS, [init_spread(R, key, c, d) for d in DIMS], key,
             linestyle="--" if key == "none" else "-")

    ax.annotate("", xy=(18.6, 4.18), xytext=(18.6, 2.26),
                arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=0.8))
    ax.text(17.6, 3.22, "how much the\ninitializer\nstill matters",
            fontsize=7.5, color=INK_2, ha="right", va="center")

    style_axes(ax, "initializer mean-rank spread")
    ax.set_ylim(0, 4.7)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2, handlelength=2.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3-spread.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = load()
    fig_triggers(R)
    fig_gbnm(R)
    fig_spread(R)
    for f in sorted(OUT.glob("*.pdf")):
        print(f"wrote {f}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
