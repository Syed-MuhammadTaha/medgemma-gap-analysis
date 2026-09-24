"""All charts for the write-up."""

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"
GRAY, LIGHT_GRAY, GRID = "#c9c8c2", "#ecebe6", "#e4e3df"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _figure(size, title, subtitle):
    fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, weight="bold", color=INK)
    fig.text(0.02, 0.905, subtitle, fontsize=9, color=MUTED)
    return fig, ax


def _save(fig, path):
    fig.tight_layout(rect=(0, 0, 0.99, 0.9))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def metric_paradox(rows, labels, path):
    """Exp 2a dot plot: one row per variant, one dot per report, big dot = mean, dashed line = unchanged report."""
    mean = lambda v: np.mean([r["rougeL"] for r in rows if r["variant"] == v])
    correct = [v for v in labels if v.startswith("paraphrase")]
    order = correct + sorted((v for v in labels if v.startswith("flip")), key=mean)

    fig, ax = _figure((9, 5.2), "ROUGE-L scores wrong pathology reports above correct ones",
                      f"{len({r['case_id'] for r in rows})} TCGA breast cancer reports. "
                      "Each dot is one report; the large dot is the mean.")
    rng = np.random.default_rng(0)
    for y, v in enumerate(order):
        color = BLUE if v in correct else ORANGE
        xs = [r["rougeL"] for r in rows if r["variant"] == v]
        ax.scatter(xs, y + rng.uniform(-0.18, 0.18, len(xs)), s=14, color=color, alpha=0.35, linewidths=0)
        ax.scatter(mean(v), y, s=90, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"{mean(v):.3f}", (mean(v), y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)

    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.text(0.998, -0.75, "unchanged report = 1.0", ha="right", va="center", fontsize=8.5, color=MUTED)
    ax.axhline(len(correct) - 0.5, color=GRID, linewidth=1)
    for y, text, color in [(-0.75, "Correct rewordings (facts unchanged)", BLUE),
                           (len(correct) - 0.25, "Wrong reports (one fact flipped)", ORANGE)]:
        ax.text(0.005, y, text, transform=ax.get_yaxis_transform(), fontsize=9, color=color,
                va="center", weight="bold")
    ax.set_yticks(range(len(order)), [labels[v] for v in order], fontsize=9.5, color=INK)
    ax.set_ylim(len(order) - 0.5, -1.0)
    ax.set_xlim(min(r["rougeL"] for r in rows) - 0.01, 1.008)
    ax.set_xlabel("ROUGE-L against the original report (higher = judged more similar)", fontsize=9.5, color=MUTED)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _save(fig, path)


def fact_accuracy(counts, facts, visible, verdicts, n_cases, path):
    """Exp 1 stacked bars: per fact, how many cases MedGemma got right, wrong or didn't state."""
    colors = {"correct": BLUE, "wrong": RED, "not stated": GRAY, "no reference": LIGHT_GRAY}
    order = visible + [f for f in facts if f not in visible]
    ys = [0, 1, 2, 3.8, 4.8, 5.8]  # gap between the two groups

    fig, ax = _figure((9.5, 5.2), "What MedGemma says about six treatment-critical facts",
                      f"{n_cases} TCGA breast cancer cases, up to 32 tissue patches each, "
                      "compared with the pathologist's report.")
    for y, fact in zip(ys, order):
        left = 0
        for v in verdicts:
            n = counts[fact][v]
            if not n:
                continue
            ax.barh(y, n, left=left, height=0.62, color=colors[v], edgecolor=SURFACE, linewidth=2)
            if n >= 2:
                ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=9,
                        color="white" if v in ("correct", "wrong") else INK)
            left += n

    for y, text in [(-0.7, "Visible in tissue patches"),
                    (3.1, "Not determinable from one slide: anything MedGemma states here is invented")]:
        ax.text(0, y, text, fontsize=9.5, color=INK, weight="bold", va="center")
    ax.set_yticks(ys, [facts[f] for f in order], fontsize=9.5, color=INK)
    ax.set_ylim(6.4, -1.1)
    ax.set_xlim(0, n_cases)
    ax.set_xlabel(f"Cases (of {n_cases})", fontsize=9.5, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[v]) for v in verdicts]
    ax.legend(handles, verdicts.values(), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2,
              frameon=False, fontsize=9, labelcolor=MUTED)
    _save(fig, path)

