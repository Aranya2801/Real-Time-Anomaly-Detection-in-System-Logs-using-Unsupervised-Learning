"""
Aggregate the raw run CSVs in results/ into markdown tables and figures.

    python experiments/make_report.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("results")
ORDER = ["RareEvent", "PCA", "IsolationForest", "OneClassSVM", "LOF", "AutoEncoder", "DeepLog"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
MARKERS = dict(zip(ORDER, ["o", "s", "^", "D", "v", "P", "X"]))

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
})


def fmt(mean, std):
    return f"{mean:.3f}" if pd.isna(std) or std < 5e-4 else f"{mean:.3f} ± {std:.3f}"


def main_table(df):
    out = []
    for setting, rate in [("clean (normal-only training)", "0.0"), ("natural (unlabelled training)", "natural")]:
        sub = df[df.rate == rate]
        g = sub.drop(columns=["rate"]).groupby("model").agg(["mean", "std"])
        for m in ORDER:
            if m not in g.index:
                continue
            row = g.loc[m]
            out.append({
                "training data": setting, "model": m,
                "precision": fmt(row.precision["mean"], row.precision["std"]),
                "recall": fmt(row.recall["mean"], row.recall["std"]),
                "F1": fmt(row.f1["mean"], row.f1["std"]),
                "PR-AUC": fmt(row.pr_auc["mean"], row.pr_auc["std"]),
                "ROC-AUC": fmt(row.roc_auc["mean"], row.roc_auc["std"]),
                "recall (distinguishable)": fmt(row.recall_distinguishable["mean"],
                                                row.recall_distinguishable["std"]),
            })
    return pd.DataFrame(out)


def contamination_plot(df):
    agg = df.groupby(["model", "rate"]).agg(c=("contamination", "mean"), f1=("f1", "mean"),
                                            f1s=("f1", "std"), pr=("pr_auc", "mean"),
                                            prs=("pr_auc", "std")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, (col, sd, title) in zip(axes, [("f1", "f1s", "F1 at label-free threshold"),
                                          ("pr", "prs", "PR-AUC (threshold-free)")]):
        for m in ORDER:
            a = agg[agg.model == m].sort_values("c")
            if a.empty:
                continue
            x = a.c * 100
            ax.plot(x, a[col], color=COLORS[m], marker=MARKERS[m], markersize=6, linewidth=2,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=m)
            ax.fill_between(x, a[col] - a[sd].fillna(0), a[col] + a[sd].fillna(0),
                            color=COLORS[m], alpha=0.12, linewidth=0)
        ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
        ax.set_xlabel("anomalies in training data (%)")
        ax.set_ylim(0, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=7,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("HDFS: how training-set contamination hurts each detector", x=0.01, ha="left",
                 fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(R / "contamination.png", dpi=160)

    tbl = agg.pivot_table(index="model", columns="rate", values="pr").reindex(ORDER)
    tbl.columns = [f"{float(c) * 100:.1f}%" if c != "natural" else "natural (~3%)" for c in tbl.columns]
    return tbl.round(3)


def ablation(abl):
    abl = abl.assign(rate=abl.rate.replace({"0": "0.0"}))
    main = abl[abl.rate.isin(["0.0", "natural"])]
    t = (main.groupby(["use_eos", "rate"])[["precision", "recall", "f1", "pr_auc"]].mean()
         .round(3).reset_index())
    t["use_eos"] = t.use_eos.map({True: "with END token", False: "vanilla DeepLog"})
    t["rate"] = t.rate.map({"0.0": "clean", "natural": "natural"})

    tg = abl[abl.rate == "topg"].groupby(["use_eos", "top_g"])[["precision", "recall", "f1"]].mean()
    tg = tg.round(3).reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for eos, color, marker in [(False, "#eb6834", "s"), (True, "#2a78d6", "o")]:
        a = tg[tg.use_eos == eos]
        lbl = "with END token" if eos else "vanilla DeepLog"
        ax.plot(a.top_g, a.f1, color=color, marker=marker, linewidth=2, markersize=6,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=lbl)
        best = a.loc[a.f1.idxmax()]
        ax.annotate(f"{lbl} (best g={int(best.top_g)}, F1={best.f1:.2f})", (best.top_g, best.f1),
                    xytext=(0, 9 if eos else -16), textcoords="offset points", fontsize=8,
                    color="#52514e", ha="center")
    ax.set_xlabel("g (event counts as normal if in top-g predictions)")
    ax.set_title("DeepLog F1 vs top-g, clean training", loc="left", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 19)
    fig.tight_layout()
    fig.savefig(R / "deeplog_topg.png", dpi=160)
    return t, tg


if __name__ == "__main__":
    runs = pd.read_csv(R / "hdfs_contamination_runs.csv", dtype={"rate": str})
    abl = pd.read_csv(R / "deeplog_ablation_runs.csv", dtype={"rate": str})
    parts = ["# Results (generated by experiments/make_report.py)\n"]
    parts += ["## Main table\n", main_table(runs).to_markdown(index=False), ""]
    parts += ["## PR-AUC vs training contamination\n", contamination_plot(runs).to_markdown(), ""]
    t, tg = ablation(abl)
    parts += ["## DeepLog END-token ablation\n", t.to_markdown(index=False), ""]
    parts += ["## DeepLog top-g sensitivity (clean)\n", tg.to_markdown(index=False), ""]
    pa = pd.read_csv(R / "parser_accuracy.csv")
    parts += ["## Parser grouping accuracy\n", pa.to_markdown(index=False), ""]
    (R / "RESULTS.md").write_text("\n".join(parts))
    print("\n".join(parts))
