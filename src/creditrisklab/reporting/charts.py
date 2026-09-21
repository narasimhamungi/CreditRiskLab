"""Charts for the validation report.

Four charts, each answering a question a credit reviewer actually asks. No chart is
produced for decoration: a score histogram with ten positives is noise, so the score
distribution is plotted as a strip rather than a density, which is honest about n.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

STYLE = {"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "font.size": 9}


def _save(fig, out_dir: Path, name: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def roc_chart(y_true, scores_by_model: dict[str, np.ndarray], out_dir: Path) -> str:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 4.2))
        for label, scores in scores_by_model.items():
            mask = ~np.isnan(np.asarray(scores, dtype="float64"))
            if len(np.unique(np.asarray(y_true)[mask])) < 2:
                continue
            fpr, tpr, _ = roc_curve(np.asarray(y_true)[mask], np.asarray(scores)[mask])
            ax.plot(fpr, tpr, label=label, linewidth=1.6)
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Out-of-fold ROC: model vs. published benchmark")
        ax.legend(loc="lower right", frameon=False)
        return _save(fig, out_dir, "roc.png")


def calibration_chart(reliability: pd.DataFrame, out_dir: Path) -> str:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect calibration")
        ax.scatter(
            reliability["mean_predicted_pd"],
            reliability["observed_default_rate"],
            s=reliability["n"] * 6,
            alpha=0.75,
            label="bucket (size = n)",
        )
        ax.set_xlabel("Mean predicted PD (pre prior-correction)")
        ax.set_ylabel("Observed default rate")
        ax.set_title("Calibration by predicted-PD bucket")
        ax.legend(frameon=False)
        return _save(fig, out_dir, "calibration.png")


def score_strip(predictions: pd.DataFrame, out_dir: Path) -> str:
    """Strip plot, not a density: with this many positives a KDE invents structure."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        rng = np.random.default_rng(7)
        for label, colour, name in ((0, "#4878A8", "survived 12m"), (1, "#C0392B", "defaulted within 12m")):
            subset = predictions.loc[predictions["label"] == label, "pd"]
            if subset.empty:
                continue
            jitter = rng.normal(loc=label, scale=0.05, size=len(subset))
            ax.scatter(subset, jitter, s=22, alpha=0.75, color=colour, label=f"{name} (n={len(subset)})")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["non-default", "default"])
        ax.set_xscale("log")
        ax.set_xlabel("Calibrated, prior-corrected PD (log scale)")
        ax.set_title("PD by realised outcome")
        ax.legend(frameon=False, loc="upper left")
        return _save(fig, out_dir, "pd_by_outcome.png")


def grade_chart(grade_table: pd.DataFrame, out_dir: Path) -> str:
    with plt.rc_context(STYLE):
        fig, ax1 = plt.subplots(figsize=(5.6, 3.8))
        ax1.bar(grade_table["grade"].astype(str), grade_table["n"], color="#B8C6D6", label="observations")
        ax1.set_xlabel("Risk grade")
        ax1.set_ylabel("Observations")
        ax2 = ax1.twinx()
        ax2.plot(
            grade_table["grade"].astype(str),
            grade_table["observed_default_rate"],
            marker="o",
            color="#C0392B",
            linewidth=1.6,
            label="observed default rate",
        )
        ax2.set_ylabel("Observed default rate")
        ax2.grid(False)
        ax1.set_title("Masterscale: population and realised default rate")
        return _save(fig, out_dir, "grades.png")


def build_all(result, out_dir: str | Path) -> list[str]:
    out_dir = Path(out_dir)
    paths: list[str] = []
    preds = result.predictions
    if preds is None or preds.empty:
        return paths
    scores = {"Logistic (out-of-fold)": preds["raw_pd"].to_numpy()}
    if "z_double_prime" in preds:
        z = preds["z_double_prime"].astype("float64")
        scores["Altman Z'' (benchmark)"] = (-z.fillna(z.median())).to_numpy()
    paths.append(roc_chart(preds["label"].to_numpy(), scores, out_dir))
    reliability = result.calibration.get("reliability")
    if reliability is not None and not reliability.empty:
        paths.append(calibration_chart(reliability, out_dir))
    paths.append(score_strip(preds, out_dir))
    if result.grade_table is not None and not result.grade_table.empty:
        paths.append(grade_chart(result.grade_table, out_dir))
    return paths
