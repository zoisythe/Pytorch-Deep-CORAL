"""Rebuild the concise Beamer deck from archived results, without training."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DECK = Path(__file__).resolve().parents[1]
DATA, FIG = DECK / "data", DECK / "figures"
ROOT = DECK.parents[2]
BLUE, GOLD, ROSE, TEAL, INK, GRAY = (
    "#004488", "#DDAA33", "#BB5566", "#337C80", "#222244", "#87909F"
)
COLORS = {"no_adaptation": GRAY, "global": BLUE, "local": ROSE, "lmmd": TEAL}
LABELS = {"no_adaptation": "No adaptation", "global": "Global CORAL",
          "local": "Local CORAL", "lmmd": "LMMD"}
FIGURE_SOURCES = {}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.edgecolor": "#D5D9E1", "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#E1E5EC", "grid.linewidth": 0.65,
    "figure.facecolor": "#FCFCFF", "axes.facecolor": "#FCFCFF",
    "savefig.facecolor": "#FCFCFF", "pdf.fonttype": 42,
    "legend.frameon": False, "lines.linewidth": 2.1,
})


def rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def select(table, key, value):
    return [r for r in table if r[key] == value]


def arr(table, key):
    return np.array([f(r, key) for r in table])


def save(name, fig, inputs):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=.07)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)
    FIGURE_SOURCES[name] = inputs


def line(ax, table, key, label, color, *, percent=False, style="-"):
    ax.plot(arr(table, "epoch"), arr(table, key) * (100 if percent else 1),
            label=label, color=color, linestyle=style)
    ax.set_xlabel("Epoch")
    ax.grid(axis="y")


def snapshot_metadata(value):
    if isinstance(value, dict):
        return {key: snapshot_metadata(item) for key, item in value.items()
                if "sha256" not in key.lower() and key != "source_snapshot_hashes_match"}
    if isinstance(value, list):
        return [snapshot_metadata(item) for item in value]
    return value


def sources(refresh):
    manifest_path = DATA / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for source in manifest["sources"]:
        path = DATA / source["snapshot"]
        if refresh:
            origin = ROOT / source["source"]
            if origin.suffix == ".json":
                content = snapshot_metadata(json.loads(origin.read_text(encoding="utf-8")))
                path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                shutil.copyfile(origin, path)
        if not path.is_file():
            raise FileNotFoundError(path)
        source["bytes"] = path.stat().st_size
    manifest = snapshot_metadata(manifest)
    manifest["deck_date"] = "2026-10-16"
    paper = "https://www.jmlr.org/papers/v13/gretton12a.html"
    if paper not in manifest["papers"]:
        manifest["papers"].append(paper)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate(alex, resnet, sub, rlmmd):
    inventory = []
    for family, table, key in [
        ("AlexNet CORAL", alex, "run_id"), ("ResNet50 CORAL", resnet, "setting"),
        ("AlexNet subdomain", sub, "setting"),
    ]:
        for setting in dict.fromkeys(row[key] for row in table):
            series = select(table, key, setting)
            inventory.append({"family": family, "run_id": setting, "epochs": len(series),
                              "updates": int(sum(arr(series, "train_steps"))),
                              "target_accuracy": f(series[-1], "target_accuracy")})
    best = max(rlmmd, key=lambda row: f(row, "target_accuracy"))
    inventory.append({"family": "ResNet50 LMMD", "run_id": "author_code_seed42",
                      "epochs": len(rlmmd), "updates": int(sum(arr(rlmmd, "train_steps"))),
                      "target_accuracy": f(best, "target_accuracy")})
    assert len(inventory) == 27
    assert sum(int(r["epochs"]) for r in inventory) == 1593
    assert sum(int(r["updates"]) for r in inventory) == 22968
    for table in [alex, resnet, sub, rlmmd]:
        assert all(int(f(r, "target_samples")) == 795 for r in table)
        assert all(abs(f(r, "target_accuracy") - f(r, "target_correct") / 795) < 1e-10 for r in table)
    for setting in ["global_quarter", "local", "mixed", "lmmd"]:
        reference, series = select(sub, "setting", "global"), select(sub, "setting", setting)
        for first, second in zip(reference[:10], series[:10]):
            for key in ["classification_loss", "global_loss", "target_accuracy"]:
                assert abs(f(first, key) - f(second, key)) < 1e-6
    assert int(best["epoch"]) == 42 and int(rlmmd[-1]["epoch"]) == 72
    joint = select(alex, "run_id", "joint_adaptive_delayed_100")
    assert round(f(joint[-1], "lambda_end"), 2) == 6.94
    assert round(f(joint[-1], "CORAL_loss"), 4) == .0167
    for name in ["alex_lmmd_author_validation.json", "resnet_lmmd_author_validation.json"]:
        record = json.loads((DATA / name).read_text(encoding="utf-8"))
        assert record["passed"] and record["absolute_loss_error"] == 0
    with (DATA / "experiment_inventory.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)
    return inventory


def tables():
    out = []

    def command(name, items):
        body = "\n".join(" & ".join(item) + r" \\" for item in items)
        out.append("\\newcommand{\\" + name + "}{%\n" + body + "\n}")

    def pct(value):
        return f"{100 * value:.2f}" + r"\%"

    table = {r["run_id"]: r for r in rows("alex_lambda.csv")}
    settings = [
        ("fc8_fixed_50", "fc8：固定"),
        ("fc8_adaptive_full_50", "fc8：第 1 轮起自适应"),
        ("fc8_adaptive_delayed_50", "fc8：第 41 轮起自适应"),
        ("joint_fixed_100", "conv5+fc8：固定"),
        ("joint_adaptive_delayed_100", "conv5+fc8：第 50 轮起自适应"),
    ]
    command("LambdaRows", [[label, pct(f(table[key], "best_target_accuracy")),
                            pct(f(table[key], "final_target_accuracy")),
                            f"{f(table[key], 'final_lambda'):.2f}"] for key, label in settings])
    command("ResnetRows", [["No CORAL" if r["setting"] == "no_coral" else r["setting"].replace("_", "+"), r["stop_epoch"],
                            pct(f(r, "target_accuracy")), pct(f(r, "baseline_same_epoch_accuracy")),
                            f"{f(r, 'improvement_pp'):+.2f} pp"] for r in rows("resnet_summary.csv")])
    sub = {r["setting"]: r for r in rows("subdomain_summary.csv")}
    terms = [("no_adaptation", "无适应", "$0$"), ("global", "global", r"$0.5\G$"),
             ("global_quarter", r"global\_quarter", r"$0.25\G$"),
             ("local", "local", r"$0.5\LC$"), ("mixed", "mixed", r"$0.25\G+0.25\LC$")]
    command("SubRows", [[label, term, pct(f(sub[key], "target_accuracy"))] for key, label, term in terms])
    diagnostic = rows("subdomain_diagnostics.csv")
    raw = {r["setting"]: r for r in diagnostic if r["representation"] == "raw"}
    contrast = {r["setting"]: r for r in diagnostic if r["representation"] == "contrast"}
    command("GeometryRows", [[LABELS[s], pct(f(raw[s], "target_accuracy")),
                              f"{f(raw[s], 'true_local_covariance_distance'):.5f}",
                              f"{f(contrast[s], 'centroid_cosine_distance'):.3f}",
                              f"{f(contrast[s], 'target_soft_class_purity'):.3f}"]
                             for s in ["global", "local", "lmmd"]])
    (DATA / "tables.tex").write_text("% Percentages only for displayed classification outcomes.\n" +
                                     "\n\n".join(out) + "\n", encoding="utf-8")


def figures(alex, sub, rlmmd):
    # Reproduction: the first slide contains this image and no other text.
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.2), layout="constrained")
    for run, label, color in [("no_adaptation_fixed_50", "No adaptation", GRAY),
                              ("fc8_fixed_50", "CORAL fc8", BLUE)]:
        series = select(alex, "run_id", run)
        line(axes[0, 0], series, "target_accuracy", label, color, percent=True)
        line(axes[0, 1], series, "source_accuracy", label, color, percent=True)
        line(axes[1, 0], series, "classification_loss", label, color)
        line(axes[1, 1], series, "CORAL_fc8", label, color)
    for ax, title in zip(axes.flat, ["Target accuracy (%)", "Source accuracy (%)",
                                    "Source CE", "fc8 covariance distance"]):
        ax.set_title(title, loc="left")
    axes[0, 0].set_ylim(0, 65)
    axes[0, 1].set_ylim(0, 105)
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    axes[0, 0].legend(loc="lower right", fontsize=10)
    save("baseline", fig, ["alex_epochs.csv"])

    layer = list(reversed(rows("alex_layers.csv")))
    fig, ax = plt.subplots(figsize=(10.8, 4.7), layout="constrained")
    values = [100 * f(r, "final_target_accuracy") for r in layer]
    colors = [TEAL if r["setting"] == "conv5+fc8" else BLUE if "fc8" in r["setting"] else GRAY for r in layer]
    ax.barh([r["setting"] for r in layer], values, color=colors, height=.65)
    baseline = next(f(r, "final_target_accuracy") * 100 for r in layer if r["setting"] == "No adaptation")
    ax.axvline(baseline, color=INK, ls="--", lw=1)
    for i, value in enumerate(values):
        ax.text(value + .45, i, f"{value:.2f}%", va="center", fontsize=12)
    ax.set(xlim=(0, 63), xlabel="Target accuracy at epoch 50 (%)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    save("alex_layers", fig, ["alex_layers.csv"])

    # Use end-of-epoch lambda, raw CORAL (not weighted CORAL), and accuracy.
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), layout="constrained")
    for run, label, color, style in [
        ("joint_fixed_100", "Fixed lambda", BLUE, "-"),
        ("joint_adaptive_delayed_100", "Adaptive from 50", ROSE, "--"),
    ]:
        series = [r for r in select(alex, "run_id", run) if f(r, "epoch") >= 35]
        for ax, key, percentage in zip(axes, ["lambda_end", "CORAL_loss", "target_accuracy"], [False, False, True]):
            line(ax, series, key, label, color, percent=percentage, style=style)
    for ax in axes:
        ax.axvline(50, color=GRAY, ls=":", lw=1)
        ax.set_xlim(35, 101)
    axes[0].set_title("Lambda", loc="left")
    axes[0].set_ylim(0, 7.7)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].annotate("6.94", (100, 6.9355845), xytext=(-32, 3), textcoords="offset points", color=ROSE)
    axes[1].set(title="Raw CORAL loss", yscale="log", ylim=(.01, .15))
    axes[2].set(title="Target accuracy (%)", ylim=(0, 65))
    save("lambda_failure", fig, ["alex_epochs.csv"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), layout="constrained",
                              gridspec_kw={"width_ratios": [1, 1.45]})
    summary = {r["setting"]: r for r in rows("subdomain_summary.csv")}
    settings = ["no_adaptation", "global", "local", "lmmd"]
    values = [100 * f(summary[s], "target_accuracy") for s in settings]
    axes[0].bar(range(4), values, color=[COLORS[s] for s in settings], width=.62)
    for i, value in enumerate(values):
        axes[0].text(i, value + 1, f"{value:.2f}%", ha="center", fontsize=11)
    axes[0].set_xticks(range(4), ["No\nadaptation", "Global\nCORAL", "Local\nCORAL", "LMMD"], fontsize=10)
    axes[0].set(ylabel="Target accuracy (%)", ylim=(0, 78), title="Epoch 50")
    axes[0].grid(axis="y")
    axes[0].set_axisbelow(True)
    for setting, style in zip(settings, [":", "-", "-.", "--"]):
        line(axes[1], select(sub, "setting", setting), "target_accuracy", LABELS[setting],
             COLORS[setting], percent=True, style=style)
    axes[1].axvline(10, color=GRAY, ls=":", lw=1)
    axes[1].set(ylabel="Target accuracy (%)", ylim=(0, 78), title="Target accuracy by epoch")
    axes[1].legend(ncol=2, loc="lower right", fontsize=9)
    save("alex_lmmd_accuracy", fig, ["subdomain_summary.csv", "subdomain_epochs.csv"])

    fig, ax = plt.subplots(figsize=(10.8, 4.2), layout="constrained")
    line(ax, rlmmd, "target_accuracy", "Target", BLUE, percent=True)
    line(ax, rlmmd, "source_accuracy", "Source", GRAY, percent=True, style="--")
    best = max(rlmmd, key=lambda r: f(r, "target_accuracy"))
    point = (f(best, "epoch"), 100 * f(best, "target_accuracy"))
    ax.plot(*point, "o", color=TEAL, ms=7)
    ax.annotate("93.71% at epoch 42", point, xytext=(27, 84), fontsize=12,
                arrowprops={"arrowstyle": "-", "color": TEAL})
    ax.axvline(72, color=ROSE, ls=":", lw=1.2)
    ax.text(4, 70, "Stop at 72: 93.58%", ha="left", color=ROSE, fontsize=11)
    ax.set(ylabel="Accuracy (%)", ylim=(65, 102), xlim=(0, 75))
    ax.legend(loc="lower right", fontsize=11)
    save("resnet_lmmd_accuracy", fig, ["resnet_lmmd_epochs.csv"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-snapshots", action="store_true")
    args = parser.parse_args()
    sources(args.refresh_snapshots)
    alex, resnet, sub, rlmmd = [rows(name) for name in [
        "alex_epochs.csv", "resnet_epochs.csv", "subdomain_epochs.csv", "resnet_lmmd_epochs.csv"]]
    inventory = validate(alex, resnet, sub, rlmmd)
    tables()
    figures(alex, sub, rlmmd)
    (DATA / "figure_sources.json").write_text(json.dumps(FIGURE_SOURCES, indent=2) + "\n", encoding="utf-8")
    report = {"runs": len(inventory), "epochs": sum(r["epochs"] for r in inventory),
              "updates": sum(r["updates"] for r in inventory), "figures": len(FIGURE_SOURCES),
              "classification_display": "accuracy_percent_only", "training_performed": False,
              "checks": ["source snapshot files exist", "all epoch accuracies match archived outcomes",
                         "27-run inventory reconciles", "ten warmup epochs match",
                         "LMMD best/stop epochs match", "adaptive lambda and raw loss endpoints match"]}
    (DATA / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
