"""Freeze evidence and rebuild paper figures/tables. Never imports training code."""

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

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
DATA = PAPER / "data"
FIG = PAPER / "figures"
BLUE, TEAL, ROSE, GOLD, GRAY = "#245781", "#237D75", "#AF4E64", "#BA8625", "#818B98"
SOURCES = {
    "data/alex_layers.csv": "reports/data/layer_results.csv",
    "data/alex_lambda.csv": "reports/data/lambda_results.csv",
    "data/alex_epochs.csv": "reports/data/epoch_metrics.csv",
    "data/alex_steps.csv": "reports/data/step_metrics.csv",
    "data/alex_manifest.json": "reports/data/run_manifest.json",
    "data/early_stop.csv": "reports/data/early_stop_results.csv",
    "data/early_stop_grid.csv": "reports/data/early_stop_rule_comparison.csv",
    "data/early_stop_epochs.csv": "reports/data/early_stop_epochs.csv",
    "data/resnet_summary.csv": "reports/resnet50/summary.csv",
    "data/resnet_epochs.csv": "reports/resnet50/epoch_metrics.csv",
    "data/subdomain_summary.csv": "reports/subdomain_coral/lmmd/summary.csv",
    "data/subdomain_epochs.csv": "reports/subdomain_coral/lmmd/epoch_metrics.csv",
    "data/subdomain_diagnostics.csv": "reports/subdomain_coral/analysis/diagnostics.csv",
    "data/subdomain_classes.csv": "reports/subdomain_coral/analysis/class_diagnostics.csv",
    "data/subdomain_pairs.csv": "reports/subdomain_coral/analysis/comparisons.csv",
    "data/subdomain_phases.csv": "reports/subdomain_coral/analysis/training_phases.csv",
    "data/resnet_lmmd_epochs.csv": "reports/resnet50_lmmd/data/metrics.csv",
    "data/resnet_lmmd_summary.json": "reports/resnet50_lmmd/data/summary.json",
    "data/experiment_inventory.csv": "reports/PPT/2/data/experiment_inventory.csv",
    "reference/main.pdf": "reports/PPT/2/main.pdf",
    "reference/deck_sources.json": "reports/PPT/2/data/sources.json",
    "reference/alex_report.md": "reports/2026-10-08-coral-experiments.md",
    "reference/subdomain_report.md": "reports/subdomain_coral/analysis/report.md",
    "reference/subdomain_protocol.md": "reports/subdomain_coral/protocol.md",
    "reference/alex_lmmd_protocol.md": "reports/subdomain_coral/lmmd/protocol.md",
    "reference/resnet_report.md": "reports/resnet50/report.md",
    "reference/resnet_lmmd_report.md": "reports/resnet50_lmmd/report.md",
    "reference/resnet_lmmd_protocol.md": "reports/resnet50_lmmd/protocol.md",
    "reference/coral.py": "src/core/coral.py",
    "reference/subdomain_coral.py": "src/core/subdomain_coral.py",
    "reference/lmmd.py": "src/core/lmmd.py",
    "reference/alexnet.py": "src/core/model.py",
    "reference/adaptive_weight.py": "src/experiments/adaptive_lambda/weight.py",
    "reference/alex_lmmd_author_validation.json": "reports/subdomain_coral/lmmd/author_validation.json",
    "reference/resnet_lmmd_author_validation.json": "reports/resnet50_lmmd/author_validation.json",
    "reference/subdomain_verification.json": "reports/subdomain_coral/verification.json",
    "reference/geometry_verification.json": "reports/subdomain_coral/analysis/verification.json",
    "reference/resnet_lmmd_verification.json": "reports/resnet50_lmmd/data/verification.json",
    "reference/dsan_main.py": "reports/resnet50_lmmd/reference/main.py",
    "reference/dsan_model.py": "reports/resnet50_lmmd/reference/DSAN.py",
    "reference/dsan_data_loader.py": "reports/resnet50_lmmd/reference/data_loader.py",
    "reference/dsan_lmmd.py": "reports/resnet50_lmmd/reference/lmmd.py",
}


def snapshot_metadata(value):
    if isinstance(value, dict):
        return {key: snapshot_metadata(item) for key, item in value.items()
                if "sha256" not in key.lower() and key != "source_snapshot_hashes_match"}
    if isinstance(value, list):
        return [snapshot_metadata(item) for item in value]
    return value


def freeze(refresh):
    manifest_file = DATA / "sources.json"
    if refresh:
        entries = []
        for snapshot, source in SOURCES.items():
            dest, origin = PAPER / snapshot, ROOT / source
            dest.parent.mkdir(parents=True, exist_ok=True)
            if origin.suffix == ".json":
                content = snapshot_metadata(json.loads(origin.read_text(encoding="utf-8")))
                dest.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            else:
                shutil.copyfile(origin, dest)
            entries.append({"snapshot": snapshot, "source": source, "bytes": dest.stat().st_size})
        manifest_file.write_text(json.dumps({
            "prepared_on": "2026-10-30", "experiment_evidence_through": "2026-10-12",
            "training_performed": False, "source_deck": "reports/PPT/2/main.pdf", "sources": entries,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = snapshot_metadata(json.loads(manifest_file.read_text(encoding="utf-8")))
    for entry in manifest["sources"]:
        path = PAPER / entry["snapshot"]
        if not path.is_file():
            raise FileNotFoundError(path)
        entry["bytes"] = path.stat().st_size
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def rows(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def f(row, key):
    return float(row[key])


def select(table, key, value):
    return [row for row in table if row[key] == value]


def arr(table, key):
    return np.asarray([f(row, key) for row in table])


def dump(name, value):
    (DATA / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


FIG_SOURCES = {}


def save(name, fig, sources):
    fig.savefig(FIG / (name + ".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(FIG / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    FIG_SOURCES[name] = sources


def line(ax, table, key, label, color, scale=1, style="-"):
    ax.plot(arr(table, "epoch"), arr(table, key) * scale, label=label, color=color, ls=style)
    ax.set_xlabel("Epoch")
    ax.grid(axis="y", alpha=0.5)


def figures():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "legend.frameon": False, "grid.color": "#CAD0D8", "grid.linewidth": 0.5,
        "pdf.fonttype": 42, "lines.linewidth": 1.65})
    epochs = rows("alex_epochs.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.2), layout="constrained")
    for run, label, color in [("no_adaptation_fixed_50", "No adaptation", GRAY),
                              ("fc8_fixed_50", "CORAL fc8", BLUE)]:
        table = select(epochs, "run_id", run)
        for ax, key, title, scale in zip(axs.flat,
                ["target_accuracy", "source_accuracy", "classification_loss", "CORAL_fc8"],
                ["(a) Target accuracy (%)", "(b) Source accuracy (%)", "(c) Source CE", "(d) fc8 CORAL distance"],
                [100, 100, 1, 1]):
            line(ax, table, key, label, color, scale)
            ax.set_title(title, loc="left")
    axs[0, 0].set_ylim(0, 70)
    axs[0, 1].set_ylim(0, 105)
    axs[1, 0].set_yscale("log")
    axs[1, 1].set_yscale("log")
    axs[0, 0].legend(loc="lower right")
    save("baseline", fig, ["alex_epochs.csv"])

    layers = rows("alex_layers.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    labels = [r["setting"].replace("No adaptation", "No adaptation") for r in layers]
    values = arr(layers, "final_target_accuracy") * 100
    ax.barh(np.arange(len(layers)), values, color=[TEAL if x == "conv5+fc8" else BLUE if x == "fc8" else GRAY for x in labels], height=.63)
    ax.set_yticks(np.arange(len(layers)), labels)
    ax.invert_yaxis()
    ref = f(next(r for r in layers if r["setting"] == "fc8"), "final_target_accuracy") * 100
    ax.axvline(ref, ls="--", lw=1, color=ROSE, label=f"fc8: {ref:.2f}%")
    for y, value in enumerate(values):
        ax.text(value + .5, y, f"{value:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 64)
    ax.set_xlabel("Target accuracy at epoch 50 (%)")
    ax.legend(loc="lower right")
    save("layers", fig, ["alex_layers.csv"])

    fixed = select(epochs, "run_id", "joint_fixed_100")
    adaptive = select(epochs, "run_id", "joint_adaptive_delayed_100")
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.0), layout="constrained")
    for table, label, color in [(fixed, "Fixed", BLUE), (adaptive, "Adaptive", ROSE)]:
        line(axs[0, 0], table, "target_accuracy", label, color, 100)
        line(axs[0, 1], table, "lambda_used", label, color)
        line(axs[0, 2], table, "CORAL_loss", label, color)
        line(axs[1, 0], table, "weighted_CORAL_loss", label, color)
    line(axs[1, 1], adaptive, "gradient_CE_ema", "CE", BLUE)
    line(axs[1, 1], adaptive, "gradient_CORAL_ema", "CORAL", ROSE)
    line(axs[1, 2], adaptive, "gradient_cosine", "Mean cosine", ROSE)
    for ax, title in zip(axs.flat, ["(a) Target accuracy (%)", "(b) Mean used lambda", "(c) Raw CORAL loss", "(d) Weighted CORAL loss", "(e) EMA gradient norm", "(f) Gradient cosine"]):
        ax.set_title(title, loc="left")
        ax.axvline(50, color=GRAY, ls=":", lw=.8)
    axs[0, 0].legend(loc="lower right")
    axs[1, 1].legend()
    axs[0, 2].set_yscale("log")
    axs[1, 1].set_yscale("log")
    axs[1, 2].axhline(0, color=GRAY, lw=.7)
    axs[1, 2].set_ylim(-1, 1)
    save("adaptive", fig, ["alex_epochs.csv"])

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
    line(axs[0], fixed, "target_accuracy", "Fixed joint", BLUE, 100)
    axs[0].axvline(58, color=TEAL, ls="--", label="Replay stop: 58")
    axs[0].scatter([58], [f(fixed[57], "target_accuracy")*100], color=TEAL, zorder=3)
    axs[0].set_ylabel("Target accuracy (%)")
    axs[0].legend()
    ce, weighted = arr(fixed, "classification_loss"), arr(fixed, "weighted_CORAL_loss")
    def ema(values):
        out = [values[0]]
        for value in values[1:]:
            out.append(.5*out[-1] + .5*value)
        return np.asarray(out)
    gap = lambda a, b: np.abs(a-b)/np.maximum(np.maximum(a, b), 1e-12)
    axs[1].plot(arr(fixed, "epoch"), gap(ce, weighted)*100, color=GRAY, ls=":", label="Raw")
    axs[1].plot(arr(fixed, "epoch"), gap(ema(ce), ema(weighted))*100, color=BLUE, label="EMA (0.5)")
    axs[1].axhline(25, color=ROSE, ls="--", label="25% threshold")
    axs[1].axvline(58, color=TEAL, ls="--")
    axs[1].set(xlabel="Epoch", ylabel="Relative loss gap (%)")
    axs[1].legend()
    save("stopping", fig, ["alex_epochs.csv", "early_stop.csv"])

    sub = rows("subdomain_epochs.csv")
    summary = rows("subdomain_summary.csv")
    settings = ["no_adaptation", "global", "global_quarter", "local", "mixed", "lmmd"]
    colors = [GRAY, BLUE, GOLD, ROSE, "#8A6D9C", TEAL]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    for setting, color in zip(settings, colors):
        line(axs[0], select(sub, "setting", setting), "target_accuracy", setting, color, 100)
    axs[0].axvline(10, ls=":", color=GRAY)
    axs[0].set_ylabel("Target accuracy (%)")
    axs[0].legend(ncol=2, fontsize=6.6)
    vals = [f(next(r for r in summary if r["setting"] == s), "target_accuracy")*100 for s in settings]
    axs[1].barh(np.arange(6), vals, color=colors)
    axs[1].set_yticks(np.arange(6), settings)
    axs[1].invert_yaxis()
    for y, value in enumerate(vals):
        axs[1].text(value+.6, y, f"{value:.2f}", va="center", fontsize=8)
    axs[1].set(xlabel="Accuracy at epoch 50 (%)", xlim=(0, 77))
    save("subdomain", fig, ["subdomain_summary.csv", "subdomain_epochs.csv"])

    diag = select(rows("subdomain_diagnostics.csv"), "representation", "raw")
    diag = [next(r for r in diag if r["setting"] == s) for s in ["global", "local", "lmmd"]]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.25), layout="constrained")
    for ax, key, title in zip(axs, ["true_local_covariance_distance", "centroid_cosine_distance", "target_mean_logit_norm", "target_soft_class_purity"],
            ["(a) True-class covariance", "(b) Center cosine distance", "(c) Target logit norm", "(d) Soft-class purity"]):
        vals = arr(diag, key)
        ax.bar(np.arange(3), vals, color=[BLUE, ROSE, TEAL], width=.65)
        ax.set_xticks(np.arange(3), ["Global", "Local", "LMMD"], rotation=30, ha="right")
        ax.set_title(title, fontsize=8, loc="left")
        ax.margins(y=.23)
        for x, value in enumerate(vals):
            ax.text(x, value, f"{value:.3f}", va="bottom", ha="center", fontsize=7)
    save("geometry", fig, ["subdomain_diagnostics.csv"])

    res = rows("resnet_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
    y = np.arange(len(res))
    v1, v2 = arr(res, "baseline_same_epoch_accuracy")*100, arr(res, "target_accuracy")*100
    ax.barh(y-.17, v1, height=.32, label="No CORAL, same epoch", color=GRAY)
    ax.barh(y+.17, v2, height=.32, label="Reported stop", color=BLUE)
    ax.set_yticks(y, [r["setting"].replace("_", "+")+f" (e={r['stop_epoch']})" for r in res])
    ax.invert_yaxis()
    ax.set(xlim=(0, 94), xlabel="Target accuracy (%)")
    for i, r in enumerate(res):
        ax.text(max(v1[i], v2[i])+.5, i, f"{f(r, 'improvement_pp'):+.2f} pp", va="center", fontsize=8)
    ax.legend(loc="lower left", ncol=2)
    save("resnet_coral", fig, ["resnet_summary.csv"])

    lm = rows("resnet_lmmd_epochs.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), layout="constrained")
    line(axs[0], lm, "target_accuracy", "Target", TEAL, 100)
    line(axs[0], lm, "source_accuracy", "Source", BLUE, 100)
    axs[0].axvline(42, color=GRAY, ls=":")
    axs[0].scatter([42], [100*f(lm[41], "target_accuracy")], color=TEAL, zorder=3)
    axs[0].set(title="(a) Accuracy (%)", ylim=(50, 102))
    axs[0].legend()
    line(axs[1], lm, "classification_loss", "CE", BLUE)
    line(axs[1], lm, "weighted_lmmd_loss", "Weighted LMMD", TEAL)
    axs[1].set(title="(b) Training losses", yscale="log")
    axs[1].legend()
    line(axs[2], lm, "lmmd_weight", "Scheduled lambda", GOLD)
    axs[2].set(title="(c) Scheduled weight", ylim=(0, .52))
    save("resnet_lmmd", fig, ["resnet_lmmd_epochs.csv"])
    dump("figure_sources.json", FIG_SOURCES)


def tex_escape(value):
    return str(value).replace("_", r"\_").replace("%", r"\%")


def tables():
    layers, weights = rows("alex_layers.csv"), rows("alex_lambda.csv")
    sub, res = rows("subdomain_summary.csv"), rows("resnet_summary.csv")
    tables_dir = PAPER / "tables"
    tables_dir.mkdir(exist_ok=True)
    def write(name, lines):
        (tables_dir / (name + ".tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    ref = f(next(r for r in layers if r["setting"] == "fc8"), "final_target_accuracy")
    write("layers", [f"{tex_escape(r['setting'])} & {100*f(r,'final_target_accuracy'):.2f} & {100*(f(r,'final_target_accuracy')-ref):+.2f} & {100*f(r,'best_target_accuracy'):.2f} / {r['best_target_epoch']} " + r"\\" for r in layers])
    labels = [r"fc8：固定", r"conv5+fc8：固定", r"fc8：第 1 轮起自适应", r"fc8：第 41 轮起自适应", r"conv5+fc8：第 50 轮起自适应"]
    write("weights", [f"{label} & {r['epochs']} & {100*f(r,'best_target_accuracy'):.2f} / {r['best_target_epoch']} & {100*f(r,'final_target_accuracy'):.2f} & {f(r,'final_lambda'):.3f} " + r"\\" for label, r in zip(labels, weights)])
    terms = {"no_adaptation": "$0$", "global": "$0.5G$", "global_quarter": "$0.25G$", "local": "$0.5L_C$", "mixed": "$0.25G+0.25L_C$", "lmmd": "$0.5M$"}
    write("subdomain", [f"{tex_escape(r['setting'])} & {terms[r['setting']]} & {100*f(r,'target_accuracy'):.2f} & {100*f(r,'target_macro_accuracy'):.2f} " + r"\\" for r in sub])
    write("resnet", [f"{tex_escape(r['setting'])} & {r['stop_epoch']} & {100*f(r,'target_accuracy'):.2f} & {100*f(r,'baseline_same_epoch_accuracy'):.2f} & {f(r,'improvement_pp'):+.2f} " + r"\\" for r in res])
    diag = select(rows("subdomain_diagnostics.csv"), "representation", "raw")
    diag = [next(r for r in diag if r["setting"] == s) for s in ["global", "local", "lmmd"]]
    write("geometry", [f"{r['setting']} & {f(r,'true_local_covariance_distance'):.5f} & {f(r,'centroid_cosine_distance'):.3f} & {f(r,'target_mean_logit_norm'):.2f} & {f(r,'target_soft_class_purity'):.3f} " + r"\\" for r in diag])
    inventory = rows("experiment_inventory.csv")
    write("inventory", [f"{i+1} & {tex_escape(r['family'])} & {tex_escape(r['run_id'])} & {r['epochs']} & {r['updates']} " + r"\\" for i, r in enumerate(inventory)])
    # Expand table rows directly in alignments; LaTeX input hooks can insert
    # tokens between the last row terminator and booktabs' noalign rules.
    names = ["layers", "weights", "subdomain", "resnet", "geometry", "inventory"]
    macros = ["\\newcommand{\\" + name.title() + "Rows}{%\n" +
              (tables_dir / (name + ".tex")).read_text(encoding="utf-8") + "}\n"
              for name in names]
    (tables_dir / "results.tex").write_text("\n".join(macros), encoding="utf-8")


def validate(manifest):
    inv = rows("experiment_inventory.csv")
    epochs = rows("alex_epochs.csv")
    layers, weights, res = rows("alex_layers.csv"), rows("alex_lambda.csv"), rows("resnet_summary.csv")
    sub = rows("subdomain_summary.csv")
    check_count = 0
    for table, accuracy, correct in [(layers, "final_target_accuracy", "final_target_correct"),
                                    (weights, "final_target_accuracy", "final_target_correct"),
                                    (res, "target_accuracy", "correct"),
                                    (sub, "target_accuracy", "target_correct")]:
        for r in table:
            assert abs(f(r, accuracy)-f(r, correct)/795) < 1e-12
            check_count += 1
    for r in layers + weights:
        run = select(epochs, "run_id", r["run_id"])
        assert len(run) == int(r["epochs"])
        assert abs(f(run[-1], "target_accuracy")-f(r, "final_target_accuracy")) < 1e-12
        best = max(run, key=lambda r: f(r, "target_accuracy"))
        assert int(f(best, "epoch")) == int(r["best_target_epoch"])
    fixed = select(epochs, "run_id", "joint_fixed_100")
    adaptive = select(epochs, "run_id", "joint_adaptive_delayed_100")
    for a, b in zip(fixed[:49], adaptive[:49]):
        for key in ["classification_loss", "CORAL_loss", "target_accuracy"]:
            assert abs(f(a, key)-f(b, key)) < 1e-6
    # Recompute the stopping decision without target labels.
    ema_a = ema_b = None
    streak, first_stop = 0, None
    gap = lambda a, b: abs(a-b)/max(a, b, 1e-12)
    for r in fixed:
        a, b = f(r, "classification_loss"), f(r, "weighted_CORAL_loss")
        ema_a = a if ema_a is None else .5*ema_a+.5*a
        ema_b = b if ema_b is None else .5*ema_b+.5*b
        ok = int(f(r, "epoch")) >= 20 and gap(a, b) <= .25 and gap(ema_a, ema_b) <= .25
        streak = streak+1 if ok else 0
        if streak == 2:
            first_stop = int(f(r, "epoch"))
            break
    assert first_stop == 58
    # Cross-check same-epoch ResNet comparisons against the full baseline curve.
    res_epochs = rows("resnet_epochs.csv")
    baseline = select(res_epochs, "setting", "no_coral")
    for r in res:
        b = next(v for v in baseline if int(f(v, "epoch")) == int(r["stop_epoch"]))
        assert abs(f(b, "target_accuracy")-f(r, "baseline_same_epoch_accuracy")) < 1e-12
        assert abs(100*(f(r, "target_accuracy")-f(b, "target_accuracy"))-f(r, "improvement_pp")) < 1e-10
    def at(e):
        return next(r for r in adaptive if int(f(r, "epoch")) == e)
    diagnostics = {str(e): {k: f(at(e), k) for k in ["CORAL_loss", "classification_loss", "weighted_CORAL_loss", "lambda_used", "gradient_cosine", "gradient_CE_ema", "gradient_CORAL_ema"]} for e in [50, 58, 100]}
    for r in diagnostics.values():
        r["ema_norm_ratio"] = r["gradient_CE_ema"]/r["gradient_CORAL_ema"]
    assert len(inv) == 27
    # Inventory accuracy is a family-specific headline, not always terminal accuracy.
    inventory = {"runs": len(inv), "epochs": sum(int(r["epochs"]) for r in inv),
                 "updates": sum(int(r["updates"]) for r in inv)}
    result = {"training_performed": False, "snapshots_verified": len(manifest["sources"]),
        "accuracy_count_checks": check_count, "inventory": inventory,
        "fixed_adaptive_shared_first_49_epochs": True, "replay_first_stop": first_stop,
        "same_epoch_resnet_comparisons_verified": True, "adaptive_diagnostics": diagnostics,
        "note": "Existing run verification JSONs are archived evidence; this script rechecks aggregates, curves, and formulas stated above, without rerunning training or model inference."}
    dump("validation.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-snapshots", action="store_true", help="Explicitly replace frozen inputs from repository archives")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    manifest = freeze(args.refresh_snapshots)
    validate(manifest)
    tables()
    figures()


if __name__ == "__main__":
    main()
