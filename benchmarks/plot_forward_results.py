"""Plot IGD figures from a recorded forward Pareto benchmark document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_ALL_STRATEGIES = (
    ("uniform", "Uniform", "#777777"),
    ("global_kde", "Global KDE", "#4C78A8"),
    ("gmm_kde_historical", "GMM-KDE (historical repair)", "#B279A2"),
    ("gmm_kde_stabilized", "GMM-KDE (relative floor)", "#59A14F"),
    ("shell_contrastive_kde", "Shell contrast", "#E45756"),
    ("pymoo_nsga2_batch_matched", "NSGA-II (3 batches)", "#72B7B2"),
    ("pymoo_nsga2_typical", "NSGA-II (200 batches)", "#F2CF5B"),
)


def _save_figure(figure: Any, output: Path) -> None:
    svg = output.with_suffix(".svg")
    figure.savefig(svg, bbox_inches="tight")
    lines = svg.read_text(encoding="utf-8").splitlines()
    svg.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )
    figure.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")


def _plot(
    document: dict[str, Any],
    strategies: tuple[tuple[str, str, str], ...],
    problems: tuple[str, ...],
    output: Path,
    *,
    logarithmic: bool,
    title: str,
) -> None:
    aggregates = {
        (item["problem"], item["strategy"]): item for item in document["aggregates"]
    }
    runs: dict[tuple[str, str], list[float]] = {}
    for run in document["runs"]:
        if run["status"] == "passed":
            key = (run["problem"], run["strategy"])
            runs.setdefault(key, []).append(run["inverted_generational_distance"])

    figure, axes = plt.subplots(figsize=(14, 7.5))
    positions = np.arange(len(problems), dtype=np.float64)
    width = 0.78 / len(strategies)
    center = (len(strategies) - 1) / 2.0
    for strategy_index, (strategy, label, color) in enumerate(strategies):
        bar_positions = positions + (strategy_index - center) * width
        medians = [
            np.nan
            if aggregates[problem, strategy]["median_igd"] is None
            else aggregates[problem, strategy]["median_igd"]
            for problem in problems
        ]
        bars = axes.bar(
            bar_positions,
            medians,
            width * 0.92,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=2,
        )
        for position, problem, bar in zip(
            bar_positions,
            problems,
            bars,
            strict=True,
        ):
            values = runs.get((problem, strategy), [])
            if values:
                offsets = np.linspace(-width * 0.22, width * 0.22, len(values))
                axes.scatter(
                    position + offsets,
                    values,
                    s=14,
                    color="black",
                    alpha=0.65,
                    zorder=3,
                )
            aggregate = aggregates[problem, strategy]
            passed = aggregate["passed"]
            total = aggregate["runs"]
            if passed < total:
                if passed:
                    height = bar.get_height() * (1.22 if logarithmic else 1.04)
                    text = f"{passed}/{total}"
                else:
                    height = 2e-4 if logarithmic else 0.02
                    text = "FAIL"
                axes.text(
                    position,
                    height,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    if logarithmic:
        axes.set_yscale("log")
        scale = ", log scale"
    else:
        scale = ""
    axes.set_ylabel(f"Inverted generational distance (lower is better{scale})")
    axes.set_title(title)
    labels = tuple(
        problem.removeprefix("IRREGULAR-").removesuffix("-20D") for problem in problems
    )
    axes.set_xticks(positions, labels)
    axes.grid(axis="y", which="both", alpha=0.25, zorder=0)
    axes.legend(
        ncol=min(5, len(strategies)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
    )
    figure.tight_layout()
    _save_figure(figure, output)
    plt.close(figure)


def _plot_mode_balance(
    document: dict[str, Any],
    problems: tuple[str, ...],
    output: Path,
) -> None:
    relevant = tuple(
        problem
        for problem in problems
        if any(
            aggregate["problem"] == problem
            and aggregate["median_decision_mode_balance"] is not None
            for aggregate in document["aggregates"]
        )
    )
    if not relevant:
        return
    figure, axes = plt.subplots(figsize=(13, 6.5))
    positions = np.arange(len(relevant), dtype=np.float64)
    width = 0.84 / len(_ALL_STRATEGIES)
    center = (len(_ALL_STRATEGIES) - 1) / 2.0
    for strategy_index, (strategy, label, color) in enumerate(_ALL_STRATEGIES):
        bar_positions = positions + (strategy_index - center) * width
        groups = [
            [
                run["decision_mode_balance"]
                for run in document["runs"]
                if run["problem"] == problem
                and run["strategy"] == strategy
                and run["status"] == "passed"
                and run["decision_mode_balance"] is not None
            ]
            for problem in relevant
        ]
        medians = [np.median(values) if values else np.nan for values in groups]
        axes.bar(
            bar_positions,
            medians,
            width * 0.92,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=2,
        )
        for position, values in zip(bar_positions, groups, strict=True):
            if values:
                offsets = np.linspace(-width * 0.22, width * 0.22, len(values))
                axes.scatter(
                    position + offsets,
                    values,
                    s=14,
                    color="black",
                    alpha=0.65,
                    zorder=3,
                )
    labels = tuple(
        problem.removeprefix("IRREGULAR-").removesuffix("-20D") for problem in relevant
    )
    axes.set_ylim(0.0, 1.02)
    axes.set_ylabel("Decision-mode balance (higher is better; zero is collapse)")
    axes.set_title("Balance of equivalent disconnected Pareto pullback modes")
    axes.set_xticks(positions, labels)
    axes.grid(axis="y", alpha=0.25, zorder=0)
    axes.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
    )
    figure.tight_layout()
    target = output.with_name(output.name + "_mode_balance")
    _save_figure(figure, target)
    plt.close(figure)


def _plot_valid_fraction(
    document: dict[str, Any],
    problems: tuple[str, ...],
    output: Path,
) -> None:
    relevant = tuple(
        problem
        for problem in problems
        if any(
            run["problem"] == problem and run["valid_evaluation_fraction"] < 1.0
            for run in document["runs"]
        )
    )
    if not relevant:
        return
    figure, axes = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(relevant), dtype=np.float64)
    width = 0.78 / len(_ALL_STRATEGIES)
    center = (len(_ALL_STRATEGIES) - 1) / 2.0
    for strategy_index, (strategy, label, color) in enumerate(_ALL_STRATEGIES):
        bar_positions = positions + (strategy_index - center) * width
        groups = [
            [
                run["valid_evaluation_fraction"]
                for run in document["runs"]
                if run["problem"] == problem
                and run["strategy"] == strategy
                and run["status"] == "passed"
            ]
            for problem in relevant
        ]
        medians = [np.median(values) if values else np.nan for values in groups]
        axes.bar(
            bar_positions,
            medians,
            width * 0.92,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.35,
            zorder=2,
        )
        for position, values in zip(bar_positions, groups, strict=True):
            if values:
                offsets = np.linspace(-width * 0.22, width * 0.22, len(values))
                axes.scatter(
                    position + offsets,
                    values,
                    s=14,
                    color="black",
                    alpha=0.65,
                    zorder=3,
                )
    labels = tuple(
        problem.removeprefix("IRREGULAR-").removesuffix("-20D") for problem in relevant
    )
    axes.set_ylim(0.0, 1.02)
    axes.set_ylabel("Valid forward-evaluation fraction (higher is better)")
    axes.set_title("Forward-evaluation yield in problems with invalid regions")
    axes.set_xticks(positions, labels)
    axes.grid(axis="y", alpha=0.25, zorder=0)
    axes.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=False,
    )
    figure.tight_layout()
    target = output.with_name(output.name + "_valid_fraction")
    _save_figure(figure, target)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_prefix", type=Path)
    arguments = parser.parse_args()
    document = json.loads(arguments.input.read_text(encoding="utf-8"))
    arguments.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    problems = tuple(document["problems"])
    budget = sum(document["stage_counts"])
    seed_count = max(run["seed"] for run in document["runs"]) + 1

    _plot(
        document,
        _ALL_STRATEGIES,
        problems,
        arguments.output_prefix,
        logarithmic=True,
        title=(
            f"{budget:,} forward evaluations per successful run; "
            f"dots are {seed_count} deterministic seeds"
        ),
    )
    _plot(
        document,
        _ALL_STRATEGIES[:-1],
        problems,
        arguments.output_prefix.with_name(
            arguments.output_prefix.name + "_three_batch"
        ),
        logarithmic=False,
        title=f"{budget:,} forward evaluations with three adaptation batches",
    )
    _plot_mode_balance(document, problems, arguments.output_prefix)
    _plot_valid_fraction(document, problems, arguments.output_prefix)


if __name__ == "__main__":
    main()
