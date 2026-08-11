# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import PathPatch
from scripts.tts_comparison_report.reporting.metrics import DistributionMetricSpec, DistributionMetricsRegistry
from scripts.tts_comparison_report.reporting.models import BucketData, StatTestResult


@dataclass
class BoxPlotsConfig:
    """Styling and layout configuration for generated benchmark box plots."""

    font_family: str = "sans-serif"
    font_list: list[str] = field(default_factory=lambda: ["Arial", "Helvetica", "DejaVu Sans"])

    linewidth: float = 0.4
    default_model_color: str = "#36454F"
    winner_model_color: str = "#7393B3"
    box_alpha: float = 0.35
    grid_alpha: float = 0.4
    fontsize: int = 6
    fontsize_title: int = 8

    widths: float = 0.6
    mean_marker: str = "o"
    mean_marker_color: str = "#CD5C5C"
    mean_marker_size: float = 4.0
    median_color: str = "black"
    whisker_color: str = "#666666"
    cap_color: str = "#666666"
    outlier_color: str = "#708090"
    outlier_marker: str = "o"
    outlier_markersize: float = 3.0
    outlier_alpha: float = 0.5


def _style_boxplot(
    bp: dict[str, PathPatch],
    metric: DistributionMetricSpec,
    samples: list[np.ndarray],
    cfg: BoxPlotsConfig,
) -> None:
    means = [values.mean() for values in samples]
    best = min(means) if metric.lower_is_better else max(means)
    for i, patch in enumerate(bp["boxes"]):
        color = cfg.winner_model_color if means[i] == best else cfg.default_model_color

        patch.set_facecolor(color)
        patch.set_alpha(cfg.box_alpha)
        patch.set_edgecolor(color)
        patch.set_linewidth(cfg.linewidth)


def _add_mean_ci_labels(
    ax: Axes,
    samples: list[np.ndarray],
    metric: DistributionMetricSpec,
    cfg: BoxPlotsConfig,
) -> None:
    for x, values in enumerate(samples, start=1):
        mean, median = values.mean(), np.median(values)
        sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
        ci95 = 1.96 * sem
        label = f"{mean:.3f} ± {ci95:.3f}"

        if metric.plot_range is not None:
            range_ = metric.plot_range[1] - metric.plot_range[0]
        else:
            range_ = values.max() - values.min()

        x_offset = 0.02
        y_offset = 0.03 * range_

        if median > mean and mean - y_offset > 0:
            y_offset = -y_offset

        ax.text(x + x_offset, mean + y_offset, label, ha="left", va="center", fontsize=cfg.fontsize)


def _configure_boxplot_axis(
    ax: Axes,
    metric: DistributionMetricSpec,
    model_names: list[str],
    cfg: BoxPlotsConfig,
) -> None:
    ax.set_title(metric.report_name, fontsize=cfg.fontsize_title)
    ax.set_xticks(list(range(1, len(model_names) + 1)))
    ax.set_xticklabels(model_names)
    ax.tick_params(axis="x", labelsize=cfg.fontsize)
    ax.tick_params(axis="y", labelsize=cfg.fontsize)
    ax.grid(True, axis="y", linestyle="dotted", alpha=cfg.grid_alpha)

    for spine in ax.spines.values():
        spine.set_linewidth(cfg.linewidth)

    ax.tick_params(axis="both", width=cfg.linewidth)

    if metric.plot_range is not None:
        ax.set_ylim(metric.plot_range[0], metric.plot_range[1])


def prepare_boxplots(
    bucket_baseline: BucketData,
    bucket_candidates: list[BucketData] | BucketData,
    stat_test_results: list[StatTestResult] | dict[str, list[StatTestResult]],
    cfg: BoxPlotsConfig,
    benchmark_name: Optional[str] = None,
) -> BytesIO:
    """Create an in-memory box plot figure for summary or benchmark-level metrics.

    Args:
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.
        stat_test_results: Statistical test results used to highlight the winning model.
        cfg: Plot styling and layout configuration.
        benchmark_name: Benchmark name. If omitted, metric samples are aggregated
            across all benchmarks.

    Returns:
        PNG image stored in an in-memory bytes buffer.
    """
    if isinstance(bucket_candidates, BucketData):
        bucket_candidates = [bucket_candidates]
    buckets = [bucket_baseline, *bucket_candidates]
    model_names = [bucket.name for bucket in buckets]
    num_rows = sum(m.add_to_box_plot for m in DistributionMetricsRegistry)
    fig_height = max(2.0 * num_rows, 4.5)

    with plt.rc_context({"font.family": cfg.font_family, "font.sans-serif": cfg.font_list}):
        fig_width = max(6, 1.5 * len(buckets) + 3)
        fig, axs = plt.subplots(num_rows, 1, figsize=(fig_width, fig_height), squeeze=False)
        axs = axs.flatten()
        plot_idx = 0

        for metric in DistributionMetricsRegistry:
            if not metric.add_to_box_plot:
                continue

            samples = [
                np.asarray(bucket.get_metric_samples(metric.key, benchmark_name), dtype=float) for bucket in buckets
            ]

            ax = axs[plot_idx]
            plot_idx += 1

            bp = ax.boxplot(
                samples,
                positions=list(range(1, len(samples) + 1)),
                widths=cfg.widths,
                patch_artist=True,
                showmeans=True,
                meanline=False,
                meanprops={
                    "marker": cfg.mean_marker,
                    "markerfacecolor": cfg.mean_marker_color,
                    "markeredgecolor": cfg.mean_marker_color,
                    "markersize": cfg.mean_marker_size,
                },
                medianprops={
                    "color": cfg.median_color,
                    "linewidth": cfg.linewidth,
                },
                whiskerprops={
                    "color": cfg.whisker_color,
                    "linewidth": cfg.linewidth,
                },
                capprops={
                    "color": cfg.cap_color,
                    "linewidth": cfg.linewidth,
                },
                boxprops={
                    "linewidth": cfg.linewidth,
                },
                flierprops={
                    "marker": cfg.outlier_marker,
                    "markerfacecolor": cfg.outlier_color,
                    "markeredgecolor": cfg.outlier_color,
                    "markersize": cfg.outlier_markersize,
                    "alpha": cfg.outlier_alpha,
                },
            )

            _style_boxplot(bp, metric, samples, cfg)
            _add_mean_ci_labels(ax, samples, metric, cfg)
            _configure_boxplot_axis(ax, metric, model_names, cfg)

        fig.tight_layout(rect=[0, 0, 1, 0.985])

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)

    return buffer
