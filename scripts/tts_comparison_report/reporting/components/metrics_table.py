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
import html

import numpy as np
from scripts.tts_comparison_report.reporting.metrics import MetricSpec, MetricsRegistry
from scripts.tts_comparison_report.reporting.models import BucketData


def _format_metric_values(values: list[float], metric: MetricSpec) -> list[str]:
    scaled = [metric.multiplier * value for value in values]
    best = None
    if metric.lower_is_better is not None:
        best = min(scaled) if metric.lower_is_better else max(scaled)
    output = []
    for value in scaled:
        value_str = html.escape(f"{round(value, metric.round_digits)}{metric.units}")
        output.append(f"<strong>{value_str}</strong>" if best is not None and value == best else value_str)
    return output


def prepare_benchmark_metrics_table_rows(
    benchmark_name: str,
    bucket_baseline: BucketData,
    bucket_candidates: list[BucketData] | BucketData,
) -> list[list[str]]:
    """Prepare formatted metric rows for one benchmark comparison table.

    Args:
        benchmark_name: Name of the benchmark to render.
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.

    Returns:
        Table rows containing metric names and formatted baseline/candidate values.

    Raises:
        ValueError: If a required metric is missing for the benchmark.
    """
    if isinstance(bucket_candidates, BucketData):
        bucket_candidates = [bucket_candidates]
    buckets = [bucket_baseline, *bucket_candidates]
    rows = []

    for metric in MetricsRegistry:
        values = [bucket.get_metric_avg_value(metric.key, benchmark_name) for bucket in buckets]
        if any(value is None for value in values):
            if metric.optional:
                continue
            raise ValueError(f"Unknown metric '{metric.key}' for benchmark '{benchmark_name}'.")

        rows.append([html.escape(metric.report_name), *_format_metric_values(values, metric)])

    return rows


def prepare_summary_metrics_table_rows(
    bucket_baseline: BucketData,
    bucket_candidates: list[BucketData] | BucketData,
) -> list[list[str]]:
    """Prepare formatted metric rows for the summary comparison table.

    Args:
        bucket_baseline: Baseline bucket data.
        bucket_candidate: Candidate bucket data.

    Returns:
        Table rows containing metric names and formatted macro-averaged
        baseline/candidate values.

    Raises:
        ValueError: If a required metric is missing for any benchmark included
            in the summary.
    """
    if isinstance(bucket_candidates, BucketData):
        bucket_candidates = [bucket_candidates]
    buckets = [bucket_baseline, *bucket_candidates]
    rows = []

    for metric in MetricsRegistry:
        if not metric.include_in_summary:
            continue

        model_values: list[list[float]] = [[] for _ in buckets]
        skip = False

        for benchmark_name in bucket_baseline.benchmarks:
            values = [bucket.get_metric_avg_value(metric.key, benchmark_name) for bucket in buckets]
            if any(value is None for value in values):
                if metric.optional:
                    skip = True
                    break
                raise ValueError(f"Unknown metric '{metric.key}' for benchmark '{benchmark_name}'.")
            for model_metric_values, value in zip(model_values, values):
                model_metric_values.append(value)

        if skip:
            continue

        averages = [float(np.mean(values)) for values in model_values]
        rows.append([html.escape(metric.report_name), *_format_metric_values(averages, metric)])

    return rows
