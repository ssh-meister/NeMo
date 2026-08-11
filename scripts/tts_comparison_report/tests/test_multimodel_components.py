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
from pathlib import Path

import pytest

from scripts.tts_comparison_report.reporting.components import audio_report, metrics_table
from scripts.tts_comparison_report.reporting.components.stat_tests import _run_single_stat_test, run_stat_tests
from scripts.tts_comparison_report.reporting.constants import TEMPLATES_DIR
from scripts.tts_comparison_report.reporting.helpers import generate_s3_prefix
from scripts.tts_comparison_report.reporting.metrics import DistributionMetricSpec, MetricSpec
from scripts.tts_comparison_report.reporting.models import (
    BenchmarkData,
    BenchmarkSampleMeta,
    BucketData,
    BucketStructure,
    ExpirationInfo,
    TaskInfo,
    UploadedAudioPairInfo,
    Winner,
)
from scripts.tts_comparison_report.reporting.renderer import Renderer, TemplateName


def _bucket(name, value):
    benchmark = BenchmarkData(
        name="libritts_test_clean",
        metrics={"metric": value},
        filewise_metrics=[{"distribution": value}, {"distribution": value + 0.01}],
    )
    return BucketData(name=name, path=Path(f"/{name}"), benchmarks={benchmark.name: benchmark})


def test_n_way_metric_rows_highlight_global_best(monkeypatch):
    monkeypatch.setattr(
        metrics_table,
        "MetricsRegistry",
        [MetricSpec("metric", "Metric", lower_is_better=True, round_digits=2)],
    )
    baseline, candidate_a, candidate_b = _bucket("base", 0.3), _bucket("a", 0.2), _bucket("b", 0.1)

    rows = metrics_table.prepare_benchmark_metrics_table_rows(
        "libritts_test_clean", baseline, [candidate_a, candidate_b]
    )

    assert len(rows[0]) == 4
    assert "<strong>0.1</strong>" == rows[0][-1]
    assert "<strong>" not in rows[0][1]


def test_stats_pair_each_candidate_with_baseline(monkeypatch):
    import scripts.tts_comparison_report.reporting.components.stat_tests as stat_tests

    monkeypatch.setattr(
        stat_tests,
        "DistributionMetricsRegistry",
        [DistributionMetricSpec("distribution", "Distribution", lower_is_better=True)],
    )
    baseline, candidate_a, candidate_b = _bucket("base", 0.3), _bucket("a", 0.2), _bucket("b", 0.1)

    results = {candidate.name: run_stat_tests(baseline, candidate) for candidate in [candidate_a, candidate_b]}

    assert set(results) == {"a", "b"}
    assert all(len(candidate_results) == 1 for candidate_results in results.values())


def test_mann_whitney_detects_better_candidate():
    winner, _, _ = _run_single_stat_test(
        baseline=[0.8] * 20,
        candidate=[0.1] * 20,
        lower_is_better=True,
    )
    assert winner == Winner.candidate


class _AudioBucket:
    def __init__(self, name, suffix=""):
        self.name = name
        self.paths = {"predicted_audio_0": Path(f"/{name}/sample{suffix}.wav")}
        self.meta = {
            "predicted_audio_0": BenchmarkSampleMeta(
                name="predicted_audio_0",
                gt_text="hello",
                context_path=Path(f"/{name}/context.wav"),
                sample_id="same-id",
            )
        }

    def get_benchmark_audio_paths(self, benchmark_name):
        return self.paths

    def get_benchmark_sample_meta(self, benchmark_name, bucket_structure):
        return self.meta


def test_audio_grid_contains_all_models():
    pairs = audio_report.prepare_audio_pairs(
        _AudioBucket("base"),
        [_AudioBucket("a"), _AudioBucket("b")],
        BucketStructure(),
        ["libritts_test_clean"],
        1,
    )
    assert list(pairs["libritts_test_clean"][0].model_paths) == ["base", "a", "b"]


def test_audio_rejects_mismatched_sample_sets():
    candidate = _AudioBucket("candidate")
    candidate.paths = {"predicted_audio_1": Path("/candidate/other.wav")}
    with pytest.raises(ValueError, match="Audio sample sets differ"):
        audio_report.prepare_audio_pairs(
            _AudioBucket("base"),
            [candidate],
            BucketStructure(),
            ["libritts_test_clean"],
            1,
        )


def test_dynamic_audio_template_renders_every_model():
    renderer = Renderer(TEMPLATES_DIR)
    pair = UploadedAudioPairInfo(
        context_url="context.wav",
        model_urls={"base": "base.wav", "a": "a.wav", "b": "b.wav"},
        text="hello",
    )
    html = renderer.render(
        TemplateName.audio_report_pair,
        context_url=pair.context_url,
        model_urls=pair.model_urls,
        text=pair.text,
    )
    assert html.count("<audio") == 4
    assert "b.wav" in html


def test_s3_prefix_is_safe_unique_and_includes_all_models():
    task = TaskInfo("NEMOTTS-1", "NEMOTTS-1", "https://jira/NEMOTTS-1")
    expiration = ExpirationInfo(1, "2030-01-01T00-00-00Z", "2030")
    first = generate_s3_prefix(
        Path("/runs/Base Model"),
        [Path("/runs/Candidate A"), Path("/other/Candidate B")],
        task,
        expiration,
    )
    second = generate_s3_prefix(
        Path("/different/Base Model"),
        [Path("/runs/Candidate A"), Path("/other/Candidate B")],
        task,
        expiration,
    )
    assert " " not in first
    assert "Base-Model_vs_Candidate-A_and_Candidate-B" in first
    assert first != second
