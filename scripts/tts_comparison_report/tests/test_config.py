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
import textwrap

import pytest

from scripts.tts_comparison_report.config import load_config
from scripts.tts_comparison_report.generate_report import _create_argparser, main


def _write_config(tmp_path, candidates=None):
    path = tmp_path / "report.yaml"
    candidate_yaml = candidates or '- {name: candidate, path: "${oc.env:REPORT_ROOT}/candidate"}'
    path.write_text(
        textwrap.dedent(
            """
            models:
              baseline: {name: baseline, path: "${oc.env:REPORT_ROOT}/baseline"}
              candidates:
            """
        )
        + textwrap.indent(candidate_yaml, "    ")
        + textwrap.dedent(
            """
            evaluation:
              benchmarks: [libritts_test_clean]
              results_subdir: results
            report:
              audio: false
              audio_benchmarks: [libritts_test_clean]
              samples_per_benchmark: 30
            storage:
              endpoint: https://s3.example
              bucket: reports
              region: us-west-2
            task_id: NEMOTTS-1
            """
        ),
        encoding="utf-8",
    )
    return path


def test_load_config_resolves_env_and_dotlist(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORT_ROOT", "/evaluation")
    cfg = load_config(
        _write_config(tmp_path),
        ["report.audio=true", "report.samples_per_benchmark=7"],
    )

    assert cfg.models.baseline.path == "/evaluation/baseline"
    assert cfg.report.audio is True
    assert cfg.report.samples_per_benchmark == 7


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        ("[]", "At least one candidate"),
        (
            "- {name: baseline, path: /other}",
            "Model names must be non-empty and unique",
        ),
        (
            '- {name: candidate, path: "${oc.env:REPORT_ROOT}/baseline"}',
            "Model paths must be non-empty and unique",
        ),
    ],
)
def test_config_rejects_invalid_models(tmp_path, monkeypatch, candidates, message):
    monkeypatch.setenv("REPORT_ROOT", "/evaluation")
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path, candidates))


def test_legacy_parser_preserves_single_candidate_flags():
    args = _create_argparser().parse_args(
        [
            "--baseline_name",
            "base",
            "--baseline_path",
            "/base",
            "--candidate_name",
            "candidate",
            "--candidate_path",
            "/candidate",
            "--s3_endpoint",
            "https://s3.example",
            "--s3_bucket",
            "reports",
            "--s3_region",
            "us-west-2",
        ]
    )

    assert args.config is None
    assert args.candidate_name == "candidate"
    assert args.audio_report is False


def test_config_rejects_legacy_flag_mixing(tmp_path):
    with pytest.raises(SystemExit):
        main(["--config", str(tmp_path / "report.yaml"), "--baseline_name", "base"])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("report.samples_per_benchmark=0", "greater than 0"),
        ("report.audio_benchmarks=[riva_hard_digits]", "not included"),
        ("remote.hostname=host", "must be provided together"),
    ],
)
def test_config_validation_rules(tmp_path, monkeypatch, override, message):
    monkeypatch.setenv("REPORT_ROOT", "/evaluation")
    with pytest.raises(ValueError, match=message):
        load_config(_write_config(tmp_path), ["report.audio=true", override])
