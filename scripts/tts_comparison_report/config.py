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
from pathlib import Path
from typing import Optional

from omegaconf import MISSING, DictConfig, OmegaConf

from scripts.tts_comparison_report.reporting.constants import DUMMY_TASK_ID, SUPPORTED_BENCHMARK_NAMES


@dataclass
class ModelConfig:
    name: str = MISSING
    path: str = MISSING


@dataclass
class ModelsConfig:
    baseline: ModelConfig = field(default_factory=ModelConfig)
    candidates: list[ModelConfig] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    benchmarks: list[str] = field(default_factory=lambda: list(SUPPORTED_BENCHMARK_NAMES))
    results_subdir: str = "results"


@dataclass
class ReportConfig:
    audio: bool = False
    audio_benchmarks: list[str] = field(
        default_factory=lambda: ["libritts_test_clean", "riva_hard_digits", "riva_hard_letters"]
    )
    samples_per_benchmark: int = 30


@dataclass
class StorageConfig:
    endpoint: str = MISSING
    bucket: str = MISSING
    region: str = MISSING


@dataclass
class RemoteConfig:
    hostname: Optional[str] = None
    username: Optional[str] = None


@dataclass
class TTSComparisonConfig:
    models: ModelsConfig = field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    task_id: str = DUMMY_TASK_ID
    remote: Optional[RemoteConfig] = None


def validate_config(cfg: DictConfig) -> None:
    """Validate a resolved report configuration."""
    candidates = list(cfg.models.candidates)
    if not candidates:
        raise ValueError("At least one candidate model must be configured.")

    models = [cfg.models.baseline, *candidates]
    names = [str(model.name).strip() for model in models]
    paths = [str(model.path).strip() for model in models]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Model names must be non-empty and unique.")
    if any(not path for path in paths) or len(paths) != len(set(paths)):
        raise ValueError("Model paths must be non-empty and unique.")

    benchmarks = list(cfg.evaluation.benchmarks)
    if not benchmarks:
        raise ValueError("At least one evaluation benchmark must be configured.")
    unknown = sorted(set(benchmarks) - set(SUPPORTED_BENCHMARK_NAMES))
    if unknown:
        raise ValueError(f"Unknown benchmark name(s): {', '.join(unknown)}.")
    if len(benchmarks) != len(set(benchmarks)):
        raise ValueError("Evaluation benchmark names must be unique.")

    audio_benchmarks = list(cfg.report.audio_benchmarks)
    if cfg.report.audio:
        if not audio_benchmarks:
            raise ValueError("At least one audio benchmark must be configured when audio is enabled.")
        missing = sorted(set(audio_benchmarks) - set(benchmarks))
        if missing:
            raise ValueError(
                "Audio benchmark(s) are not included in evaluation benchmarks: " + ", ".join(missing) + "."
            )
        if cfg.report.samples_per_benchmark <= 0:
            raise ValueError("report.samples_per_benchmark must be greater than 0.")

    remote = cfg.remote
    if remote is not None and bool(remote.hostname) != bool(remote.username):
        raise ValueError("remote.hostname and remote.username must be provided together.")


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> DictConfig:
    """Load, merge, resolve, and validate a structured OmegaConf configuration."""
    schema = OmegaConf.structured(TTSComparisonConfig)
    loaded = OmegaConf.load(path)
    dotlist = OmegaConf.from_dotlist(overrides or [])
    cfg = OmegaConf.merge(schema, loaded, dotlist)
    OmegaConf.resolve(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    validate_config(cfg)
    return cfg
