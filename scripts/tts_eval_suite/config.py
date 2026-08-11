# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Structured configuration for the four-model TTS evaluation suite."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from omegaconf import MISSING, DictConfig, OmegaConf


@dataclass
class ModelConfig:
    name: str = MISSING
    display_name: str = MISSING
    model_type: str = MISSING
    nemo_path: str = MISSING
    codec_path: str = MISSING
    temperature: float = 0.7
    extra_args: list[str] = field(default_factory=list)


@dataclass
class GroupConfig:
    config_path: str = MISSING
    benchmarks: list[str] = field(default_factory=list)
    expected_records: dict[str, int] = field(default_factory=dict)
    audio_benchmarks: list[str] = field(default_factory=list)


@dataclass
class AssetsConfig:
    eval_root: str = MISSING
    tokenizer_path: str = MISSING
    titanet_path: str = MISSING
    eou_model_path: str = MISSING


@dataclass
class InferenceConfig:
    python: str = "python"
    script: str = "examples/tts/magpietts_inference.py"
    batch_size: int = 32
    topk: int = 80
    cfg_scale: float = 2.5
    num_repeats: int = 1
    use_cfg: bool = True
    run_evaluation: bool = True
    disable_fcd: bool = True
    extra_args: list[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    root: str = MISSING
    logs_dir: str = "${output.root}/logs"
    report_configs_dir: str = "${output.root}/report_configs"


@dataclass
class StorageConfig:
    endpoint: str = '${oc.env:S3_ENDPOINT,""}'
    bucket: str = '${oc.env:S3_BUCKET,""}'
    region: str = "${oc.env:S3_REGION,us-west-2}"


@dataclass
class ReportsConfig:
    enabled: bool = True
    audio: bool = True
    samples_per_benchmark: int = 20
    task_id: str = "NEMOTTS-0000"
    storage: StorageConfig = field(default_factory=StorageConfig)


@dataclass
class SuiteConfig:
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    models: list[ModelConfig] = field(default_factory=list)
    groups: dict[str, GroupConfig] = field(default_factory=dict)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    reports: ReportsConfig = field(default_factory=ReportsConfig)


def validate_config(cfg: DictConfig) -> None:
    if len(cfg.models) != 4:
        raise ValueError(f"Exactly four models are required, got {len(cfg.models)}.")
    names = [str(model.name) for model in cfg.models]
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique.")
    invalid_types = sorted({str(model.model_type) for model in cfg.models} - {"magpie", "easy_magpie"})
    if invalid_types:
        raise ValueError(f"Unsupported model type(s): {', '.join(invalid_types)}.")
    if set(cfg.groups) != {"english", "multilingual"}:
        raise ValueError("Groups must be exactly: english, multilingual.")
    for group_name, group in cfg.groups.items():
        if not group.benchmarks or len(group.benchmarks) != len(set(group.benchmarks)):
            raise ValueError(f"Group {group_name} must contain unique benchmarks.")
        if set(group.audio_benchmarks) - set(group.benchmarks):
            raise ValueError(f"Group {group_name} has audio benchmarks outside its benchmark list.")
    if cfg.inference.num_repeats != 1:
        raise ValueError("This report suite requires exactly one repeat.")


def load_config(path: str | Path, overrides: Optional[list[str]] = None) -> DictConfig:
    """Load YAML, apply Hydra dotlist overrides, resolve interpolation, and validate."""
    schema = OmegaConf.structured(SuiteConfig)
    cfg = OmegaConf.merge(schema, OmegaConf.load(path), OmegaConf.from_dotlist(overrides or []))
    OmegaConf.resolve(cfg)
    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    validate_config(cfg)
    return cfg


def to_plain(value: Any) -> Any:
    """Convert an OmegaConf node to ordinary Python containers."""
    return OmegaConf.to_container(value, resolve=True)
