# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.tts_eval_suite.config import load_config
from scripts.tts_eval_suite.runner import (
    PreflightError,
    build_inference_command,
    is_complete,
    jobs,
    preflight,
    report_config,
    write_report_configs,
)

_ROOT = Path(__file__).parents[3]
_CONFIG = _ROOT / "scripts/tts_eval_suite/configs/four_model_v2607.yaml"


def _index(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_config_overrides_and_matrix(monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_OUTPUT_ROOT", str(tmp_path / "outputs"))
    cfg = load_config(_CONFIG, ["inference.batch_size=7", "reports.audio=false"])

    assert cfg.inference.batch_size == 7
    assert cfg.reports.audio is False
    assert jobs(cfg) == [
        (model, group)
        for model in (
            "magpie_v2607",
            "emtts_2605_pretrained",
            "easy_magpie_ytc_yodas_150k",
            "easy_magpie_all_granary_150k",
        )
        for group in ("english", "multilingual")
    ]


def test_commands_are_model_specific(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_OUTPUT_ROOT", str(tmp_path))
    cfg = load_config(_CONFIG)
    magpie = build_inference_command(cfg, "magpie_v2607", "english")
    easy = build_inference_command(cfg, "emtts_2605_pretrained", "multilingual")

    assert _index(magpie, "--model_type") == "magpie"
    assert "--phoneme_input_type" not in magpie
    assert "--phoneme_tokenizer_path" not in magpie
    assert _index(magpie, "--topk") == "80"
    assert _index(magpie, "--cfg_scale") == "2.5"
    assert "--disable_fcd" in magpie
    assert _index(easy, "--model_type") == "easy_magpie"
    assert _index(easy, "--phoneme_input_type") == "predicted"
    assert _index(easy, "--phoneme_sampling_method") == "argmax"
    assert "--disable_cas_for_context_text" not in easy


def _tiny_cfg(tmp_path: Path):
    cfg = load_config(_CONFIG)
    eval_root = tmp_path / "eval"
    audio = eval_root / "audio"
    models = tmp_path / "models"
    audio.mkdir(parents=True)
    models.mkdir()
    (audio / "target.wav").touch()
    (audio / "context.wav").touch()
    manifest = eval_root / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"audio_filepath": "target.wav", "context_audio_filepath": "context.wav", "text": "hello"}) + "\n",
        encoding="utf-8",
    )
    asr = eval_root / "asr.nemo"
    asr.touch()
    dataset = {
        "manifest_path": "manifest.jsonl",
        "audio_dir": "audio",
        "asr_model": {"name": "asr.nemo", "type": "nemo"},
    }
    english_config = tmp_path / "english.json"
    multilingual_config = tmp_path / "multilingual.json"
    english_config.write_text(json.dumps({"libritts": dataset}), encoding="utf-8")
    multilingual_config.write_text(json.dumps({"de_DE_cmltts": dataset}), encoding="utf-8")
    for name in ("tokenizer.json", "titanet.nemo", "codec.nemo", "inference.py"):
        (models / name).touch()
    (models / "eou").mkdir()
    for index, model in enumerate(cfg.models):
        model.nemo_path = str(models / f"model-{index}.nemo")
        Path(model.nemo_path).touch()
        model.codec_path = str(models / "codec.nemo")
    cfg.assets.eval_root = str(eval_root)
    cfg.assets.tokenizer_path = str(models / "tokenizer.json")
    cfg.assets.titanet_path = str(models / "titanet.nemo")
    cfg.assets.eou_model_path = str(models / "eou")
    cfg.inference.script = str(models / "inference.py")
    cfg.output.root = str(tmp_path / "output")
    cfg.output.logs_dir = str(tmp_path / "output/logs")
    cfg.output.report_configs_dir = str(tmp_path / "output/report_configs")
    cfg.groups.english.config_path = str(english_config)
    cfg.groups.english.benchmarks = ["libritts"]
    cfg.groups.english.audio_benchmarks = ["libritts"]
    cfg.groups.multilingual.config_path = str(multilingual_config)
    cfg.groups.multilingual.benchmarks = ["de_DE_cmltts"]
    cfg.groups.multilingual.audio_benchmarks = ["de_DE_cmltts"]
    cfg.groups.multilingual.expected_records = {"de_DE_cmltts": 1}
    return cfg


def test_preflight_accepts_jsonl_and_reports_missing_reference(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    preflight(cfg, "magpie_v2607", "english")
    (Path(cfg.assets.eval_root) / "audio/context.wav").unlink()

    with pytest.raises(PreflightError, match="context_audio_filepath"):
        preflight(cfg, "magpie_v2607", "english")


def test_completion_requires_marker_and_all_metrics(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    root = Path(cfg.output.root) / "magpie_v2607/english"
    benchmark = root / "configuration_libritts"
    benchmark.mkdir(parents=True)
    (root / ".complete.json").write_text("{}")
    (benchmark / "libritts_metrics_0.json").write_text("{}")
    assert not is_complete(cfg, "magpie_v2607", "english")
    (benchmark / "libritts_filewise_metrics_0.json").write_text("[]")
    assert is_complete(cfg, "magpie_v2607", "english")


def test_report_configs_match_buckets_and_benchmarks(tmp_path):
    cfg = _tiny_cfg(tmp_path)
    generated = write_report_configs(cfg)
    english = report_config(cfg, "english")

    assert len(generated) == 2
    assert english["evaluation"] == {"benchmarks": ["libritts"], "results_subdir": ""}
    assert english["models"]["baseline"]["path"].endswith("magpie_v2607/english")
    assert len(english["models"]["candidates"]) == 3
    loaded = OmegaConf.load(generated[0])
    assert loaded.evaluation.results_subdir == ""
