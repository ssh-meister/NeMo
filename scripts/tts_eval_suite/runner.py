# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Resumable orchestration for the four-model TTS evaluation suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tts_eval_suite.config import load_config

_AUDIO_KEYS = ("audio_filepath", "context_audio_filepath")
_S3_CREDENTIALS = ("S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY")
_REPORT_BENCHMARKS = {
    "de_DE_cmltts",
    "es_ES_cmltts",
    "fr_FR_cmltts",
    "it_IT_cmltts",
    "libritts",
    "riva_en",
    "riva_en_hard_sentences",
    "riva_en_qa",
    "riva_en_qa_longform",
    "riva_en_short_sentences",
}


class PreflightError(RuntimeError):
    """Raised when suite inputs are incomplete or inconsistent."""


def jobs(cfg) -> list[tuple[str, str]]:
    return [(str(model.name), str(group)) for model in cfg.models for group in cfg.groups]


def get_model(cfg, model_name: str):
    for model in cfg.models:
        if model.name == model_name:
            return model
    raise ValueError(f"Unknown model {model_name!r}; choose from: {', '.join(m.name for m in cfg.models)}")


def output_dir(cfg, model_name: str, group_name: str) -> Path:
    return Path(cfg.output.root) / model_name / group_name


def _benchmark_dir(root: Path, benchmark: str) -> Path | None:
    matches = [path for path in root.glob(f"*_{benchmark}") if path.is_dir()]
    exact = root / benchmark
    if exact.is_dir():
        matches.append(exact)
    return sorted(set(matches))[0] if matches else None


def is_complete(cfg, model_name: str, group_name: str) -> bool:
    root = output_dir(cfg, model_name, group_name)
    marker = root / ".complete.json"
    if not marker.is_file():
        return False
    for benchmark in cfg.groups[group_name].benchmarks:
        benchmark_dir = _benchmark_dir(root, benchmark)
        if benchmark_dir is None:
            return False
        if not (benchmark_dir / f"{benchmark}_metrics_0.json").is_file():
            return False
        if not (benchmark_dir / f"{benchmark}_filewise_metrics_0.json").is_file():
            return False
    return True


def build_inference_command(cfg, model_name: str, group_name: str) -> list[str]:
    model = get_model(cfg, model_name)
    group = cfg.groups[group_name]
    command = [
        str(cfg.inference.python),
        str(cfg.inference.script),
        "--model_type",
        str(model.model_type),
        "--nemo_files",
        str(model.nemo_path),
        "--codecmodel_path",
        str(model.codec_path),
        "--datasets_json_path",
        str(group.config_path),
        "--datasets_base_path",
        str(cfg.assets.eval_root),
        "--datasets",
        ",".join(group.benchmarks),
        "--out_dir",
        str(output_dir(cfg, model_name, group_name)),
        "--batch_size",
        str(cfg.inference.batch_size),
        "--temperature",
        str(model.temperature),
        "--topk",
        str(cfg.inference.topk),
        "--cfg_scale",
        str(cfg.inference.cfg_scale),
        "--num_repeats",
        str(cfg.inference.num_repeats),
        "--sv_model",
        "titanet",
        "--sv_model_path",
        str(cfg.assets.titanet_path),
        "--eou_model_name",
        str(cfg.assets.eou_model_path),
    ]
    for enabled, flag in (
        (cfg.inference.use_cfg, "--use_cfg"),
        (cfg.inference.run_evaluation, "--run_evaluation"),
        (cfg.inference.disable_fcd, "--disable_fcd"),
    ):
        if enabled:
            command.append(flag)
    if model.model_type == "easy_magpie":
        command.extend(
            [
                "--phoneme_input_type",
                "predicted",
                "--phoneme_sampling_method",
                "argmax",
                "--phoneme_tokenizer_path",
                str(cfg.assets.tokenizer_path),
            ]
        )
    command.extend(str(arg) for arg in cfg.inference.extra_args)
    command.extend(str(arg) for arg in model.extra_args)
    return command


def _load_manifest(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("manifest must contain JSON objects")
    return records


def _selected(cfg, model_name: str | None, group_name: str | None) -> Iterable[tuple[str, str]]:
    for current_model, current_group in jobs(cfg):
        if model_name is not None and current_model != model_name:
            continue
        if group_name is not None and current_group != group_name:
            continue
        yield current_model, current_group


def preflight(cfg, model_name: str | None = None, group_name: str | None = None, reports: bool = False) -> None:
    """Validate selected models, canonical datasets, references, and report prerequisites."""
    errors: list[str] = []
    eval_root = Path(cfg.assets.eval_root)
    selected = list(_selected(cfg, model_name, group_name))
    selected_models = {name for name, _ in selected}
    selected_groups = {name for _, name in selected}

    required_paths = {
        "EVAL_ROOT": eval_root,
        "IPA tokenizer": Path(cfg.assets.tokenizer_path),
        "TitaNet": Path(cfg.assets.titanet_path),
        "EoU model": Path(cfg.assets.eou_model_path),
        "inference script": Path(cfg.inference.script),
    }
    for model in cfg.models:
        if model.name in selected_models:
            required_paths[f"{model.name} model"] = Path(model.nemo_path)
            required_paths[f"{model.name} codec"] = Path(model.codec_path)
    for label, path in required_paths.items():
        if not path.exists():
            errors.append(f"missing {label}: {path}")

    for group_name in selected_groups:
        group = cfg.groups[group_name]
        config_path = Path(group.config_path)
        if not config_path.is_file():
            errors.append(f"missing {group_name} config: {config_path}")
            continue
        try:
            datasets = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"invalid {group_name} config {config_path}: {error}")
            continue
        if list(datasets) != list(group.benchmarks):
            errors.append(f"{group_name} config keys do not match configured benchmark order")
        unknown = sorted(set(group.benchmarks) - _REPORT_BENCHMARKS)
        if unknown:
            errors.append(f"unsupported {group_name} benchmark(s): {', '.join(unknown)}")
        for benchmark in group.benchmarks:
            meta = datasets.get(benchmark)
            if not isinstance(meta, dict):
                errors.append(f"{group_name}/{benchmark}: missing config object")
                continue
            manifest = Path(meta.get("manifest_path", ""))
            audio_dir = Path(meta.get("audio_dir", ""))
            manifest = manifest if manifest.is_absolute() else eval_root / manifest
            audio_dir = audio_dir if audio_dir.is_absolute() else eval_root / audio_dir
            if not manifest.is_file():
                errors.append(f"{benchmark}: missing manifest {manifest}")
                continue
            if not audio_dir.is_dir():
                errors.append(f"{benchmark}: missing audio directory {audio_dir}")
                continue
            asr = meta.get("asr_model", {})
            asr_name = Path(asr.get("name", "")) if isinstance(asr, dict) else Path()
            if asr_name.suffix == ".nemo":
                asr_path = asr_name if asr_name.is_absolute() else eval_root / asr_name
                if not asr_path.is_file():
                    errors.append(f"{benchmark}: missing ASR model {asr_path}")
            try:
                records = _load_manifest(manifest)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                errors.append(f"{benchmark}: invalid manifest {manifest}: {error}")
                continue
            expected = group.expected_records.get(benchmark)
            if expected is not None and len(records) != expected:
                errors.append(f"{benchmark}: expected {expected} records, found {len(records)}")
            for index, record in enumerate(records):
                for key in _AUDIO_KEYS:
                    value = record.get(key)
                    if not value:
                        errors.append(f"{benchmark}[{index}]: missing {key}")
                        continue
                    audio_path = Path(value)
                    audio_path = audio_path if audio_path.is_absolute() else audio_dir / audio_path
                    if not audio_path.is_file():
                        errors.append(f"{benchmark}[{index}]: missing {key} {audio_path}")

    for current_model, current_group in selected:
        destination = output_dir(cfg, current_model, current_group)
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            errors.append(f"cannot create output {destination}: {error}")
    if reports:
        for variable in _S3_CREDENTIALS:
            if not os.environ.get(variable):
                errors.append(f"missing report credential environment variable {variable}")
        for key in ("endpoint", "bucket", "region"):
            if not str(cfg.reports.storage[key]).strip():
                errors.append(f"missing reports.storage.{key}")
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:100])
        remainder = len(errors) - 100
        if remainder > 0:
            preview += f"\n- ... and {remainder} more error(s)"
        raise PreflightError(f"Preflight failed with {len(errors)} error(s):\n{preview}")


def run_inference(cfg, model_name: str, group_name: str, resume: bool = True) -> None:
    preflight(cfg, model_name, group_name)
    destination = output_dir(cfg, model_name, group_name)
    if resume and is_complete(cfg, model_name, group_name):
        print(f"Already complete: {model_name}/{group_name}")
        return
    log_path = Path(cfg.output.logs_dir) / f"{model_name}__{group_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_inference_command(cfg, model_name, group_name)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nARGV: " + json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    marker = destination / ".complete.json"
    marker.write_text(json.dumps({"model": model_name, "group": group_name}, indent=2) + "\n", encoding="utf-8")
    if not is_complete(cfg, model_name, group_name):
        marker.unlink(missing_ok=True)
        raise RuntimeError(f"Inference exited successfully but expected artifacts are missing in {destination}")


def report_config(cfg, group_name: str) -> dict:
    group = cfg.groups[group_name]
    baseline, *candidates = cfg.models

    def model_entry(model) -> dict:
        return {
            "name": str(model.display_name),
            "path": str(output_dir(cfg, str(model.name), group_name)),
        }

    return {
        "models": {"baseline": model_entry(baseline), "candidates": [model_entry(model) for model in candidates]},
        "evaluation": {"benchmarks": list(group.benchmarks), "results_subdir": ""},
        "report": {
            "audio": bool(cfg.reports.audio),
            "audio_benchmarks": list(group.audio_benchmarks),
            "samples_per_benchmark": int(cfg.reports.samples_per_benchmark),
        },
        "storage": {
            "endpoint": str(cfg.reports.storage.endpoint),
            "bucket": str(cfg.reports.storage.bucket),
            "region": str(cfg.reports.storage.region),
        },
        "task_id": str(cfg.reports.task_id),
    }


def write_report_configs(cfg) -> list[Path]:
    root = Path(cfg.output.report_configs_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for group_name in cfg.groups:
        path = root / f"{group_name}.yaml"
        OmegaConf.save(OmegaConf.create(report_config(cfg, group_name)), path)
        paths.append(path)
    return paths


def run_reports(cfg) -> None:
    preflight(cfg, reports=True)
    incomplete = [f"{model}/{group}" for model, group in jobs(cfg) if not is_complete(cfg, model, group)]
    if incomplete:
        raise RuntimeError("Cannot generate reports; incomplete jobs: " + ", ".join(incomplete))
    for report_path in write_report_configs(cfg):
        subprocess.run(
            [
                str(cfg.inference.python),
                "scripts/tts_comparison_report/generate_report.py",
                "--config",
                str(report_path),
            ],
            check=True,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "inference", "reports", "all"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--group", choices=("english", "multilingual"))
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args, overrides = parser.parse_known_args(argv)
    invalid = [value for value in overrides if value.startswith("-") or "=" not in value]
    if invalid:
        parser.error("unrecognized arguments: " + " ".join(invalid))
    cfg = load_config(args.config, overrides)
    if args.action == "preflight":
        preflight(cfg, args.model, args.group, reports=False)
        print("Preflight passed.")
    elif args.action == "inference":
        if not args.model or not args.group:
            raise SystemExit("inference requires --model and --group")
        run_inference(cfg, args.model, args.group, resume=not args.no_resume)
    elif args.action == "reports":
        run_reports(cfg)
    else:
        preflight(cfg)
        for model_name, group_name in jobs(cfg):
            run_inference(cfg, model_name, group_name, resume=not args.no_resume)
        run_reports(cfg)


if __name__ == "__main__":
    main()
