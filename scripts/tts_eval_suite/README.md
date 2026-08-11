# Four-model TTS evaluation suite

This suite evaluates:

1. MagpieTTS v2607 (baseline);
2. EMTTS 2605 pretrained;
3. EasyMagpie YTC+YODAS 150k;
4. EasyMagpie All-Granary 150k.

It runs the repository's `examples/tts/magpietts_inference.py` procedure and produces separate English and
multilingual comparison reports. Magpie uses its native 21.5-fps causal codec; all EasyMagpie models use the
25-fps spectral codec. Shared compatible controls are top-k 80, CFG 2.5, TitaNet, one repeat, evaluation
enabled, and FCD disabled. EasyMagpie uses predicted phonemes with temperature 0.7.

The canonical data is read directly from:

```text
/lustre/fsw/nemotron_speech_tts/data/evaluation_datasets
```

No dataset or audio is copied. VCTK is deliberately excluded.

## Dataset groups

- English: `libritts`, `riva_en`, `riva_en_hard_sentences`, `riva_en_short_sentences`, `riva_en_qa`,
  `riva_en_qa_longform`, matching the current `magpie_pretraining_data` English config semantics.
- Multilingual: `de_DE_cmltts`, `es_ES_cmltts`, `fr_FR_cmltts`, `it_IT_cmltts`, exactly 100 records each.

Suite-owned JSON files contain relative paths under `EVAL_ROOT`. Preflight validates the model, native codec,
IPA tokenizer, TitaNet, EoU model, manifests, audio directories, every target/context audio reference, local
ASR `.nemo` reference, benchmark membership, and selected output paths. Both JSON arrays and JSONL manifests
are accepted.

## Configuration and local commands

Run from the NeMo repository root:

```bash
export NEMO_ROOT=$PWD
export WORKSPACE_ROOT=/lustre/fsw/nemotron_speech_asr/users/ameister/easymagpie_workspace
export EVAL_ROOT=/lustre/fsw/nemotron_speech_tts/data/evaluation_datasets
export EVAL_OUTPUT_ROOT=$WORKSPACE_ROOT/evaluation/tts_four_model_v2607
CONFIG=scripts/tts_eval_suite/configs/four_model_v2607.yaml

python scripts/tts_eval_suite/runner.py preflight --config "$CONFIG"
```

Inference does not require S3 credentials. Run one resumable matrix cell with:

```bash
python scripts/tts_eval_suite/runner.py inference --config "$CONFIG" \
  --model magpie_v2607 --group english
```

Run all eight inference cells locally, followed by reports:

```bash
export S3_ENDPOINT=https://your-s3-endpoint
export S3_BUCKET=your-bucket
export S3_ACCESS_KEY_ID=...
export S3_SECRET_ACCESS_KEY=...
python scripts/tts_eval_suite/runner.py all --config "$CONFIG"
```

Trailing Hydra dotlist overrides are supported:

```bash
python scripts/tts_eval_suite/runner.py preflight --config "$CONFIG" \
  inference.batch_size=16 output.root=/alternate/evaluation/root
```

Every inference job appends its exact argv and output to
`$EVAL_OUTPUT_ROOT/logs/<model>__<group>.log`. A job is skipped only when its completion marker and every
benchmark's aggregate and filewise metrics exist. Use `--no-resume` to force a rerun.

## EOS Slurm

The wrappers use account `nemotron_speech_tts`, the workspace NeMo 25.11 container, one exclusive EOS node
per array task, and explicit read-only mounts for the repository, models, experiments, and `EVAL_ROOT`.
They intentionally avoid unsupported EOS GPU/GRES directives; each single inference process uses the first
visible GPU on its allocated node.
The inference wrapper uses EOS's four-hour job limit. Re-submit an unfinished matrix cell if it reaches that
limit; completed cells are skipped automatically.

Preflight first; it also creates the output mount source:

```bash
python scripts/tts_eval_suite/runner.py preflight --config "$CONFIG"
```

Submit the eight GPU jobs and a dependent CPU report job:

```bash
ARRAY_JOB=$(sbatch --parsable scripts/tts_eval_suite/slurm/inference_array.sh)
sbatch --dependency=afterok:"$ARRAY_JOB" scripts/tts_eval_suite/slurm/reports.sh
```

Export `S3_ENDPOINT`, `S3_BUCKET`, `S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY` before the report submission.
The report phase writes generated report YAMLs below `$EVAL_OUTPUT_ROOT/report_configs`; each model path points
to its matching group bucket and uses `results_subdir: ""`.
