#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#SBATCH --account=nemotron_speech_tts
#SBATCH --partition=batch
#SBATCH --job-name=nemotron_speech_tts-easymagpie_eval.inference
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --export=ALL

set -euo pipefail

NEMO_ROOT="${NEMO_ROOT:-/lustre/fsw/nemotron_speech_asr/users/ameister/easymagpie_workspace/code/ZeroShotEMTTS/NeMo}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/lustre/fsw/nemotron_speech_asr/users/ameister/easymagpie_workspace}"
EVAL_ROOT="${EVAL_ROOT:-/lustre/fsw/nemotron_speech_tts/data/evaluation_datasets}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${WORKSPACE_ROOT}/evaluation/tts_four_model_v2607}"
CONFIG="${CONFIG:-${NEMO_ROOT}/scripts/tts_eval_suite/configs/four_model_v2607.yaml}"
CONTAINER="${WORKSPACE_ROOT}/containers/nemo_25.11-260122.sqsh"
EXPERIMENT_ROOT="/lustre/fsw/nemotron_speech_asr/users/ameister/experiments/EasyMagpie"

models=(magpie_v2607 emtts_2605_pretrained easy_magpie_ytc_yodas_150k easy_magpie_all_granary_150k)
groups=(english multilingual)
model="${models[$((SLURM_ARRAY_TASK_ID / 2))]}"
group="${groups[$((SLURM_ARRAY_TASK_ID % 2))]}"

mounts="${NEMO_ROOT}:${NEMO_ROOT}:ro"
mounts+=",${WORKSPACE_ROOT}/models:${WORKSPACE_ROOT}/models:ro"
mounts+=",${WORKSPACE_ROOT}/evaluation:${WORKSPACE_ROOT}/evaluation"
mounts+=",${EXPERIMENT_ROOT}:${EXPERIMENT_ROOT}:ro"
mounts+=",${EVAL_ROOT}:${EVAL_ROOT}:ro"
mounts+=",/home/ameister:/home/ameister"

srun --no-container-mount-home \
  --container-image="${CONTAINER}" \
  --container-mounts="${mounts}" \
  env PYTHONPATH="${NEMO_ROOT}" NEMO_ROOT="${NEMO_ROOT}" WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
  EVAL_ROOT="${EVAL_ROOT}" EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
  python "${NEMO_ROOT}/scripts/tts_eval_suite/runner.py" inference \
  --config "${CONFIG}" --model "${model}" --group "${group}"
