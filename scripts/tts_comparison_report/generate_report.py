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
import logging
import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Optional

from paramiko import AutoAddPolicy, SSHClient
from paramiko.sftp_client import SFTPClient
from scripts.tts_comparison_report.config import load_config
from scripts.tts_comparison_report.reporting import (
    DUMMY_TASK_ID,
    SUPPORTED_BENCHMARK_NAMES,
    TEMPLATES_DIR,
    BaseStorage,
    BucketStructure,
    LocalStorage,
    Orchestrator,
    Renderer,
    S3Client,
    S3Config,
    SFTPStorage,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.captureWarnings(True)


_REMOTE_PASSWORD: str = "REMOTE_PASSWORD"
_S3_ACCESS_KEY_ID: str = "S3_ACCESS_KEY_ID"
_S3_SECRET_ACCESS_KEY: str = "S3_SECRET_ACCESS_KEY"

_DEFAULT_BENCHMARK_NAMES: str = ",".join(SUPPORTED_BENCHMARK_NAMES)


def _create_argparser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Script for generating MagpieTTS evaluation comparison reports",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Primary YAML configuration file.")
    parser.add_argument(
        "--baseline_name",
        type=str,
        required=False,
        help="Name of the baseline model that will be used in report.",
    )
    parser.add_argument(
        "--baseline_path",
        type=str,
        required=False,
        help="Path to the generated evaluation bucket for the baseline model.",
    )
    parser.add_argument(
        "--candidate_name",
        type=str,
        required=False,
        help="Name of the candidate model that will be used in report.",
    )
    parser.add_argument(
        "--candidate_path",
        type=str,
        required=False,
        help="Path to the generated evaluation bucket for the candidate model.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default=_DEFAULT_BENCHMARK_NAMES,
        help="Comma-separated list of benchmarks used in the evaluation report.",
    )
    parser.add_argument(
        "--s3_endpoint",
        type=str,
        required=False,
        help="S3 endpoint URL used for uploading the audio report.",
    )
    parser.add_argument(
        "--s3_bucket",
        type=str,
        required=False,
        help="Name of the S3 bucket where the audio report HTML and audio files will be uploaded.",
    )
    parser.add_argument(
        "--s3_region",
        type=str,
        required=False,
        help="AWS region name for the S3 client.",
    )
    parser.add_argument(
        "--remote_hostname",
        type=str,
        default=None,
        help="Name of the remote host, if the generated buckets are located there.",
    )
    parser.add_argument(
        "--remote_username",
        type=str,
        default=None,
        help="Name of the user on the remote host.",
    )
    parser.add_argument(
        "--task_id",
        type=str,
        default=DUMMY_TASK_ID,
        help="Jira task number associated with this report.",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="results",
        help="Subdirectory inside the bucket root that contains evaluation outputs produced by `magpietts_inference`.",
    )
    parser.add_argument(
        "--audio_report",
        action='store_true',
        help="Generate additional report with side-by-side audio comparison.",
    )
    parser.add_argument(
        "--audio_report_benchmarks",
        type=str,
        default="libritts_test_clean,riva_hard_digits,riva_hard_letters",
        help="Comma-separated list of benchmarks to include in the audio report.",
    )
    parser.add_argument(
        "--samples_per_benchmark",
        type=int,
        default=30,
        help="Number of samples per benchmark in the audio report.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style dotlist overrides applied after --config (for example report.audio=true).",
    )
    return parser


def _get_benchmarks_list(benchmarks: str) -> list[str]:
    return [x.strip() for x in benchmarks.split(",") if x.strip()]


def _validate_benchmarks(benchmarks: list[str]) -> None:
    if not benchmarks:
        raise ValueError("Empty list of benchmark names was provided.")

    supported_set = set(SUPPORTED_BENCHMARK_NAMES)

    for name in benchmarks:
        if name not in supported_set:
            raise ValueError(f"Unknown benchmark name: '{name}'.")


def _validate_audio_report_benchmarks(
    benchmarks: list[str],
    audio_report_benchmarks: list[str],
) -> None:
    if not audio_report_benchmarks:
        raise ValueError("Empty list of benchmark names was provided for the audio report.")

    supported_set = set(benchmarks)

    for name in audio_report_benchmarks:
        if name not in supported_set:
            raise ValueError(f"Benchmark name for audio report '{name}' is not included in evaluation benchmarks.")


def main(argv: Optional[list[str]] = None) -> None:
    """Parse CLI arguments, generate comparison reports, and upload them to S3.

    This function serves as the command-line entry point for the report
    generation workflow. It validates user input, initializes storage and S3
    clients, runs the report orchestrator, and logs the resulting report URLs.

    Raises:
        ValueError: If required environment variables are missing or CLI
            arguments are invalid.
        RuntimeError: If report generation or upload does not complete
            successfully.
    """
    logger = logging.getLogger(__name__)

    parser = _create_argparser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)

    legacy_option_names = {
        "--baseline_name",
        "--baseline_path",
        "--candidate_name",
        "--candidate_path",
        "--benchmarks",
        "--s3_endpoint",
        "--s3_bucket",
        "--s3_region",
        "--remote_hostname",
        "--remote_username",
        "--task_id",
        "--results_subdir",
        "--audio_report",
        "--audio_report_benchmarks",
        "--samples_per_benchmark",
    }
    if args.config:
        mixed = sorted({token.split("=", 1)[0] for token in raw_argv if token.split("=", 1)[0] in legacy_option_names})
        if mixed:
            parser.error("--config cannot be combined with legacy flags: " + ", ".join(mixed))
        cfg = load_config(args.config, args.overrides)
        baseline_name = str(cfg.models.baseline.name)
        baseline_path = Path(str(cfg.models.baseline.path)).resolve()
        candidates = [(str(model.name), Path(str(model.path)).resolve()) for model in cfg.models.candidates]
        benchmarks = list(cfg.evaluation.benchmarks)
        results_subdir = str(cfg.evaluation.results_subdir)
        audio_report = bool(cfg.report.audio)
        audio_report_benchmarks = list(cfg.report.audio_benchmarks) if audio_report else None
        samples_per_benchmark = int(cfg.report.samples_per_benchmark)
        s3_endpoint, s3_bucket, s3_region = str(cfg.storage.endpoint), str(cfg.storage.bucket), str(cfg.storage.region)
        task_id = str(cfg.task_id)
        remote_hostname = str(cfg.remote.hostname) if cfg.remote and cfg.remote.hostname else None
        remote_username = str(cfg.remote.username) if cfg.remote and cfg.remote.username else None
    else:
        if args.overrides:
            parser.error("Dotlist overrides require --config.")
        required = [
            "baseline_name",
            "baseline_path",
            "candidate_name",
            "candidate_path",
            "s3_endpoint",
            "s3_bucket",
            "s3_region",
        ]
        missing = [f"--{name}" for name in required if getattr(args, name) is None]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(missing))
        baseline_name = args.baseline_name
        baseline_path = Path(args.baseline_path).resolve()
        candidates = [(args.candidate_name, Path(args.candidate_path).resolve())]
        benchmarks = _get_benchmarks_list(args.benchmarks)
        _validate_benchmarks(benchmarks)
        results_subdir = args.results_subdir
        audio_report = args.audio_report
        audio_report_benchmarks = None
        if audio_report:
            audio_report_benchmarks = _get_benchmarks_list(args.audio_report_benchmarks)
            _validate_audio_report_benchmarks(benchmarks, audio_report_benchmarks)
            if args.samples_per_benchmark <= 0:
                raise ValueError("Number of samples per benchmark for the audio report must be greater than 0.")
        samples_per_benchmark = args.samples_per_benchmark
        s3_endpoint, s3_bucket, s3_region = args.s3_endpoint, args.s3_bucket, args.s3_region
        task_id = args.task_id
        remote_hostname, remote_username = args.remote_hostname, args.remote_username

    bucket_structure = BucketStructure()
    bucket_structure.eval_output_subdir = results_subdir

    storage: BaseStorage
    s3_client: Optional[S3Client] = None
    ssh_client: Optional[SSHClient] = None
    sftp: Optional[SFTPClient] = None
    eval_report_url: Optional[str] = None
    audio_report_url: Optional[str] = None

    s3_key_id = os.getenv(_S3_ACCESS_KEY_ID)
    s3_secret_key = os.getenv(_S3_SECRET_ACCESS_KEY)

    if s3_key_id is None or s3_secret_key is None:
        raise ValueError(
            f"Environment variables '{_S3_ACCESS_KEY_ID}' and '{_S3_SECRET_ACCESS_KEY}' "
            "must be set for uploading reports to S3."
        )

    s3_cfg = S3Config(
        bucket=s3_bucket,
        endpoint_url=s3_endpoint,
        region_name=s3_region,
    )
    s3_client = S3Client(
        cfg=s3_cfg,
        aws_access_key_id=s3_key_id,
        aws_secret_access_key=s3_secret_key,
    )

    if task_id == DUMMY_TASK_ID:
        logger.warning("\nWARNING: It is recommended to assign the evaluation report to a specific ticket!")

    if any(baseline_path == candidate_path for _, candidate_path in candidates):
        logger.warning(
            "\nWARNING: Baseline and candidate paths are identical. "
            "Comparison report is not meaningful in this case!"
        )

    logger.info(
        f"\nComparing baseline '{baseline_name}' against candidate(s) "
        + ", ".join(f"'{name}'" for name, _ in candidates)
    )

    try:
        if remote_hostname is not None or remote_username is not None:
            if remote_username is None:
                raise ValueError("'remote_username' must be provided when using remote access.")

            if remote_hostname is None:
                raise ValueError("'remote_hostname' must be provided when using remote access.")

            remote_password = os.getenv(_REMOTE_PASSWORD)

            if remote_password is None:
                raise ValueError(f"Environment variable '{_REMOTE_PASSWORD}' is not set.")

            logger.info(f"\nSetting remote connection with host: {remote_hostname}")

            ssh_client = SSHClient()
            ssh_client.set_missing_host_key_policy(policy=AutoAddPolicy())
            ssh_client.connect(
                hostname=remote_hostname,
                username=remote_username,
                password=remote_password,
            )
            sftp = ssh_client.open_sftp()
            storage = SFTPStorage(sftp)

        else:
            storage = LocalStorage()

        renderer = Renderer(templates_dir=TEMPLATES_DIR)

        orchestrator = Orchestrator(
            bucket_structure=bucket_structure,
            storage=storage,
            s3_client=s3_client,
            renderer=renderer,
            logger=logger,
        )
        eval_report_url, audio_report_url = orchestrator.run(
            baseline_name=baseline_name,
            baseline_path=baseline_path,
            candidates=candidates,
            benchmarks=benchmarks,
            generate_audio_report=audio_report,
            audio_report_benchmarks=audio_report_benchmarks,
            samples_per_benchmark=samples_per_benchmark,
            task_id=task_id,
        )

    finally:
        if sftp is not None:
            sftp.close()

        if ssh_client is not None:
            ssh_client.close()

        if s3_client is not None:
            s3_client.close()

    if eval_report_url is None:
        raise RuntimeError("Failed to generate evaluation report and upload it to S3.")

    if audio_report and audio_report_url is None:
        raise RuntimeError("Failed to upload audio report to S3 and create URL.")

    if audio_report_url is not None:
        logger.info(f"\nAudio report is available at:\n{audio_report_url}")

    logger.info(f"\nEvaluation report is available at:\n{eval_report_url}")

    logger.info("\nSave the links and open in your browser!\n")


if __name__ == "__main__":
    main()
