from __future__ import annotations

import argparse
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from mlx_asr_pipeline.pipeline import (
    DEFAULT_ALIGNER_MODEL,
    DEFAULT_ASR_MODEL,
    DEFAULT_CHUNK_SECONDS,
    run_pipeline,
)

_NAME = "mlx-asr-pipeline"


def _package_version() -> str:
    try:
        return version(_NAME)
    except PackageNotFoundError:
        return "0.1.0"


class _CliParser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        print(
            f"error: {message}\nRun '{_NAME} --help' for usage.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    parser = _CliParser(
        prog=_NAME,
        add_help=False,
        description=(
            f"{_NAME} — Native Apple Silicon MLX ASR and forced-alignment subtitle pipeline."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            f"  {_NAME} talk.mp4\n"
            f'  {_NAME} talk.mp4 -o talk.en.srt -l English -c "Grokbot PStack TLA+ Lean"\n'
            f"  {_NAME} clip.mp4 -o - --quiet > clip.srt"
        ),
    )
    parser.add_argument(
        "input", metavar="INPUT", nargs="?", help="Input audio or video file"
    )
    parser.add_argument("-h", "--help", action="store_true", help="Print help")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output SRT path (default: <input_stem>.srt; use '-' for stdout)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        help="Force recognition language (default: auto-detect)",
    )
    parser.add_argument(
        "-c",
        "--context",
        default="",
        help="Domain terms / prompt context to bias recognition",
    )
    parser.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help=f"ASR model ID or local path [default: {DEFAULT_ASR_MODEL}]",
    )
    parser.add_argument(
        "--aligner-model",
        default=DEFAULT_ALIGNER_MODEL,
        help=f"ForcedAligner model ID or local path [default: {DEFAULT_ALIGNER_MODEL}]",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=int,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Macro-chunk size for checkpoint resume; 0 disables [default: {DEFAULT_CHUNK_SECONDS}]",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Checkpoint directory (default: auto temp dir per input)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing checkpoint and start from scratch",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary to stdout",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress output on stderr",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-chunk transcript previews on stderr",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Print version",
    )
    return parser


def run(args: list[str] | None = None) -> int:
    argv = sys.argv[1:] if args is None else args
    parser = _build_parser()
    if not argv:
        parser.print_help()
        return 0

    try:
        parsed = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if parsed.help:
        parser.print_help()
        return 0
    if parsed.version:
        print(f"{_NAME} {_package_version()}")
        return 0

    if not parsed.input:
        print(
            f"error: missing required argument: INPUT\nRun '{_NAME} --help' for usage.",
            file=sys.stderr,
        )
        return 2

    if parsed.chunk_seconds < 0:
        print(
            f"error: --chunk-seconds must be >= 0\nRun '{_NAME} --help' for usage.",
            file=sys.stderr,
        )
        return 2

    if parsed.output == "-" and parsed.json:
        print(
            f"error: --output - cannot be combined with --json\nRun '{_NAME} --help' for usage.",
            file=sys.stderr,
        )
        return 2

    if parsed.quiet:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")

    try:
        result = run_pipeline(
            parsed.input,
            output_file=parsed.output,
            language=parsed.language,
            context=parsed.context,
            asr_model=parsed.asr_model,
            aligner_model=parsed.aligner_model,
            chunk_seconds=parsed.chunk_seconds,
            work_dir=parsed.work_dir,
            resume=not parsed.no_resume,
            quiet=parsed.quiet,
            verbose=parsed.verbose,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if parsed.output == "-":
        sys.stdout.write(result.srt_text)
        sys.stdout.flush()
    elif parsed.json:
        print(
            json.dumps(
                {
                    "output": result.output,
                    "language": result.language,
                    "duration_sec": result.duration_sec,
                    "cues": result.cues,
                    "elapsed_sec": result.elapsed_sec,
                },
                ensure_ascii=False,
            )
        )
    else:
        print(result.output)

    return 0


def entrypoint() -> None:
    raise SystemExit(run())
