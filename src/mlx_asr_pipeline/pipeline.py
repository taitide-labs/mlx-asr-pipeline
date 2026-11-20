from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from mlx_asr_pipeline.subtitles import render_srt, segments_to_cues, write_srt

DEFAULT_ASR_MODEL = "moona3k/mlx-qwen3-asr-1.7b-8bit"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_CHUNK_SECONDS = 300
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class PipelineResult:
    output: str
    language: str
    duration_sec: float
    cues: int
    elapsed_sec: float
    srt_text: str


def default_work_dir(input_path: Path) -> Path:
    resolved = input_path.resolve()
    stat = resolved.stat()
    fingerprint = f"{resolved}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    digest = hashlib.sha256(fingerprint).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "mlx-asr-pipeline" / digest


def find_macro_chunk_boundaries(
    audio: np.ndarray,
    *,
    sr: int = SAMPLE_RATE,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    search_window_sec: float = 15.0,
) -> list[tuple[int, int]]:
    total_samples = len(audio)
    if total_samples == 0:
        return []
    if chunk_seconds <= 0:
        return [(0, total_samples)]

    target_samples = chunk_seconds * sr
    if total_samples <= target_samples:
        return [(0, total_samples)]

    half_window = int(search_window_sec * sr)
    rms_win = max(1, int(0.5 * sr))
    step = max(1, rms_win // 2)

    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < total_samples:
        nominal_end = start + target_samples
        if nominal_end >= total_samples:
            boundaries.append((start, total_samples))
            break

        search_start = max(start + sr, nominal_end - half_window)
        search_end = min(total_samples - sr, nominal_end + half_window)
        best_split = nominal_end

        if search_end - search_start > rms_win:
            min_energy = float("inf")
            for pos in range(search_start, search_end - rms_win, step):
                segment = audio[pos : pos + rms_win]
                energy = float(np.sqrt(np.mean(segment * segment)))
                if energy < min_energy:
                    min_energy = energy
                    best_split = pos + rms_win // 2

        if best_split <= start:
            best_split = min(total_samples, nominal_end)

        boundaries.append((start, best_split))
        start = best_split

    return boundaries


def _checkpoint_fingerprint(
    *,
    input_path: Path,
    asr_model: str,
    aligner_model: str,
    language: str | None,
    context: str,
    chunk_seconds: int,
) -> dict[str, Any]:
    stat = input_path.resolve().stat()
    return {
        "input": str(input_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "asr_model": asr_model,
        "aligner_model": aligner_model,
        "language": language,
        "context": context,
        "chunk_seconds": chunk_seconds,
    }


def _load_checkpoint(
    checkpoint_file: Path,
    expected_fingerprint: dict[str, Any],
) -> tuple[int, list[dict[str, Any]], str]:
    if not checkpoint_file.exists():
        return 0, [], ""
    try:
        data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return 0, [], ""

    if data.get("fingerprint") != expected_fingerprint:
        return 0, [], ""

    next_chunk = int(data.get("next_chunk", 0))
    cues = list(data.get("cues", []))
    language = str(data.get("language", ""))
    return next_chunk, cues, language


def _save_checkpoint(
    checkpoint_file: Path,
    *,
    fingerprint: dict[str, Any],
    next_chunk: int,
    cues: list[dict[str, Any]],
    language: str,
) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = checkpoint_file.with_suffix(".json.tmp")
    payload = {
        "fingerprint": fingerprint,
        "next_chunk": next_chunk,
        "language": language,
        "cues": cues,
    }
    tmp_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp_file.replace(checkpoint_file)


def _log(msg: str, *, quiet: bool, stream: TextIO | None = None) -> None:
    if quiet:
        return
    target = sys.stderr if stream is None else stream
    print(msg, file=target, flush=True)


def run_pipeline(
    input_file: str | Path,
    *,
    output_file: str | Path | None = None,
    language: str | None = None,
    context: str = "",
    asr_model: str = DEFAULT_ASR_MODEL,
    aligner_model: str = DEFAULT_ALIGNER_MODEL,
    chunk_seconds: int = DEFAULT_CHUNK_SECONDS,
    work_dir: str | Path | None = None,
    resume: bool = True,
    quiet: bool = False,
    verbose: bool = False,
    session_factory: Any = None,
    aligner_factory: Any = None,
    audio_loader: Any = None,
    cue_grouper: Callable[..., list[dict[str, Any]]] | None = None,
) -> PipelineResult:
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if audio_loader is None:
        from mlx_qwen3_asr.audio import load_audio_np

        audio_loader = load_audio_np

    write_to_stdout = str(output_file) == "-"
    resolved_output: Path | None
    if write_to_stdout:
        resolved_output = None
    elif output_file is not None:
        resolved_output = Path(output_file)
    else:
        resolved_output = input_path.with_suffix(".srt")

    ckpt_dir = Path(work_dir) if work_dir is not None else default_work_dir(input_path)
    checkpoint_file = ckpt_dir / "checkpoint.json"
    fingerprint = _checkpoint_fingerprint(
        input_path=input_path,
        asr_model=asr_model,
        aligner_model=aligner_model,
        language=language,
        context=context,
        chunk_seconds=chunk_seconds,
    )

    t0 = time.monotonic()
    _log(f"Decoding audio (16kHz mono): {input_path}", quiet=quiet)
    audio_np = np.asarray(audio_loader(str(input_path)), dtype=np.float32)
    duration_sec = len(audio_np) / SAMPLE_RATE
    boundaries = find_macro_chunk_boundaries(
        audio_np, sr=SAMPLE_RATE, chunk_seconds=chunk_seconds
    )
    total_chunks = len(boundaries)
    _log(
        f"Audio duration: {duration_sec:.1f}s ({total_chunks} macro-chunk(s))",
        quiet=quiet,
    )

    if resume:
        start_idx, all_cues, detected_language = _load_checkpoint(
            checkpoint_file, fingerprint
        )
        if start_idx > 0:
            _log(
                f"Resuming from macro-chunk {start_idx + 1}/{total_chunks} ({len(all_cues)} cues)",
                quiet=quiet,
            )
    else:
        start_idx, all_cues, detected_language = 0, [], ""

    if start_idx < total_chunks:
        if session_factory is None or aligner_factory is None:
            from mlx_qwen3_asr import ForcedAligner, Session

            if session_factory is None:
                session_factory = Session
            if aligner_factory is None:
                aligner_factory = ForcedAligner

        t_load = time.monotonic()
        _log(f"Loading ASR ({asr_model}) & Aligner ({aligner_model})...", quiet=quiet)
        session = session_factory(asr_model)
        aligner = aligner_factory(aligner_model)
        _log(f"Models ready in {time.monotonic() - t_load:.2f}s", quiet=quiet)

        for idx in range(start_idx, total_chunks):
            s_idx, e_idx = boundaries[idx]
            offset_sec = s_idx / SAMPLE_RATE
            chunk_audio = audio_np[s_idx:e_idx]
            t_chunk = time.monotonic()

            result = session.transcribe(
                chunk_audio,
                context=context,
                language=language,
                return_timestamps=True,
                forced_aligner=aligner,
                verbose=verbose and not quiet,
            )

            chunk_lang = str(getattr(result, "language", "") or language or "")
            if not detected_language and chunk_lang:
                detected_language = chunk_lang

            segments = getattr(result, "segments", None) or []
            chunk_cues = segments_to_cues(
                segments,
                language=chunk_lang,
                text=getattr(result, "text", None),
                offset_sec=offset_sec,
                grouper=cue_grouper,
            )
            all_cues.extend(chunk_cues)

            if resolved_output is not None:
                write_srt(all_cues, resolved_output)

            _save_checkpoint(
                checkpoint_file,
                fingerprint=fingerprint,
                next_chunk=idx + 1,
                cues=all_cues,
                language=detected_language,
            )

            _log(
                f"[{idx + 1}/{total_chunks}] "
                f"{offset_sec:.1f}s -> {e_idx / SAMPLE_RATE:.1f}s "
                f"({time.monotonic() - t_chunk:.1f}s | {len(all_cues)} cues)",
                quiet=quiet,
            )

    srt_text = render_srt(all_cues)
    if resolved_output is not None and not resolved_output.exists():
        write_srt(all_cues, resolved_output)

    elapsed_sec = time.monotonic() - t0
    if resolved_output is not None:
        rtf = duration_sec / elapsed_sec if elapsed_sec > 0 else 0.0
        _log(
            f"Done in {elapsed_sec:.1f}s ({rtf:.1f}x realtime) -> {resolved_output}",
            quiet=quiet,
        )

    return PipelineResult(
        output="-" if resolved_output is None else str(resolved_output),
        language=detected_language or (language or "unknown"),
        duration_sec=round(duration_sec, 3),
        cues=len(all_cues),
        elapsed_sec=round(elapsed_sec, 3),
        srt_text=srt_text,
    )
