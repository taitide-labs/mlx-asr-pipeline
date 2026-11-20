from __future__ import annotations

import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

from mlx_asr_pipeline.pipeline import (
    SAMPLE_RATE,
    find_macro_chunk_boundaries,
    run_pipeline,
)
from mlx_asr_pipeline.subtitles import (
    format_srt_timestamp,
    render_srt,
    restore_punctuation_resilient,
    segments_to_cues,
    validate_timeline,
)


def _has_mlx_runtime() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _fallback_grouper(
    segments: list[dict[str, Any]], *, language: str = "", text: str | None = None
) -> list[dict[str, Any]]:
    if not segments:
        return []
    return [
        {
            "text": " ".join(str(s["text"]) for s in segments),
            "start": segments[0]["start"],
            "end": segments[-1]["end"],
        }
    ]


class _FakeSession:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        context: str = "",
        language: str | None = None,
        return_timestamps: bool = False,
        forced_aligner: Any = None,
        verbose: bool = False,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "samples": len(audio),
                "context": context,
                "language": language,
                "return_timestamps": return_timestamps,
                "forced_aligner": forced_aligner,
                "verbose": verbose,
            }
        )
        chunk_idx = len(self.calls)
        return SimpleNamespace(
            text=f"Hello world {chunk_idx}.",
            language=language or "English",
            segments=[
                {"text": "Hello", "start": 0.1, "end": 0.5},
                {"text": "world", "start": 0.55, "end": 0.9},
                {"text": str(chunk_idx), "start": 0.95, "end": 1.3},
            ],
        )


class _FakeAligner:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


def test_format_and_validate_srt() -> None:
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(3661.25) == "01:01:01,250"

    cues = [
        {"start": 0.1, "end": 1.2, "text": "First line."},
        {"start": 1.2, "end": 2.4, "text": "Second line."},
    ]
    srt = render_srt(cues)
    assert "1\n00:00:00,100 --> 00:00:01,200\nFirst line.\n" in srt
    assert validate_timeline(cues) == {"count": 2, "zeros": 0, "overlaps": 0}
    assert validate_timeline(
        [
            {"start": 0.0, "end": 2.0, "text": "A"},
            {"start": 1.5, "end": 1.5, "text": "B"},
        ]
    ) == {"count": 2, "zeros": 1, "overlaps": 1}


def test_segments_to_cues_preserves_punctuation_across_symbols() -> None:
    segments = [
        {"text": "TLA", "start": 0.0, "end": 0.0},
        {"text": "is", "start": 0.5, "end": 0.7},
        {"text": "100", "start": 0.7, "end": 1.0},
        {"text": "great", "start": 1.0, "end": 1.4},
    ]
    restored = restore_punctuation_resilient(segments, "TLA+ is 100% great!")
    assert restored[-1]["text"] == "great!"

    grouper = None if _has_mlx_runtime() else _fallback_grouper
    cues = segments_to_cues(
        segments, language="English", text="TLA+ is 100% great!", grouper=grouper
    )
    assert len(cues) >= 1
    assert cues[-1]["text"].endswith("!")
    assert all(cue["end"] > cue["start"] for cue in cues)


def test_find_macro_chunk_boundaries_picks_silence() -> None:
    sr = 1000
    chunk_seconds = 30
    total_sec = 70
    rng = np.random.default_rng(42)
    audio = (rng.normal(0.0, 0.2, total_sec * sr)).astype(np.float32)
    # Plant a clean silence window around 27.0s-28.0s inside [15s, 45s] search range.
    audio[27 * sr : 28 * sr] = 0.0

    boundaries = find_macro_chunk_boundaries(
        audio, sr=sr, chunk_seconds=chunk_seconds, search_window_sec=5.0
    )
    assert len(boundaries) >= 2
    assert boundaries[0][0] == 0
    assert boundaries[-1][1] == len(audio)
    assert 27 * sr <= boundaries[0][1] <= 28 * sr
    for (s1, e1), (s2, e2) in itertools.pairwise(boundaries):
        assert e1 == s2
        assert s1 < e1
        assert s2 < e2


def test_find_macro_chunk_boundaries_disabled() -> None:
    audio = np.zeros(SAMPLE_RATE * 10, dtype=np.float32)
    assert find_macro_chunk_boundaries(audio, chunk_seconds=0) == [(0, len(audio))]
    assert find_macro_chunk_boundaries(np.array([], dtype=np.float32)) == []


def test_run_pipeline_and_checkpoint_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeSession.calls = []
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"fake-media")
    out_srt = tmp_path / "out.srt"
    work_dir = tmp_path / "ckpt"

    total_samples = SAMPLE_RATE * 65
    fake_audio = np.ones(total_samples, dtype=np.float32) * 0.01

    grouper = None if _has_mlx_runtime() else _fallback_grouper

    res1 = run_pipeline(
        media,
        output_file=out_srt,
        language="English",
        context="Grokbot PStack",
        chunk_seconds=30,
        work_dir=work_dir,
        quiet=True,
        session_factory=_FakeSession,
        aligner_factory=_FakeAligner,
        audio_loader=lambda _: fake_audio,
        cue_grouper=grouper,
    )
    assert res1.cues >= 2
    assert res1.language == "English"
    assert out_srt.is_file()
    assert len(_FakeSession.calls) >= 2
    assert _FakeSession.calls[0]["context"] == "Grokbot PStack"
    assert isinstance(_FakeSession.calls[0]["forced_aligner"], _FakeAligner)

    ckpt_data = json.loads((work_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt_data["next_chunk"] >= 2

    # Resuming a finished checkpoint should skip inference completely.
    prev_calls = len(_FakeSession.calls)
    res2 = run_pipeline(
        media,
        output_file="-",
        language="English",
        context="Grokbot PStack",
        chunk_seconds=30,
        work_dir=work_dir,
        quiet=True,
        session_factory=_FakeSession,
        aligner_factory=_FakeAligner,
        audio_loader=lambda _: fake_audio,
        cue_grouper=grouper,
    )
    assert len(_FakeSession.calls) == prev_calls
    assert "Hello world 1." in res2.srt_text

    # Disabling resume re-runs inference.
    run_pipeline(
        media,
        output_file=out_srt,
        language="English",
        context="Grokbot PStack",
        chunk_seconds=30,
        work_dir=work_dir,
        resume=False,
        quiet=True,
        session_factory=_FakeSession,
        aligner_factory=_FakeAligner,
        audio_loader=lambda _: fake_audio,
        cue_grouper=grouper,
    )
    assert len(_FakeSession.calls) > prev_calls
