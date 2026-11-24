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
    trim_chunk_silence,
)
from mlx_asr_pipeline.subtitles import (
    DEFAULT_MIN_DURATION_SEC,
    format_srt_timestamp,
    group_subtitle_cues,
    is_context_echo,
    render_srt,
    restore_punctuation_resilient,
    segments_to_cues,
    validate_timeline,
)


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
        {"text": "TLA", "start": 0.0, "end": 0.4},
        {"text": "is", "start": 0.5, "end": 0.7},
        {"text": "100", "start": 0.7, "end": 1.0},
        {"text": "great", "start": 1.0, "end": 1.4},
    ]
    restored = restore_punctuation_resilient(
        segments, "TLA+ is 100% great!", language="English"
    )
    assert [s["text"] for s in restored] == ["TLA+", "is", "100%", "great!"]

    cues = segments_to_cues(segments, language="English", text="TLA+ is 100% great!")
    assert cues == [{"start": 0.0, "end": 1.4, "text": "TLA+ is 100% great!"}]

    fallback_cues = segments_to_cues(
        segments,
        language="English",
        text="TLA+ is 100% great!",
        grouper=_fallback_grouper,
    )
    assert len(fallback_cues) == 1
    assert fallback_cues[0]["text"] == "TLA+ is 100% great!"


def test_restore_punctuation_preserves_internal_symbols_and_hyphens() -> None:
    text = "Next.js prevents anti-patterns, property-based bugs, 11,000 PRs, well—that works!"
    segments = [
        {"text": "Next", "start": 0.0, "end": 0.2},
        {"text": "js", "start": 0.2, "end": 0.4},
        {"text": "prevents", "start": 0.45, "end": 0.8},
        {"text": "antipatterns", "start": 0.85, "end": 1.3},
        {"text": "propertybased", "start": 1.35, "end": 1.8},
        {"text": "bugs", "start": 1.85, "end": 2.1},
        {"text": "11000", "start": 2.15, "end": 2.5},
        {"text": "PRs", "start": 2.55, "end": 2.8},
        {"text": "wellthat", "start": 2.85, "end": 3.2},
        {"text": "works", "start": 3.25, "end": 3.6},
    ]
    restored = restore_punctuation_resilient(segments, text, language="English")
    assert [s["text"] for s in restored] == text.split()
    assert restored[0]["start"] == 0.0
    assert restored[0]["end"] == 0.4

    zh_segments = [
        {"text": "我", "start": 0.0, "end": 0.1},
        {"text": "们", "start": 0.1, "end": 0.2},
        {"text": "用", "start": 0.2, "end": 0.3},
        {"text": "TLA", "start": 0.3, "end": 0.6},
        {"text": "做", "start": 0.6, "end": 0.8},
        {"text": "100", "start": 0.8, "end": 1.0},
        {"text": "验", "start": 1.0, "end": 1.2},
        {"text": "证", "start": 1.2, "end": 1.4},
    ]
    zh_expected = "我们用TLA+做100%验证。"
    zh_restored = restore_punctuation_resilient(
        zh_segments, zh_expected, language="Chinese"
    )
    assert "".join(s["text"] for s in zh_restored) == zh_expected


def test_group_subtitle_cues_two_line_wrap_and_timing() -> None:
    sentence = "Today I am going to convince you that you can turn that slop into gold."
    words = sentence.split()
    segments = [
        {"text": w, "start": i * 0.2, "end": i * 0.2 + 0.15}
        for i, w in enumerate(words)
    ]
    grouped = group_subtitle_cues(segments, language="English")
    assert len(grouped) == 1
    lines = grouped[0]["text"].splitlines()
    assert len(lines) == 2
    assert all(len(line) <= 42 for line in lines)

    flash_segments = [
        {"text": "Okay.", "start": 0.0, "end": 0.2},
        {"text": "Next", "start": 2.0, "end": 2.1},
        {"text": "thing.", "start": 2.1, "end": 2.25},
    ]
    cues = segments_to_cues(
        flash_segments, language="English", text="Okay. Next thing."
    )
    assert len(cues) == 2
    assert cues[0]["end"] - cues[0]["start"] >= DEFAULT_MIN_DURATION_SEC - 1e-6
    assert cues[1]["end"] - cues[1]["start"] >= DEFAULT_MIN_DURATION_SEC - 1e-6
    assert validate_timeline(cues) == {"count": 2, "zeros": 0, "overlaps": 0}


def test_context_echo_filter_removes_prompt_leak_but_keeps_speech() -> None:
    context = "Grokbot PStack TLA+ Lean"
    assert is_context_echo("Grokbot PStack TLA Lean.", context)
    assert not is_context_echo("I work on Grokbot at xAI.", context)
    assert not is_context_echo("Grokbot.", context)

    zh_context = "学生管理 形式验证 变异测试"
    assert is_context_echo("学生管理形式验证变异测试。", zh_context)
    assert not is_context_echo("今天讲学生管理和形式验证。", zh_context)
    assert not is_context_echo("学生管理。", zh_context)

    segments = [
        {"text": "Grokbot", "start": 0.0, "end": 0.3},
        {"text": "PStack", "start": 0.3, "end": 0.6},
        {"text": "TLA", "start": 0.6, "end": 0.9},
        {"text": "Lean", "start": 0.9, "end": 1.2},
        {"text": "Hello", "start": 5.0, "end": 5.3},
        {"text": "Grokbot.", "start": 5.3, "end": 5.7},
    ]
    cues = segments_to_cues(
        segments,
        language="English",
        text="Grokbot PStack TLA Lean. Hello Grokbot.",
        context=context,
    )
    assert [c["text"] for c in cues] == ["Hello Grokbot."]


def test_chinese_subtitle_grouping_two_line_wrap_and_timing() -> None:
    zh_context = "学生管理 形式验证 变异测试"
    first_sentence = "今天我们用TLA+做100%形式验证，但是成本真的很高。"
    text = f"学生管理形式验证变异测试。{first_sentence}好的。"
    raw_words = [
        "学",
        "生",
        "管",
        "理",
        "形",
        "式",
        "验",
        "证",
        "变",
        "异",
        "测",
        "试",
        "今",
        "天",
        "我",
        "们",
        "用",
        "TLA",
        "做",
        "100",
        "形",
        "式",
        "验",
        "证",
        "但",
        "是",
        "成",
        "本",
        "真",
        "的",
        "很",
        "高",
        "好",
        "的",
    ]
    segments: list[dict[str, Any]] = []
    for i, w in enumerate(raw_words):
        shift = 5.0 if i >= 32 else (2.0 if i >= 12 else 0.0)
        segments.append(
            {"text": w, "start": i * 0.12 + shift, "end": i * 0.12 + 0.1 + shift}
        )

    cues = segments_to_cues(segments, language="Chinese", text=text, context=zh_context)
    assert len(cues) == 2
    assert cues[0]["text"].replace("\n", "") == first_sentence
    lines = cues[0]["text"].splitlines()
    assert len(lines) == 2
    assert all(sum(2 if ord(ch) > 127 else 1 for ch in line) <= 36 for line in lines)
    assert cues[1]["text"] == "好的。"
    assert cues[1]["end"] - cues[1]["start"] >= DEFAULT_MIN_DURATION_SEC - 1e-6
    assert validate_timeline(cues) == {"count": 2, "zeros": 0, "overlaps": 0}


def test_trim_chunk_silence_skips_dead_silence_and_preserves_offset() -> None:
    silent = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    trimmed_silent, offset_silent = trim_chunk_silence(silent, sr=SAMPLE_RATE)
    assert len(trimmed_silent) == 0
    assert offset_silent == 0.0

    mixed = np.zeros(SAMPLE_RATE * 5, dtype=np.float32)
    mixed[2 * SAMPLE_RATE : 3 * SAMPLE_RATE] = 0.05
    trimmed_mixed, offset_mixed = trim_chunk_silence(mixed, sr=SAMPLE_RATE)
    assert 1.7 <= offset_mixed <= 1.8
    assert 1.3 * SAMPLE_RATE <= len(trimmed_mixed) <= 1.6 * SAMPLE_RATE


def test_find_macro_chunk_boundaries_picks_silence() -> None:
    sr = 1000
    chunk_seconds = 30
    total_sec = 70
    rng = np.random.default_rng(42)
    audio = (rng.normal(0.0, 0.2, total_sec * sr)).astype(np.float32)
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
    )
    assert res1.cues >= 2
    assert res1.language == "English"
    assert out_srt.is_file()
    assert len(_FakeSession.calls) >= 2
    assert _FakeSession.calls[0]["context"] == "Grokbot PStack"
    assert isinstance(_FakeSession.calls[0]["forced_aligner"], _FakeAligner)

    ckpt_data = json.loads((work_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert ckpt_data["next_chunk"] >= 2

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
    )
    assert len(_FakeSession.calls) == prev_calls
    assert "Hello world 1." in res2.srt_text

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
    )
    assert len(_FakeSession.calls) > prev_calls
