from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_MIN_CUE_DURATION_SEC = 0.08


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000.0))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _is_punct(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


def _is_opening_punct(ch: str) -> bool:
    return unicodedata.category(ch) in {"Ps", "Pi"}


def _is_kept_align_char(ch: str) -> bool:
    if ch == "'":
        return True
    cat = unicodedata.category(ch)
    return cat.startswith(("L", "N"))


def restore_punctuation_resilient(
    segments: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    # mlx_qwen3_asr's built-in restore_punctuation aborts the whole chunk when
    # encountering Symbol tokens (e.g. '+', '$', '%') stripped by ForcedAligner.
    n = len(text)
    i = 0
    restored: list[dict[str, Any]] = []

    for seg in segments:
        raw_seg_text = str(seg.get("text", "")).strip()
        chars = [c for c in raw_seg_text if _is_kept_align_char(c)]
        if not chars:
            restored.append(seg)
            continue

        leading: list[str] = []
        while i < n and not _is_kept_align_char(text[i]):
            if _is_punct(text[i]):
                leading.append(text[i])
            i += 1

        matched_from = i
        matched = True
        cursor = i
        for c in chars:
            while cursor < n and not _is_kept_align_char(text[cursor]):
                cursor += 1
            if cursor < n and text[cursor].casefold() == c.casefold():
                cursor += 1
            else:
                matched = False
                break

        if not matched:
            needle = "".join(chars).casefold()
            window_end = min(n, matched_from + 120)
            window_chars: list[tuple[int, str]] = [
                (idx, text[idx].casefold())
                for idx in range(matched_from, window_end)
                if _is_kept_align_char(text[idx])
            ]
            joined_window = "".join(ch for _, ch in window_chars)
            found_at = joined_window.find(needle)
            if found_at != -1:
                start_pos = window_chars[found_at][0]
                leading = [
                    text[k]
                    for k in range(matched_from, start_pos)
                    if _is_punct(text[k])
                ]
                i = window_chars[found_at + len(needle) - 1][0] + 1
            else:
                restored.append({**seg, "text": raw_seg_text})
                continue
        else:
            i = cursor

        trailing: list[str] = []
        while i < n and not _is_kept_align_char(text[i]):
            if _is_punct(text[i]) and not _is_opening_punct(text[i]):
                trailing.append(text[i])
            elif text[i].isspace():
                break
            i += 1

        restored.append(
            {**seg, "text": "".join(leading) + raw_seg_text + "".join(trailing)}
        )

    return restored


def segments_to_cues(
    segments: list[dict[str, Any]],
    *,
    language: str = "",
    text: str | None = None,
    offset_sec: float = 0.0,
    grouper: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    if grouper is None:
        from mlx_qwen3_asr.writers import group_subtitle_segments

        grouper = group_subtitle_segments
    prepared = restore_punctuation_resilient(segments, text) if text else segments
    grouped = grouper(prepared, language=language, text=None)
    cues: list[dict[str, Any]] = []
    for seg in grouped:
        cue_text = str(seg.get("text", "")).strip()
        if not cue_text:
            continue
        start = max(0.0, float(seg["start"]) + offset_sec)
        end = max(start, float(seg["end"]) + offset_sec)
        cues.append({"start": start, "end": end, "text": cue_text})

    for idx, cue in enumerate(cues):
        if cue["end"] <= cue["start"]:
            next_start = cues[idx + 1]["start"] if idx + 1 < len(cues) else None
            target_end = cue["start"] + _MIN_CUE_DURATION_SEC
            if next_start is not None and next_start > cue["start"]:
                target_end = min(target_end, next_start)
            cue["end"] = target_end

    return cues


def render_srt(cues: Sequence[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for index, cue in enumerate(cues, start=1):
        start = format_srt_timestamp(float(cue["start"]))
        end = format_srt_timestamp(float(cue["end"]))
        text = str(cue["text"]).strip()
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n\n"


def write_srt(cues: Sequence[Mapping[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    tmp_path.write_text(render_srt(cues), encoding="utf-8")
    tmp_path.replace(output_path)


def validate_timeline(cues: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    zeros = 0
    overlaps = 0
    prev_end: float | None = None
    for cue in cues:
        start = float(cue["start"])
        end = float(cue["end"])
        if end <= start:
            zeros += 1
        if prev_end is not None and start < prev_end - 1e-3:
            overlaps += 1
        prev_end = end
    return {"count": len(cues), "zeros": zeros, "overlaps": overlaps}
