from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

_MIN_CUE_DURATION_SEC = 0.08
DEFAULT_MIN_DURATION_SEC = 0.833
DEFAULT_MAX_DURATION_SEC = 7.0
DEFAULT_MAX_GAP_SEC = 0.8
DEFAULT_MIN_GAP_SEC = 0.08
DEFAULT_MAX_CPS = 20.0
DEFAULT_MAX_LINES = 2
DEFAULT_LATIN_MAX_LINE_WIDTH = 42
DEFAULT_CJK_MAX_LINE_WIDTH = 36

_NO_SPACE_LANGUAGES = frozenset(
    {"chinese", "zh", "zh-cn", "zh-tw", "cantonese", "yue", "japanese", "ja", "jp"}
)

_LINE_BREAK_BEFORE_WORDS = frozenset(
    {
        "and",
        "but",
        "or",
        "so",
        "yet",
        "nor",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "while",
        "because",
        "although",
        "though",
        "if",
        "unless",
        "until",
        "before",
        "after",
        "to",
        "in",
        "on",
        "at",
        "for",
        "from",
        "with",
        "of",
        "by",
        "about",
        "as",
        "into",
        "through",
    }
)

_WEAK_LINE_END_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
    }
)

_CLOSING_CHARS = "\"'”’」』）)】》〉]"
_SENTENCE_END_RE = re.compile(r"[.!?。！？…][" + re.escape(_CLOSING_CHARS) + r"]*$")
_CLAUSE_END_RE = re.compile(r"[,;:—，、；：][" + re.escape(_CLOSING_CHARS) + r"]*$")
_LEADING_ATTACHED_PUNCT = set("([{\"'“‘「『（【《〈")
_TRAILING_ATTACHED_PUNCT = set(",.;:!?%‰°'\"”’」』）】》〉])")
_CJK_LINE_BREAK_BEFORE = (
    "但是",
    "不过",
    "所以",
    "因此",
    "因为",
    "如果",
    "而且",
    "并且",
    "或者",
    "然后",
    "同时",
    "另外",
    "而是",
    "虽然",
)
_CJK_WEAK_LINE_START = frozenset("的了着过吗呢吧啊嘛呀")
_CONTEXT_SPLIT_RE = re.compile(r"[\s,，;；、/|]+")
_CUE_SPLIT_RE = re.compile(r"[\s,，.。!！?？;；:：、/|—–-]+")


def format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000.0))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _is_no_space_language(language: str | None) -> bool:
    key = str(language or "").strip().casefold().replace("_", "-")
    return key in _NO_SPACE_LANGUAGES


def _is_kept_align_char(ch: str) -> bool:
    if ch == "'":
        return True
    return unicodedata.category(ch).startswith(("L", "N"))


def _kept_align_key(text: str) -> str:
    return "".join(ch.casefold() for ch in text if _is_kept_align_char(ch))


def _is_punct_or_symbol(ch: str) -> bool:
    return unicodedata.category(ch).startswith(("P", "S"))


def _is_opening_punct(ch: str) -> bool:
    return unicodedata.category(ch) in {"Ps", "Pi"}


def _display_width(text: str) -> int:
    return sum(
        2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
        for c in text
        if c != "\n"
    )


def _ends_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _ends_clause(text: str) -> bool:
    return bool(_CLAUSE_END_RE.search(text.strip()))


def _align_space_delimited_text(
    segments: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    raw_words = text.split()
    if not raw_words or not segments:
        return segments

    seg_keys = [_kept_align_key(str(seg.get("text", ""))) for seg in segments]
    aligned: list[dict[str, Any]] = []
    seg_idx = 0
    n_segs = len(segments)

    for word in raw_words:
        word_key = _kept_align_key(word)
        while seg_idx < n_segs and not seg_keys[seg_idx]:
            seg_idx += 1

        if not word_key:
            anchor = (
                aligned[-1]["end"] if aligned else float(segments[0].get("start", 0.0))
            )
            aligned.append({"text": word, "start": anchor, "end": anchor})
            continue

        matched_start_idx: int | None = None
        matched_end_idx: int | None = None

        if seg_idx < n_segs:
            acc = ""
            probe = seg_idx
            while probe < n_segs and len(acc) < len(word_key):
                if seg_keys[probe]:
                    acc += seg_keys[probe]
                probe += 1
            if acc == word_key:
                matched_start_idx = seg_idx
                matched_end_idx = probe - 1
                seg_idx = probe
            else:
                resync_limit = min(n_segs, seg_idx + 6)
                for cand in range(seg_idx + 1, resync_limit):
                    if not seg_keys[cand]:
                        continue
                    cand_acc = ""
                    cand_probe = cand
                    while cand_probe < n_segs and len(cand_acc) < len(word_key):
                        if seg_keys[cand_probe]:
                            cand_acc += seg_keys[cand_probe]
                        cand_probe += 1
                    if cand_acc == word_key:
                        matched_start_idx = cand
                        matched_end_idx = cand_probe - 1
                        seg_idx = cand_probe
                        break

        if matched_start_idx is not None and matched_end_idx is not None:
            first_seg = segments[matched_start_idx]
            last_seg = segments[matched_end_idx]
            start_t = float(first_seg.get("start", 0.0))
            end_t = max(start_t, float(last_seg.get("end", start_t)))
            entry = {**first_seg, "text": word, "start": start_t, "end": end_t}
            aligned.append(entry)
        else:
            if aligned:
                anchor = float(aligned[-1]["end"])
                speaker = aligned[-1].get("speaker")
            else:
                anchor = float(segments[min(seg_idx, n_segs - 1)].get("start", 0.0))
                speaker = segments[min(seg_idx, n_segs - 1)].get("speaker")
            entry = {"text": word, "start": anchor, "end": anchor}
            if speaker is not None:
                entry["speaker"] = speaker
            aligned.append(entry)

    return aligned


def _align_no_space_text(
    segments: list[dict[str, Any]], text: str
) -> list[dict[str, Any]]:
    # Preserve exact transcript slices including symbols ('+', '%', '$') that
    # ForcedAlignTextProcessor strips during alignment.
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
            if _is_punct_or_symbol(text[i]):
                leading.append(text[i])
            i += 1

        matched_from = i
        matched = True
        cursor = i
        first_pos: int | None = None
        last_pos: int | None = None

        for c in chars:
            while cursor < n and not _is_kept_align_char(text[cursor]):
                cursor += 1
            if cursor < n and text[cursor].casefold() == c.casefold():
                if first_pos is None:
                    first_pos = cursor
                last_pos = cursor
                cursor += 1
            else:
                matched = False
                break

        if not matched or first_pos is None or last_pos is None:
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
                first_pos = window_chars[found_at][0]
                last_pos = window_chars[found_at + len(needle) - 1][0]
                leading = [
                    text[k]
                    for k in range(matched_from, first_pos)
                    if _is_punct_or_symbol(text[k])
                ]
                i = last_pos + 1
            else:
                restored.append({**seg, "text": raw_seg_text})
                continue
        else:
            i = cursor

        core_slice = text[first_pos : last_pos + 1]
        trailing: list[str] = []
        while i < n and not _is_kept_align_char(text[i]):
            if _is_punct_or_symbol(text[i]) and not _is_opening_punct(text[i]):
                trailing.append(text[i])
            elif text[i].isspace():
                break
            i += 1

        restored.append(
            {**seg, "text": "".join(leading) + core_slice + "".join(trailing)}
        )

    return restored


def restore_punctuation_resilient(
    segments: list[dict[str, Any]],
    text: str,
    *,
    language: str = "",
) -> list[dict[str, Any]]:
    if not segments or not text.strip():
        return segments
    if _is_no_space_language(language):
        return _align_no_space_text(segments, text)
    return _align_space_delimited_text(segments, text)


def is_context_echo(cue_text: str, context: str) -> bool:
    context_terms = {
        key
        for token in _CONTEXT_SPLIT_RE.split(context)
        if (key := _kept_align_key(token))
    }
    if len(context_terms) < 2:
        return False

    cue_terms = [
        key
        for token in _CUE_SPLIT_RE.split(cue_text)
        if (key := _kept_align_key(token))
    ]
    if (
        len(cue_terms) >= 2
        and len(set(cue_terms)) >= 2
        and all(term in context_terms for term in cue_terms)
    ):
        return True

    flat = _kept_align_key(cue_text)
    if not flat or len(flat) > 128:
        return False

    n = len(flat)
    state: list[str | None] = [None] * (n + 1)
    start_marker = ""
    multi_marker = "*"
    state[0] = start_marker

    for pos in range(n):
        cur = state[pos]
        if cur is None:
            continue
        for term in context_terms:
            if not flat.startswith(term, pos):
                continue
            nxt = pos + len(term)
            if cur == start_marker:
                cand = term
            elif cur == multi_marker or cur != term:
                cand = multi_marker
            else:
                cand = term

            existing = state[nxt]
            if existing is None or cand == multi_marker:
                state[nxt] = cand
            elif existing != cand:
                state[nxt] = multi_marker

    return state[n] == multi_marker


def _join_tokens(tokens: Sequence[str], *, no_space: bool) -> str:
    if no_space:
        return "".join(tokens)
    return " ".join(tokens)


def _split_candidates(text: str, *, no_space: bool) -> list[tuple[str, str]]:
    if not no_space:
        words = text.split()
        return [
            (" ".join(words[:idx]), " ".join(words[idx:]))
            for idx in range(1, len(words))
        ]

    chars = list(text)
    candidates: list[tuple[str, str]] = []
    for idx in range(1, len(chars)):
        left_last = chars[idx - 1]
        right_first = chars[idx]
        if right_first in _TRAILING_ATTACHED_PUNCT or right_first.isspace():
            continue
        if left_last in _LEADING_ATTACHED_PUNCT or left_last.isspace():
            continue
        if (
            left_last.isascii()
            and left_last.isalnum()
            and right_first.isascii()
            and right_first.isalnum()
        ):
            continue
        left = "".join(chars[:idx]).rstrip()
        right = "".join(chars[idx:]).lstrip()
        if left and right:
            candidates.append((left, right))
    return candidates


def wrap_cue_lines(
    text: str,
    *,
    max_line_width: int = DEFAULT_LATIN_MAX_LINE_WIDTH,
    max_lines: int = DEFAULT_MAX_LINES,
    no_space: bool = False,
) -> str:
    clean = " ".join(text.split()) if not no_space else text.strip()
    if max_lines <= 1 or _display_width(clean) <= max_line_width:
        return clean

    best_pair: tuple[str, str] | None = None
    best_score = float("inf")

    for left, right in _split_candidates(clean, no_space=no_space):
        w_left = _display_width(left)
        w_right = _display_width(right)
        if w_left > max_line_width or w_right > max_line_width:
            continue

        score = float(abs(w_left - w_right))
        if w_right >= w_left:
            score -= 1.0

        if _ends_sentence(left):
            score -= 18.0
        elif _ends_clause(left):
            score -= 12.0

        if not no_space:
            left_last_key = _kept_align_key(left.split()[-1])
            right_first_key = _kept_align_key(right.split()[0])
            if right_first_key in _LINE_BREAK_BEFORE_WORDS:
                score -= 6.0
            if left_last_key in _WEAK_LINE_END_WORDS:
                score += 8.0
        else:
            if right.startswith(_CJK_LINE_BREAK_BEFORE):
                score -= 6.0
            if right[0] in _CJK_WEAK_LINE_START:
                score += 6.0

        if score < best_score:
            best_score = score
            best_pair = (left, right)

    if best_pair is None:
        return clean
    return f"{best_pair[0]}\n{best_pair[1]}"


def _can_fit_in_lines(
    text: str,
    *,
    max_line_width: int,
    max_lines: int,
    no_space: bool,
) -> bool:
    if _display_width(text) <= max_line_width:
        return True
    if max_lines <= 1:
        return False
    wrapped = wrap_cue_lines(
        text,
        max_line_width=max_line_width,
        max_lines=max_lines,
        no_space=no_space,
    )
    if "\n" not in wrapped:
        return False
    lines = wrapped.splitlines()
    return len(lines) <= max_lines and all(
        _display_width(line) <= max_line_width for line in lines
    )


def _merge_short_tails(
    groups: list[list[dict[str, Any]]],
    *,
    no_space: bool,
    max_line_width: int,
    max_lines: int,
    max_duration_sec: float,
) -> list[list[dict[str, Any]]]:
    if len(groups) < 2:
        return groups

    merged: list[list[dict[str, Any]]] = [groups[0]]
    short_width_threshold = max(4, max_line_width // 3)

    for group in groups[1:]:
        prev = merged[-1]
        prev_text = _join_tokens([item["text"] for item in prev], no_space=no_space)
        cur_text = _join_tokens([item["text"] for item in group], no_space=no_space)
        gap = float(group[0]["start"]) - float(prev[-1]["end"])
        cur_dur = float(group[-1]["end"]) - float(group[0]["start"])
        combined_dur = float(group[-1]["end"]) - float(prev[0]["start"])

        same_speaker = prev[-1].get("speaker") == group[0].get("speaker")
        is_short_tail = (
            cur_dur < 0.5 or _display_width(cur_text) < short_width_threshold
        )

        if (
            same_speaker
            and gap < 0.35
            and not _ends_sentence(prev_text)
            and is_short_tail
            and combined_dur <= max_duration_sec
        ):
            combined_tokens = [*prev, *group]
            combined_text = _join_tokens(
                [item["text"] for item in combined_tokens], no_space=no_space
            )
            if _can_fit_in_lines(
                combined_text,
                max_line_width=max_line_width,
                max_lines=max_lines,
                no_space=no_space,
            ):
                merged[-1] = combined_tokens
                continue

        merged.append(group)

    return merged


def _extend_cue_durations(
    cues: list[dict[str, Any]],
    *,
    min_duration_sec: float,
    max_duration_sec: float,
    min_gap_sec: float,
    max_cps: float,
) -> list[dict[str, Any]]:
    for idx, cue in enumerate(cues):
        start = float(cue["start"])
        end = max(start, float(cue["end"]))
        width = _display_width(str(cue["text"]))
        cps_target = width / max_cps if max_cps > 0 else 0.0
        desired_dur = min(max_duration_sec, max(min_duration_sec, cps_target))

        if end - start < desired_dur:
            target_end = start + desired_dur
            if idx + 1 < len(cues):
                next_start = float(cues[idx + 1]["start"])
                max_allowed_end = next_start - min_gap_sec
                if max_allowed_end > end:
                    end = min(target_end, max_allowed_end)
            else:
                end = target_end

        if end <= start:
            next_start = float(cues[idx + 1]["start"]) if idx + 1 < len(cues) else None
            fallback_end = start + _MIN_CUE_DURATION_SEC
            if next_start is not None and next_start > start:
                fallback_end = min(fallback_end, next_start)
            end = fallback_end

        cue["start"] = start
        cue["end"] = end

    return cues


def group_subtitle_cues(
    segments: Sequence[Mapping[str, Any]],
    *,
    language: str = "",
    max_line_width: int | None = None,
    max_lines: int = DEFAULT_MAX_LINES,
    max_duration_sec: float = DEFAULT_MAX_DURATION_SEC,
    max_gap_sec: float = DEFAULT_MAX_GAP_SEC,
) -> list[dict[str, Any]]:
    no_space = _is_no_space_language(language)
    line_width = (
        max_line_width
        if max_line_width is not None
        else (DEFAULT_CJK_MAX_LINE_WIDTH if no_space else DEFAULT_LATIN_MAX_LINE_WIDTH)
    )

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for seg in segments:
        seg_text = str(seg.get("text", "")).strip()
        if not seg_text:
            continue
        start = float(seg.get("start", 0.0))
        end = max(start, float(seg.get("end", start)))
        item = {
            "text": seg_text,
            "start": start,
            "end": end,
            "speaker": seg.get("speaker"),
        }

        if not current:
            current.append(item)
            continue

        prev = current[-1]
        current_text = _join_tokens([tok["text"] for tok in current], no_space=no_space)
        candidate_text = _join_tokens(
            [*[tok["text"] for tok in current], seg_text], no_space=no_space
        )
        current_dur = float(prev["end"]) - float(current[0]["start"])

        hard_break = (
            start - float(prev["end"]) >= max_gap_sec
            or end - float(current[0]["start"]) > max_duration_sec
            or item["speaker"] != prev["speaker"]
            or _ends_sentence(str(prev["text"]))
            or (
                _ends_clause(str(prev["text"]))
                and (_display_width(current_text) >= line_width or current_dur >= 2.5)
            )
        )
        width_overflow = not _can_fit_in_lines(
            candidate_text,
            max_line_width=line_width,
            max_lines=max_lines,
            no_space=no_space,
        )

        if hard_break or width_overflow:
            groups.append(current)
            current = [item]
        else:
            current.append(item)

    if current:
        groups.append(current)

    groups = _merge_short_tails(
        groups,
        no_space=no_space,
        max_line_width=line_width,
        max_lines=max_lines,
        max_duration_sec=max_duration_sec,
    )

    cues: list[dict[str, Any]] = []
    for tokens in groups:
        flat_text = _join_tokens([tok["text"] for tok in tokens], no_space=no_space)
        wrapped_text = wrap_cue_lines(
            flat_text,
            max_line_width=line_width,
            max_lines=max_lines,
            no_space=no_space,
        )
        cues.append(
            {
                "start": float(tokens[0]["start"]),
                "end": float(tokens[-1]["end"]),
                "text": wrapped_text,
            }
        )
    return cues


def segments_to_cues(
    segments: list[dict[str, Any]],
    *,
    language: str = "",
    text: str | None = None,
    offset_sec: float = 0.0,
    context: str = "",
    grouper: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    if not segments:
        return []

    prepared = (
        restore_punctuation_resilient(segments, text, language=language)
        if text
        else segments
    )
    used_builtin_grouper = grouper is None
    if grouper is None:
        grouped = group_subtitle_cues(prepared, language=language)
    else:
        grouped = grouper(prepared, language=language, text=None)

    cues: list[dict[str, Any]] = []
    for seg in grouped:
        cue_text = str(seg.get("text", "")).strip()
        if not cue_text:
            continue
        if context and is_context_echo(cue_text, context):
            continue
        start = max(0.0, float(seg["start"]) + offset_sec)
        end = max(start, float(seg["end"]) + offset_sec)
        cues.append({"start": start, "end": end, "text": cue_text})

    if used_builtin_grouper:
        return _extend_cue_durations(
            cues,
            min_duration_sec=DEFAULT_MIN_DURATION_SEC,
            max_duration_sec=DEFAULT_MAX_DURATION_SEC,
            min_gap_sec=DEFAULT_MIN_GAP_SEC,
            max_cps=DEFAULT_MAX_CPS,
        )

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
