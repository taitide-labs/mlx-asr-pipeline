from __future__ import annotations

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_asr_pipeline import youtube_cookies


def test_yt_cookies_version_and_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert youtube_cookies.run(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "yt-cookies 0.1.0"

    assert youtube_cookies.run(["--help"]) == 0
    assert "usage: yt-cookies" in capsys.readouterr().out


def test_yt_cookies_conflicting_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert youtube_cookies.run(["--quiet", "--json"]) == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_resolve_output_path_permissions(tmp_path: Path) -> None:
    explicit = tmp_path / "nested" / "cookies.txt"
    resolved = youtube_cookies.resolve_output_path(str(explicit))
    assert resolved == str(explicit.resolve())
    mode = stat.S_IMODE(Path(resolved).stat().st_mode)
    assert mode == 0o600


def test_export_cookies_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cookie_path = tmp_path / "cookies.txt"

    def fake_runner(*args: Any, **kwargs: Any) -> SimpleNamespace:
        payload = {
            "profileName": "02-0xtimi2233",
            "profileId": "Profile 6",
            "cookieCount": 42,
            "outputPath": str(cookie_path),
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(
        youtube_cookies,
        "export_cookies",
        lambda profile, output, urls=None: (
            youtube_cookies.export_cookies.__wrapped__(  # type: ignore[attr-defined]
                profile, output, urls
            )
            if hasattr(youtube_cookies.export_cookies, "__wrapped__")
            else fake_export(profile, output, fake_runner)
        ),
    )

    assert youtube_cookies.run(["-o", str(cookie_path), "--quiet"]) == 0
    assert capsys.readouterr().out.strip() == str(cookie_path.resolve())

    assert youtube_cookies.run(["-o", str(cookie_path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["cookie_count"] == 42
    assert data["profile_id"] == "Profile 6"


def fake_export(profile: str, output: str, runner: Any) -> dict[str, Any]:
    target_path = youtube_cookies.resolve_output_path(output)
    proc = runner(
        ["ego-browser", "nodejs"], input="", text=True, capture_output=True, check=False
    )
    data = dict(json.loads(proc.stdout))
    data["outputPath"] = target_path
    data["profileName"] = profile
    return data
