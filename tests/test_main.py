import tomllib
from pathlib import Path

import pytest

from mlx_asr_pipeline import run


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    expected = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )["project"]["version"]
    assert run(["--version"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == f"mlx-asr-pipeline {expected}"
    assert captured.err == ""


def test_help_and_empty_args(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--help"]) == 0
    help_out = capsys.readouterr().out
    assert "usage: mlx-asr-pipeline" in help_out

    assert run([]) == 0
    empty_out = capsys.readouterr().out
    assert empty_out == help_out


def test_unknown_argument(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--unknown"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: unrecognized arguments: --unknown" in captured.err


def test_conflicting_stdout_and_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["clip.mp4", "-o", "-", "--json"]) == 2
    captured = capsys.readouterr()
    assert "cannot be combined" in captured.err


def test_missing_input_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["/nonexistent/path/audio.mp4", "--quiet"]) == 1
    captured = capsys.readouterr()
    assert "Input file not found" in captured.err
