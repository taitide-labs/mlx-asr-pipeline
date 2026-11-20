from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROFILE_NAME = ""
OUTPUT_FILE = ""
COOKIE_URLS = [
    "https://www.youtube.com",
    "https://youtube.com",
    "https://accounts.google.com",
    "https://www.google.com",
]

_TOOL_NAME = "yt-cookies"
_PKG_NAME = "mlx-asr-pipeline"

NODE_SCRIPT_TEMPLATE = r"""
const fs = await import("node:fs/promises");
const CONFIG = __CONFIG_JSON__;

const allProfiles = await profiles();
if (!allProfiles.length) {
  throw new Error("No available Ego Browser profile detected");
}

let target = null;
if (CONFIG.profileName) {
  const needle = CONFIG.profileName.trim();
  target = allProfiles.find((p) => p.name === needle || p.id === needle)
    || allProfiles.find((p) => p.name.toLowerCase() === needle.toLowerCase() || p.id.toLowerCase() === needle.toLowerCase());
  if (!target) {
    const available = allProfiles.map((p) => `${p.name} (${p.id}${p.isDefault ? ", default" : ""})`).join(", ");
    throw new Error(`Profile "${needle}" not found. Available: ${available}`);
  }
} else {
  target = allProfiles.find((p) => p.isDefault) || allProfiles[0];
}

const spaceName = `yt-cookie-export-${Date.now()}`;
const task = await taskSpace(spaceName, { profileId: target.id });

try {
  const page = task.page("p1");
  const { cookies } = await page.cdp("Network.getCookies", { urls: CONFIG.urls });

  const lines = ["# Netscape HTTP Cookie File"];
  for (const c of cookies) {
    const domain = c.domain;
    const includeSubdomains = domain.startsWith(".") ? "TRUE" : "FALSE";
    const path = c.path || "/";
    const secure = c.secure ? "TRUE" : "FALSE";
    const expires = c.expires && c.expires > 0 ? Math.floor(c.expires) : 0;
    lines.push([domain, includeSubdomains, path, secure, expires, c.name, c.value].join("\t"));
  }

  await fs.writeFile(CONFIG.outputPath, lines.join("\n") + "\n", { mode: 0o600 });
  console.log(JSON.stringify({
    profileName: target.name,
    profileId: target.id,
    cookieCount: cookies.length,
    outputPath: CONFIG.outputPath,
  }));
} finally {
  await task.finish({ keep: [] }).catch(() => {});
}
"""


def _package_version() -> str:
    try:
        return version(_PKG_NAME)
    except PackageNotFoundError:
        return "0.1.0"


def resolve_output_path(configured_path: str) -> str:
    if configured_path:
        abs_path = Path(configured_path).expanduser().resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.touch(exist_ok=True)
        os.chmod(abs_path, 0o600)
        return str(abs_path)

    fd, temp_path = tempfile.mkstemp(prefix="yt_cookies_", suffix=".txt")
    os.close(fd)
    os.chmod(temp_path, 0o600)
    return temp_path


def export_cookies(
    profile_name: str,
    output_path: str,
    urls: list[str] | None = None,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    target_path = resolve_output_path(output_path)
    config_json = json.dumps(
        {
            "profileName": profile_name,
            "outputPath": target_path,
            "urls": COOKIE_URLS if urls is None else urls,
        },
        ensure_ascii=False,
    )
    node_script = NODE_SCRIPT_TEMPLATE.replace("__CONFIG_JSON__", config_json)

    try:
        proc = runner(
            ["ego-browser", "nodejs"],
            input=node_script,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        if (
            not output_path
            and os.path.exists(target_path)
            and os.path.getsize(target_path) == 0
        ):
            os.remove(target_path)
        raise RuntimeError("ego-browser not found on PATH.") from exc

    if proc.returncode != 0:
        if (
            not output_path
            and os.path.exists(target_path)
            and os.path.getsize(target_path) == 0
        ):
            os.remove(target_path)
        err_msg = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ego-browser failed:\n{err_msg}")

    combined = "\n".join(filter(None, [proc.stdout, proc.stderr]))
    for line in reversed(combined.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return dict(json.loads(stripped))

    raise RuntimeError("Failed to parse ego-browser export output.")


class _CliParser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        print(
            f"error: {message}\nRun '{_TOOL_NAME} --help' for usage.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _build_parser() -> argparse.ArgumentParser:
    parser = _CliParser(
        prog=_TOOL_NAME,
        add_help=False,
        description=(
            f"{_TOOL_NAME} — Export YouTube/Google session cookies from Ego Browser "
            "to a 0600 Netscape file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            f"  COOKIE_PATH=$({_TOOL_NAME} -q)\n"
            f'  {_TOOL_NAME} --profile "Profile 1" --json'
        ),
    )
    parser.add_argument("-h", "--help", action="store_true", help="Print help")
    parser.add_argument(
        "--profile",
        default=PROFILE_NAME,
        help="Ego Browser profile name or ID (default: browser default profile)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=OUTPUT_FILE,
        help="Output cookie file path (default: secure temp file)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable metadata to stdout",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print the exported file path to stdout",
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
    try:
        parsed = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2

    if parsed.help:
        parser.print_help()
        return 0
    if parsed.version:
        print(f"{_TOOL_NAME} {_package_version()}")
        return 0

    if parsed.quiet and parsed.json:
        print(
            f"error: --quiet cannot be combined with --json\nRun '{_TOOL_NAME} --help' for usage.",
            file=sys.stderr,
        )
        return 2

    try:
        result = export_cookies(parsed.profile, parsed.output)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if parsed.json:
        payload = {
            "profile_name": result["profileName"],
            "profile_id": result["profileId"],
            "cookie_count": result["cookieCount"],
            "output_path": result["outputPath"],
        }
        print(json.dumps(payload, ensure_ascii=False))
        return 0

    if parsed.quiet:
        print(result["outputPath"])
        return 0

    print(f"Profile: {result['profileName']} ({result['profileId']})")
    print(f"Cookies exported: {result['cookieCount']}")
    print(f"File: {result['outputPath']}")
    return 0


def entrypoint() -> None:
    raise SystemExit(run())
