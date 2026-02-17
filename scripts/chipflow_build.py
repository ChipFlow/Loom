#!/usr/bin/env python3
"""Submit a ChipFlow design for synthesis and download build artifacts.

Uses the ChipFlow API with GitHub token authentication.

Usage:
    # Submit and wait for build
    python scripts/chipflow_build.py submit <design_dir> [--wait] [--build-mode synth_only]

    # Check build status
    python scripts/chipflow_build.py status <build_id>

    # Download build artifacts
    python scripts/chipflow_build.py download <build_id> <output_dir>

Examples:
    # Submit mcu_soc for synthesis
    python scripts/chipflow_build.py submit vendor/chipflow-examples/mcu_soc --wait --build-mode synth_only

    # Download results
    python scripts/chipflow_build.py download <build_id> tests/mcu_soc/build/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tarfile
import time
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

API_ORIGIN = os.environ.get("CHIPFLOW_API_ORIGIN", "https://build.chipflow.com")


def get_api_key() -> tuple[str, str]:
    """Get ChipFlow API key via GitHub token exchange.

    Returns (username, api_key).
    """
    # Check env var first
    api_key = os.environ.get("CHIPFLOW_API_KEY")
    if api_key:
        username = os.environ.get("CHIPFLOW_USERNAME", "")
        return username, api_key

    # Get GitHub token via gh CLI
    try:
        gh_token = subprocess.check_output(
            ["gh", "auth", "token"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.error("Failed to get GitHub token. Run 'gh auth login' first.")
        sys.exit(1)

    # Exchange for ChipFlow API key
    import urllib.request

    req = urllib.request.Request(
        f"{API_ORIGIN}/auth/github-token",
        data=json.dumps({"github_token": gh_token}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            api_key = data.get("api_key") or data.get("access_token")
            username = data.get("username", "")
            if not api_key:
                log.error(f"No API key in response: {data}")
                sys.exit(1)
            return username, api_key
    except Exception as e:
        log.error(f"Failed to exchange GitHub token for API key: {e}")
        sys.exit(1)


def api_request(method: str, path: str, username: str, api_key: str,
                **kwargs) -> dict | bytes | str:
    """Make an authenticated API request."""
    import urllib.request
    import base64

    url = f"{API_ORIGIN}{path}"
    auth = base64.b64encode(f"{username}:{api_key}".encode()).decode()

    if "files" in kwargs:
        # Multipart form data
        import uuid
        boundary = uuid.uuid4().hex
        body = b""
        for name, value in kwargs.get("data", {}).items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
            body += f"{value}\r\n".encode()
        for name, (filename, content) in kwargs["files"].items():
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
            body += b"Content-Type: application/octet-stream\r\n\r\n"
            body += content + b"\r\n"
        body += f"--{boundary}--\r\n".encode()

        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    elif "json_data" in kwargs:
        body = json.dumps(kwargs["json_data"]).encode()
        req = urllib.request.Request(url, data=body, method=method)
        req.add_header("Content-Type", "application/json")
    else:
        req = urllib.request.Request(url, method=method)

    req.add_header("Authorization", f"Basic {auth}")

    resp = urllib.request.urlopen(req)
    content_type = resp.headers.get("Content-Type", "")
    raw = resp.read()
    if "application/json" in content_type:
        return json.loads(raw)
    elif "text/" in content_type:
        return raw.decode()
    return raw


def cmd_submit(args):
    """Submit a design for synthesis."""
    design_dir = Path(args.design_dir)

    # Find the RTLIL and config files
    build_dir = design_dir / "build"
    rtlil_path = build_dir / f"{design_dir.name}.il"

    # We may need the chipflow_examples.il naming convention
    if not rtlil_path.exists():
        # Try chipflow-lib naming: project_name.il
        for il_file in build_dir.glob("*.il"):
            rtlil_path = il_file
            break

    if not rtlil_path.exists():
        log.error(f"RTLIL file not found in {build_dir}. Run 'chipflow silicon prepare' first.")
        sys.exit(1)

    # Load pins.lock as config
    pins_lock_path = design_dir / "pins.lock"
    if not pins_lock_path.exists():
        log.error(f"pins.lock not found at {pins_lock_path}. Run 'chipflow pin lock' first.")
        sys.exit(1)

    log.info(f"Submitting {design_dir.name}")
    log.info(f"  RTLIL: {rtlil_path} ({rtlil_path.stat().st_size / 1024:.0f} KB)")
    log.info(f"  Config: {pins_lock_path}")

    username, api_key = get_api_key()
    log.info(f"Authenticated as: {username}")

    with open(rtlil_path, "rb") as f:
        rtlil_data = f.read()
    with open(pins_lock_path, "rb") as f:
        config_data = f.read()

    form_data = {
        "projectId": design_dir.name,
        "name": f"gem-build-{int(time.time())}",
    }
    if args.build_mode:
        form_data["build_mode"] = args.build_mode

    result = api_request(
        "POST", "/build/submit",
        username, api_key,
        data=form_data,
        files={
            "rtlil": ("design.il", rtlil_data),
            "config": ("config.json", config_data),
        },
    )

    build_id = result["build_id"]
    log.info(f"Build submitted: {build_id}")
    log.info(f"Build URL: {API_ORIGIN}/build/{build_id}")

    if args.wait:
        log.info("Waiting for build to complete...")
        while True:
            time.sleep(10)
            status = api_request("GET", f"/build/{build_id}/status", username, api_key)
            build_status = status.get("status", "unknown")
            log.info(f"  Status: {build_status}")

            if build_status == "completed":
                log.info("Build completed successfully!")
                if args.output_dir:
                    download_artifacts(build_id, Path(args.output_dir), username, api_key)
                break
            elif build_status == "failed":
                log.error("Build failed!")
                # Try to get logs
                try:
                    logs = api_request("GET", f"/build/{build_id}/logs", username, api_key)
                    if isinstance(logs, str):
                        log.error(f"Build logs:\n{logs[-2000:]}")
                except Exception:
                    pass
                sys.exit(1)
    else:
        print(build_id)


def download_artifacts(build_id: str, output_dir: Path, username: str, api_key: str):
    """Download build artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download tarball
    log.info(f"Downloading artifacts for {build_id}...")
    try:
        data = api_request("GET", f"/build/{build_id}/artifacts/tarball", username, api_key)
        if isinstance(data, bytes):
            tarball_path = output_dir / f"{build_id}.tar.gz"
            with open(tarball_path, "wb") as f:
                f.write(data)
            log.info(f"  Saved tarball: {tarball_path} ({len(data) / 1024:.0f} KB)")

            # Extract
            results_dir = output_dir / "build" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(path=output_dir)
            log.info(f"  Extracted to: {output_dir}")
    except Exception as e:
        log.warning(f"  Failed to download tarball: {e}")

    # Try downloading individual artifacts
    for artifact in ["verilog", "report", "sdc"]:
        try:
            data = api_request(
                "GET", f"/build/{build_id}/artifacts/{artifact}",
                username, api_key
            )
            if isinstance(data, bytes):
                ext = {"verilog": ".v", "report": ".json", "sdc": ".sdc"}[artifact]
                path = output_dir / f"{artifact}{ext}"
                with open(path, "wb") as f:
                    f.write(data)
                log.info(f"  Saved {artifact}: {path} ({len(data) / 1024:.0f} KB)")
        except Exception as e:
            log.debug(f"  Could not download {artifact}: {e}")


def cmd_status(args):
    """Check build status."""
    username, api_key = get_api_key()
    status = api_request("GET", f"/build/{args.build_id}/status", username, api_key)
    print(json.dumps(status, indent=2))


def cmd_download(args):
    """Download build artifacts."""
    username, api_key = get_api_key()
    download_artifacts(args.build_id, Path(args.output_dir), username, api_key)


def main() -> int:
    parser = argparse.ArgumentParser(description="ChipFlow build management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Submit
    sub = subparsers.add_parser("submit", help="Submit design for synthesis")
    sub.add_argument("design_dir", help="Path to design directory (with pins.lock and build/)")
    sub.add_argument("--wait", action="store_true", help="Wait for build completion")
    sub.add_argument("--build-mode", choices=["full", "synth_only"], default=None,
                     help="Build mode (default: full)")
    sub.add_argument("--output-dir", help="Download artifacts to this directory on completion")
    sub.set_defaults(func=cmd_submit)

    # Status
    sub = subparsers.add_parser("status", help="Check build status")
    sub.add_argument("build_id", help="Build UUID")
    sub.set_defaults(func=cmd_status)

    # Download
    sub = subparsers.add_parser("download", help="Download build artifacts")
    sub.add_argument("build_id", help="Build UUID")
    sub.add_argument("output_dir", help="Output directory")
    sub.set_defaults(func=cmd_download)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
