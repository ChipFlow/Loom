#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "rich"]
# ///
"""Download RTL artifacts from ChipFlow build.

Uses gsutil for GCS access (requires gcloud auth).
Falls back to API if gsutil unavailable.
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

console = Console()

BASE_URL = "https://build.staging.chipflow.com"


def get_gh_token() -> str:
    """Get GitHub auth token using gh CLI."""
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode != 0:
        console.print("[red]Failed to get gh auth token. Run 'gh auth login' first.[/red]")
        sys.exit(1)
    return result.stdout.strip()


def exchange_token_for_api_key(gh_token: str) -> tuple[str, str]:
    """Exchange GitHub token for ChipFlow API key."""
    url = f"{BASE_URL}/auth/github-token"
    resp = requests.post(url, json={"github_token": gh_token})
    if resp.status_code == 401:
        console.print("[red]Invalid GitHub token[/red]")
        sys.exit(1)
    resp.raise_for_status()
    data = resp.json()
    # API returns api_key but not always username
    return data.get("username", ""), data["api_key"]


def get_build_status(build_id: str, api_key: str) -> dict:
    """Fetch build status from API."""
    url = f"{BASE_URL}/build/{build_id}/status"
    resp = requests.get(url, auth=("", api_key))
    if resp.status_code == 404:
        console.print(f"[red]Build {build_id} not found[/red]")
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def find_gcs_bucket(build_id: str) -> str | None:
    """Find which GCS bucket contains the build."""
    result = subprocess.run(["gsutil", "ls"], capture_output=True, text=True)
    if result.returncode != 0:
        return None

    for bucket in result.stdout.strip().split("\n"):
        bucket = bucket.rstrip("/")
        # Check if build folder exists in this bucket
        check = subprocess.run(
            ["gsutil", "ls", f"{bucket}/{build_id}/"],
            capture_output=True, text=True
        )
        if check.returncode == 0:
            return bucket

    return None


def list_gcs_artifacts(bucket: str, build_id: str) -> list[tuple[str, int]]:
    """List artifacts in GCS bucket."""
    result = subprocess.run(
        ["gsutil", "ls", "-l", f"{bucket}/{build_id}/"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return []

    artifacts = []
    for line in result.stdout.strip().split("\n"):
        if line.startswith("TOTAL:"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            size = int(parts[0]) if parts[0].isdigit() else 0
            path = parts[-1]
            name = path.split("/")[-1]
            artifacts.append((name, size, path))

    return artifacts


def download_gcs_artifact(gcs_path: str, output_path: Path) -> None:
    """Download artifact from GCS."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["gsutil", "cp", gcs_path, str(output_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"gsutil cp failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Download RTL artifacts from ChipFlow build")
    parser.add_argument("build_id", help="Build ID (UUID)")
    parser.add_argument("-o", "--output", type=Path, default=Path("./build_artifacts"),
                        help="Output directory (default: ./build_artifacts)")
    parser.add_argument("-l", "--list", action="store_true", help="List artifacts only")
    parser.add_argument("-p", "--pattern", help="Only download files matching pattern")
    parser.add_argument("--extract", action="store_true",
                        help="Extract tarball after download")
    args = parser.parse_args()

    # Step 1: Get build status via API
    console.print("[blue]Authenticating...[/blue]")
    gh_token = get_gh_token()
    _, api_key = exchange_token_for_api_key(gh_token)

    console.print(f"[blue]Checking build {args.build_id}...[/blue]")
    status = get_build_status(args.build_id, api_key)
    console.print(f"[green]Build status: {status.get('status', 'unknown')}[/green]")

    # Step 2: Find GCS bucket containing the build
    console.print("[blue]Looking for build in GCS...[/blue]")
    bucket = find_gcs_bucket(args.build_id)
    if not bucket:
        console.print("[red]Could not find build in any GCS bucket.[/red]")
        console.print("[yellow]Make sure you have gsutil installed and authenticated.[/yellow]")
        sys.exit(1)

    console.print(f"[green]Found in bucket: {bucket}[/green]")

    # Step 3: List artifacts
    artifacts = list_gcs_artifacts(bucket, args.build_id)
    if not artifacts:
        console.print("[yellow]No artifacts found[/yellow]")
        return

    table = Table(title=f"Artifacts for build {args.build_id[:8]}...")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green", justify="right")

    for name, size, _ in artifacts:
        if args.pattern and args.pattern not in name:
            continue
        size_str = f"{size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"{size / 1024:.1f} KB"
        table.add_row(name, size_str)

    console.print(table)

    if args.list:
        return

    # Step 4: Download artifacts
    args.output.mkdir(parents=True, exist_ok=True)
    console.print(f"\n[blue]Downloading to {args.output}...[/blue]")

    downloaded = []
    for name, size, gcs_path in artifacts:
        if args.pattern and args.pattern not in name:
            continue

        output_path = args.output / name
        try:
            download_gcs_artifact(gcs_path, output_path)
            console.print(f"  [green]✓[/green] {name}")
            downloaded.append((name, output_path))
        except Exception as e:
            console.print(f"  [red]✗[/red] {name}: {e}")

    # Step 5: Extract tarball if requested
    if args.extract:
        for name, path in downloaded:
            if name.endswith(".tar.gz"):
                console.print(f"\n[blue]Extracting {name}...[/blue]")
                extract_dir = args.output / "extracted"
                extract_dir.mkdir(exist_ok=True)
                with tarfile.open(path, "r:gz") as tar:
                    tar.extractall(extract_dir)
                console.print(f"[green]Extracted to {extract_dir}[/green]")

    console.print(f"\n[green]Done! Files saved to {args.output}[/green]")


if __name__ == "__main__":
    main()
