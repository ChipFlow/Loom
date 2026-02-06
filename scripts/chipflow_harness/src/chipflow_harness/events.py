"""Event comparison for simulation verification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


@dataclass
class Event:
    """A simulation event."""

    time_ps: int
    peripheral: str
    event_type: str
    payload: Any


def parse_events_json(path: Path) -> list[Event]:
    """Parse events from JSON file."""
    with open(path) as f:
        data = json.load(f)

    events = []
    for item in data.get("events", []):
        events.append(
            Event(
                time_ps=item.get("time_ps", 0),
                peripheral=item.get("peripheral", ""),
                event_type=item.get("type", ""),
                payload=item.get("payload"),
            )
        )
    return events


def compare_events(
    reference: list[Event],
    actual: list[Event],
    tolerance_ps: int = 1000,
) -> tuple[bool, list[str]]:
    """Compare reference events against actual events.

    Returns (success, list of error messages).
    """
    errors: list[str] = []

    ref_idx = 0
    for act in actual:
        if ref_idx >= len(reference):
            break

        ref = reference[ref_idx]

        # Check if events match
        if act.peripheral == ref.peripheral and act.event_type == ref.event_type:
            # Check payload
            if act.payload != ref.payload:
                errors.append(
                    f"Payload mismatch at {act.time_ps}ps: "
                    f"expected {ref.payload}, got {act.payload}"
                )

            # Check timing
            time_diff = abs(act.time_ps - ref.time_ps)
            if time_diff > tolerance_ps:
                errors.append(
                    f"Timing mismatch for {act.peripheral}.{act.event_type}: "
                    f"expected ~{ref.time_ps}ps, got {act.time_ps}ps "
                    f"(diff: {time_diff}ps)"
                )

            ref_idx += 1

    # Check for missing events
    if ref_idx < len(reference):
        missing = reference[ref_idx:]
        for ev in missing:
            errors.append(f"Missing event: {ev.peripheral}.{ev.event_type} = {ev.payload}")

    return len(errors) == 0, errors


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare simulation events")
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to reference events JSON",
    )
    parser.add_argument(
        "--actual",
        type=Path,
        required=True,
        help="Path to actual events JSON from simulation",
    )
    parser.add_argument(
        "--tolerance-ps",
        type=int,
        default=1000,
        help="Timing tolerance in picoseconds (default: 1000)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        reference = parse_events_json(args.reference)
        actual = parse_events_json(args.actual)
    except Exception as e:
        log.error(f"Failed to parse events: {e}")
        return 1

    log.info(f"Reference: {len(reference)} events")
    log.info(f"Actual: {len(actual)} events")

    success, errors = compare_events(reference, actual, args.tolerance_ps)

    if success:
        log.info("All events matched!")
        return 0
    else:
        log.error(f"Found {len(errors)} mismatches:")
        for err in errors:
            log.error(f"  {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
