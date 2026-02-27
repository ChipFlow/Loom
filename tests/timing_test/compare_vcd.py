# /// script
# /// requires-python = ">=3.10"
# /// dependencies = []
# ///
"""Compare VCD files from two simulators for timing correctness.

Parses two VCD files and compares signal transitions at matching timestamps.
Reports timing differences with configurable tolerance.

Usage:
    uv run compare_vcd.py <reference.vcd> <candidate.vcd> [options]
    uv run compare_vcd.py --timing-stdout <ref_stdout> <cand_stdout> [options]

In --timing-stdout mode, parses RESULT: lines from simulator stdout instead
of VCD files. Expected format:
    RESULT: clk_to_q=350
    RESULT: chain_delay=973
    RESULT: total_delay=1323
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VcdSignal:
    code: str
    name: str
    width: int
    scope: str


@dataclass
class VcdTransition:
    time: int
    signal_code: str
    value: str


@dataclass
class VcdData:
    signals: dict[str, VcdSignal] = field(default_factory=dict)  # code -> signal
    transitions: list[VcdTransition] = field(default_factory=list)
    timescale_ps: int = 1  # timescale in picoseconds


def parse_timescale(line: str) -> int:
    """Convert timescale string to picoseconds."""
    line = line.strip()
    match = re.match(r"(\d+)\s*(s|ms|us|ns|ps|fs)", line)
    if not match:
        return 1
    value = int(match.group(1))
    unit = match.group(2)
    multipliers = {
        "s": 10**12,
        "ms": 10**9,
        "us": 10**6,
        "ns": 10**3,
        "ps": 1,
        "fs": 0,  # sub-ps not supported
    }
    return value * multipliers.get(unit, 1)


def parse_vcd(path: Path) -> VcdData:
    """Parse a VCD file and extract signal definitions and transitions."""
    data = VcdData()
    scope_stack: list[str] = []

    with open(path) as f:
        lines = f.readlines()

    i = 0
    in_header = True
    in_dumpvars = False

    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line:
            continue

        if in_header:
            if line.startswith("$timescale"):
                # May be on same line or next line
                ts_text = line[len("$timescale"):].strip()
                if "$end" in ts_text:
                    ts_text = ts_text[:ts_text.index("$end")].strip()
                else:
                    while i < len(lines):
                        next_line = lines[i].strip()
                        i += 1
                        if "$end" in next_line:
                            ts_text += " " + next_line[:next_line.index("$end")].strip()
                            break
                        ts_text += " " + next_line
                data.timescale_ps = parse_timescale(ts_text)
                continue

            if line.startswith("$scope"):
                parts = line.split()
                if len(parts) >= 3:
                    scope_stack.append(parts[2])
                continue

            if line.startswith("$upscope"):
                if scope_stack:
                    scope_stack.pop()
                continue

            if line.startswith("$var"):
                parts = line.split()
                if len(parts) >= 5:
                    width = int(parts[2])
                    code = parts[3]
                    name = parts[4]
                    scope = ".".join(scope_stack)
                    data.signals[code] = VcdSignal(
                        code=code, name=name, width=width, scope=scope
                    )
                continue

            if line.startswith("$enddefinitions"):
                in_header = False
                continue
            continue

        # Post-header: transitions
        if line.startswith("$dumpvars"):
            in_dumpvars = True
            continue

        if line == "$end":
            in_dumpvars = False
            continue

        if line.startswith("$"):
            continue

        if line.startswith("#"):
            current_time = int(line[1:]) * data.timescale_ps
            continue

        # Value change: 0X, 1X, xX, or bVALUE X
        if line[0] in "01xXzZ" and len(line) >= 2 and not line.startswith("b"):
            value = line[0]
            code = line[1:]
            data.transitions.append(
                VcdTransition(time=current_time, signal_code=code, value=value)
            )
        elif line.startswith("b"):
            parts = line.split()
            if len(parts) == 2:
                value = parts[0]
                code = parts[1]
                data.transitions.append(
                    VcdTransition(time=current_time, signal_code=code, value=value)
                )

    return data


def get_signal_by_name(vcd: VcdData, name: str) -> VcdSignal | None:
    """Find a signal by name (case-insensitive, ignoring scope)."""
    name_lower = name.lower()
    for sig in vcd.signals.values():
        if sig.name.lower() == name_lower:
            return sig
    return None


def get_transitions_for_signal(
    vcd: VcdData, signal: VcdSignal
) -> list[tuple[int, str]]:
    """Get all (time_ps, value) transitions for a signal."""
    return [
        (t.time, t.value)
        for t in vcd.transitions
        if t.signal_code == signal.code
    ]


def compare_signal_transitions(
    ref_transitions: list[tuple[int, str]],
    cand_transitions: list[tuple[int, str]],
    signal_name: str,
    tolerance_ps: int,
) -> list[str]:
    """Compare transitions of a signal between reference and candidate.

    Returns list of difference descriptions. Empty list means match.
    """
    diffs: list[str] = []

    # Filter out x/X/z/Z initial values for comparison
    ref_clean = [(t, v) for t, v in ref_transitions if v in ("0", "1")]
    cand_clean = [(t, v) for t, v in cand_transitions if v in ("0", "1")]

    if not ref_clean and not cand_clean:
        return diffs

    if not ref_clean:
        diffs.append(f"  {signal_name}: reference has no transitions, candidate has {len(cand_clean)}")
        return diffs

    if not cand_clean:
        diffs.append(f"  {signal_name}: candidate has no transitions, reference has {len(ref_clean)}")
        return diffs

    # Compare transition counts
    if len(ref_clean) != len(cand_clean):
        diffs.append(
            f"  {signal_name}: different transition counts: ref={len(ref_clean)}, cand={len(cand_clean)}"
        )

    # Compare matching transitions
    for idx in range(min(len(ref_clean), len(cand_clean))):
        ref_time, ref_val = ref_clean[idx]
        cand_time, cand_val = cand_clean[idx]

        if ref_val != cand_val:
            diffs.append(
                f"  {signal_name}[{idx}]: value mismatch at ref_t={ref_time}ps "
                f"(ref={ref_val}, cand={cand_val})"
            )
        elif abs(ref_time - cand_time) > tolerance_ps:
            diffs.append(
                f"  {signal_name}[{idx}]: timing mismatch: "
                f"ref={ref_time}ps, cand={cand_time}ps, "
                f"diff={cand_time - ref_time}ps (tolerance={tolerance_ps}ps)"
            )

    return diffs


def compare_vcds(
    ref_path: Path,
    cand_path: Path,
    signals: list[str],
    tolerance_ps: int,
) -> tuple[bool, list[str]]:
    """Compare two VCD files on specified signals.

    Returns (pass, messages).
    """
    ref_vcd = parse_vcd(ref_path)
    cand_vcd = parse_vcd(cand_path)

    messages: list[str] = []
    all_pass = True

    for sig_name in signals:
        ref_sig = get_signal_by_name(ref_vcd, sig_name)
        cand_sig = get_signal_by_name(cand_vcd, sig_name)

        if ref_sig is None:
            messages.append(f"WARNING: Signal '{sig_name}' not found in reference VCD")
            continue
        if cand_sig is None:
            messages.append(f"WARNING: Signal '{sig_name}' not found in candidate VCD")
            continue

        ref_trans = get_transitions_for_signal(ref_vcd, ref_sig)
        cand_trans = get_transitions_for_signal(cand_vcd, cand_sig)

        diffs = compare_signal_transitions(ref_trans, cand_trans, sig_name, tolerance_ps)
        if diffs:
            all_pass = False
            messages.append(f"FAIL: {sig_name} has timing differences:")
            messages.extend(diffs)
        else:
            messages.append(f"PASS: {sig_name} matches within {tolerance_ps}ps tolerance")

    return all_pass, messages


def parse_timing_results(text: str) -> dict[str, int]:
    """Parse RESULT: key=value lines from simulator stdout."""
    results: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"RESULT:\s*(\w+)=(\d+)", line)
        if match:
            results[match.group(1)] = int(match.group(2))
    return results


def compare_timing_stdout(
    ref_text: str,
    cand_text: str,
    tolerance_ps: int,
) -> tuple[bool, list[str]]:
    """Compare timing results from simulator stdout.

    Parses RESULT: lines and compares values with tolerance.
    """
    ref_results = parse_timing_results(ref_text)
    cand_results = parse_timing_results(cand_text)

    messages: list[str] = []
    all_pass = True

    if not ref_results:
        messages.append("ERROR: No RESULT: lines found in reference output")
        return False, messages

    if not cand_results:
        messages.append("ERROR: No RESULT: lines found in candidate output")
        return False, messages

    all_keys = sorted(set(ref_results.keys()) | set(cand_results.keys()))
    for key in all_keys:
        if key not in ref_results:
            messages.append(f"WARNING: '{key}' only in candidate ({cand_results[key]}ps)")
            continue
        if key not in cand_results:
            messages.append(f"WARNING: '{key}' only in reference ({ref_results[key]}ps)")
            continue

        ref_val = ref_results[key]
        cand_val = cand_results[key]
        diff = abs(ref_val - cand_val)

        if diff > tolerance_ps:
            all_pass = False
            messages.append(
                f"FAIL: {key}: ref={ref_val}ps, cand={cand_val}ps, "
                f"diff={diff}ps > tolerance={tolerance_ps}ps"
            )
        else:
            messages.append(
                f"PASS: {key}: ref={ref_val}ps, cand={cand_val}ps, "
                f"diff={diff}ps <= tolerance={tolerance_ps}ps"
            )

    return all_pass, messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare VCD files or timing results from two simulators"
    )
    parser.add_argument(
        "file1",
        type=Path,
        help="Reference file (VCD or stdout log)",
    )
    parser.add_argument(
        "file2",
        type=Path,
        help="Candidate file (VCD or stdout log)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=50,
        help="Timing tolerance in picoseconds (default: 50)",
    )
    parser.add_argument(
        "--signals",
        nargs="+",
        default=["Q", "q1"],
        help="Signal names to compare in VCD mode (default: Q q1)",
    )
    parser.add_argument(
        "--timing-stdout",
        action="store_true",
        help="Compare RESULT: lines from stdout instead of VCD signals",
    )

    args = parser.parse_args()

    if args.timing_stdout:
        ref_text = args.file1.read_text()
        cand_text = args.file2.read_text()
        passed, messages = compare_timing_stdout(ref_text, cand_text, args.tolerance)
    else:
        passed, messages = compare_vcds(
            args.file1, args.file2, args.signals, args.tolerance
        )

    for msg in messages:
        print(msg)

    if passed:
        print(f"\nAll comparisons passed (tolerance={args.tolerance}ps)")
        return 0
    else:
        print(f"\nSome comparisons FAILED (tolerance={args.tolerance}ps)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
