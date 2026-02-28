# /// script
# /// requires-python = ">=3.10"
# /// dependencies = []
# ///
"""Compare Loom and CVC simulation outputs for MCU SoC timing validation.

Reads both VCDs, extracts gpio_out signals at each clock edge, and reports
differences. Handles format differences:
  - Loom: individual 1-bit signals with multi-char VCD codes (e.g., %-)
  - CVC: 44-bit bus gpio_out[43:0] with potential X values

Usage:
    python3 compare_outputs.py <loom.vcd> <cvc.vcd> [--skip-cycles N]
"""

import re
import sys
from pathlib import Path


class VCDResult:
    """Parsed gpio_out changes and metadata from a VCD file."""
    def __init__(self, changes: list[tuple[int, int]], last_time: int):
        self.changes = changes  # [(timestamp, value), ...]
        self.last_time = last_time  # last timestamp in VCD


def parse_loom_vcd(path: Path, width: int = 44) -> VCDResult:
    """Parse Loom VCD and return gpio_out value changes + last timestamp.

    Loom VCD has individual 1-bit signals like:
        $var wire 1 %- gpio_out[1] $end
    """
    code_to_bit: dict[str, int] = {}
    in_header = True
    results: list[tuple[int, int]] = []
    current_time = 0
    last_time = 0
    state = [0] * width

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if in_header:
                if stripped == "$enddefinitions $end":
                    in_header = False
                    continue
                m = re.match(
                    r'\$var\s+wire\s+1\s+(\S+)\s+gpio_out\s*\[(\d+)\]\s+\$end',
                    stripped,
                )
                if m:
                    code_to_bit[m.group(1)] = int(m.group(2))
                continue

            if stripped.startswith('#'):
                current_time = int(stripped[1:])
                last_time = current_time
                continue

            if stripped and stripped[0] in '01xXzZ':
                val_char = stripped[0]
                code = stripped[1:]
                if code in code_to_bit:
                    bit = code_to_bit[code]
                    new_val = 1 if val_char == '1' else 0
                    if state[bit] != new_val:
                        state[bit] = new_val
                        val = sum(state[i] << i for i in range(width))
                        results.append((current_time, val))

    return VCDResult(results, last_time)


def parse_cvc_vcd(path: Path) -> VCDResult:
    """Parse CVC VCD and return gpio_out value changes + last timestamp.

    CVC VCD has 44-bit bus:
        $var wire 44 ! gpio_out [43:0] $end
    X values are treated as 0.
    """
    gpio_out_code = None
    in_header = True
    results: list[tuple[int, int]] = []
    current_time = 0
    last_time = 0

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if in_header:
                if stripped == "$enddefinitions $end":
                    in_header = False
                    continue
                m = re.match(
                    r'\$var\s+wire\s+44\s+(\S+)\s+gpio_out\s',
                    stripped,
                )
                if m:
                    gpio_out_code = m.group(1)
                continue

            if stripped.startswith('#'):
                current_time = int(stripped[1:])
                last_time = current_time
                continue

            if stripped.startswith('b') and gpio_out_code:
                parts = stripped.split()
                if len(parts) == 2 and parts[1] == gpio_out_code:
                    binary = parts[0][1:]
                    val = 0
                    for ch in binary:
                        val <<= 1
                        if ch == '1':
                            val |= 1
                    results.append((current_time, val))

    return VCDResult(results, last_time)


def main() -> None:
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <loom.vcd> <cvc.vcd> [--skip-cycles N]",
            file=sys.stderr,
        )
        sys.exit(1)

    loom_path = Path(sys.argv[1])
    cvc_path = Path(sys.argv[2])
    skip_cycles = 0
    if "--skip-cycles" in sys.argv:
        idx = sys.argv.index("--skip-cycles")
        skip_cycles = int(sys.argv[idx + 1])

    clock_period = 40000  # ps

    print(f"Parsing Loom VCD: {loom_path}")
    loom_result = parse_loom_vcd(loom_path)
    loom_changes = loom_result.changes
    print(f"  {len(loom_changes)} gpio_out value changes, last_time={loom_result.last_time}ps")

    print(f"Parsing CVC VCD: {cvc_path}")
    cvc_result = parse_cvc_vcd(cvc_path)
    cvc_changes = cvc_result.changes
    print(f"  {len(cvc_changes)} gpio_out value changes, last_time={cvc_result.last_time}ps")

    # Use the minimum of both last timestamps as the comparison range
    max_time = min(loom_result.last_time, cvc_result.last_time)

    # Calculate number of cycles
    num_cycles = max_time // clock_period + 1

    # Build per-cycle values by interpolating from change lists
    def build_cycle_values(
        changes: list[tuple[int, int]], num_cycles: int
    ) -> list[int]:
        """Build value at each rising clock edge from a list of (time, value)."""
        values = [0] * num_cycles
        change_idx = 0
        current_val = 0
        for cycle in range(num_cycles):
            t = cycle * clock_period
            while change_idx < len(changes) and changes[change_idx][0] <= t:
                current_val = changes[change_idx][1]
                change_idx += 1
            values[cycle] = current_val
        return values

    loom_values = build_cycle_values(loom_changes, num_cycles)
    cvc_values = build_cycle_values(cvc_changes, num_cycles)

    print(f"\nComparing {num_cycles} cycles (skip first {skip_cycles}):")
    print(f"  Clock period: {clock_period} ps")

    matches = 0
    mismatches = 0
    first_mismatch = None

    for cycle in range(skip_cycles, num_cycles):
        lv = loom_values[cycle]
        cv = cvc_values[cycle]
        if lv == cv:
            matches += 1
        else:
            mismatches += 1
            if mismatches <= 10:
                diff = lv ^ cv
                diff_bits = [i for i in range(44) if diff & (1 << i)]
                t = cycle * clock_period
                print(f"\n  MISMATCH at cycle {cycle} (t={t}ps):")
                print(f"    Loom: 0x{lv:011x}")
                print(f"    CVC:  0x{cv:011x}")
                print(f"    Diff bits: {diff_bits}")
                if first_mismatch is None:
                    first_mismatch = cycle
            elif mismatches == 11:
                print(f"\n  ... (suppressing further mismatches)")

    total = matches + mismatches
    print(f"\n{'='*60}")
    print(f"Comparison Summary (gpio_out, cycles {skip_cycles}-{num_cycles-1}):")
    print(f"  Cycles compared: {total}")
    if total > 0:
        print(f"  Matches:    {matches} ({100*matches/total:.1f}%)")
        print(f"  Mismatches: {mismatches} ({100*mismatches/total:.1f}%)")
    else:
        print("  No cycles to compare")
    if first_mismatch is not None:
        print(f"  First mismatch at cycle: {first_mismatch}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
