# /// script
# /// requires-python = ">=3.10"
# /// dependencies = []
# ///
"""Strip TIMINGCHECK, INTERCONNECT, and empty DELAY blocks from SDF for CVC.

CVC requires matching specify-block timing checks for every SDF TIMINGCHECK
entry, and has trouble with escaped port names in INTERCONNECT entries.
For initial Loom vs CVC waveform comparison, IOPATH delays suffice.

Strips:
  - TIMINGCHECK blocks (no matching specify checks)
  - INTERCONNECT lines (escaped port names)
  - Empty (DELAY (ABSOLUTE)) blocks left after INTERCONNECT removal
  - The final trailing (CELL ...) at wrapper scope (CVC can't resolve it)

Two-pass approach: first pass strips TIMINGCHECK + INTERCONNECT, second
pass removes empty DELAY blocks and the trailing wrapper-scope CELL.

Usage:
    python3 strip_sdf_checks.py <input.sdf> <output.sdf>
"""

import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.sdf> <output.sdf>", file=sys.stderr)
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # --- Pass 1: Strip TIMINGCHECK blocks and INTERCONNECT lines ---
    in_timingcheck = False
    paren_depth = 0
    timingcheck_removed = 0
    interconnect_removed = 0
    pass1_lines: list[str] = []

    with open(input_path) as fin:
        for line in fin:
            if not in_timingcheck:
                stripped = line.strip()
                if stripped == "(TIMINGCHECK":
                    in_timingcheck = True
                    paren_depth = 1
                    timingcheck_removed += 1
                    continue
                if stripped.startswith("(INTERCONNECT "):
                    interconnect_removed += 1
                    continue
                pass1_lines.append(line)
            else:
                for ch in line:
                    if ch == '(':
                        paren_depth += 1
                    elif ch == ')':
                        paren_depth -= 1
                timingcheck_removed += 1
                if paren_depth <= 0:
                    in_timingcheck = False

    # --- Pass 2: Remove problematic CELL blocks ---
    # Parse CELL blocks by parenthesis depth and remove:
    #   1. CELLs with empty DELAY/ABSOLUTE (no IOPATH entries)
    #   2. CELLs with escaped $ in INSTANCE name (CVC can't resolve them)
    empty_delay_removed = 0
    escaped_inst_removed = 0
    output_lines: list[str] = []
    i = 0
    while i < len(pass1_lines):
        line = pass1_lines[i]
        stripped = line.strip()
        if stripped == "(CELL":
            # Collect the entire CELL block by tracking paren depth
            cell_lines = [line]
            depth = 1
            i += 1
            while i < len(pass1_lines) and depth > 0:
                cell_line = pass1_lines[i]
                cell_lines.append(cell_line)
                for ch in cell_line:
                    if ch == '(':
                        depth += 1
                    elif ch == ')':
                        depth -= 1
                i += 1
            # Decide whether to keep this CELL block
            block = "".join(cell_lines)
            if '(IOPATH' not in block:
                empty_delay_removed += 1
                continue
            inst_match = re.search(r'\(INSTANCE\s+([^)]*)\)', block)
            if inst_match:
                inst_name = inst_match.group(1).strip()
                if '\\$' in inst_name:
                    escaped_inst_removed += 1
                    continue
            output_lines.extend(cell_lines)
        else:
            output_lines.append(line)
            i += 1

    text = "".join(output_lines)

    with open(output_path, 'w') as fout:
        fout.write(text)

    lines_kept = text.count('\n')
    total_removed = timingcheck_removed + interconnect_removed
    print(f"Stripped SDF: ~{lines_kept} lines kept, {total_removed} lines removed (pass 1)")
    print(f"  TIMINGCHECK: {timingcheck_removed} lines")
    print(f"  INTERCONNECT: {interconnect_removed} lines")
    print(f"  Empty DELAY CELLs removed: {empty_delay_removed}")
    print(f"  Escaped-$ INSTANCE CELLs removed: {escaped_inst_removed}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
