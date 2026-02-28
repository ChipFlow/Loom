# /// script
# requires-python = ">=3.10"
# ///
"""Generate combined SKY130 cell behavioral Verilog with specify blocks.

Parses a post-PnR netlist to extract all cell types used, then assembles
a single Verilog file containing UDP definitions, base behavioral modules
(with specify blocks merged in), and sized wrapper modules. The output is
suitable for CVC SDF-annotated simulation.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
CELL_LIB = PROJECT_ROOT / "sky130_fd_sc_hd"
CELLS_DIR = CELL_LIB / "cells"
MODELS_DIR = CELL_LIB / "models"
NETLIST = PROJECT_ROOT / "tests" / "mcu_soc" / "data" / "6_final.v"
OUTPUT = SCRIPT_DIR / "sky130_cells.v"

# Cells that are physical-only (no logic): generate empty stubs.
STUB_FAMILIES: set[str] = {"decap", "tapvpwrvgnd", "fill"}

# Regex to extract cell type names from netlist instantiation lines.
CELL_RE = re.compile(r"\bsky130_fd_sc_hd__([a-zA-Z0-9_]+)\b")

# Regex to split a cell type like "inv_1" into family "inv" and size "1".
# Handles multi-segment families like "lpflow_inputiso1p_1" and
# exotic names like "clkdlybuf4s15_2" or "a2111oi_0".
# Strategy: the size is always the last `_<digits>` segment.
SIZE_RE = re.compile(r"^(.+)_(\d+)$")

# Lines to strip from source files.
STRIP_LINE_RES = [
    re.compile(r"^\s*`ifndef\b"),
    re.compile(r"^\s*`define\b"),
    re.compile(r"^\s*`endif\b"),
    re.compile(r"^\s*`timescale\b"),
    re.compile(r"^\s*`default_nettype\b"),
    re.compile(r"^\s*`include\b"),
    re.compile(r"^\s*`celldefine\b"),
    re.compile(r"^\s*`endcelldefine\b"),
]

# Regex to find include directives and extract the UDP name.
INCLUDE_RE = re.compile(
    r'`include\s+"(?:\.\./)*models/([^/]+)/sky130_fd_sc_hd__([^"]+)\.v"'
)


def should_strip(line: str) -> bool:
    """Return True if this line should be removed during cleaning."""
    return any(r.match(line) for r in STRIP_LINE_RES)


def clean_source(text: str) -> str:
    """Remove include guards, timescale, default_nettype, celldefine, etc."""
    lines = text.splitlines()
    cleaned = [line for line in lines if not should_strip(line)]
    return "\n".join(cleaned)


def extract_include_udps(text: str) -> list[str]:
    """Extract UDP directory names from include directives in source."""
    return [m.group(1) for m in INCLUDE_RE.finditer(text)]


def extract_non_power_wrapper(text: str) -> str | None:
    """Extract the non-USE_POWER_PINS version of a sized wrapper.

    The sized wrapper files have this structure:
        `ifdef USE_POWER_PINS
        ... power-pin version ...
        `else
        ... non-power version (uses supply1/supply0) ...
        `endif

    We want the `else` branch, cleaned of directives.
    """
    # Find the `else branch
    else_match = re.search(r"`else\s*//.*?$", text, re.MULTILINE)
    endif_match = re.search(r"`endif\s*//\s*USE_POWER_PINS", text, re.MULTILINE)

    if else_match and endif_match:
        block = text[else_match.end() : endif_match.start()]
        return clean_source(block).strip()

    # If no ifdef/else/endif structure, just clean the whole thing
    log.warning("No USE_POWER_PINS ifdef structure found, using full file")
    return clean_source(text).strip()


def extract_specify_block(text: str) -> str | None:
    """Extract the specify...endspecify block from a specify.v file.

    Returns the full block including specify/endspecify keywords, or None.
    Strips timing check statements ($setuphold, $recrem, $width, $removal,
    $recovery, $hold, $setup) that reference condition signals (AWAKE, COND0,
    etc.) which may not exist in the non-power-pins behavioral model.
    We only need path delay statements for SDF IOPATH annotation.
    """
    # The specify.v files have a license header then specify...endspecify.
    match = re.search(r"^(specify\b.*?^endspecify)\b", text, re.MULTILINE | re.DOTALL)
    if not match:
        return None

    block = match.group(1)
    # Strip timing check lines that reference undefined condition signals.
    # Keep only path delay statements (lines with => or *>).
    timing_check_re = re.compile(
        r"^\s*\$(setuphold|recrem|width|removal|recovery|hold|setup)\b.*$",
        re.MULTILINE,
    )
    block = timing_check_re.sub("", block)

    # Remove blank lines that result from stripping
    lines = [l for l in block.splitlines() if l.strip()]
    return "\n".join(lines)


def wire_delayed_signals(text: str) -> str:
    """Add 'assign X_delayed = X;' for all wire X_delayed declarations.

    SKY130 behavioral models declare wires like D_delayed, CLK_delayed,
    RESET_B_delayed that are intended to be driven by specify-block internal
    paths. Without these connections, the delayed wires stay X and prevent
    any logic from evaluating.

    This function finds all 'wire X_delayed;' declarations and adds
    'assign X_delayed = X;' right after them, connecting them directly
    to the corresponding input port.
    """
    delayed_re = re.compile(r"^(\s*)wire\s+(\w+_delayed)\s*;", re.MULTILINE)
    matches = list(delayed_re.finditer(text))
    if not matches:
        return text

    # Build assign statements and insert after the wire declarations
    for m in reversed(matches):
        indent = m.group(1)
        delayed_name = m.group(2)
        # Strip _delayed suffix to get the original signal name
        original = delayed_name.removesuffix("_delayed")
        assign_line = f"\n{indent}assign {delayed_name} = {original};"
        text = text[:m.end()] + assign_line + text[m.end():]

    return text


def merge_specify_into_behavioral(behavioral: str, specify: str) -> str:
    """Insert a specify block just before endmodule in the behavioral code."""
    # Find the last endmodule
    idx = behavioral.rfind("endmodule")
    assert idx >= 0, "No endmodule found in behavioral model"

    # Insert specify block with proper indentation
    indented_specify = "\n".join(
        f"    {line}" if line.strip() else "" for line in specify.splitlines()
    )

    return behavioral[:idx] + "\n" + indented_specify + "\n\n" + behavioral[idx:]


def read_udp(udp_dir_name: str) -> str | None:
    """Read a UDP primitive definition, extracting just the primitive block.

    UDP files have:
        `ifdef NO_PRIMITIVES
        `include "blackbox..."
        `else
        primitive ... endprimitive
        `endif

    We want the primitive...endprimitive block only.
    """
    udp_dir = MODELS_DIR / udp_dir_name
    # Find the .v file (not .blackbox.v, .tb.v, etc.)
    candidates = list(udp_dir.glob("sky130_fd_sc_hd__*.v"))
    main_v = [
        f
        for f in candidates
        if not any(
            suffix in f.name for suffix in [".blackbox.", ".tb.", ".symbol."]
        )
    ]
    if not main_v:
        log.error("No main .v file found for UDP %s", udp_dir_name)
        return None

    text = main_v[0].read_text()

    # Extract the primitive block (skip the NO_PRIMITIVES/blackbox branch)
    match = re.search(
        r"^(primitive\b.*?^endprimitive)\b", text, re.MULTILINE | re.DOTALL
    )
    if not match:
        log.error("No primitive block found in %s", main_v[0])
        return None

    udp_text = match.group(1)

    # Initialize DFF UDPs to 0 (matching Loom's 0-initialization).
    # Without this, CVC starts all DFF outputs at X and X-propagation
    # prevents the design from functioning during comparison.
    udp_text = re.sub(
        r"(\n\s*reg Q;)",
        r"\1\n    initial Q = 1'b0;",
        udp_text,
    )

    return udp_text


def parse_netlist_cell_types(netlist_path: Path) -> set[str]:
    """Extract all unique sky130_fd_sc_hd__* cell type names from the netlist.

    Returns the suffix after `sky130_fd_sc_hd__`, e.g. "inv_1", "dfxtp_2".
    Only matches cell instantiation lines (cell_type instance_name (...)).
    """
    cell_types: set[str] = set()
    with open(netlist_path) as f:
        for line in f:
            for m in CELL_RE.finditer(line):
                cell_types.add(m.group(1))
    return cell_types


def split_family_size(cell_type: str) -> tuple[str, str | None]:
    """Split 'inv_1' into ('inv', '1'), 'clkdlybuf4s15_2' into ('clkdlybuf4s15', '2').

    For cells that might not follow the pattern, returns (cell_type, None).
    We verify the family directory exists before accepting the split.
    """
    m = SIZE_RE.match(cell_type)
    if m:
        family_candidate = m.group(1)
        size = m.group(2)
        # Verify this family directory actually exists
        if (CELLS_DIR / family_candidate).is_dir():
            return family_candidate, size
        # Try without the split (the whole thing might be the family name)

    # No size suffix, or the directory doesn't exist with the split
    if (CELLS_DIR / cell_type).is_dir():
        return cell_type, None

    log.warning("Cannot find cell directory for %s", cell_type)
    return cell_type, None


def generate_stub_module(module_name: str, ports: str = "") -> str:
    """Generate an empty stub module for physical-only cells."""
    if ports:
        return (
            f"module {module_name} (\n"
            f"{ports}\n"
            f");\n"
            f"endmodule\n"
        )
    return f"module {module_name} ();\nendmodule\n"


def main() -> int:
    assert NETLIST.exists(), f"Netlist not found: {NETLIST}"
    assert CELLS_DIR.is_dir(), f"Cell library not found: {CELLS_DIR}"
    assert MODELS_DIR.is_dir(), f"Models directory not found: {MODELS_DIR}"

    # ── Step 1: Parse netlist for cell types ─────────────────────────
    log.info("Parsing netlist: %s", NETLIST)
    cell_types = parse_netlist_cell_types(NETLIST)
    log.info("Found %d unique cell types", len(cell_types))

    # ── Step 2: Split into families and sizes ────────────────────────
    families: dict[str, set[str]] = {}  # family -> set of sizes
    unsized_cells: set[str] = set()  # families used without size suffix
    unresolved: set[str] = set()

    for ct in sorted(cell_types):
        family, size = split_family_size(ct)
        if not (CELLS_DIR / family).is_dir():
            unresolved.add(ct)
            continue
        if family not in families:
            families[family] = set()
        if size is not None:
            families[family].add(size)
        else:
            unsized_cells.add(family)

    if unresolved:
        log.warning("Unresolved cell types (no directory): %s", sorted(unresolved))

    log.info("Found %d cell families", len(families))
    for family in sorted(families):
        sizes = sorted(families[family])
        if sizes:
            log.info("  %s: sizes %s", family, ", ".join(sizes))
        else:
            log.info("  %s: (unsized)", family)

    # ── Step 3: Read behavioral models and specify blocks ────────────
    udp_names: list[str] = []  # ordered, unique UDP dir names
    udp_set: set[str] = set()
    base_modules: list[str] = []  # cleaned behavioral + specify merged
    sized_wrappers: list[str] = []  # cleaned sized wrappers
    stub_modules: list[str] = []  # stubs for physical-only cells

    for family in sorted(families):
        family_dir = CELLS_DIR / family

        # Check if this is a stub-only family
        if family in STUB_FAMILIES:
            # Generate stubs for the base module and each sized wrapper
            behavioral_path = family_dir / f"sky130_fd_sc_hd__{family}.behavioral.v"
            if behavioral_path.exists():
                # Use the actual behavioral model (it is already empty)
                raw = behavioral_path.read_text()
                cleaned = clean_source(raw).strip()
                # Remove comments that are just copyright
                stub_modules.append(cleaned)
            else:
                stub_modules.append(
                    generate_stub_module(f"sky130_fd_sc_hd__{family}")
                )

            for size in sorted(families[family]):
                sized_path = (
                    family_dir / f"sky130_fd_sc_hd__{family}_{size}.v"
                )
                if sized_path.exists():
                    raw = sized_path.read_text()
                    wrapper = extract_non_power_wrapper(raw)
                    if wrapper:
                        sized_wrappers.append(wrapper)
                    else:
                        log.warning(
                            "Could not extract wrapper for %s_%s, generating stub",
                            family, size,
                        )
                        stub_modules.append(
                            generate_stub_module(
                                f"sky130_fd_sc_hd__{family}_{size}"
                            )
                        )
                else:
                    log.warning(
                        "No sized wrapper for %s_%s, generating stub",
                        family, size,
                    )
                    stub_modules.append(
                        generate_stub_module(f"sky130_fd_sc_hd__{family}_{size}")
                    )
            continue

        # ── Read behavioral model ────────────────────────────────
        behavioral_path = family_dir / f"sky130_fd_sc_hd__{family}.behavioral.v"
        if not behavioral_path.exists():
            log.warning(
                "No behavioral model for family %s, generating stub", family
            )
            stub_modules.append(
                generate_stub_module(f"sky130_fd_sc_hd__{family}")
            )
            # Also stub the sized wrappers
            for size in sorted(families[family]):
                stub_modules.append(
                    generate_stub_module(f"sky130_fd_sc_hd__{family}_{size}")
                )
            continue

        raw_behavioral = behavioral_path.read_text()

        # Collect UDP includes before cleaning
        for udp_dir in extract_include_udps(raw_behavioral):
            if udp_dir not in udp_set:
                udp_set.add(udp_dir)
                udp_names.append(udp_dir)

        cleaned_behavioral = clean_source(raw_behavioral).strip()
        cleaned_behavioral = wire_delayed_signals(cleaned_behavioral)

        # ── Read specify block (if exists) ───────────────────────
        specify_path = family_dir / f"sky130_fd_sc_hd__{family}.specify.v"
        if specify_path.exists():
            raw_specify = specify_path.read_text()
            specify_block = extract_specify_block(raw_specify)
            if specify_block:
                cleaned_behavioral = merge_specify_into_behavioral(
                    cleaned_behavioral, specify_block
                )
            else:
                log.warning("Could not extract specify block for %s", family)

        base_modules.append(cleaned_behavioral)

        # ── Read specify block for this family (to insert into wrappers) ──
        family_specify: str | None = None
        specify_path = family_dir / f"sky130_fd_sc_hd__{family}.specify.v"
        if specify_path.exists():
            raw_specify = specify_path.read_text()
            family_specify = extract_specify_block(raw_specify)

        # ── Read sized wrappers ──────────────────────────────────
        for size in sorted(families[family]):
            sized_path = family_dir / f"sky130_fd_sc_hd__{family}_{size}.v"
            if not sized_path.exists():
                log.warning("Missing sized wrapper: %s_%s", family, size)
                continue

            raw_wrapper = sized_path.read_text()
            wrapper = extract_non_power_wrapper(raw_wrapper)
            if wrapper:
                # CVC needs specify blocks in the sized wrapper (not just
                # the base module) for SDF IOPATH matching.
                if family_specify:
                    wrapper = merge_specify_into_behavioral(
                        wrapper, family_specify
                    )
                sized_wrappers.append(wrapper)
            else:
                log.error("Failed to extract wrapper for %s_%s", family, size)

    # ── Step 4: Read UDP definitions ─────────────────────────────────
    udp_blocks: list[str] = []
    for udp_dir_name in udp_names:
        udp_text = read_udp(udp_dir_name)
        if udp_text:
            udp_blocks.append(udp_text)
            log.info("Read UDP: %s", udp_dir_name)
        else:
            log.error("Failed to read UDP: %s", udp_dir_name)

    # ── Step 5: Write combined output ────────────────────────────────
    log.info("Writing output: %s", OUTPUT)

    with open(OUTPUT, "w") as f:
        f.write("// Auto-generated SKY130 cell models for CVC simulation\n")
        f.write("// Generated by gen_cell_models.py\n")
        f.write(f"// Cell families: {len(families)}\n")
        f.write(f"// Sized wrappers: {len(sized_wrappers)}\n")
        f.write(f"// UDP primitives: {len(udp_blocks)}\n")
        f.write("//\n")
        f.write("// DO NOT EDIT - regenerate with:\n")
        f.write("//   uv run tests/mcu_soc/cvc/gen_cell_models.py\n")
        f.write("\n")
        f.write("`timescale 1ps/1ps\n")
        f.write("\n")

        # UDP primitives first (must be defined before modules that use them)
        if udp_blocks:
            f.write(
                "// "
                + "=" * 70
                + "\n"
            )
            f.write("// UDP Primitive Definitions\n")
            f.write(
                "// "
                + "=" * 70
                + "\n\n"
            )
            for block in udp_blocks:
                f.write(block)
                f.write("\n\n")

        # Base behavioral modules (with specify blocks)
        if base_modules:
            f.write(
                "// "
                + "=" * 70
                + "\n"
            )
            f.write("// Base Behavioral Modules (with specify blocks)\n")
            f.write(
                "// "
                + "=" * 70
                + "\n\n"
            )
            for module in base_modules:
                f.write(module)
                f.write("\n\n")

        # Stub modules for physical-only cells
        if stub_modules:
            f.write(
                "// "
                + "=" * 70
                + "\n"
            )
            f.write("// Stub Modules (physical-only cells)\n")
            f.write(
                "// "
                + "=" * 70
                + "\n\n"
            )
            for module in stub_modules:
                f.write(module)
                f.write("\n\n")

        # Sized wrapper modules
        if sized_wrappers:
            f.write(
                "// "
                + "=" * 70
                + "\n"
            )
            f.write("// Sized Wrapper Modules\n")
            f.write(
                "// "
                + "=" * 70
                + "\n\n"
            )
            for module in sized_wrappers:
                f.write(module)
                f.write("\n\n")

    output_size = OUTPUT.stat().st_size
    log.info(
        "Done! Output: %s (%d bytes, %.1f KB)",
        OUTPUT,
        output_size,
        output_size / 1024,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
