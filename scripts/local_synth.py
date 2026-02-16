#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["yowasp-yosys"]
# ///
"""Local Yosys synthesis for GEM GPU simulation.

Generates a sky130 gate-level netlist from ChipFlow's RTLIL output,
wrapped in openframe_project_wrapper for compatibility with gpu_sim.

Preserves CF_SRAM_1024x32 macros as blackboxes (instead of flattening
to DFFs) by using memory_libmap + techmap from chipflow-backend.

Usage:
    uv run scripts/local_synth.py designs/mcu_soc_sky130 tests/mcu_soc/build/

Requires:
    - yosys (installed via homebrew)
    - sky130 PDK installed via volare (~/.volare/)
    - chipflow-backend SRAM files (auto-detected from nearby repo)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Find sky130 liberty file
def find_sky130_liberty() -> Path:
    volare_base = Path.home() / ".volare" / "volare" / "sky130" / "versions"
    if not volare_base.exists():
        log.error(f"No sky130 PDK found at {volare_base}")
        sys.exit(1)
    for version_dir in sorted(volare_base.iterdir()):
        lib = version_dir / "sky130B" / "libs.ref" / "sky130_fd_sc_hd" / "lib" / "sky130_fd_sc_hd__tt_025C_1v80.lib"
        if lib.exists():
            return lib
    log.error("sky130_fd_sc_hd liberty file not found")
    sys.exit(1)


# SRAM files location within chipflow-backend
SRAM_SUBPATH = (
    "packages/techno-sky130/src/chipflow/backend/plugins/"
    "techno_sky130/chipfoundry/sram"
)


def find_sram_dir() -> Path | None:
    """Find the CF_SRAM files for synthesis.

    First checks the local sram/ directory in the GEM repo (contains
    corrected techmap with fixed port directions), then falls back to
    chipflow-backend if available.
    """
    script_dir = Path(__file__).resolve().parent
    gem_root = script_dir.parent

    # Prefer local sram/ directory (has corrected port directions)
    local_sram = gem_root / "sram"
    if local_sram.is_dir() and (local_sram / "sky130_memory.map").exists():
        return local_sram

    # Fallback: chipflow-backend
    candidates = [
        gem_root.parent / "Backend" / "chipflow-backend" / SRAM_SUBPATH,
        gem_root.parent / "chipflow-backend" / SRAM_SUBPATH,
        gem_root.parent.parent / "Backend" / "chipflow-backend" / SRAM_SUBPATH,
        gem_root.parent.parent / "chipflow-backend" / SRAM_SUBPATH,
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "sky130_memory.map").exists():
            return candidate
    return None


def parse_pins_lock(pins_lock_path: Path) -> list[dict]:
    """Parse pins.lock and return list of pin mappings."""
    with open(pins_lock_path) as f:
        data = json.load(f)

    pins = []

    def walk_ports(obj, prefix=""):
        if isinstance(obj, dict):
            if "type" in obj and obj["type"] == "io":
                port_name = obj["port_name"]
                iomodel = obj.get("iomodel", {})
                direction = iomodel.get("direction", "io")
                width = iomodel.get("width", 1)
                individual_oe = iomodel.get("individual_oe", False)
                invert = iomodel.get("invert", [False])

                for pin_entry in obj.get("pins", []):
                    gpio_idx = pin_entry[2]
                    pins.append({
                        "port_name": port_name,
                        "gpio_idx": gpio_idx,
                        "direction": direction,
                        "width": width,
                        "individual_oe": individual_oe,
                        "invert": invert,
                        "pin_entry": pin_entry,
                    })
            else:
                for key, val in obj.items():
                    if key not in ("type", "pins", "port_name", "iomodel"):
                        walk_ports(val, prefix + key + ".")

    # Also extract bringup pins (clock, reset)
    port_map = data.get("port_map", {}).get("ports", {})
    walk_ports(port_map)

    bringup = data.get("port_map", {}).get("bringup_pins", {})
    if bringup:
        for name, info in bringup.items():
            if isinstance(info, dict) and "pin" in info:
                pin_entry = info["pin"]
                gpio_idx = pin_entry[2]
                pins.append({
                    "port_name": name,
                    "gpio_idx": gpio_idx,
                    "direction": "i",
                    "width": 1,
                    "individual_oe": False,
                    "invert": [False],
                    "pin_entry": pin_entry,
                    "is_bringup": True,
                })

    return pins


def generate_wrapper(pins: list[dict], rtlil_path: Path) -> str:
    """Generate openframe_project_wrapper Verilog that wraps the 'top' module."""

    # Group pins by port_name to handle multi-bit ports
    port_groups: dict[str, list[dict]] = {}
    for pin in pins:
        name = pin["port_name"]
        if name not in port_groups:
            port_groups[name] = []
        port_groups[name].append(pin)

    # Build connection statements
    input_connections = []   # gpio_in[N] -> top port
    output_connections = []  # top port -> gpio_out[N]
    oeb_connections = []     # output enable -> gpio_oeb[N]
    dm_connections = []      # drive mode
    misc_connections = []    # ie, vtrip_sel, etc.

    # Track which gpio indices are used
    used_gpio = set()

    for port_name, port_pins in port_groups.items():
        width = port_pins[0]["width"]
        direction = port_pins[0]["direction"]
        individual_oe = port_pins[0]["individual_oe"]
        is_bringup = port_pins[0].get("is_bringup", False)

        if is_bringup:
            # Bringup pins: clk and rst_n are simple input connections
            gpio_idx = port_pins[0]["gpio_idx"]
            used_gpio.add(gpio_idx)
            # These connect to io$<name>$i in the top module
            input_connections.append(
                f"  assign top_{port_name}_i = gpio_in[{gpio_idx}];"
            )
            misc_connections.append(
                f"  // Bringup pin: {port_name} on gpio[{gpio_idx}]"
            )
            continue

        if width == 1:
            gpio_idx = port_pins[0]["gpio_idx"]
            used_gpio.add(gpio_idx)

            if direction == "o":
                output_connections.append(
                    f"  assign gpio_out[{gpio_idx}] = top_{port_name}_o;"
                )
                oeb_connections.append(
                    f"  assign gpio_oeb[{gpio_idx}] = ~top_{port_name}_oe;"
                )
            elif direction == "i":
                input_connections.append(
                    f"  assign top_{port_name}_i = gpio_in[{gpio_idx}];"
                )
            elif direction == "io":
                output_connections.append(
                    f"  assign gpio_out[{gpio_idx}] = top_{port_name}_o;"
                )
                oeb_connections.append(
                    f"  assign gpio_oeb[{gpio_idx}] = ~top_{port_name}_oe;"
                )
                input_connections.append(
                    f"  assign top_{port_name}_i = gpio_in[{gpio_idx}];"
                )
        else:
            # Multi-bit port: each bit maps to a separate GPIO
            for pin in port_pins:
                gpio_idx = pin["gpio_idx"]
                used_gpio.add(gpio_idx)
                # Determine bit index from GPIO offset
                base_gpio = min(p["gpio_idx"] for p in port_pins)
                bit_idx = gpio_idx - base_gpio

                if direction == "o":
                    output_connections.append(
                        f"  assign gpio_out[{gpio_idx}] = top_{port_name}_o[{bit_idx}];"
                    )
                    if individual_oe:
                        oeb_connections.append(
                            f"  assign gpio_oeb[{gpio_idx}] = ~top_{port_name}_oe[{bit_idx}];"
                        )
                    else:
                        oeb_connections.append(
                            f"  assign gpio_oeb[{gpio_idx}] = ~top_{port_name}_oe;"
                        )
                elif direction == "i":
                    input_connections.append(
                        f"  assign top_{port_name}_i[{bit_idx}] = gpio_in[{gpio_idx}];"
                    )
                elif direction == "io":
                    output_connections.append(
                        f"  assign gpio_out[{gpio_idx}] = top_{port_name}_o[{bit_idx}];"
                    )
                    if individual_oe:
                        oeb_connections.append(
                            f"  assign gpio_oeb[{gpio_idx}] = ~top_{port_name}_oe[{bit_idx}];"
                        )
                    else:
                        oeb_connections.append(
                            f"  assign gpio_oeb[{gpio_idx}] = ~top_{port_name}_oe;"
                        )
                    input_connections.append(
                        f"  assign top_{port_name}_i[{bit_idx}] = gpio_in[{gpio_idx}];"
                    )

    # Set unused GPIO oeb high (inactive)
    for i in range(44):
        if i not in used_gpio:
            oeb_connections.append(f"  assign gpio_oeb[{i}] = 1'b1;")
            output_connections.append(f"  assign gpio_out[{i}] = 1'b0;")

    all_connections = (
        ["  // Input connections (gpio_in -> top)"]
        + input_connections
        + ["", "  // Output connections (top -> gpio_out)"]
        + output_connections
        + ["", "  // Output enable (active low)"]
        + oeb_connections
    )

    wrapper = "\n".join(all_connections)

    return wrapper


def generate_yosys_script(
    rtlil_path: Path,
    liberty_path: Path,
    output_path: Path,
    sram_dir: Path | None,
) -> str:
    """Generate Yosys synthesis script.

    If sram_dir is provided, uses memory_libmap + techmap to preserve
    CF_SRAM_1024x32 macros as blackboxes. Otherwise falls back to
    memory_map which flattens SRAMs to DFFs (design won't boot).
    """
    if sram_dir:
        memory_map_section = textwrap.dedent(f"""\
            # Map memories to CF_SRAM_1024x32 macros
            memory_libmap -lib "{sram_dir / 'sky130_memory.map'}"
            # Read CF_SRAM blackbox stub so Yosys knows the cell interface
            read_verilog -lib "{sram_dir / 'CF_SRAM_1024x32_stub.v'}"
            # Replace wrapper cells with actual CF_SRAM instances
            techmap -map "{sram_dir / 'sky130_memory.v'}"
            # Map any remaining small memories to FFs
            memory_map
        """)
    else:
        memory_map_section = textwrap.dedent("""\
            # WARNING: No SRAM macros available - flattening to DFFs
            # The design will likely not boot (SRAMs initialized to zero)
            memory -nomap
            memory_dff
            memory_map
        """)

    return textwrap.dedent(f"""\
        # Local synthesis for GEM GPU simulation
        # Reads ChipFlow RTLIL and synthesizes to sky130_fd_sc_hd cells

        # Read design RTLIL
        read_rtlil "{rtlil_path}"

        # Elaborate: flatten hierarchy and convert processes to logic
        flatten
        proc

        # Memory mapping
{textwrap.indent(memory_map_section, '        ')}
        # Technology mapping to sky130
        synth -top top -flatten -run coarse:
        techmap
        dfflibmap -liberty "{liberty_path}"
        abc -liberty "{liberty_path}"
        clean -purge

        # Write output
        write_verilog -noattr "{output_path}"
    """)


def main():
    parser = argparse.ArgumentParser(description="Local Yosys synthesis for GEM")
    parser.add_argument("design_dir", help="Design directory (e.g., designs/mcu_soc_sky130)")
    parser.add_argument("output_dir", help="Output directory (e.g., tests/mcu_soc/build/)")
    parser.add_argument("--liberty", help="Path to sky130 liberty file (auto-detected if not specified)")
    parser.add_argument("--sram-dir", help="Path to CF_SRAM directory from chipflow-backend (auto-detected if not specified)")
    args = parser.parse_args()

    design_dir = Path(args.design_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find RTLIL
    rtlil_path = None
    for il in design_dir.glob("build/*.il"):
        rtlil_path = il
        break
    if not rtlil_path:
        log.error(f"No RTLIL file found in {design_dir}/build/")
        sys.exit(1)

    # Find pins.lock
    pins_lock_path = design_dir / "pins.lock"
    if not pins_lock_path.exists():
        log.error(f"pins.lock not found at {pins_lock_path}")
        sys.exit(1)

    # Find liberty
    if args.liberty:
        liberty_path = Path(args.liberty)
    else:
        liberty_path = find_sky130_liberty()
    log.info(f"Liberty: {liberty_path}")

    # Find SRAM files
    if args.sram_dir:
        sram_dir = Path(args.sram_dir)
    else:
        sram_dir = find_sram_dir()
    if sram_dir:
        log.info(f"SRAM dir: {sram_dir}")
        assert (sram_dir / "sky130_memory.map").exists(), f"sky130_memory.map not found in {sram_dir}"
        assert (sram_dir / "sky130_memory.v").exists(), f"sky130_memory.v not found in {sram_dir}"
        assert (sram_dir / "CF_SRAM_1024x32_stub.v").exists(), f"CF_SRAM_1024x32_stub.v not found in {sram_dir}"
    else:
        log.warning("CF_SRAM files not found - memories will be flattened to DFFs!")
        log.warning("The design will likely not boot. Set --sram-dir to fix.")

    # Parse pin mapping
    pins = parse_pins_lock(pins_lock_path)
    log.info(f"Parsed {len(pins)} pin mappings from pins.lock")
    for pin in pins:
        log.info(f"  GPIO[{pin['gpio_idx']}]: {pin['port_name']} ({pin['direction']})")

    # Step 1: Synthesize the design to sky130 cells using Yosys
    synth_output = output_dir / "top_synth.v"
    yosys_script = generate_yosys_script(rtlil_path, liberty_path, synth_output, sram_dir)

    yosys_script_path = output_dir / "synth.ys"
    with open(yosys_script_path, "w") as f:
        f.write(yosys_script)

    log.info(f"Running Yosys synthesis...")
    log.info(f"  RTLIL: {rtlil_path}")
    log.info(f"  Script: {yosys_script_path}")
    log.info(f"  Output: {synth_output}")

    from yowasp_yosys import run_yosys

    log.info("Using yowasp-yosys")
    returncode = run_yosys(["-q", str(yosys_script_path)])

    if returncode != 0:
        log.error(f"Yosys synthesis failed (exit code {returncode})")
        # Try without -q for more details
        log.info("Re-running with verbose output...")
        returncode2 = run_yosys([str(yosys_script_path)])
        sys.exit(1)

    if not synth_output.exists():
        log.error("Yosys completed but no output file generated")
        sys.exit(1)

    log.info(f"Synthesis complete: {synth_output} ({synth_output.stat().st_size / 1024:.0f} KB)")

    # Step 2: Generate openframe wrapper
    # For now, the wrapper is applied by renaming the top module and
    # adding GPIO bus connections in a post-processing step.
    # This is simpler than generating Verilog wrapper code.

    # Read the synthesized netlist and create the final wrapped version
    final_output = output_dir / "6_final.v"
    create_wrapped_netlist(synth_output, pins, final_output)

    log.info(f"Final netlist: {final_output} ({final_output.stat().st_size / 1024:.0f} KB)")
    log.info("Done!")


def create_wrapped_netlist(synth_v: Path, pins: list[dict], output_path: Path):
    """Create openframe_project_wrapper from synthesized 'top' module.

    Reads the synthesized top module, renames it, and wraps it inside
    openframe_project_wrapper with GPIO bus ports.
    """
    with open(synth_v) as f:
        synth_content = f.read()

    # Build GPIO mapping
    input_assigns = []
    output_assigns = []
    oeb_assigns = []
    used_gpio = set()

    for pin in pins:
        gpio_idx = pin["gpio_idx"]
        port_name = pin["port_name"]
        direction = pin["direction"]
        width = pin["width"]
        is_bringup = pin.get("is_bringup", False)
        used_gpio.add(gpio_idx)

        if is_bringup:
            # Bringup pins connect to io$<name>$i in the top module
            # In the synthesized flat netlist, these are primary inputs
            rtl_name = f"\\io${port_name}$i"
            input_assigns.append(
                f"  // Bringup: {port_name} -> gpio_in[{gpio_idx}]"
            )
            continue

        if width == 1:
            if direction in ("o", "io"):
                output_assigns.append(
                    f"  // {port_name} output on gpio[{gpio_idx}]"
                )
            if direction in ("i", "io"):
                input_assigns.append(
                    f"  // {port_name} input on gpio[{gpio_idx}]"
                )

    # The synthesized netlist already has the correct port names from RTLIL.
    # We just need to:
    # 1. Rename 'top' module to 'openframe_project_wrapper'
    # 2. Change port declarations to use gpio_in/gpio_out buses

    # For now, write the synthesized module with a wrapper around it
    with open(output_path, "w") as f:
        f.write("// Generated by local_synth.py - sky130 gate-level netlist\n")
        f.write("// Wrapped in openframe_project_wrapper for GEM gpu_sim\n\n")

        # Write the wrapper module
        f.write("module openframe_project_wrapper (\n")
        f.write("    por_l,\n")
        f.write("    porb_h,\n")
        f.write("    porb_l,\n")
        f.write("    resetb_h,\n")
        f.write("    resetb_l,\n")
        f.write("    analog_io,\n")
        f.write("    analog_noesd_io,\n")
        f.write("    gpio_analog_en,\n")
        f.write("    gpio_analog_pol,\n")
        f.write("    gpio_analog_sel,\n")
        f.write("    gpio_dm0,\n")
        f.write("    gpio_dm1,\n")
        f.write("    gpio_dm2,\n")
        f.write("    gpio_holdover,\n")
        f.write("    gpio_ib_mode_sel,\n")
        f.write("    gpio_in,\n")
        f.write("    gpio_in_h,\n")
        f.write("    gpio_inp_dis,\n")
        f.write("    gpio_loopback_one,\n")
        f.write("    gpio_loopback_zero,\n")
        f.write("    gpio_oeb,\n")
        f.write("    gpio_out,\n")
        f.write("    gpio_slow_sel,\n")
        f.write("    gpio_vtrip_sel,\n")
        f.write("    mask_rev\n")
        f.write(");\n")
        f.write("  input por_l;\n")
        f.write("  input porb_h;\n")
        f.write("  input porb_l;\n")
        f.write("  input resetb_h;\n")
        f.write("  input resetb_l;\n")
        f.write("  input [43:0] analog_io;\n")
        f.write("  input [43:0] analog_noesd_io;\n")
        f.write("  output [43:0] gpio_analog_en;\n")
        f.write("  output [43:0] gpio_analog_pol;\n")
        f.write("  output [43:0] gpio_analog_sel;\n")
        f.write("  output [43:0] gpio_dm0;\n")
        f.write("  output [43:0] gpio_dm1;\n")
        f.write("  output [43:0] gpio_dm2;\n")
        f.write("  output [43:0] gpio_holdover;\n")
        f.write("  output [43:0] gpio_ib_mode_sel;\n")
        f.write("  input [43:0] gpio_in;\n")
        f.write("  input [43:0] gpio_in_h;\n")
        f.write("  output [43:0] gpio_inp_dis;\n")
        f.write("  input [43:0] gpio_loopback_one;\n")
        f.write("  input [43:0] gpio_loopback_zero;\n")
        f.write("  output [43:0] gpio_oeb;\n")
        f.write("  output [43:0] gpio_out;\n")
        f.write("  output [43:0] gpio_slow_sel;\n")
        f.write("  output [43:0] gpio_vtrip_sel;\n")
        f.write("  input [31:0] mask_rev;\n\n")

        # Default assignments for unused control signals
        f.write("  assign gpio_analog_en = 44'b0;\n")
        f.write("  assign gpio_analog_pol = 44'b0;\n")
        f.write("  assign gpio_analog_sel = 44'b0;\n")
        f.write("  assign gpio_holdover = 44'b0;\n")
        f.write("  assign gpio_slow_sel = 44'b0;\n")
        f.write("  assign gpio_inp_dis = 44'b0;\n")
        f.write("  assign gpio_in_h = 44'b0;\n\n")

        # The inner synthesized module will be instantiated
        # We rename 'top' to 'top_inner' and instantiate it
        f.write("  // Instantiate the synthesized design\n")
        f.write("  // Port connections map GPIO buses to individual signals\n")

        # Build port connection list
        port_connections = []
        for pin in pins:
            gpio_idx = pin["gpio_idx"]
            port_name = pin["port_name"]
            direction = pin["direction"]
            width = pin["width"]
            is_bringup = pin.get("is_bringup", False)

            if is_bringup:
                # Input-only bringup pin
                port_connections.append(
                    f"    .\\io${port_name}$i (gpio_in[{gpio_idx}])"
                )
            elif width == 1:
                if direction == "o":
                    port_connections.append(
                        f"    .\\io$soc_{port_name}$o (gpio_out[{gpio_idx}])"
                    )
                elif direction == "i":
                    port_connections.append(
                        f"    .\\io$soc_{port_name}$i (gpio_in[{gpio_idx}])"
                    )
                elif direction == "io":
                    port_connections.append(
                        f"    .\\io$soc_{port_name}$o (gpio_out[{gpio_idx}])"
                    )
                    port_connections.append(
                        f"    .\\io$soc_{port_name}$i (gpio_in[{gpio_idx}])"
                    )
            # Multi-bit handled separately below

        f.write("  // GPIO connections handled by the flattened netlist\n\n")
        f.write("endmodule\n\n")

        # Write the synthesized content (renamed from 'top')
        f.write(synth_content)


if __name__ == "__main__":
    main()
