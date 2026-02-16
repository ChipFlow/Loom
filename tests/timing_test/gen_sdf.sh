#!/usr/bin/env bash
# Generate SDF timing file for the minimal_build design using OpenSTA (via OpenROAD docker).
#
# This produces gate-only timing (no parasitic wire delays since we don't have
# a SPEF file from routing). Cell IOPATH delays come from the Liberty .lib.
#
# Usage:
#   bash tests/timing_test/gen_sdf.sh [corner]
#
# Corners: tt (default), ss, ff
#   tt = sky130_fd_sc_hd__tt_025C_1v80.lib  (typical)
#   ss = sky130_fd_sc_hd__ss_100C_1v60.lib  (slow-slow)
#   ff = sky130_fd_sc_hd__ff_n40C_1v95.lib  (fast-fast)
#
# Output: tests/timing_test/minimal_build/6_final.sdf
#
# Requires: docker with openroad/ubuntu22.04-builder-gcc:e9e8ef image

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/minimal_build"

CORNER="${1:-tt}"

# Map corner name to liberty filename
case "$CORNER" in
    tt) LIB_FILE="sky130_fd_sc_hd__tt_025C_1v80.lib" ;;
    ss) LIB_FILE="sky130_fd_sc_hd__ss_100C_1v60.lib" ;;
    ff) LIB_FILE="sky130_fd_sc_hd__ff_n40C_1v95.lib" ;;
    *)  echo "Unknown corner: $CORNER (use tt, ss, or ff)"; exit 1 ;;
esac

VOLARE_BASE="$HOME/.volare/volare/sky130/versions/c6d73a35f524070e85faff4a6a9eef49553ebc2b"
LIB_PATH="$VOLARE_BASE/sky130B/libs.ref/sky130_fd_sc_hd/lib/$LIB_FILE"

if [ ! -f "$LIB_PATH" ]; then
    echo "Error: Liberty file not found: $LIB_PATH"
    echo "Install sky130 PDK via volare: pip install volare && volare enable c6d73a35"
    exit 1
fi

if [ ! -f "$BUILD_DIR/6_final.v" ]; then
    echo "Error: Netlist not found: $BUILD_DIR/6_final.v"
    exit 1
fi

DOCKER_IMAGE="openroad/ubuntu22.04-builder-gcc:e9e8ef"

# Check docker image exists
if ! docker image inspect "$DOCKER_IMAGE" > /dev/null 2>&1; then
    echo "Error: Docker image $DOCKER_IMAGE not found."
    echo "Pull it with: docker pull $DOCKER_IMAGE"
    exit 1
fi

OUTPUT_SDF="$BUILD_DIR/6_final.sdf"

echo "=== Generating SDF for minimal_build ==="
echo "Corner:  $CORNER ($LIB_FILE)"
echo "Netlist: $BUILD_DIR/6_final.v"
echo "Liberty: $LIB_PATH"
echo "Output:  $OUTPUT_SDF"
echo ""

# Create OpenSTA TCL script
STA_SCRIPT=$(mktemp /tmp/gen_sdf_XXXXXX.tcl)
cat > "$STA_SCRIPT" << 'EOF'
# OpenSTA script to generate SDF from gate-level netlist + Liberty
# Runs inside docker container with paths mounted at /work and /pdk

read_liberty /pdk/lib.lib
read_verilog /work/6_final.v
link_design openframe_project_wrapper

# Read timing constraints
read_sdc /work/6_1_fill.sdc

# Report basic timing info
report_checks -path_delay max -sort_by_slack
report_checks -path_delay min -sort_by_slack

# Write SDF (typ corner — the Liberty file determines the actual corner)
write_sdf -divider . /work/6_final.sdf

puts "SDF written to /work/6_final.sdf"
exit
EOF

echo "Running OpenSTA via docker..."
docker run --rm \
    -v "$BUILD_DIR:/work:rw" \
    -v "$LIB_PATH:/pdk/lib.lib:ro" \
    -v "$STA_SCRIPT:/work/gen_sdf.tcl:ro" \
    "$DOCKER_IMAGE" \
    /OpenROAD/src/sta/app/sta /work/gen_sdf.tcl

rm -f "$STA_SCRIPT"

if [ -f "$OUTPUT_SDF" ]; then
    # Count cells in SDF
    CELL_COUNT=$(grep -c "(CELLTYPE" "$OUTPUT_SDF" 2>/dev/null || echo "?")
    ESCAPED_COUNT=$(grep -c 'INSTANCE.*\\' "$OUTPUT_SDF" 2>/dev/null || echo "0")
    FILE_SIZE=$(du -h "$OUTPUT_SDF" | cut -f1)
    echo ""
    echo "=== SDF generation complete ==="
    echo "File: $OUTPUT_SDF ($FILE_SIZE, $CELL_COUNT cells)"
    if [ "$ESCAPED_COUNT" -gt 0 ] 2>/dev/null; then
        echo "Note: $ESCAPED_COUNT instances have escaped identifiers (\\$ \\.)."
        echo "      These require SDF parser escape-stripping for correct matching."
    fi
    echo ""
    echo "Usage with timing_sim_cpu:"
    echo "  cargo run -r --bin timing_sim_cpu -- --config tests/timing_test/sim_config.json --sdf $OUTPUT_SDF"
    echo ""
    echo "Note: This SDF has gate-only timing (no parasitic wire delays)."
    echo "For wire delays, provide a SPEF file from PnR routing."
else
    echo "Error: SDF file was not generated"
    exit 1
fi
