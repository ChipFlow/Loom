#!/usr/bin/env bash
# Run CVC SDF-annotated simulation for MCU SoC timing validation via Docker.
#
# Usage:
#   bash tests/mcu_soc/cvc/run_cvc.sh
#
# Prerequisites:
#   1. Generate stimulus VCD:
#      cargo run -r --features metal --bin loom -- cosim \
#        tests/mcu_soc/data/6_final.v \
#        --config tests/mcu_soc/sim_config_sky130.json \
#        --top-module openframe_project_wrapper \
#        --max-cycles 1000 \
#        --stimulus-vcd tests/mcu_soc/cvc/stimulus.vcd
#
#   2. Convert stimulus to Verilog:
#      python3 tests/mcu_soc/cvc/convert_stimulus.py \
#        tests/mcu_soc/cvc/stimulus.vcd \
#        tests/mcu_soc/cvc/stimulus_gen.v
#
#   3. Generate cell models:
#      python3 tests/mcu_soc/cvc/gen_cell_models.py
#
# Outputs:
#   tests/mcu_soc/cvc/output/cvc_output.log     (CVC compilation + sim log)
#   tests/mcu_soc/cvc/output/cvc_output.vcd      (GPIO output waveforms)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DATA_DIR="$REPO_ROOT/tests/mcu_soc/data"
CVC_DIR="$SCRIPT_DIR"
OUTPUT_DIR="$CVC_DIR/output"
IMAGE_NAME="loom-cvc"

mkdir -p "$OUTPUT_DIR"

# Check prerequisites
for f in "$CVC_DIR/stimulus_gen.v" "$CVC_DIR/sky130_cells.v" "$CVC_DIR/tb_cvc.v"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        echo "See script header for prerequisites."
        exit 1
    fi
done

for f in "$DATA_DIR/6_final.v" "$CVC_DIR/6_final_nocheck.sdf"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Missing $f"
        exit 1
    fi
done

# Build the CVC image if not present
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "=== Building CVC Docker image (first time only) ==="
    docker build --platform linux/amd64 -t "$IMAGE_NAME" "$REPO_ROOT/tests/timing_test/cvc"
    echo ""
fi

echo "=== Running CVC: MCU SoC with SDF ==="
echo "This may take a long time for a 19MB netlist..."

# CVC compiles Verilog to native code (./cvcsim), then we run it.
# Mount both the CVC directory and data directory into the container.
# Working directory is /cvc so relative paths in tb_cvc.v resolve.
docker run --rm --platform linux/amd64 \
    -v "$CVC_DIR:/cvc:rw" \
    -v "$DATA_DIR:/data:ro" \
    -w /cvc \
    --entrypoint /bin/sh \
    "$IMAGE_NAME" \
    -c 'ln -sf /data/6_final.v . && cvc64 +typdelays tb_cvc.v sky130_cells.v cf_sram.v 6_final.v && ./cvcsim' 2>&1 | tee "$OUTPUT_DIR/cvc_output.log"

# Move cvcsim binary out of cvc dir
for f in cvcsim cvc_output.vcd; do
    if [ -f "$CVC_DIR/$f" ]; then
        mv "$CVC_DIR/$f" "$OUTPUT_DIR/$f"
    fi
done
# Clean up symlinks
rm -f "$CVC_DIR/6_final.v"

echo ""
echo "=== CVC Simulation Complete ==="
if [ -f "$OUTPUT_DIR/cvc_output.vcd" ]; then
    echo "Output VCD: $OUTPUT_DIR/cvc_output.vcd"
    echo "Size: $(du -h "$OUTPUT_DIR/cvc_output.vcd" | cut -f1)"
fi
