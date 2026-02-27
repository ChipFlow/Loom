#!/usr/bin/env bash
# Run CVC SDF-annotated simulation for multi_depth timing validation via Docker.
#
# Usage:
#   bash tests/timing_test/multi_depth/run_cvc_multi_depth.sh
#
# Builds the CVC Docker image (cached after first run), then runs the
# multi_depth testbench with SDF back-annotation. Reports per-group
# arrival times for comparison against Loom's timing simulation.
#
# Outputs:
#   tests/timing_test/multi_depth/cvc_output.log              (stdout from CVC sim)
#   tests/timing_test/multi_depth/cvc_multi_depth_output.vcd   (VCD waveform)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
MULTI_DEPTH_DIR="$SCRIPT_DIR"
CVC_DIR="$REPO_ROOT/tests/timing_test/cvc"
IMAGE_NAME="loom-cvc"

# Build the CVC image if not present (reuses same Dockerfile as inv_chain)
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "=== Building CVC Docker image (first time only) ==="
    docker build --platform linux/amd64 -t "$IMAGE_NAME" "$CVC_DIR"
    echo ""
fi

echo "=== Running CVC: multi_depth with SDF ==="

# CVC compiles Verilog to native code (./cvcsim), then we run it.
# Working directory is /design so $sdf_annotate relative paths resolve.
docker run --rm --platform linux/amd64 \
    -v "$MULTI_DEPTH_DIR:/design:rw" \
    -w /design \
    --entrypoint /bin/sh \
    "$IMAGE_NAME" \
    -c 'cvc64 +typdelays tb_cvc.v multi_depth.v && ./cvcsim' 2>&1 | tee "$MULTI_DEPTH_DIR/cvc_output.log"

# Clean up compiled binary from design dir
rm -f "$MULTI_DEPTH_DIR/cvcsim"

echo ""
echo "=== CVC Multi-Depth Results ==="

if grep -q "RESULT: grp_e_delay=" "$MULTI_DEPTH_DIR/cvc_output.log"; then
    grep "RESULT:" "$MULTI_DEPTH_DIR/cvc_output.log"
    echo ""
    echo "Per-group arrival times from CLK posedge:"
    for grp in a b c d e; do
        delay=$(grep "RESULT: grp_${grp}_delay=" "$MULTI_DEPTH_DIR/cvc_output.log" | sed 's/.*=//')
        echo "  Group ${grp}: ${delay}ps"
    done
else
    echo "ERROR: CVC did not produce expected RESULT: lines"
    echo "--- Full output ---"
    cat "$MULTI_DEPTH_DIR/cvc_output.log"
    exit 1
fi

if [ -f "$MULTI_DEPTH_DIR/cvc_multi_depth_output.vcd" ]; then
    echo ""
    echo "VCD output: $MULTI_DEPTH_DIR/cvc_multi_depth_output.vcd"
fi
