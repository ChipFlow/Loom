#!/usr/bin/env bash
# Run CVC SDF-annotated simulation for inv_chain timing validation via Docker.
#
# Usage:
#   bash tests/timing_test/cvc/run_cvc.sh
#
# Builds the CVC Docker image (cached after first run), then runs the
# inv_chain testbench with SDF back-annotation. Compares the reported
# total delay against Loom's expected 1323ps.
#
# Outputs:
#   tests/timing_test/cvc/output/cvc_output.log            (stdout from CVC sim)
#   tests/timing_test/cvc/output/cvc_inv_chain_output.vcd   (VCD waveform)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INV_CHAIN_DIR="$REPO_ROOT/tests/timing_test/inv_chain_pnr"
OUTPUT_DIR="$SCRIPT_DIR/output"
IMAGE_NAME="loom-cvc"

mkdir -p "$OUTPUT_DIR"

# Build the CVC image if not present
if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo "=== Building CVC Docker image (first time only) ==="
    docker build --platform linux/amd64 -t "$IMAGE_NAME" "$SCRIPT_DIR"
    echo ""
fi

echo "=== Running CVC: inv_chain with SDF ==="

# CVC compiles Verilog to native code (./cvcsim), then we run it.
# Working directory is /design so $sdf_annotate relative paths resolve.
# The $dumpfile in tb_cvc.v writes to cvc_inv_chain_output.vcd (relative).
docker run --rm --platform linux/amd64 \
    -v "$INV_CHAIN_DIR:/design:rw" \
    -w /design \
    --entrypoint /bin/sh \
    "$IMAGE_NAME" \
    -c 'cvc64 +typdelays tb_cvc.v inv_chain.v && ./cvcsim' 2>&1 | tee "$OUTPUT_DIR/cvc_output.log"

# Move outputs from design dir to output dir
for f in cvcsim cvc_inv_chain_output.vcd; do
    if [ -f "$INV_CHAIN_DIR/$f" ]; then
        mv "$INV_CHAIN_DIR/$f" "$OUTPUT_DIR/$f"
    fi
done

echo ""
echo "=== CVC Results ==="

if grep -q "RESULT: total_delay=" "$OUTPUT_DIR/cvc_output.log"; then
    grep "RESULT:" "$OUTPUT_DIR/cvc_output.log"
    echo ""

    CVC_TOTAL=$(grep "RESULT: total_delay=" "$OUTPUT_DIR/cvc_output.log" | sed 's/.*=//')
    CVC_CLK_TO_Q=$(grep "RESULT: clk_to_q=" "$OUTPUT_DIR/cvc_output.log" | sed 's/.*=//')
    CVC_CHAIN=$(grep "RESULT: chain_delay=" "$OUTPUT_DIR/cvc_output.log" | sed 's/.*=//')
    LOOM_TOTAL=1323
    echo "CVC:  clk_to_q=${CVC_CLK_TO_Q}ps  chain=${CVC_CHAIN}ps  total=${CVC_TOTAL}ps"
    echo "Loom: clk_to_q=350ps  chain=973ps  total=${LOOM_TOTAL}ps"
    echo ""

    # Loom uses max(rise, fall) per cell — a conservative approximation since
    # the GPU kernel processes 32 packed signals and can't track per-signal
    # transition direction. CVC tracks actual rise/fall transitions.
    # Expected overestimate: 8 inverters × 10ps IOPATH + 8 wires × 1ps = 88ps.
    DIFF=$((LOOM_TOTAL - CVC_TOTAL))
    echo "Difference: ${DIFF}ps (Loom conservative overestimate)"
    if [ "$DIFF" -ge 0 ] && [ "$DIFF" -le 200 ]; then
        echo "PASS: Loom within expected conservative bound"
    else
        echo "FAIL: Unexpected difference (expected 0-200ps overestimate)"
        exit 1
    fi
else
    echo "ERROR: CVC did not produce expected RESULT: lines"
    echo "--- Full output ---"
    cat "$OUTPUT_DIR/cvc_output.log"
    exit 1
fi

if [ -f "$OUTPUT_DIR/cvc_inv_chain_output.vcd" ]; then
    echo ""
    echo "VCD output: $OUTPUT_DIR/cvc_inv_chain_output.vcd"
fi
