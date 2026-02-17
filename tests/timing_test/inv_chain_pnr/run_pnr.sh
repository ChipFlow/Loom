#!/usr/bin/env bash
# Run OpenLane2 P&R on the inv_chain test design.
#
# Usage:
#   bash tests/timing_test/inv_chain_pnr/run_pnr.sh
#
# Outputs:
#   tests/timing_test/inv_chain_pnr/openlane_runs/  (full run)
#   tests/timing_test/inv_chain_pnr/6_final.v       (post-layout netlist, copied)
#   tests/timing_test/inv_chain_pnr/6_final.sdf     (SDF timing, copied)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OPENLANE_IMAGE="ghcr.io/efabless/openlane2:2.4.0.dev1"
PDK_VOLUME="openlane-pdk"

echo "=== OpenLane2 P&R: inv_chain ==="
echo "Design dir: $SCRIPT_DIR"
echo "Image:      $OPENLANE_IMAGE"

docker run --rm \
  -v "$PDK_VOLUME:/root/.volare" \
  -v "$SCRIPT_DIR:/design" \
  -w /design \
  "$OPENLANE_IMAGE" \
  openlane /design/config.json

# Find the latest run directory
LATEST_RUN=$(ls -td "$SCRIPT_DIR"/openlane_runs/RUN_* 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No run directory found"
    exit 1
fi

echo "Latest run: $LATEST_RUN"

# Copy final outputs
if [ -f "$LATEST_RUN/results/finishing/6_final.v" ]; then
    cp "$LATEST_RUN/results/finishing/6_final.v" "$SCRIPT_DIR/6_final.v"
    echo "Copied 6_final.v"
fi

if [ -f "$LATEST_RUN/results/finishing/6_final.sdf" ]; then
    cp "$LATEST_RUN/results/finishing/6_final.sdf" "$SCRIPT_DIR/6_final.sdf"
    echo "Copied 6_final.sdf"
fi

echo "=== Done ==="
