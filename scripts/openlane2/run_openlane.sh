#!/usr/bin/env bash
# Run OpenLane2 P&R via Docker on a ChipFlow-synthesized design.
#
# Usage:
#   bash scripts/openlane2/run_openlane.sh <design_dir> [config.json]
#
# Example:
#   bash scripts/openlane2/run_openlane.sh designs/mcu_soc_sky130
#
# Expects:
#   <design_dir>/build/*.il  (RTLIL from chipflow silicon prepare)
#   <design_dir>/pins.lock   (pin assignments)
#
# Outputs:
#   <design_dir>/openlane_runs/  (OpenLane run results)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DESIGN_DIR="${1:?Usage: $0 <design_dir> [config.json]}"
DESIGN_DIR="$(cd "$DESIGN_DIR" && pwd)"
CONFIG="${2:-$SCRIPT_DIR/config.json}"

OPENLANE_IMAGE="ghcr.io/efabless/openlane2:2.4.0.dev1"
PDK_VOLUME="openlane-pdk"

echo "=== OpenLane2 P&R ==="
echo "Design: $DESIGN_DIR"
echo "Config: $CONFIG"
echo "Image:  $OPENLANE_IMAGE"

# Mount design dir and config into container
docker run --rm \
  -v "$PDK_VOLUME:/root/.volare" \
  -v "$DESIGN_DIR:/design" \
  -v "$CONFIG:/design/config.json:ro" \
  -w /design \
  "$OPENLANE_IMAGE" \
  openlane /design/config.json

echo "=== Done ==="
echo "Results in: $DESIGN_DIR/openlane_runs/"
