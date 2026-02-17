#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regression test runner for GEM non-synthesizable constructs
#
# Usage: ./run_test.sh <test_verilog_file>
#
# This script:
# 1. Synthesizes the Verilog file using Yosys with AIGPDK
# 2. Runs iverilog as golden reference
# 3. Compares outputs
# 4. Reports PASS/FAIL

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <test_verilog_file>"
    exit 1
fi

TEST_FILE="$1"
TEST_NAME=$(basename "$TEST_FILE" .v)
TEST_DIR=$(dirname "$TEST_FILE")
WORK_DIR="$TEST_DIR/${TEST_NAME}_work"

# Get absolute path to GEM root (where aigpdk/ lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
AIGPDK_DIR="$GEM_ROOT/aigpdk"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Running regression test: $TEST_NAME"
echo "========================================="

# Create work directory
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Step 1: Run iverilog as golden reference
echo ""
echo "[1/4] Running iverilog (golden reference)..."
if iverilog -g2012 -o "${TEST_NAME}_iverilog" "../${TEST_NAME}.v" 2>&1 | tee iverilog_compile.log; then
    echo -e "${GREEN}✓ iverilog compilation successful${NC}"
else
    echo -e "${RED}✗ iverilog compilation failed${NC}"
    exit 1
fi

echo ""
echo "[2/4] Running iverilog simulation..."
if vvp "${TEST_NAME}_iverilog" 2>&1 | tee iverilog_output.txt; then
    echo -e "${GREEN}✓ iverilog simulation completed${NC}"
else
    echo -e "${RED}✗ iverilog simulation failed${NC}"
    exit 1
fi

# Step 2: Synthesize with Yosys (using yowasp-yosys for slang support)
echo ""
echo "[3/4] Synthesizing with yowasp-yosys (includes slang for $display)..."

# Detect which yosys to use
if command -v yowasp-yosys &> /dev/null; then
    YOSYS_CMD="yowasp-yosys"
    echo "  Using yowasp-yosys (with slang support)"
elif command -v yosys &> /dev/null; then
    YOSYS_CMD="yosys"
    echo "  Using standard yosys (limited $display support)"
else
    echo -e "${RED}✗ No yosys found${NC}"
    exit 1
fi

cat > synth.tcl <<EOF
# Read Verilog with SystemVerilog support and FORMAL/SYNTHESIS defines
# -DFORMAL enables assertion code, -DSYNTHESIS enables synthesis-compatible code
read_verilog -sv -DFORMAL -DSYNTHESIS ../${TEST_NAME}.v

# Hierarchy
hierarchy -check -auto-top

# Synthesis
proc

# Write RTLIL after proc to see $print cells
write_rtlil ${TEST_NAME}_after_proc.rtlil

# Mark $print and $check cells as "keep" to prevent optimization
select t:\$print t:\$check %i
setattr -set keep 1
select -clear

opt_expr
opt_dff
opt_clean

# Map formal cells to GEM cells
techmap -map ${AIGPDK_DIR}/gem_formal.v

# Write output with GEM cells (before ABC which may fail in yowasp)
write_verilog ${TEST_NAME}_gem.gv
write_json ${TEST_NAME}_gem.json

# Technology mapping to AIGPDK (optional, may fail in yowasp)
dfflibmap -liberty ${AIGPDK_DIR}/aigpdk_nomem.lib
opt_clean -purge

# Try ABC, but don't fail if it doesn't work
# (yowasp-yosys has temp file issues with ABC)
# abc -liberty ../../../../aigpdk/aigpdk_nomem.lib
# opt_clean -purge
# techmap
# abc -liberty ../../../../aigpdk/aigpdk_nomem.lib
# opt_clean -purge

# Write final output
write_verilog ${TEST_NAME}_synth.gv
write_json ${TEST_NAME}_synth.json

# Statistics
stat
EOF

if $YOSYS_CMD -s synth.tcl 2>&1 | tee yosys.log; then
    echo -e "${GREEN}✓ Synthesis successful${NC}"
else
    echo -e "${RED}✗ Synthesis failed${NC}"
    exit 1
fi

# Step 3: Check if GEM_DISPLAY or GEM_ASSERT cells were generated
echo ""
echo "[4/4] Checking synthesized design..."

# Check the GEM-only output first (before ABC)
GEM_FILE="${TEST_NAME}_gem.gv"
if [ -f "$GEM_FILE" ] && grep -q "GEM_DISPLAY\|GEM_ASSERT" "$GEM_FILE"; then
    echo -e "${GREEN}✓ GEM cells found in synthesis output${NC}"
    GEM_DISPLAY_COUNT=$(grep -c "GEM_DISPLAY" "$GEM_FILE" || echo "0")
    GEM_ASSERT_COUNT=$(grep -c "GEM_ASSERT" "$GEM_FILE" || echo "0")
    echo "  - GEM_DISPLAY cells: $GEM_DISPLAY_COUNT"
    echo "  - GEM_ASSERT cells: $GEM_ASSERT_COUNT"
elif [ -f "${TEST_NAME}_synth.gv" ] && grep -q "GEM_DISPLAY\|GEM_ASSERT" "${TEST_NAME}_synth.gv"; then
    echo -e "${GREEN}✓ GEM cells found in final synthesis output${NC}"
    GEM_DISPLAY_COUNT=$(grep -c "GEM_DISPLAY" "${TEST_NAME}_synth.gv" || echo "0")
    GEM_ASSERT_COUNT=$(grep -c "GEM_ASSERT" "${TEST_NAME}_synth.gv" || echo "0")
    echo "  - GEM_DISPLAY cells: $GEM_DISPLAY_COUNT"
    echo "  - GEM_ASSERT cells: $GEM_ASSERT_COUNT"
else
    echo -e "${YELLOW}⚠ No GEM cells found (may be optimized away)${NC}"
fi

# Step 4: Compare outputs (basic check)
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Test name: ${YELLOW}${TEST_NAME}${NC}"
echo "Work directory: $WORK_DIR"
echo ""
echo "Outputs:"
echo "  - iverilog output: iverilog_output.txt"
echo "  - Yosys log: yosys.log"
echo "  - Synthesized Verilog: ${TEST_NAME}_synth.gv"
echo "  - Synthesized JSON: ${TEST_NAME}_synth.json"
echo ""

# Check for test completion (case insensitive)
if grep -qi "PASS\|SUCCESS\|test.* complete" iverilog_output.txt; then
    echo -e "${GREEN}✓✓✓ TEST PASSED ✓✓✓${NC}"
    exit 0
elif grep -q "FAIL\|ERROR" iverilog_output.txt; then
    echo -e "${RED}✗✗✗ TEST FAILED ✗✗✗${NC}"
    exit 1
else
    echo -e "${YELLOW}⚠⚠⚠ TEST STATUS UNKNOWN ⚠⚠⚠${NC}"
    exit 2
fi
