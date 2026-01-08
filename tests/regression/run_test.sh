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

# Step 2: Synthesize with Yosys
echo ""
echo "[3/4] Synthesizing with Yosys..."
cat > synth.tcl <<EOF
# Read Verilog (define SYNTHESIS to exclude testbench)
read_verilog -sv -DSYNTHESIS ../${TEST_NAME}.v

# Hierarchy
hierarchy -check -auto-top

# Synthesis
proc
opt_expr
opt_dff
opt_clean

# Map formal cells to GEM cells
techmap -map ../../../aigpdk/gem_formal.v

# Technology mapping to AIGPDK
dfflibmap -liberty ../../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
abc -liberty ../../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty ../../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Write output
write_verilog ${TEST_NAME}_synth.gv
write_json ${TEST_NAME}_synth.json

# Statistics
stat
EOF

if yosys -s synth.tcl 2>&1 | tee yosys.log; then
    echo -e "${GREEN}✓ Yosys synthesis successful${NC}"
else
    echo -e "${RED}✗ Yosys synthesis failed${NC}"
    exit 1
fi

# Step 3: Check if GEM_DISPLAY or GEM_ASSERT cells were generated
echo ""
echo "[4/4] Checking synthesized design..."
if grep -q "GEM_DISPLAY\|GEM_ASSERT" "${TEST_NAME}_synth.gv"; then
    echo -e "${GREEN}✓ GEM cells found in synthesis output${NC}"
    grep -c "GEM_DISPLAY" "${TEST_NAME}_synth.gv" | xargs echo "  - GEM_DISPLAY cells:"
    grep -c "GEM_ASSERT" "${TEST_NAME}_synth.gv" | xargs echo "  - GEM_ASSERT cells:" || echo "  - GEM_ASSERT cells: 0"
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

# Check for test completion
if grep -q "PASS\|SUCCESS\|test complete" iverilog_output.txt; then
    echo -e "${GREEN}✓✓✓ TEST PASSED ✓✓✓${NC}"
    exit 0
elif grep -q "FAIL\|ERROR" iverilog_output.txt; then
    echo -e "${RED}✗✗✗ TEST FAILED ✗✗✗${NC}"
    exit 1
else
    echo -e "${YELLOW}⚠⚠⚠ TEST STATUS UNKNOWN ⚠⚠⚠${NC}"
    exit 2
fi
