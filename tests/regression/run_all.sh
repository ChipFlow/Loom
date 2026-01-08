#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run all regression tests
#
# Usage: ./run_all.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "GEM Regression Test Suite"
echo "========================================="
echo ""

# Find all test files
TEST_FILES=$(find . -name "*.v" -type f ! -name "*_synth.gv" ! -path "*/.*" | sort)
TEST_COUNT=$(echo "$TEST_FILES" | wc -l)
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo "Found $TEST_COUNT test(s)"
echo ""

# Run each test
for test_file in $TEST_FILES; do
    test_name=$(basename "$test_file" .v)
    test_dir=$(dirname "$test_file")

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}Testing: $test_file${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if ./run_test.sh "$test_file"; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        exit_code=$?
        if [ $exit_code -eq 2 ]; then
            echo -e "${YELLOW}⚠ $test_name STATUS UNKNOWN${NC}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
        else
            echo -e "${RED}✗ $test_name FAILED${NC}"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    fi
    echo ""
done

# Summary
echo "========================================="
echo "Test Summary"
echo "========================================="
echo -e "Total:   $TEST_COUNT"
echo -e "${GREEN}Passed:  $PASS_COUNT${NC}"
echo -e "${RED}Failed:  $FAIL_COUNT${NC}"
echo -e "${YELLOW}Unknown: $SKIP_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}✓✓✓ ALL TESTS PASSED ✓✓✓${NC}"
    exit 0
else
    echo -e "${RED}✗✗✗ SOME TESTS FAILED ✗✗✗${NC}"
    exit 1
fi
