#!/usr/bin/env bash
# Run SKY130 timing simulation on post-layout netlist
#
# Usage: ./scripts/run_sky130_timing_sim.sh [options]
#
# Options:
#   --setup-pdk     Download SKY130 PDK using volare (requires pip)
#   --pdk-root DIR  Use existing PDK at DIR (default: ~/.volare)
#   --max-cycles N  Limit simulation cycles (default: 10)
#   --verbose       Enable verbose output

set -euo pipefail

# Default values
SETUP_PDK=false
PDK_ROOT="${PDK_ROOT:-$HOME/.volare}"
MAX_CYCLES=10
VERBOSE=false
REPORT_VIOLATIONS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --setup-pdk)
            SETUP_PDK=true
            shift
            ;;
        --pdk-root)
            PDK_ROOT="$2"
            shift 2
            ;;
        --max-cycles)
            MAX_CYCLES="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --report-violations)
            REPORT_VIOLATIONS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --setup-pdk          Download SKY130 PDK using volare"
            echo "  --pdk-root DIR       Use existing PDK at DIR (default: ~/.volare)"
            echo "  --max-cycles N       Limit simulation cycles (default: 10)"
            echo "  --verbose            Enable verbose output"
            echo "  --report-violations  Report timing violations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Test files
NETLIST="$PROJECT_ROOT/tests/timing_test/6_final.v"
INPUT_VCD="$PROJECT_ROOT/tests/timing_test/6_final_test_input.vcd"

# Check test files exist
if [[ ! -f "$NETLIST" ]]; then
    echo "Error: Netlist not found: $NETLIST"
    echo "Please ensure the test files are available."
    exit 1
fi

if [[ ! -f "$INPUT_VCD" ]]; then
    echo "Error: Input VCD not found: $INPUT_VCD"
    echo "Please ensure the test files are available."
    exit 1
fi

# Setup PDK if requested
if [[ "$SETUP_PDK" == "true" ]]; then
    echo "Setting up SKY130 PDK using volare..."

    # Install volare if not present
    if ! command -v volare &> /dev/null; then
        echo "Installing volare..."
        pip3 install --upgrade volare
    fi

    # Use a known working version of the PDK
    # This is the version used by OpenLane 2.x
    PDK_VERSION="bdc9412b3e468c102d01b7cf6571e31e6d283fcf"

    echo "Downloading SKY130 PDK version $PDK_VERSION..."
    volare enable --pdk sky130 "$PDK_VERSION"

    echo "PDK setup complete at $PDK_ROOT"
fi

# Find liberty file
LIBERTY_FILE=""
POSSIBLE_PATHS=(
    "$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
    "$PDK_ROOT/volare/sky130/versions/*/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
    "$PDK_ROOT/sky130/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
)

for path in "${POSSIBLE_PATHS[@]}"; do
    # Handle glob patterns
    for expanded_path in $path; do
        if [[ -f "$expanded_path" ]]; then
            LIBERTY_FILE="$expanded_path"
            break 2
        fi
    done
done

# Build the project
echo "Building GEM timing simulator..."
cd "$PROJECT_ROOT"
cargo build --release --bin timing_sim_cpu

# Construct command
CMD="cargo run --release --bin timing_sim_cpu --"
CMD="$CMD $NETLIST $INPUT_VCD"
CMD="$CMD --clock-period 25000"
CMD="$CMD --max-cycles $MAX_CYCLES"

if [[ -n "$LIBERTY_FILE" ]]; then
    echo "Using liberty file: $LIBERTY_FILE"
    CMD="$CMD --liberty $LIBERTY_FILE"
else
    echo "Warning: No liberty file found, using default SKY130 timing values"
    echo "Run with --setup-pdk to download the PDK with accurate timing data"
fi

if [[ "$VERBOSE" == "true" ]]; then
    CMD="$CMD --verbose"
fi

if [[ "$REPORT_VIOLATIONS" == "true" ]]; then
    CMD="$CMD --report-violations"
fi

# Run simulation
echo ""
echo "Running timing simulation..."
echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "Timing simulation complete!"
