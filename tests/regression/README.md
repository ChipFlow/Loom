# GEM Regression Test Suite

This directory contains regression tests for GEM's non-synthesizable constructs support, including `$display`, `$finish`, and assertions.

## Directory Structure

```
regression/
├── counter/           # Counter-based tests
│   ├── simple_counter.v      # Basic counter with $display and $finish
│   └── format_test.v         # Tests various $display format specifiers
├── alu/              # ALU tests
│   └── simple_alu.v          # ALU with assertions and $display
├── fifo/             # FIFO tests (planned)
├── run_test.sh       # Single test runner
├── run_all.sh        # Run all regression tests
└── README.md         # This file
```

## Running Tests

### Run All Tests

```bash
cd tests/regression
./run_all.sh
```

### Run Single Test

```bash
cd tests/regression
./run_test.sh counter/simple_counter.v
```

## Test Format

Each test is a self-contained Verilog file with:
- A design module
- A testbench module
- `$display` statements for output
- `$finish` to terminate simulation
- Optional assertions for verification

## How It Works

The test runner:
1. **Runs iverilog** as a golden reference to verify the test logic
2. **Synthesizes with Yosys** using AIGPDK and GEM formal cell mappings
3. **Checks for GEM cells** (`GEM_DISPLAY`, `GEM_ASSERT`) in synthesis output
4. **Compares results** and reports PASS/FAIL

## Adding New Tests

1. Create a new `.v` file in the appropriate subdirectory
2. Include both design and testbench modules
3. Use `$display` for output and `$finish` to end simulation
4. Include "PASS", "SUCCESS", or "test complete" in output for pass detection
5. Run `./run_all.sh` to verify

## Example Test Structure

```verilog
// Design module
module my_design(...);
    // Your design logic
    always @(posedge clk) begin
        if (condition) begin
            $display("Test milestone reached");
        end
        if (done) begin
            $display("Test complete");
            $finish;
        end
    end
endmodule

// Testbench
module testbench;
    // Instantiate design
    // Generate stimulus
    // Check for completion
endmodule
```

## Current Test Coverage

### Counter Tests
- ✓ `simple_counter.v` - Basic counter with $display/$finish
- ✓ `format_test.v` - Format specifier testing (decimal, hex, binary)

### ALU Tests
- ✓ `simple_alu.v` - 4-bit ALU with assertions and multi-arg $display

### Planned Tests
- [ ] FIFO with overflow/underflow assertions
- [ ] State machine with $display state tracking
- [ ] Multi-clock domain test
- [ ] Memory tests with $display
- [ ] Complex format specifiers (width, padding, alignment)

## Requirements

- **iverilog** (for golden reference simulation) - Required
- **Yosys** (v0.49+) or **yowasp-yosys** (for synthesis verification) - Required
- AIGPDK library files in `../../aigpdk/`

### Yosys $display Support

**Standard Yosys (v0.49+) DOES support `$display` in `always` blocks!**

- **Yosys proc pass**: Converts $display statements to `$print` cells automatically
- **GEM techmap**: Maps `$print` cells to `GEM_DISPLAY` cells (see `aigpdk/gem_formal.v`)
- **Works with**: Both standard Yosys and yowasp-yosys (WebAssembly-packaged Yosys)

The key requirement is that the `gem_formal.v` techmap handles multi-bit trigger signals correctly (e.g., `{rst, clk}`). Our implementation uses `TRG[0]` as the clock regardless of trigger width.

### Current Test Capabilities

These regression tests verify:
1. ✓ **Functional correctness** via iverilog simulation
2. ✓ **Synthesizability** of design logic
3. ✓ **GEM cell generation** - $print → GEM_DISPLAY conversion working!

The counter tests demonstrate successful GEM_DISPLAY cell generation with:
- Format string preservation via `gem_format` attribute
- Argument width tracking via `gem_args_width` attribute
- Conditional display with enable signals
- Multi-argument format strings

## Output Files

Each test creates a work directory containing:
- `iverilog_output.txt` - Golden reference output
- `yosys.log` - Synthesis log
- `<test>_synth.gv` - Synthesized Verilog
- `<test>_synth.json` - Synthesized JSON with GEM cell attributes
