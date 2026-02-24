# Troubleshooting VCD Input Issues

This guide helps debug VCD input problems where GEM simulations produce incorrect results or warn about missing signals.

## VCD Scope Auto-Detection (Recommended)

**NEW**: GEM now automatically detects the correct VCD scope containing your design's ports. In most cases, you don't need to specify `--input-vcd-scope` manually.

### How Auto-Detection Works

When you run `loom sim` without specifying `--input-vcd-scope`, GEM:

1. Extracts the list of required input ports from your synthesized design
2. Searches the VCD file for scopes containing all required ports
3. Tries common DUT scope names first: `dut`, `uut`, `DUT`, `UUT`, or your module name
4. Falls back to any scope that contains all required ports
5. Logs which scope was selected for transparency

### Example Output

```
INFO No VCD scope specified - attempting auto-detection
DEBUG Searching for VCD scope containing 4 input ports
DEBUG Required ports: {"din_valid", "clk", "reset", "din"}
INFO Auto-detected VCD scope: safe_tb/uut (matched common pattern 'uut')
```

### Manual Override

If auto-detection fails or selects the wrong scope, use `--input-vcd-scope` to specify manually:

```bash
# Slash-separated path to the DUT scope
loom sim design.gv design.gemparts input.vcd output.vcd 8 \
    --input-vcd-scope "testbench/dut"

# For nested hierarchies
loom sim design.gv design.gemparts input.vcd output.vcd 8 \
    --input-vcd-scope "top_tb/subsystem/my_module"
```

**Note**: Use slash separators (`/`), not dots (`.`).

---

## Symptom: Missing Primary Input Warnings

```
WARN (GATESIM_VCDI_MISSING_PI) Primary input port (HierName(), "reset", None) not present in the VCD input
WARN (GATESIM_VCDI_MISSING_PI) Primary input port (HierName(), "din", Some(3)) not present in the VCD input
```

### Root Cause

GEM expects VCD signals at **absolute top-level** with no module hierarchy prefix. The signal names must exactly match the synthesized module's port names.

### How to Check

1. **Inspect your VCD file**:
```bash
grep '\$var' your_input.vcd | head -20
```

2. **Look for module scopes**:
```bash
grep '\$scope module' your_input.vcd
```

3. **Check synthesized module ports**:
```bash
head -20 your_design_synth.gv
```

### What GEM Expects

**Correct** - Signals at top level:
```vcd
$timescale 1ns/1ns
$var reg 1 ! clk $end
$var reg 1 " reset $end
$var reg 4 # din [3:0] $end
$var reg 1 $ din_valid $end
$var wire 1 % unlocked $end
$enddefinitions $end
$dumpvars
0"
0$
0%
1!
#10
1"
#20
b1100 #
1$
```

**Incorrect** - Signals scoped under module:
```vcd
$scope module testbench $end
  $scope module dut $end
    $var wire 1 ! clk $end
    $var wire 1 " reset $end
    $var wire 4 # din [3:0] $end
    ...
  $upscope $end
$upscope $end
```

## Solution 1: Flat VCD Generation

Create a testbench that dumps signals at absolute top level:

```verilog
module testbench;

reg clk = 0;
reg reset;
reg [3:0] din;
reg din_valid = 0;
wire unlocked;

// DUT instantiation
your_module dut (
    .clk(clk),
    .reset(reset),
    .din(din),
    .din_valid(din_valid),
    .unlocked(unlocked)
);

always #10 clk = !clk;

initial begin
    // CRITICAL: Dump signals at top level (depth 1)
    // NOT inside module hierarchy!
    $dumpfile("output.vcd");
    $dumpvars(1, clk, reset, din, din_valid, unlocked);

    // Test sequence
    reset = 1;
    #60;
    reset = 0;

    // ... your test stimulus ...

    #200;
    $finish;
end

endmodule
```

**Key Point**: `$dumpvars(1, signal1, signal2, ...)` dumps individual signals at the current scope level, **not** inside child modules.

### Compile and Run

```bash
# For synthesis-compatible testbench
iverilog -DSYNTHESIS -o sim your_design.v testbench.v
./sim

# Check VCD structure
grep '\$scope' output.vcd  # Should be minimal or none
grep '\$var' output.vcd | head -10
```

## Solution 2: Post-Process VCD (Advanced)

If you can't change the testbench, post-process the VCD to flatten hierarchy:

```python
#!/usr/bin/env python3
"""Flatten VCD hierarchy to top level"""

import sys

def flatten_vcd(input_vcd, output_vcd):
    with open(input_vcd) as inf, open(output_vcd, 'w') as outf:
        in_scope = False
        scope_depth = 0

        for line in inf:
            # Track scope depth
            if line.strip().startswith('$scope'):
                scope_depth += 1
                if scope_depth == 1:
                    continue  # Keep root scope
                in_scope = True
                continue
            elif line.strip().startswith('$upscope'):
                scope_depth -= 1
                if in_scope and scope_depth == 0:
                    in_scope = False
                continue

            # Skip signals inside nested scopes, keep only top-level
            if in_scope and line.strip().startswith('$var'):
                continue  # Skip nested module signals

            outf.write(line)

if __name__ == '__main__':
    flatten_vcd(sys.argv[1], sys.argv[2])
```

**Usage**:
```bash
python3 flatten_vcd.py hierarchical.vcd flat.vcd
```

## Solution 3: VCD Scope Option (Experimental)

GEM provides `--input-vcd-scope` to specify which module hierarchy to read:

```bash
cargo run -r --features metal --bin loom -- sim \
    design.gv parts.gemparts input.vcd output.vcd 48 \
    --input-vcd-scope module_name
```

**Known Issue**: Currently, signal matching still fails even with correct scope specified. This is under investigation.

## Diagnostic Checklist

### 1. Verify Signal Names Match

**Synthesized Module**:
```bash
grep "^module\|input\|output" design_synth.gv
```

Output:
```verilog
module safe(clk, reset, din, din_valid, unlocked);
  input clk;
  input reset;
  input [3:0] din;
  input din_valid;
  output unlocked;
```

**VCD Signals**:
```bash
grep '\$var.*\(clk\|reset\|din\|unlocked\)' input.vcd
```

Output should match synthesized port names exactly.

### 2. Check Signal Bit Widths

Multi-bit signals must have correct indices:

**Synthesized**: `input [3:0] din;`

**VCD**:
```vcd
$var reg 4 # din [3:0] $end
```

GEM expects separate indices: `din[3]`, `din[2]`, `din[1]`, `din[0]`

### 3. Verify Timestamp Format

GEM expects integer timestamps (not real numbers):

**Correct**:
```vcd
#0
#10
#20
```

**Incorrect**:
```vcd
#0.0
#10.5
#20.25
```

### 4. Check Timescale

Ensure VCD timescale matches simulation expectations:

```vcd
$timescale 1ns $end
```

or

```vcd
$timescale 1ps $end
```

Clock periods in testbench should use same time unit.

## Validation Steps

After fixing VCD issues, validate GEM is reading inputs correctly:

### 1. Run with CPU Verification

```bash
cargo run -r --features metal --bin loom -- sim \
    design.gv parts.gemparts input.vcd output.vcd 48 \
    --check-with-cpu
```

This compares GPU results against CPU gate-level simulation. Should print:
```
[INFO] sanity test passed!
```

### 2. Compare Output VCD with Reference

Run same design with iverilog:
```bash
iverilog -o reference_sim design.v testbench.v
./reference_sim  # Generates reference.vcd
```

Compare outputs:
```bash
# Check if unlocked signal toggles the same in both
grep '^[01]!' gem_output.vcd
grep '^[01]!' reference.vcd
```

### 3. Check Cycle Count

```bash
cargo run -r --features metal --bin loom -- sim \
    design.gv parts.gemparts input.vcd output.vcd 48 \
    2>&1 | grep "total number of cycles"
```

Should match your testbench's simulation time / clock period.

## Common Pitfalls

### 1. Testbench Inside \`ifndef SYNTHESIS

If testbench is only compiled when `SYNTHESIS` is not defined:

```verilog
`ifndef SYNTHESIS
module testbench;
  // ...
endmodule
`endif
```

You must compile **without** `-DSYNTHESIS` for VCD generation:
```bash
iverilog -o sim design.v testbench.v  # No -DSYNTHESIS!
```

But the DUT must be compiled **with** `-DSYNTHESIS` if it has non-synthesizable constructs:
```bash
# Separate compilation
iverilog -DSYNTHESIS -c design.v
iverilog -o sim design.v testbench.v
```

### 2. X/Z Values in VCD

GEM may not handle unknown (X) or high-impedance (Z) values correctly:

```vcd
$dumpvars
x"  # reset = X
bxxxx #  # din = XXXX
```

**Solution**: Initialize all inputs in testbench:
```verilog
initial begin
    reset = 0;  // Don't leave uninitialized
    din = 4'h0;
    din_valid = 0;
end
```

### 3. Missing Clock Signal

If VCD doesn't include clock:
```
WARN (GATESIM_VCDI_MISSING_PI) Primary input port (HierName(), "clk", None) not present
```

**Ensure**:
- Clock is generated in testbench
- Clock is included in `$dumpvars`
- Clock signal name matches synthesized netlist exactly

## Example: Working Flat VCD Testbench

```verilog
// testbench_flat.v - Generates GEM-compatible VCD
module testbench_flat;

// Declare all signals at top level
reg clk = 0;
reg reset = 1;
reg [3:0] din = 4'h0;
reg din_valid = 0;
wire unlocked;

// DUT instantiation
safe dut (
    .clk(clk),
    .reset(reset),
    .din(din),
    .din_valid(din_valid),
    .unlocked(unlocked)
);

// Clock generation
always #10 clk = !clk;  // 20ns period = 50MHz

// Test sequence
initial begin
    // CRITICAL: Dump at top level (depth 1)
    $dumpfile("safe_flat.vcd");
    $dumpvars(1, clk, reset, din, din_valid, unlocked);

    // Reset phase
    reset = 1;
    #60;  // 3 clock cycles
    reset = 0;
    #11;  // Small offset from clock edge

    // Apply test stimulus
    din = 4'hc;
    din_valid = 1;
    #20;

    din = 4'h0;
    #20;

    din = 4'hd;
    #20;

    din = 4'he;
    #20;

    din_valid = 0;
    #40;

    $finish;
end

endmodule
```

**Compile and test**:
```bash
# Compile (DUT must be SYNTHESIS-compatible)
iverilog -DSYNTHESIS -o sim safe.v testbench_flat.v

# Run simulation
./sim

# Verify VCD structure
echo "=== VCD Scopes ==="
grep '\$scope' safe_flat.vcd

echo -e "\n=== VCD Signals ==="
grep '\$var' safe_flat.vcd

# Should show signals at top level, no nested $scope modules
```

## Still Having Issues?

1. **Enable debug logging**:
   ```bash
   RUST_LOG=debug,vcd_ng=trace cargo run -r --features metal --bin loom -- sim <args> 2>&1 | tee debug.log
   ```

2. **Check with minimal test**:
   - Create simplest possible design (single DFF)
   - Generate flat VCD
   - Verify GEM can read it correctly

3. **Report issue** with:
   - Synthesized `.gv` file
   - Input VCD file
   - GEM command line
   - Error messages or unexpected output

---

**Document Version**: 1.0
**Last Updated**: 2025-01-08
**Related**: simulation-architecture.md
