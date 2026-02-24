# Loom Documentation

Welcome to the documentation for Loom, a GPU-accelerated RTL logic simulator.

Use the sidebar to navigate between topics, or start with the [Getting Started](usage.md) guide.

## Documents

### Core Documentation

- **[Simulation Architecture](simulation-architecture.md)**: Detailed explanation of Loom's internal architecture
  - Pipeline stages (NetlistDB → AIG → StagedAIG → Partitions → FlattenedScript → GPU)
  - Data structures and representations
  - VCD input/output format requirements
  - Assertion and display support infrastructure
  - Performance characteristics
  - Known issues and limitations

- **[Timing Simulation](timing-simulation.md)**: CPU-based timing simulation with Liberty/SDF delays
- **[Timing Violations](timing-violations.md)**: GPU-side setup/hold violation detection

### Troubleshooting Guides

- **[Troubleshooting VCD](troubleshooting-vcd.md)**: Debugging VCD input issues
  - VCD hierarchy requirements
  - Signal naming and matching
  - Solutions for flat VCD generation
  - Diagnostic checklist
  - Working examples

## Quick Reference

### VCD Input Requirements (Critical!)

Loom expects VCD signals at **absolute top-level** (no module hierarchy):

```verilog
// ✓ Correct testbench
initial begin
    $dumpfile("output.vcd");
    $dumpvars(1, clk, reset, din, dout);  // Depth 1, explicit signals
end

// ✗ Incorrect testbench
initial begin
    $dumpfile("output.vcd");
    $dumpvars(0, testbench);  // Dumps entire hierarchy
end
```

### Debug Commands

```bash
# Enable debug logging
RUST_LOG=debug cargo run -r --features metal --bin loom -- sim <args>

# Verify with CPU simulation
cargo run -r --features metal --bin loom -- sim <args> --check-with-cpu

# Check VCD structure
grep '\$scope\|\$var' input.vcd | head -20
```

### Key Statistics

When running Loom, look for these diagnostic outputs:

```
netlist has X pins, Y aig pins, Z and gates        # AIG complexity
current: N endpoints, try M parts                  # Partition count
Built script for B blocks, reg/io state size S     # Final script
WARN (GATESIM_VCDI_MISSING_PI) ...                 # VCD issues!
```

## Investigation Methodology

This documentation was created through systematic investigation of Loom's behavior:

1. **Source Code Analysis**: Examined `src/aig.rs`, `src/flatten.rs`, `src/staging.rs`
2. **Debug Tracing**: Used `RUST_LOG=debug` to capture internal state
3. **Test Case Development**: Created minimal reproducible examples
4. **Comparative Testing**: Compared Loom vs iverilog outputs
5. **Third-Party Validation**: Tested with real-world examples (sva-playground)

## Known Issues Documented

1. **VCD Hierarchy Mismatch** (CRITICAL):
   - Loom expects flat VCD hierarchy
   - Most testbenches generate hierarchical VCDs
   - See troubleshooting-vcd.md for solutions

2. **Complex FSM Simulation**:
   - Some FSM designs don't simulate correctly
   - Under investigation (safe.v example in third_party tests)
   - May be related to synthesis optimization or reset handling

3. **Format String Preservation**:
   - Yosys may not preserve format attributes
   - Display messages show placeholders
   - Extract format strings from pre-synthesis JSON as workaround

## Contributing

When adding documentation:

1. **Be specific**: Include actual commands, file paths, code snippets
2. **Show examples**: Both working and non-working cases
3. **Link related docs**: Cross-reference other documentation files
4. **Date updates**: Update version and date at bottom of documents
5. **Test instructions**: Verify all commands actually work

## Future Documentation Needs

- [ ] Performance tuning guide (optimal `NUM_BLOCKS`, `--level-split`)
- [ ] Memory (SRAM) modeling and synthesis
- [ ] Custom cell library support beyond AIGPDK
- [ ] Multi-clock domain handling
- [ ] VCD scope option detailed behavior
- [ ] GPU kernel optimization internals
- [ ] Comparison with other simulators (VCS, Verilator, iverilog)

## Related Resources

- **Main README**: `../README.md` - Project overview and quick start
- **CLAUDE.md**: `../CLAUDE.md` - Development guidelines and architecture overview
- **Test Suite**: `../tests/` - Examples and regression tests
- **Third-Party Tests**: `../tests/regression/third_party/` - Real-world examples with attribution

---

**Last Updated**: 2026-02-16
**Maintained By**: ChipFlow + Community Contributions
