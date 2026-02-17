# Yosys synthesis script for DFF timing test
# Uses aigpdk library for GEM simulation

# Read RTL
read_verilog dff_test.v

# Hierarchy check
hierarchy -check -top dff_test

# Synthesis
proc;;
opt_expr; opt_dff; opt_clean
synth -flatten

# Technology mapping to aigpdk
dfflibmap -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Write output
write_verilog dff_test_synth.gv

# Print statistics
stat
