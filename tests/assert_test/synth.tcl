# Yosys synthesis script for assertion test
# Uses aigpdk library with GEM formal cell mapping

# Read RTL with formal constructs
read_verilog -sv -DFORMAL assert_simple.v

# Hierarchy check
hierarchy -check -top assert_simple

# Synthesis
proc;;
opt_expr; opt_dff; opt_clean
synth -flatten

# Map formal cells to GEM cells BEFORE technology mapping
# This converts $check -> GEM_ASSERT
techmap -map ../../aigpdk/gem_formal.v

# Technology mapping to aigpdk
dfflibmap -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Write output
write_verilog assert_synth.gv

# Print statistics
stat
