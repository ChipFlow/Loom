# Yosys synthesis script for $display test

# Read RTL - use yosys-slang if available
read_verilog -sv display_simple.v

# Hierarchy check
hierarchy -check -top display_simple

# Show initial cells to see $display
select -module display_simple
write_rtlil display_initial.rtlil

# Synthesis
proc;;
opt_expr; opt_dff; opt_clean

# Write RTLIL to see what $print cells look like
write_rtlil display_after_proc.rtlil

# Print cells to see $print structure
select -module display_simple
select t:$print
dump

# Map formal cells to GEM cells
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
write_verilog display_synth.gv
write_json display_synth.json

# Print statistics
stat
