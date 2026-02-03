# Yosys synthesis script for $display test - JSON output

# Read RTL
read_verilog -sv display_simple.v

# Hierarchy check
hierarchy -check -top display_simple

# Synthesis
proc;;
opt_expr; opt_dff; opt_clean
synth -flatten

# Map formal cells to GEM cells BEFORE technology mapping
techmap -map ../../aigpdk/gem_formal.v

# Technology mapping to aigpdk
dfflibmap -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Write JSON output (preserves cell structure including $print)
write_json display_synth.json

# Print statistics
stat
