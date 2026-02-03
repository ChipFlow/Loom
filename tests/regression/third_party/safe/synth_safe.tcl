# Synthesis script for safe.v to AIGPDK
# Tests FSM with assertions for backdoor detection

# Read the design
read_verilog -sv -DFORMAL -DSYNTHESIS safe.v

# Set top module
hierarchy -check -top safe

# Synthesis - proc converts assertions to $check cells
proc
opt_expr
opt_clean
opt_dff
synth -flatten

# Mark $print and $check cells as keep
select t:\$print t:\$check
setattr -set keep 1
select -clear

# Map formal cells to GEM cells
techmap -map ../../../../aigpdk/gem_formal.v

# Technology mapping to AIGPDK
dfflibmap -liberty ../../../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# ABC technology mapping
abc -liberty ../../../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty ../../../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Remap DFFs and map simple cells
dfflibmap -liberty ../../../../aigpdk/aigpdk_nomem.lib
techmap -map ../../../../tests/e2e_display_test/map_simple.v
opt_clean -purge

# Write outputs
write_verilog -noattr -noexpr safe_synth.gv
write_json safe_synth.json

# Statistics
stat
