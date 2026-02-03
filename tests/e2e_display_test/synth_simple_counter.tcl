# Synthesis script for simple_counter.v to AIGPDK
# This synthesizes the design module only (not the testbench)

# Read the design
read_verilog -sv -DSYNTHESIS simple_counter.v

# Set top module
hierarchy -check -top simple_counter

# Synthesis
proc
opt_expr
opt_clean
opt_dff

# Mark $print cells as keep (for GEM_DISPLAY)
select t:\$print
setattr -set keep 1
select -clear

# Technology mapping to AIGPDK
techmap -map ../../aigpdk/gem_formal.v
dfflibmap -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# ABC technology mapping
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge
techmap
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Remap DFFs to AIGPDK (ABC converts them back to $_DFF_* cells)
dfflibmap -liberty ../../aigpdk/aigpdk_nomem.lib
# Map simple cells ($_NOT_, $_AND_, etc) to AIGPDK gates
techmap -map map_simple.v
opt_clean -purge

# Write final gate-level outputs
write_verilog -noattr -noexpr simple_counter_synth.gv
write_json simple_counter_synth.json

# Statistics
stat
