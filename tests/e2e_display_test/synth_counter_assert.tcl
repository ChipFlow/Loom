# Synthesis script for counter_with_assert.v to AIGPDK
# Tests both $display and assertions

# Read the design with FORMAL defined to enable assertions
read_verilog -sv -DFORMAL -DSYNTHESIS counter_with_assert.v

# Set top module
hierarchy -check -top counter_with_assert

# Synthesis - proc converts assertions to $check/$assert cells
proc
opt_expr
opt_clean
opt_dff
synth -flatten

# Mark $print and $check cells as keep (for GEM_DISPLAY and GEM_ASSERT)
select t:\$print t:\$check
setattr -set keep 1
select -clear

# Map formal cells to GEM cells BEFORE technology mapping
# This converts $check -> GEM_ASSERT and $print -> GEM_DISPLAY
techmap -map ../../aigpdk/gem_formal.v

# Technology mapping to AIGPDK
dfflibmap -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Workaround for yowasp-yosys ABC temp file issue
write_verilog -noattr /tmp/counter_assert_pre_abc.v

# ABC technology mapping
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Workaround before second ABC
write_verilog -noattr /tmp/counter_assert_pre_abc2.v
techmap
abc -liberty ../../aigpdk/aigpdk_nomem.lib
opt_clean -purge

# Remap DFFs to AIGPDK (ABC converts them back to $_DFF_* cells)
dfflibmap -liberty ../../aigpdk/aigpdk_nomem.lib
# Map simple cells ($_NOT_, $_AND_, etc) to AIGPDK gates
techmap -map map_simple.v
opt_clean -purge

# Write final gate-level outputs
write_verilog -noattr -noexpr counter_assert_synth.gv
write_json counter_assert_synth.json

# Statistics
stat
