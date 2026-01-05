# $stop/$finish Test Case

This test case validates GEM's handling of simulation control system tasks.

## Test Files

- `finish_test.v` - DUT with a counter that sets a 'done' flag
- `finish_test_tb.v` - Testbench that calls $finish when done

## Running with iverilog (Golden Reference)

```bash
cd tests/stop_finish_test
iverilog -o finish_test finish_test_tb.v finish_test.v
vvp finish_test
```

This generates `finish_test.vcd` as the golden reference.

## Current Implementation Status

The infrastructure for $stop/$finish support is in place:

1. **Event Buffer** (`src/event_buffer.rs`, `csrc/event_buffer.h`)
   - EventType::Stop and EventType::Finish defined
   - EventBuffer struct for GPU→CPU communication
   - process_events() handles Stop/Finish events

2. **AIG Extension** (`src/aig.rs`)
   - SimControlType enum (Stop, Finish)
   - SimControlNode struct with condition input
   - EndpointGroup::SimControl variant
   - simcontrols collection in AIG

3. **Metal Kernel** (`csrc/kernel_v1.metal`)
   - Event buffer parameter added
   - write_sim_control_event() helper available
   - TODO: Actual event writing when script includes SimControl data

4. **metal_test.rs**
   - Event buffer allocation and processing
   - Simulation terminates on Finish, pauses on Stop

## What's Still Needed

To complete Phase 1:

1. **yosys-slang Integration**
   - Parse $check cells with STOP/FINISH flavor
   - Map to SimControlNode in AIG

2. **Script Generation**
   - Include SimControl condition in partition scripts
   - Generate event writing instructions

3. **Kernel Implementation**
   - Add event writing logic for SimControl nodes
   - Check condition and write event when true

## Testing

Once full integration is complete:

```bash
# Generate golden VCD with iverilog
cd tests/stop_finish_test
iverilog -o finish_test finish_test_tb.v finish_test.v
vvp finish_test

# Run with GEM (TBD)
# The simulation should terminate when $finish is encountered
```
