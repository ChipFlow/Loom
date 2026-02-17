# netlist-graph

Graph-based netlist analysis tool for SKY130 post-synthesis netlists.

## Installation

```bash
cd scripts/netlist_graph
uv sync
```

## Usage

### Search for nets

```bash
uv run netlist-graph search <netlist.v> <pattern>
```

### Trace drivers (backwards)

```bash
uv run netlist-graph drivers <netlist.v> <net> [-d depth]
```

### Trace loads (forwards)

```bash
uv run netlist-graph loads <netlist.v> <net> [-d depth]
```

### Find path between nets

```bash
uv run netlist-graph path <netlist.v> <source> <target>
```

### Generate trace configuration

Generate configuration for monitoring signals in timing_sim_cpu:

```bash
# Simple output
uv run netlist-graph trace <netlist.v> <signal1> <signal2> ...

# Generate Rust code snippet
uv run netlist-graph trace <netlist.v> ibus__cyc rst_n_sync.rst --rust

# Generate JSON for programmatic use
uv run netlist-graph trace <netlist.v> ibus__cyc rst_n_sync.rst --json
```

### Generate watchlist file

Create a JSON watchlist that timing_sim_cpu can load:

```bash
uv run netlist-graph watchlist <netlist.v> <output.json> <signal1> <signal2> ...
```

Example:
```bash
uv run netlist-graph watchlist design.v watch.json ibus__cyc rst_n_sync gpio_out
```

### Interactive mode

```bash
uv run netlist-graph interactive <netlist.v>
```

Commands in interactive mode:
- `d <net>` - find drivers
- `l <net>` - find loads
- `p <src> <tgt>` - find path
- `s <pattern>` - search nets
- `c <net>` - cone of influence
- `q` - quit

## Examples

```bash
# Find what drives the ibus_cyc signal
uv run netlist-graph drivers tests/timing_test/minimal_build/6_final.v "ibus__cyc" -d 8

# Trace reset synchronizer path
uv run netlist-graph drivers tests/timing_test/minimal_build/6_final.v "rst_n_sync.rst"

# Find path from reset input to CPU
uv run netlist-graph path tests/timing_test/minimal_build/6_final.v "gpio_in[40]" "ibus__cyc"

# Generate watchlist for debugging
uv run netlist-graph watchlist tests/timing_test/minimal_build/6_final.v watch.json \
    ibus__cyc rst_n_sync.rst gpio_out[0] gpio_out[1]
```

## Signal Types

The tool classifies signals based on their drivers:
- `reg` - Driven by a flip-flop (dfxtp, dfrtp, etc.)
- `mem` - Driven by SRAM
- `comb` - Combinational logic (gates, buffers)
