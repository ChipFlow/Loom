"""Command-line interface for netlist-graph."""

import logging
import sys
from pathlib import Path

import click

from netlist_graph.graph import NetlistGraph

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx, verbose):
    """Graph-based netlist analysis tool."""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command()
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
@click.argument("net", type=str)
@click.option("-d", "--depth", default=10, help="Maximum trace depth")
def drivers(netlist: Path, net: str, depth: int):
    """Trace backwards to find drivers of a net."""
    graph = NetlistGraph.from_file(netlist)
    click.echo(f"Loaded {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Search for matching nets
    matches = graph.search(net)
    if not matches:
        click.echo(f"No nets matching '{net}'")
        return

    if len(matches) > 1:
        click.echo(f"Multiple matches for '{net}':")
        for m in matches:
            click.echo(f"  {m}")
        return

    target = matches[0]
    click.echo(f"\nTracing drivers of: {target}")

    trace = graph.trace_back(target, max_depth=depth)
    for d, n, driver in trace:
        indent = "  " * d
        if driver:
            click.echo(f"{indent}<- [{driver.cell_type}] {driver.cell}")
        click.echo(f"{indent}{n}")


@main.command()
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
@click.argument("net", type=str)
@click.option("-d", "--depth", default=3, help="Maximum trace depth")
def loads(netlist: Path, net: str, depth: int):
    """Trace forwards to find loads of a net."""
    graph = NetlistGraph.from_file(netlist)
    click.echo(f"Loaded {graph.num_nodes} nodes, {graph.num_edges} edges")

    matches = graph.search(net)
    if not matches:
        click.echo(f"No nets matching '{net}'")
        return

    if len(matches) > 1:
        click.echo(f"Multiple matches for '{net}':")
        for m in matches:
            click.echo(f"  {m}")
        return

    source = matches[0]
    click.echo(f"\nTracing loads of: {source}")

    trace = graph.trace_forward(source, max_depth=depth)
    for d, n, load in trace:
        indent = "  " * d
        click.echo(f"{indent}{n}")
        if load:
            click.echo(f"{indent}-> [{load.cell_type}] {load.cell}")


@main.command()
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
@click.argument("source", type=str)
@click.argument("target", type=str)
def path(netlist: Path, source: str, target: str):
    """Find shortest path between two nets."""
    graph = NetlistGraph.from_file(netlist)
    click.echo(f"Loaded {graph.num_nodes} nodes, {graph.num_edges} edges")

    src_matches = graph.search(source)
    tgt_matches = graph.search(target)

    if len(src_matches) != 1:
        click.echo(f"Source matches: {src_matches}")
        return
    if len(tgt_matches) != 1:
        click.echo(f"Target matches: {tgt_matches}")
        return

    src = src_matches[0]
    tgt = tgt_matches[0]

    path_result = graph.shortest_path(src, tgt)
    if not path_result:
        click.echo(f"No path from {src} to {tgt}")
        return

    click.echo(f"\nPath from {src} to {tgt} ({len(path_result)} hops):")
    for net, cell, cell_type in path_result:
        click.echo(f"  {net}")
        if cell:
            click.echo(f"    -> [{cell_type}] {cell}")


@main.command()
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
@click.argument("pattern", type=str)
def search(netlist: Path, pattern: str):
    """Search for nets matching a pattern."""
    graph = NetlistGraph.from_file(netlist)

    matches = graph.search(pattern, limit=50)
    click.echo(f"Found {len(matches)} matches:")
    for m in matches:
        click.echo(f"  {m}")


@main.command()
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
@click.argument("signals", type=str, nargs=-1)
@click.option("--rust", is_flag=True, help="Generate Rust code for timing_sim_cpu")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON for programmatic use")
def trace(netlist: Path, signals: tuple[str, ...], rust: bool, json_out: bool):
    """Generate trace configuration for monitoring signals.

    Finds nets matching the given patterns and outputs configuration
    for monitoring them in timing_sim_cpu.

    Example:
        netlist-graph trace design.v ibus__cyc rst_n_sync.rst --rust
    """
    import json

    graph = NetlistGraph.from_file(netlist)

    results = []
    for pattern in signals:
        matches = graph.search(pattern)
        if not matches:
            click.echo(f"Warning: No match for '{pattern}'", err=True)
            continue

        if len(matches) > 1:
            click.echo(f"Warning: Multiple matches for '{pattern}': {matches[:5]}", err=True)
            # Use exact match if available
            exact = [m for m in matches if pattern in m]
            if len(exact) == 1:
                matches = exact
            else:
                continue

        net_name = matches[0]
        # Find what drives this net to determine signal type
        drivers = graph.find_drivers(net_name)

        signal_type = "combinational"
        driver_cell = None
        if drivers:
            driver = drivers[0]
            driver_cell = driver.cell_type
            if "dff" in driver.cell_type.lower() or "dfx" in driver.cell_type.lower():
                signal_type = "register"
            elif "sram" in driver.cell_type.lower() or "ram" in driver.cell_type.lower():
                signal_type = "memory"

        results.append({
            "pattern": pattern,
            "net_name": net_name,
            "signal_type": signal_type,
            "driver_cell": driver_cell,
        })

    if json_out:
        click.echo(json.dumps(results, indent=2))
        return

    if rust:
        click.echo("// Add to timing_sim_cpu.rs - signal monitoring")
        click.echo("// Paste this in the signal setup section\n")

        click.echo("// Signal name patterns to monitor")
        click.echo("let trace_patterns: &[(&str, &str)] = &[")
        for r in results:
            click.echo(f'    ("{r["pattern"]}", "{r["net_name"]}"),')
        click.echo("];\n")

        click.echo("// Find pins for traced signals")
        click.echo("let mut trace_pins: Vec<(String, usize)> = Vec::new();")
        click.echo("for (label, pattern) in trace_patterns {")
        click.echo("    if let Some(pin) = find_pin_by_pattern(pattern) {")
        click.echo("        trace_pins.push((label.to_string(), pin));")
        click.echo("    }")
        click.echo("}\n")

        click.echo("// In simulation loop, add:")
        click.echo("// for (label, pin) in &trace_pins {")
        click.echo('//     let val = circ_state[*pin];')
        click.echo('//     print!("{label}={val} ");')
        click.echo("// }")
        click.echo("// println!();")
        return

    # Default: simple output
    click.echo(f"\nFound {len(results)} signals to trace:")
    for r in results:
        click.echo(f"  {r['pattern']}")
        click.echo(f"    Net: {r['net_name']}")
        click.echo(f"    Type: {r['signal_type']}")
        if r['driver_cell']:
            click.echo(f"    Driver: {r['driver_cell']}")


@main.command()
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
@click.argument("signals", type=str, nargs=-1)
def watchlist(netlist: Path, output: Path, signals: tuple[str, ...]):
    """Generate a watchlist file for timing_sim_cpu.

    Creates a JSON file mapping signal names to their net names,
    which timing_sim_cpu can load with --watchlist.

    Example:
        netlist-graph watchlist design.v watch.json ibus__cyc rst_n_sync gpio_out
    """
    import json

    graph = NetlistGraph.from_file(netlist)

    watchlist_data = {"signals": []}

    for pattern in signals:
        matches = graph.search(pattern)
        if not matches:
            click.echo(f"Warning: No match for '{pattern}'", err=True)
            continue

        # Add all matches for bus signals
        for net_name in matches[:10]:  # Limit to 10 per pattern
            drivers = graph.find_drivers(net_name)
            signal_type = "comb"
            if drivers:
                cell_type = drivers[0].cell_type.lower()
                if "dff" in cell_type or "dfx" in cell_type:
                    signal_type = "reg"
                elif "sram" in cell_type or "ram" in cell_type:
                    signal_type = "mem"

            watchlist_data["signals"].append({
                "name": pattern if len(matches) == 1 else net_name,
                "net": net_name,
                "type": signal_type,
            })

    output.write_text(json.dumps(watchlist_data, indent=2))
    click.echo(f"Wrote {len(watchlist_data['signals'])} signals to {output}")


@main.command()
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
@click.argument("signal", type=str)
@click.option("-d", "--depth", default=20, help="Maximum trace depth")
@click.option("--through-regs", is_flag=True, help="Continue tracing past register boundaries")
def cone(netlist: Path, signal: str, depth: int, through_regs: bool):
    """Trace the combinational logic cone of a signal.

    SIGNAL can be a hierarchical name (e.g., top.soc.sram.wb_bus__ack),
    a raw net name (e.g., _05294_), or a glob pattern.

    Stops at register (DFF) boundaries by default. Shows the logic tree
    with cell types and pin connections.

    Examples:
        netlist-graph cone design.v top.soc.sram.wb_bus__ack
        netlist-graph cone design.v _05294_ --through-regs
    """
    graph = NetlistGraph.from_file(netlist)
    click.echo(f"Loaded {graph.num_nodes} nodes, {graph.num_edges} edges")

    # Resolve signal name
    resolved = graph.resolve_name(signal)
    if not resolved:
        matches = graph.search(signal, limit=10)
        if matches:
            click.echo(f"No unique match for '{signal}'. Candidates:")
            for m in matches:
                click.echo(f"  {graph._short_net(m)}")
        else:
            click.echo(f"No nets matching '{signal}'")
        return

    click.echo(f"\nLogic cone of: {graph._short_net(resolved)}")
    if not through_regs:
        click.echo("(stopping at register boundaries, use --through-regs to continue)\n")

    cone_data = graph.logic_cone(resolved, max_depth=depth, through_regs=through_regs)

    for depth_level, net, cell_type, input_pins in cone_data:
        indent = "  " * depth_level
        short_net = graph._short_net(net)
        short_type = graph._short_cell_type(cell_type)

        if cell_type == "(primary)":
            click.echo(f"{indent}{short_net}  [primary input]")
        elif cell_type == "(reg-output)":
            click.echo(f"{indent}{short_net}  [REG output]")
        else:
            is_reg = graph._is_register(cell_type)
            pin_str = ", ".join(
                f"{pin}={graph._short_net(src)}" for pin, src in input_pins
            )
            reg_marker = " [REG]" if is_reg else ""
            click.echo(f"{indent}{short_net} = {short_type}({pin_str}){reg_marker}")


@main.command("interactive")
@click.argument("netlist", type=click.Path(exists=True, path_type=Path))
def interactive_mode(netlist: Path):
    """Interactive query mode."""
    graph = NetlistGraph.from_file(netlist)
    click.echo(f"Loaded {graph.num_nodes} nodes, {graph.num_edges} edges")

    click.echo("\nCommands:")
    click.echo("  d <net>     - find drivers of net")
    click.echo("  l <net>     - find loads of net")
    click.echo("  p <s> <t>   - find path from s to t")
    click.echo("  s <pattern> - search for nets")
    click.echo("  c <net>     - cone of influence")
    click.echo("  q           - quit")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue

        parts = line.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd == "q":
            break

        elif cmd == "d" and len(parts) >= 2:
            matches = graph.search(parts[1])
            if len(matches) == 1:
                target = matches[0]
                click.echo(f"Drivers of {target}:")
                for driver in graph.find_drivers(target):
                    click.echo(f"  <- [{driver.cell_type}] {driver.cell} <- {driver.from_net}")
            elif matches:
                click.echo(f"Multiple matches: {matches[:10]}")
            else:
                click.echo(f"No match for {parts[1]}")

        elif cmd == "l" and len(parts) >= 2:
            matches = graph.search(parts[1])
            if len(matches) == 1:
                source = matches[0]
                click.echo(f"Loads of {source}:")
                for load in graph.find_loads(source)[:20]:
                    click.echo(f"  -> [{load.cell_type}] {load.cell} -> {load.to_net}")
            elif matches:
                click.echo(f"Multiple matches: {matches[:10]}")
            else:
                click.echo(f"No match for {parts[1]}")

        elif cmd == "p" and len(parts) >= 3:
            src_matches = graph.search(parts[1])
            tgt_matches = graph.search(parts[2])
            if len(src_matches) == 1 and len(tgt_matches) == 1:
                path_result = graph.shortest_path(src_matches[0], tgt_matches[0])
                if path_result:
                    click.echo(f"Path ({len(path_result)} hops):")
                    for net, cell, cell_type in path_result:
                        click.echo(f"  {net}")
                        if cell:
                            click.echo(f"    -> [{cell_type}] {cell}")
                else:
                    click.echo("No path found")
            else:
                click.echo(f"Source: {src_matches[:5]}")
                click.echo(f"Target: {tgt_matches[:5]}")

        elif cmd == "s" and len(parts) >= 2:
            matches = graph.search(parts[1], limit=30)
            click.echo(f"Found {len(matches)} matches:")
            for m in matches:
                click.echo(f"  {m}")

        elif cmd == "c" and len(parts) >= 2:
            matches = graph.search(parts[1])
            if len(matches) == 1:
                cone = graph.cone_of_influence(matches[0])
                click.echo(f"Cone of influence: {len(cone)} nets")
                # Show primary inputs in the cone
                primaries = [n for n in cone if not graph.find_drivers(n)]
                click.echo(f"Primary inputs in cone: {primaries[:20]}")
            elif matches:
                click.echo(f"Multiple matches: {matches[:10]}")
            else:
                click.echo(f"No match for {parts[1]}")

        else:
            click.echo(f"Unknown command: {line}")


if __name__ == "__main__":
    main()
