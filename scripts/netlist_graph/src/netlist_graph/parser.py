"""Parse Verilog netlists into graph structures."""

import logging
import re
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)

# SKY130 cell pin directions
OUTPUT_PINS = {"X", "Y", "Q", "Q_N", "COUT", "SUM", "DO", "Z"}
INPUT_PINS = {
    "A", "B", "C", "D", "A1", "A2", "A3", "B1", "B2", "C1", "D1",
    "S", "S0", "CLK", "RESET_B", "SET_B", "GATE", "EN",
    "A0", "DI", "AD", "BEN", "CLKin", "R_WB", "WLBI", "SM", "TM",
    "ScanInDR", "ScanInDL", "ScanInCC", "vpwrpc", "vpwrac",
}

# Cell pattern: cell_type instance_name (.port(net), ...);
CELL_PATTERN = re.compile(
    r"(\w+)\s+"  # cell type
    r"(\\[^\s]+|\w+)\s*"  # instance name (escaped or simple)
    r"\(([^;]+)\);",  # port connections
    re.MULTILINE | re.DOTALL,
)

# Port connection pattern
PORT_PATTERN = re.compile(r"\.(\w+)\s*\(([^)]*)\)")


def classify_pin(pin_name: str) -> str:
    """Classify a pin as 'input', 'output', or 'unknown'."""
    pin_base = pin_name.rstrip("0123456789")
    if pin_base in OUTPUT_PINS or pin_name in OUTPUT_PINS:
        return "output"
    if pin_base in INPUT_PINS or pin_name in INPUT_PINS:
        return "input"
    # Guess based on common patterns
    if pin_name.startswith("Q") or pin_name in ("X", "Y", "Z"):
        return "output"
    return "input"  # Default to input


def extract_nets(net_expr: str) -> list[str]:
    """Extract individual net names from a port expression."""
    net_expr = net_expr.strip()
    if not net_expr:
        return []

    if net_expr.startswith("{"):
        # Concatenation: {a, b, c}
        inner = net_expr[1:-1]
        nets = re.findall(r"\\[^\s,}]+|\w+\[[^\]]+\]|\w+", inner)
        return [n.strip() for n in nets if n.strip()]
    else:
        return [net_expr]


def parse_netlist(netlist_path: Path) -> nx.DiGraph:
    """
    Parse a Verilog netlist and build a directed graph.

    Nodes represent nets/wires.
    Edges represent signal flow through cells (input -> output).
    Edge attributes include cell instance name and type.
    """
    G = nx.DiGraph()
    content = netlist_path.read_text()

    cells_parsed = 0
    skip_types = {"wire", "input", "output", "inout", "module", "assign", "endmodule"}

    for match in CELL_PATTERN.finditer(content):
        cell_type = match.group(1)
        inst_name = match.group(2).strip()
        ports_str = match.group(3)

        if cell_type in skip_types:
            continue

        cells_parsed += 1

        # Parse port connections
        inputs: list[tuple[str, str]] = []   # (net, pin_name)
        outputs: list[tuple[str, str]] = []  # (net, pin_name)

        for port_match in PORT_PATTERN.finditer(ports_str):
            pin_name = port_match.group(1)
            net_expr = port_match.group(2)
            nets = extract_nets(net_expr)

            pin_type = classify_pin(pin_name)
            if pin_type == "output":
                for net in nets:
                    outputs.append((net, pin_name))
            else:
                for net in nets:
                    inputs.append((net, pin_name))

        # Add nodes
        for net, _pin in inputs + outputs:
            if net and not G.has_node(net):
                G.add_node(net, type="net")

        # Create edges from inputs to outputs through this cell
        for inp, inp_pin in inputs:
            for out, out_pin in outputs:
                if inp and out:
                    G.add_edge(
                        inp, out,
                        cell=inst_name, cell_type=cell_type,
                        in_pin=inp_pin, out_pin=out_pin,
                    )

    logger.info(f"Parsed {cells_parsed} cells")
    logger.info(f"Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G
