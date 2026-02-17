"""Graph wrapper with query methods for netlist analysis."""

import fnmatch
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from netlist_graph.parser import parse_netlist


@dataclass
class Driver:
    """A driver of a net."""

    from_net: str
    cell: str
    cell_type: str
    in_pin: str = ""
    out_pin: str = ""


@dataclass
class Load:
    """A load on a net."""

    to_net: str
    cell: str
    cell_type: str
    in_pin: str = ""
    out_pin: str = ""


class NetlistGraph:
    """Graph-based netlist analysis."""

    def __init__(self, graph: nx.DiGraph):
        self._graph = graph

    @classmethod
    def from_file(cls, netlist_path: Path) -> "NetlistGraph":
        """Load a netlist from a Verilog file."""
        graph = parse_netlist(netlist_path)
        return cls(graph)

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def resolve_name(self, name: str) -> str | None:
        """Resolve a hierarchical signal name to a net in the graph.

        Tries several naming conventions:
          - Exact match
          - With \\inst$ prefix (OpenROAD convention)
          - Glob search fallback
        """
        # Exact match
        if self._graph.has_node(name):
            return name

        # Try \\inst$ prefix (OpenROAD hierarchical naming)
        prefixed = f"\\inst${name}"
        if self._graph.has_node(prefixed):
            return prefixed

        # Try as a search pattern and take unique match
        matches = self.search(name, limit=5)
        if len(matches) == 1:
            return matches[0]

        return None

    def search(self, pattern: str, limit: int = 20) -> list[str]:
        """Search for nets matching a glob pattern.

        Brackets containing numbers (e.g., [0], [31:0]) are escaped
        to match literal Verilog bus indices.
        """
        import re

        # Escape brackets that look like Verilog bus indices [N] or [N:M]
        escaped = re.sub(r"\[(\d+)(:\d+)?\]", r"[\1\2]".replace("[", "[[]").replace("]", "[]]"), pattern)
        # Simpler: just escape all brackets if they contain digits
        escaped = re.sub(r"\[(\d+)\]", lambda m: f"[[]" + m.group(1) + "[]]", pattern)

        if "*" not in escaped and "?" not in escaped:
            escaped = f"*{escaped}*"
        matches = [n for n in self._graph.nodes() if fnmatch.fnmatch(n, escaped)]
        return sorted(matches)[:limit]

    def has_net(self, net: str) -> bool:
        """Check if a net exists."""
        return self._graph.has_node(net)

    def find_drivers(self, net: str) -> list[Driver]:
        """Find all cells that drive a net."""
        if not self._graph.has_node(net):
            return []
        drivers = []
        for pred in self._graph.predecessors(net):
            edge_data = self._graph.get_edge_data(pred, net)
            drivers.append(
                Driver(
                    from_net=pred,
                    cell=edge_data.get("cell", "?"),
                    cell_type=edge_data.get("cell_type", "?"),
                    in_pin=edge_data.get("in_pin", ""),
                    out_pin=edge_data.get("out_pin", ""),
                )
            )
        return drivers

    def find_loads(self, net: str) -> list[Load]:
        """Find all cells that are driven by a net."""
        if not self._graph.has_node(net):
            return []
        loads = []
        for succ in self._graph.successors(net):
            edge_data = self._graph.get_edge_data(net, succ)
            loads.append(
                Load(
                    to_net=succ,
                    cell=edge_data.get("cell", "?"),
                    cell_type=edge_data.get("cell_type", "?"),
                    in_pin=edge_data.get("in_pin", ""),
                    out_pin=edge_data.get("out_pin", ""),
                )
            )
        return loads

    def trace_back(self, net: str, max_depth: int = 10) -> list[tuple[int, str, Driver | None]]:
        """
        Trace backwards from a net to find its driver chain.

        Returns list of (depth, net, driver) tuples.
        """
        result = []
        visited = set()
        queue = [(net, 0, None)]

        while queue:
            current_net, depth, driver = queue.pop(0)
            if depth > max_depth or current_net in visited:
                continue
            visited.add(current_net)

            result.append((depth, current_net, driver))

            drivers = self.find_drivers(current_net)
            for d in drivers:
                queue.append((d.from_net, depth + 1, d))

        return result

    def trace_forward(self, net: str, max_depth: int = 3) -> list[tuple[int, str, Load | None]]:
        """
        Trace forwards from a net to find its load chain.

        Returns list of (depth, net, load) tuples.
        """
        result = []
        visited = set()
        queue = [(net, 0, None)]

        while queue:
            current_net, depth, load = queue.pop(0)
            if depth > max_depth or current_net in visited:
                continue
            visited.add(current_net)

            result.append((depth, current_net, load))

            loads = self.find_loads(current_net)
            for l in loads[:10]:  # Limit fanout
                queue.append((l.to_net, depth + 1, l))

        return result

    def shortest_path(self, source: str, target: str) -> list[tuple[str, str | None, str | None]]:
        """
        Find shortest path between two nets.

        Returns list of (net, cell, cell_type) tuples along the path.
        """
        try:
            path = nx.shortest_path(self._graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

        result = []
        for i, net in enumerate(path):
            if i < len(path) - 1:
                edge = self._graph.get_edge_data(net, path[i + 1])
                result.append((net, edge.get("cell"), edge.get("cell_type")))
            else:
                result.append((net, None, None))

        return result

    def cone_of_influence(self, net: str, max_depth: int = 10) -> set[str]:
        """Find all nets in the backwards cone of influence."""
        cone = set()
        queue = [net]
        depth = 0

        while queue and depth < max_depth:
            next_queue = []
            for n in queue:
                if n in cone:
                    continue
                cone.add(n)
                for d in self.find_drivers(n):
                    next_queue.append(d.from_net)
            queue = next_queue
            depth += 1

        return cone

    @staticmethod
    def _is_register(cell_type: str) -> bool:
        """Check if a cell type is a register (DFF/latch)."""
        ct = cell_type.lower()
        return any(k in ct for k in ("dff", "dfx", "dlat", "dlxtp", "sdff", "sdf"))

    @staticmethod
    def _short_cell_type(cell_type: str) -> str:
        """Shorten sky130_fd_sc_hd__nand2_1 to nand2_1."""
        for prefix in ("sky130_fd_sc_hd__", "sky130_fd_sc_hs__", "sky130_fd_sc_ms__"):
            if cell_type.startswith(prefix):
                return cell_type[len(prefix):]
        return cell_type

    @staticmethod
    def _short_net(net: str) -> str:
        """Shorten \\inst$top.soc.foo to top.soc.foo."""
        if net.startswith("\\inst$"):
            return net[6:]
        return net

    # Pins that are data inputs to DFFs (trace through these)
    _DFF_DATA_PINS = {"D", "DE", "SCD"}
    # Pins that are control inputs to DFFs (stop here)
    _DFF_CONTROL_PINS = {"CLK", "RESET_B", "SET_B", "GATE", "EN"}

    def logic_cone(
        self,
        net: str,
        max_depth: int = 20,
        through_regs: bool = False,
    ) -> list[tuple[int, str, str, list[tuple[str, str]]]]:
        """Trace the combinational logic cone of a net.

        Returns a tree as a list of (depth, net, cell_type, inputs) tuples
        where inputs is [(pin_name, source_net), ...].

        By default stops at register (DFF) output boundaries — it traces
        through the D input of a DFF but stops when it reaches a net
        driven by another DFF's Q output. Use through_regs=True to
        continue past register outputs.
        """
        result: list[tuple[int, str, str, list[tuple[str, str]]]] = []
        visited: set[str] = set()

        def _trace(net: str, depth: int):
            if depth > max_depth or net in visited:
                return
            visited.add(net)

            drivers = self.find_drivers(net)
            if not drivers:
                # Primary input — no driver
                result.append((depth, net, "(primary)", []))
                return

            # Group drivers by cell (a cell may have multiple input edges)
            cells: dict[str, list[Driver]] = {}
            for d in drivers:
                cells.setdefault(d.cell, []).append(d)

            for cell_name, cell_drivers in cells.items():
                cell_type = cell_drivers[0].cell_type
                is_reg = self._is_register(cell_type)

                # Collect all input pins for this cell
                input_pins: list[tuple[str, str]] = [
                    (d.in_pin, d.from_net) for d in cell_drivers
                ]

                result.append((depth, net, cell_type, input_pins))

                if is_reg:
                    # For registers, trace data pins (D) but not control pins (CLK)
                    for pin, src in input_pins:
                        if pin in self._DFF_DATA_PINS:
                            if through_regs:
                                _trace(src, depth + 1)
                            else:
                                # Trace through D pin's combinational logic
                                _trace(src, depth + 1)
                        else:
                            # Control pin — don't recurse
                            pass
                else:
                    # Combinational cell — check if any source is a reg output
                    for _pin, src in input_pins:
                        if not through_regs and self._is_reg_output(src):
                            result.append((depth + 1, src, "(reg-output)", []))
                        else:
                            _trace(src, depth + 1)

        _trace(net, 0)
        return result

    def _is_reg_output(self, net: str) -> bool:
        """Check if a net is driven by a register's output pin."""
        drivers = self.find_drivers(net)
        return any(self._is_register(d.cell_type) for d in drivers)

    def fanout_cone(self, net: str, max_depth: int = 5) -> set[str]:
        """Find all nets in the forwards fanout cone."""
        cone = set()
        queue = [net]
        depth = 0

        while queue and depth < max_depth:
            next_queue = []
            for n in queue:
                if n in cone:
                    continue
                cone.add(n)
                for l in self.find_loads(n):
                    next_queue.append(l.to_net)
            queue = next_queue
            depth += 1

        return cone
