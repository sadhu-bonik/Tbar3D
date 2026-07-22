import os
import ctypes
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from typing import Self

class Tbar:
    """
    A simulator for a 3D memristor crossbar array using Nodal Analysis (KCL).

    Attributes:
        n (int): Number of rows (wordlines).
        m (int): Number of columns (bitlines).
        p (int): Number of layers in the 3D structure.
        S (float): Input source conductance (Siemens).
        L (float): Output load conductance (Siemens).
        GW (float): Wire segment and via conductance (Siemens).
        G (nx.Graph): The underlying NetworkX graph structure.
        GMem (np.ndarray): A 3D array of shape (n, m, p) storing memristor conductances.
        VInput (np.ndarray): A 2D array of shape (n, p) storing input bias voltages.
        VOutput (np.ndarray): A 2D array of shape (m, p) storing output bias voltages.
    """

    def __init__(self, n: int, m: int, p: int, S: float = 1e12, L: float = 1e12, GW: float = 1e12) -> None:
        """
        Initializes the 3D crossbar array structure and sets parasitic conductances.

        Args:
            n (int): Number of rows (wordlines) in the array.
            m (int): Number of columns (bitlines) in the array.
            p (int): Number of vertical layers in the array.
            S (float, optional): Input resistor conductance from voltage source to the 
                first wordline node. Defaults to 1e12 (near-perfect conductor).
            L (float, optional): Output resistor conductance from the last bitline node 
                to the output sink. Defaults to 1e12 (near-perfect conductor).
            GW (float, optional): Wire segment and via conductance between internal 
                nodes. Defaults to 1e12.
        """
        self.n: int = n  
        self.m: int = m  
        self.p: int = p
        self.S = float(S)
        self.L = float(L)
        self.GW = float(GW)

        # Pure NumPy memory, no graph overhead
        self.GMem = np.full((p, n, m), np.nan, dtype=np.float64)
        self.VInput = np.full((n, p), np.nan, dtype=np.float64)
        self.VOutput = np.full((m, p), np.nan, dtype=np.float64)

        self._voltages = None
        self.last_cpp_solve_seconds = None
        self._load_cpp_solver()

    def _load_cpp_solver(self):
        """Loads the compiled C++ shared library."""
        # Ensure the shared library is in the same directory or adjust path.
        lib_dir = os.path.dirname(__file__)
        candidate_names = ["solver.dll", "solver.so", "solver.dylib"]
        lib_path = None
        for name in candidate_names:
            candidate_path = os.path.join(lib_dir, name)
            if os.path.exists(candidate_path):
                lib_path = candidate_path
                break
        if lib_path is None:
            raise FileNotFoundError(
                f"Cannot find C++ shared library in {lib_dir}. Please compile solver.dll/solver.so first."
            )
        
        self.clib = ctypes.CDLL(lib_path)
        self.clib.solve_kcl.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS'), # GMem
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # VInput
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'), # VOutput
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'), # VKCL out
            ctypes.c_int, ctypes.c_int, ctypes.c_int,                               # n, m, p
            ctypes.c_double, ctypes.c_double, ctypes.c_double                       # S, L, GW
        ]

    def set_parasitic_resistance(self, RW: float):
        """
        Updates the parasitic resistance

        Args:
            RW (float, optional): New wire/via resistance.
        """
        self.GW = 1.0 / RW if RW != 0 else 1e12

    def set_parasitic_conductance(self, GW: float):
        """
        Updates the parasitic conductance

        Args:
            GW (float, optional): New wire/via conductance.
        """
        self.GW = GW

    def set_input_resistance(self, S: float):
        """
        Updates the input resistance

        Args:
            S (float, optional): New input resistance.
        """
        self.S = 1.0 / S if S != 0 else 1e12

    def set_output_resistance(self, L: float):
        """
        Updates the output resistance

        Args:
            L (float, optional): New output resistance.
        """
        self.L = 1.0 / L if L != 0 else 1e12

    def set_input_conductance(self, S: float):
        """
        Updates the input conductance

        Args:
            S (float, optional): New input conductance.
        """
        self.S = S

    def set_output_conductance(self, L: float):
        self.L = float(L)

    def set_conductances(self, matrix):
        """
        Assigns memristor conductances in bulk using a 3D array.

        Args:
            matrix (np.ndarray | list): A 3D array of conductances. 

        Raises:
            ValueError: If the provided matrix does not match the shape (n, m, p).
        """
        matrix = np.asarray(matrix, dtype=float)
        expected = (self.p, self.n, self.m)
        if matrix.shape != expected:
            raise ValueError(f"matrix has shape {matrix.shape}, expected {expected} (p, n, m)")
        self.GMem = np.ascontiguousarray(matrix)

    def set_conductance(self, i: int, j: int, k: int, value: float) -> None:
        """
        Sets the conductance for a specific memristor at a given 3D coordinate.

        Args:
            i (int): Row index.
            j (int): Column index.
            k (int): Layer index.
            value (float): Conductance value in Siemens.
        """        
        self.GMem[k, i, j] = value

    def randomize_conductances(
        self,
        low: int = 0,
        high: int = 100,
        seed: int | None = None,
        only_unset: bool = True,
    ) -> None:
        """
        Fills the crossbar array with random memristor conductances.

        Args:
            low (int, optional): Minimum random conductance value. Defaults to 0.
            high (int, optional): Maximum random conductance value. Defaults to 100.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            only_unset (bool, optional): If True, only fills cells that have not been 
                manually set yet (NaN). If False, overwrites the entire array. Defaults to True.
        """
        rng = np.random.default_rng(seed)
        random_vals = rng.integers(low, high, size=self.GMem.shape).astype(np.float64)
        if only_unset:
            mask = np.isnan(self.GMem)
            self.GMem[mask] = random_vals[mask]
        else:
            self.GMem = np.ascontiguousarray(random_vals)

    def set_input_voltages(self, matrix):
        """
        Assigns input biases in bulk using a 2D array.

        Args:
            matrix (np.ndarray | list): A 2D array of voltages.

        Raises:
            ValueError: If the matrix does not match the shape (n, p).
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (self.n, self.p):
            raise ValueError(f"matrix has shape {matrix.shape}, expected {(self.n, self.p)} (n, p)")
        self.VInput = np.ascontiguousarray(matrix)

    def set_output_voltages(self, matrix):
        """
        Assigns output biases in bulk using a 2D array.

        Args:
            matrix (np.ndarray | list): A 2D array of voltages.

        Raises:
            ValueError: If the matrix does not match the shape (m, p).
        """
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (self.m, self.p):
            raise ValueError(f"matrix has shape {matrix.shape}, expected {(self.m, self.p)} (m, p)")
        self.VOutput = np.ascontiguousarray(matrix)

    def set_input_voltage(self, i: int, k: int, value: float) -> None:
        """Set a single input node I(i,k)."""
        self.VInput[i, k] = value

    def set_output_voltage(self, j: int, k: int, value: float) -> None:
        """Set a single output node O(j,k)."""
        self.VOutput[j, k] = value

    def set_bias(
        self,
        vin: float | None = None,
        vout: float | None = None,
        only_unset: bool = False,
    ) -> None:
        """
        Fills all input or output nodes with a uniform constant voltage.

        Args:
            vin (float, optional): The voltage to apply to input nodes.
            vout (float, optional): The voltage to apply to output nodes.
            only_unset (bool, optional): If True, only fills nodes that are currently NaN. 
                If False, overwrites all specified nodes. Defaults to False.
        """
        if vin is not None:
            if only_unset:
                mask = np.isnan(self.VInput)
                self.VInput[mask] = float(vin)
            else:
                self.VInput[:, :] = float(vin)
        if vout is not None:
            if only_unset:
                mask = np.isnan(self.VOutput)
                self.VOutput[mask] = float(vout)
            else:
                self.VOutput[:, :] = float(vout)

    def is_ready_to_solve(self) -> bool:
        cond_ready = not np.isnan(self.GMem).any()
        bias_ready = not np.isnan(self.VInput).any() and not np.isnan(self.VOutput).any()
        return cond_ready and bias_ready

    def _wordline_node_index(self, i: int, j: int, k: int) -> int:
        return int(2 * self.m * self.n * k + 2 * self.m * i + 2 * j)

    def _bitline_node_index(self, i: int, j: int, k: int) -> int:
        return int(2 * self.m * self.n * k + 2 * self.m * i + 2 * j + 1)

    def solve(self) -> Self:
        if not self.is_ready_to_solve():
            raise RuntimeError("Cannot solve: conductances and/or bias voltages are not fully set.")

        n, m, p = self.n, self.m, self.p
        size_kcl = 2 * n * m * p

        # Pre-allocate contiguous memory for the output
        VKCL = np.zeros(size_kcl, dtype=np.float64)

        # External C++ call
        import time
        start_time = time.perf_counter()
        self.clib.solve_kcl(
            self.GMem, self.VInput, self.VOutput, VKCL,
            n, m, p, self.S, self.L, self.GW
        )
        self.last_cpp_solve_seconds = time.perf_counter() - start_time
        print(f"solve_kcl() took {self.last_cpp_solve_seconds:.9f} seconds")

        # Store voltages for dictionary access mapping
        self._voltages = {}
        for k in range(p):
            for i in range(n):
                for j in range(m):
                    self._voltages[f"W({i},{j},{k})"] = VKCL[self._wordline_node_index(i, j, k)]
                    self._voltages[f"B({i},{j},{k})"] = VKCL[self._bitline_node_index(i, j, k)]
        for i in range(n):
            for k in range(p):
                self._voltages[f"I({i},{k})"] = self.VInput[i, k]
        for j in range(m):
            for k in range(p):
                self._voltages[f"O({j},{k})"] = self.VOutput[j, k]

        return self

    @property
    def solved(self) -> bool:
        return self._voltages is not None

    def get_voltage(self, node_name: str) -> float:
        """
        Retrieves the calculated voltage for a specific node.

        Args:
            node_name (str): The node string identifier (e.g., 'W(0,0,0)', 'I(1,0)').

        Returns:
            float: Voltage at the requested node.

        Raises:
            RuntimeError: If called before solve().
        """        
        if not self.solved:
            raise RuntimeError("Call solve() first.")
        return self._voltages[node_name]

    def _edge_weight(self, u: str, v: str) -> float:
        if u.startswith("I(") and v.startswith("W("):
            i, k = (int(x) for x in u[2:-1].split(","))
            wi, wj, wk = (int(x) for x in v[2:-1].split(","))
            if wi != i or wj != 0 or wk != k:
                raise KeyError(f"Nodes {u!r} and {v!r} are not adjacent.")
            return self.S
        if u.startswith("W(") and v.startswith("I("):
            return self._edge_weight(v, u)

        if u.startswith("O(") and v.startswith("B("):
            j, k = (int(x) for x in u[2:-1].split(","))
            bi, bj, bk = (int(x) for x in v[2:-1].split(","))
            if bi != self.n - 1 or bj != j or bk != k:
                raise KeyError(f"Nodes {u!r} and {v!r} are not adjacent.")
            return self.L
        if u.startswith("B(") and v.startswith("O("):
            return self._edge_weight(v, u)

        if u.startswith("W(") and v.startswith("B("):
            i, j, k = (int(x) for x in u[2:-1].split(","))
            bi, bj, bk = (int(x) for x in v[2:-1].split(","))
            if bi != i or bj != j or bk != k:
                raise KeyError(f"Nodes {u!r} and {v!r} are not adjacent.")
            return float(self.GMem[k, i, j])
        if u.startswith("B(") and v.startswith("W("):
            return self._edge_weight(v, u)

        # CORRECTED WORDLINE LOGIC
        if u.startswith("W(") and v.startswith("W("):
            i1, j1, k1 = (int(x) for x in u[2:-1].split(","))
            i2, j2, k2 = (int(x) for x in v[2:-1].split(","))
            
            is_j_adj = (i1 == i2 and k1 == k2 and abs(j1 - j2) == 1) # Horizontal segment
            is_k_adj = (i1 == i2 and j1 == j2 and abs(k1 - k2) == 1) # Vertical via
            
            if not (is_j_adj or is_k_adj):
                raise KeyError(f"Nodes {u!r} and {v!r} are not adjacent.")
            return self.GW

        # CORRECTED BITLINE LOGIC
        if u.startswith("B(") and v.startswith("B("):
            i1, j1, k1 = (int(x) for x in u[2:-1].split(","))
            i2, j2, k2 = (int(x) for x in v[2:-1].split(","))
            
            is_i_adj = (j1 == j2 and k1 == k2 and abs(i1 - i2) == 1) # Horizontal segment
            is_k_adj = (i1 == i2 and j1 == j2 and abs(k1 - k2) == 1) # Vertical via
            
            if not (is_i_adj or is_k_adj):
                raise KeyError(f"Nodes {u!r} and {v!r} are not adjacent.")
            return self.GW

        raise KeyError(f"Unsupported node pair: {u!r}, {v!r}")
    
    def get_current(self, u: str, v: str) -> float:
        if not self.solved:
            raise RuntimeError("Call solve() first.")
        return abs(self._voltages[u] - self._voltages[v]) * self._edge_weight(u, v)

    def show(self) -> go.Figure:
        """
        Lazily constructs a NetworkX graph representation strictly for Plotly 3D visualization.
        """
        import networkx as nx
        
        G = nx.Graph()
        n, m, p = self.n, self.m, self.p

        # 1. Build Nodes & Edges (Lazy Generation)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    G.add_node(f"W({i},{j},{k})", pos=(j, i, k), category="W")
                    G.add_node(f"B({i},{j},{k})", pos=(j + 0.5, i + 0.5, k - 0.5), category="B")
                    G.add_edge(f"W({i},{j},{k})", f"B({i},{j},{k})", category="4", weight=self.GMem[k, i, j])

                    if j < m - 1:
                        G.add_edge(f"W({i},{j},{k})", f"W({i},{j+1},{k})", category="3a", weight=self.GW)
                    if i < n - 1:
                        G.add_edge(f"B({i},{j},{k})", f"B({i+1},{j},{k})", category="3b", weight=self.GW)
                    if k < p - 1:
                        G.add_edge(f"W({i},{j},{k})", f"W({i},{j},{k+1})", category="3c", weight=self.GW)
                        G.add_edge(f"B({i},{j},{k})", f"B({i},{j},{k+1})", category="3c", weight=self.GW)

        for i in range(n):
            for k in range(p):
                G.add_node(f"I({i},{k})", pos=(-1, i, k), category="I")
                G.add_edge(f"I({i},{k})", f"W({i},0,{k})", category="1", weight=self.S)

        for j in range(m):
            for k in range(p):
                G.add_node(f"O({j},{k})", pos=(j + 0.5, n + 0.5, k - 0.5), category="O")
                G.add_edge(f"B({n-1},{j},{k})", f"O({j},{k})", category="2", weight=self.L)

        # 2. Plotly Visualization
        edge_styles = {
            "1": {"name": "Input Resistor", "color": "#00BFA5"},
            "2": {"name": "Output Resistor", "color": "#29B6F6"},
            "3a": {"name": "Wordline segment", "color": "#EC407A"},
            "3b": {"name": "Bitline segment", "color": "#5C6BC0"},
            "3c": {"name": "Vertical via", "color": "#BDC3C7"},
            "4": {"name": "Memristor", "color": "#FF9100"},
        }
        node_styles = {
            "W": {"name": "Wordline Node", "color": "#0000FF"},
            "B": {"name": "Bitline Node", "color": "#FF0000"},
            "I": {"name": "Input Node", "color": "#00FFFF"},
            "O": {"name": "Output Node", "color": "#FF00FF"},
        }

        traces = []
        for cat, style in edge_styles.items():
            ex, ey, ez, mid_x, mid_y, mid_z, mid_text = [], [], [], [], [], [], []
            for u, v, data in G.edges(data=True):
                if data["category"] != cat: continue
                x0, y0, z0 = G.nodes[u]["pos"]
                x1, y1, z1 = G.nodes[v]["pos"]
                ex += [x0, x1, None]; ey += [y0, y1, None]; ez += [z0, z1, None]
                mid_x.append((x0 + x1) / 2); mid_y.append((y0 + y1) / 2); mid_z.append((z0 + z1) / 2)

                text = f"{u} - {v}<br>G = {data['weight']:.3g}"
                if self.solved:
                    current = self.get_current(u, v)
                    text += f"<br>I = {current:.4g}"
                mid_text.append(text)

            traces.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", 
                                       line=dict(color=style["color"], width=10 if cat == "4" else 4),
                                       name=style["name"], hoverinfo="none", showlegend=(cat != "4")))
            
            marker_style = dict(size=4, color="#1A1A1A", symbol="diamond") if cat == "4" else dict(size=2, color=style["color"])
            traces.append(go.Scatter3d(x=mid_x, y=mid_y, z=mid_z, mode="markers", marker=marker_style,
                                       name=style["name"], text=mid_text, hoverinfo="text", showlegend=(cat == "4")))

        for cat, style in node_styles.items():
            nx_, ny_, nz_, ntext = [], [], [], []
            for node, data in G.nodes(data=True):
                if data["category"] != cat: continue
                x, y, z = data["pos"]
                nx_.append(x); ny_.append(y); nz_.append(z)
                val = self._voltages.get(node) if self.solved else None
                label = f"{node}<br>V = {val:.3g}" if val is not None else node
                ntext.append(label)

            traces.append(go.Scatter3d(x=nx_, y=ny_, z=nz_, mode="markers", 
                                       marker=dict(size=5, color=style["color"]),
                                       name=style["name"], text=ntext, hoverinfo="text"))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"Crossbar topology ({self.n}x{self.m}x{self.p})",
            scene=dict(xaxis_title="X (j)", yaxis_title="Y (i)", zaxis_title="Z (k)", bgcolor="white")
        )
        fig.show()
        return fig


if __name__ == "__main__":
    cb = Tbar(n=3, m=3, p=3)
    cb.randomize_conductances(low=10, high=100, seed=42)
    cb.show()
