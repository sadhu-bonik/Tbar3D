"""
crossbar_kcl.py
================

Automated graph-based formulation for an m x n x p 3D memristor crossbar,
following the (V, E, w) definition:

    V = VW ∪ VB ∪ VI ∪ VO
    E = EA ∪ EB ∪ EC ∪ ED

This module does two things:

1.  GENERIC MODE  (build_graph / kcl_equation_for_node)
    Actually builds the graph for a concrete (m, n, p) using networkx exactly
    as specified (every vertex + every weighted edge), then derives the KCL
    equation for ANY node by literally summing over its incident edges. This
    is the "automated" part -- no case-by-case logic is hard-coded here, it
    falls straight out of the graph.

2.  TEMPLATE MODE (the 18 symbolic cases)
    Every W(i,j,k) node's equation only depends on whether j is at the low
    boundary (j=0), an interior value (0<j<m-1), or the high boundary
    (j=m-1), and similarly whether k is 0 / interior / p-1. That is a 3x3 =
    9-way classification. Symmetrically every B(i,j,k) node's equation only
    depends on the boundary class of i (0 / interior / n-1) and of k
    (0 / interior / p-1) -> another 3x3 = 9 cases. 9 + 9 = 18 total, matching
    the paper's "18 Cases for KCL".

    generate_18_cases() builds a symbolic sympy equation for each of the 18
    cases from first principles (which neighbors exist), and
    verify_templates_against_graph() cross-checks every single template
    against an actual instantiated graph (mode 1) for a concrete (m,n,p) to
    prove the two derivations agree.

Run this file directly to print all 18 equations and the verification
result.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import product
import sympy as sp
import networkx as nx


# --------------------------------------------------------------------------
# 1. GENERIC GRAPH CONSTRUCTION  (exactly per the paper's V, E, w definition)
# --------------------------------------------------------------------------

def node_name(kind: str, *idx: int) -> str:
    """Canonical vertex name, e.g. node_name('W', 1, 2, 0) -> 'W_1_2_0'."""
    return f"{kind}_" + "_".join(str(x) for x in idx)


def build_graph(m: int, n: int, p: int, symbolic: bool = True) -> nx.Graph:
    """
    Build the full graph G(V, E, w) for an m x n x p crossbar.

    Vertices:
        VW: W(i,j,k)   0<=i<n, 0<=j<m, 0<=k<p
        VB: B(i,j,k)   0<=i<n, 0<=j<m, 0<=k<p
        VI: I(i,k)     0<=i<n, 0<=k<p
        VO: O(j,k)     0<=j<m, 0<=k<p

    Edges (with weights = conductance in Siemens):
        EA: (I(i,k), W(i,0,k))            weight S
        EB: (B(n-1,j,k), O(j,k))          weight L
        EC: wire segments                 weight Gwire
        ED: (W(i,j,k), B(i,j,k))          weight G(i,j,k)   [memristor]

    If symbolic=True, Gwire/S/L are sympy symbols and each memristor gets its
    own symbol G_i_j_k. If symbolic=False, arbitrary placeholder floats are
    used instead (useful for quick numeric sanity checks / simulation).
    """
    G = nx.Graph()

    if symbolic:
        Gwire = sp.Symbol("Gwire", positive=True)
        S = sp.Symbol("S", positive=True)
        L = sp.Symbol("L", positive=True)
        def Gmem(i, j, k):
            return sp.Symbol(f"G_{i}_{j}_{k}", positive=True)
    else:
        Gwire, S, L = 1.0, 2.0, 0.5
        def Gmem(i, j, k):
            return 1.0

    # --- Vertices ---
    for i, j, k in product(range(n), range(m), range(p)):
        G.add_node(node_name("W", i, j, k), kind="W", idx=(i, j, k))
        G.add_node(node_name("B", i, j, k), kind="B", idx=(i, j, k))
    for i, k in product(range(n), range(p)):
        G.add_node(node_name("I", i, k), kind="I", idx=(i, k))
    for j, k in product(range(m), range(p)):
        G.add_node(node_name("O", j, k), kind="O", idx=(j, k))

    # --- EA: input resistors : I(i,k) -- W(i,0,k) ---
    for i, k in product(range(n), range(p)):
        G.add_edge(node_name("I", i, k), node_name("W", i, 0, k), weight=S)

    # --- EB: output resistors : B(n-1,j,k) -- O(j,k) ---
    for j, k in product(range(m), range(p)):
        G.add_edge(node_name("B", n - 1, j, k), node_name("O", j, k), weight=L)

    # --- EC: wire segments ---
    for i, j, k in product(range(n), range(m - 1), range(p)):
        G.add_edge(node_name("W", i, j, k), node_name("W", i, j + 1, k), weight=Gwire)
    for i, j, k in product(range(n), range(m), range(p - 1)):
        G.add_edge(node_name("W", i, j, k), node_name("W", i, j, k + 1), weight=Gwire)
    for i, j, k in product(range(n - 1), range(m), range(p)):
        G.add_edge(node_name("B", i, j, k), node_name("B", i + 1, j, k), weight=Gwire)
    for i, j, k in product(range(n), range(m), range(p - 1)):
        G.add_edge(node_name("B", i, j, k), node_name("B", i, j, k + 1), weight=Gwire)

    # --- ED: memristors : W(i,j,k) -- B(i,j,k) ---
    for i, j, k in product(range(n), range(m), range(p)):
        G.add_edge(node_name("W", i, j, k), node_name("B", i, j, k), weight=Gmem(i, j, k))

    return G


def voltage_symbol(node: str) -> sp.Symbol:
    return sp.Symbol(f"v_{node}")


def kcl_equation_for_node(G: nx.Graph, node: str) -> sp.Eq:
    """
    Sum_i G_i * (V - V_i) = 0   for every neighbor i of `node`.

    This is a completely generic KCL builder -- it only looks at the graph's
    incident edges, exactly like the derivation in the paper. Works for any
    node (W, B, I, or O), though I/O nodes are normally the *known* boundary
    values rather than unknowns to solve for.
    """
    V = voltage_symbol(node)
    lhs = sp.Integer(0)
    for nbr in G.neighbors(node):
        Gi = G.edges[node, nbr]["weight"]
        lhs += Gi * (V - voltage_symbol(nbr))
    return sp.Eq(sp.expand(lhs), 0)


# --------------------------------------------------------------------------
# 2. THE 18 SYMBOLIC TEMPLATE CASES
# --------------------------------------------------------------------------

@dataclass
class Case:
    number: int
    node_type: str          # "W" or "B"
    description: str        # index-range description, matching the paper
    count_formula: str      # symbolic count of how many real nodes hit this case
    equation: sp.Eq         # the symbolic KCL equation, sympy Eq(lhs, 0)


def _boundary_label(pos: str) -> str:
    return {"lo": "0", "mid": "interior", "hi": "max-1"}[pos]


def generate_18_cases() -> list[Case]:
    """
    Build the 18 symbolic KCL-equation templates directly from the
    neighbor-existence rules implied by EA..ED, WITHOUT instantiating a
    concrete graph. This mirrors exactly what generate_18_cases derives from
    build_graph() (see verify_templates_against_graph below).

    Symbols used:
      W, Wjm1, Wjp1, Wkm1, Wkp1  -> W(i,j,k) and its wordline-direction /
                                     layer-direction wire neighbors
      B, Bim1, Bip1, Bkm1, Bkp1  -> B(i,j,k) and its column-direction /
                                     layer-direction wire neighbors
      I, O                       -> input / output terminal voltages
      Gwire, S, L                -> parasitic / input / output conductances
      Gijk                       -> the memristor conductance G(i,j,k)
    """
    Gwire, S, L, Gijk = sp.symbols("Gwire S L G_ijk", positive=True)

    W, Wjm1, Wjp1, Wkm1, Wkp1 = sp.symbols("W_ijk W_i_jm1_k W_i_jp1_k W_i_j_km1 W_i_j_kp1")
    B, Bim1, Bip1, Bkm1, Bkp1 = sp.symbols("B_ijk B_im1_jk B_ip1_jk B_i_j_km1 B_i_j_kp1")
    I, O = sp.symbols("I_ik O_jk")

    cases: list[Case] = []
    n_no = 0  # running case counter

    j_positions = ["lo", "mid", "hi"]   # j = 0 / interior / m-1
    k_positions = ["lo", "mid", "hi"]   # k = 0 / interior / p-1

    # ---- 9 cases for W(i,j,k):  classified by (j-position, k-position) ----
    for jp in j_positions:
        for kp in k_positions:
            n_no += 1
            neighbors = []  # list of (voltage symbol, conductance)

            # j-direction neighbors (+ input resistor only at j=0)
            if jp == "lo":
                neighbors.append((I, S))
                neighbors.append((Wjp1, Gwire))
            elif jp == "mid":
                neighbors.append((Wjm1, Gwire))
                neighbors.append((Wjp1, Gwire))
            else:  # hi
                neighbors.append((Wjm1, Gwire))

            # k-direction neighbors
            if kp == "lo":
                neighbors.append((Wkp1, Gwire))
            elif kp == "mid":
                neighbors.append((Wkm1, Gwire))
                neighbors.append((Wkp1, Gwire))
            else:  # hi
                neighbors.append((Wkm1, Gwire))

            # always: the memristor down to B(i,j,k)
            neighbors.append((B, Gijk))

            lhs = sp.expand(sum(g * (W - v) for v, g in neighbors))

            j_desc = {"lo": "j = 0", "mid": "1 <= j < m-1", "hi": "j = m-1"}[jp]
            k_desc = {"lo": "k = 0", "mid": "1 <= k < p-1", "hi": "k = p-1"}[kp]
            j_cnt = {"lo": "1", "mid": "(m-2)", "hi": "1"}[jp]
            k_cnt = {"lo": "1", "mid": "(p-2)", "hi": "1"}[kp]

            cases.append(Case(
                number=n_no,
                node_type="W",
                description=f"{{W(i,j,k) | 0<=i<n, {j_desc}, {k_desc}}}",
                count_formula=f"n * {j_cnt} * {k_cnt}",
                equation=sp.Eq(lhs, 0),
            ))

    # ---- 9 cases for B(i,j,k):  classified by (i-position, k-position) ----
    i_positions = ["lo", "mid", "hi"]   # i = 0 / interior / n-1

    for ip in i_positions:
        for kp in k_positions:
            n_no += 1
            neighbors = []

            # i-direction neighbors (+ output resistor only at i=n-1)
            if ip == "lo":
                neighbors.append((Bip1, Gwire))
            elif ip == "mid":
                neighbors.append((Bim1, Gwire))
                neighbors.append((Bip1, Gwire))
            else:  # hi
                neighbors.append((Bim1, Gwire))
                neighbors.append((O, L))

            # k-direction neighbors
            if kp == "lo":
                neighbors.append((Bkp1, Gwire))
            elif kp == "mid":
                neighbors.append((Bkm1, Gwire))
                neighbors.append((Bkp1, Gwire))
            else:  # hi
                neighbors.append((Bkm1, Gwire))

            # always: the memristor up to W(i,j,k)
            neighbors.append((W, Gijk))

            lhs = sp.expand(sum(g * (B - v) for v, g in neighbors))

            i_desc = {"lo": "i = 0", "mid": "1 <= i < n-1", "hi": "i = n-1"}[ip]
            k_desc = {"lo": "k = 0", "mid": "1 <= k < p-1", "hi": "k = p-1"}[kp]
            i_cnt = {"lo": "1", "mid": "(n-2)", "hi": "1"}[ip]
            k_cnt = {"lo": "1", "mid": "(p-2)", "hi": "1"}[kp]

            cases.append(Case(
                number=n_no,
                node_type="B",
                description=f"{{B(i,j,k) | {i_desc}, 0<=j<m, {k_desc}}}",
                count_formula=f"m * {i_cnt} * {k_cnt}",
                equation=sp.Eq(lhs, 0),
            ))

    return cases


def print_cases(cases: list[Case]) -> None:
    for c in cases:
        print(f"Case {c.number} [{c.node_type}]: {c.description}")
        print(f"    Equations = {c.count_formula}")
        print(f"    {sp.sstr(c.equation.lhs)} = 0")
        print()


# --------------------------------------------------------------------------
# 3. VERIFY the 18 templates against an actual instantiated graph
# --------------------------------------------------------------------------

def _classify_w(j: int, k: int, m: int, p: int) -> tuple[str, str]:
    jp = "lo" if j == 0 else ("hi" if j == m - 1 else "mid")
    kp = "lo" if k == 0 else ("hi" if k == p - 1 else "mid")
    return jp, kp


def _classify_b(i: int, k: int, n: int, p: int) -> tuple[str, str]:
    ip = "lo" if i == 0 else ("hi" if i == n - 1 else "mid")
    kp = "lo" if k == 0 else ("hi" if k == p - 1 else "mid")
    return ip, kp


def verify_templates_against_graph(m: int = 4, n: int = 4, p: int = 4) -> bool:
    """
    For a concrete m x n x p crossbar (needs m,n,p >= 3 so every boundary
    class -- lo/mid/hi -- actually occurs), build the real graph, derive
    each node's KCL equation directly from its edges, substitute the correct
    neighbor voltages into the matching symbolic template, and check the two
    are algebraically identical (difference simplifies to 0).
    """
    assert m >= 3 and n >= 3 and p >= 3, "need >=3 in each dimension to see all boundary classes"

    G = build_graph(m, n, p, symbolic=True)
    cases = generate_18_cases()
    w_cases = {(jp, kp): c for c in cases if c.node_type == "W"
               for jp in ["lo", "mid", "hi"] for kp in ["lo", "mid", "hi"]
               if c.description == f"{{W(i,j,k) | 0<=i<n, {({'lo':'j = 0','mid':'1 <= j < m-1','hi':'j = m-1'}[jp])}, {({'lo':'k = 0','mid':'1 <= k < p-1','hi':'k = p-1'}[kp])}}}"}
    b_cases = {(ip, kp): c for c in cases if c.node_type == "B"
               for ip in ["lo", "mid", "hi"] for kp in ["lo", "mid", "hi"]
               if c.description == f"{{B(i,j,k) | {({'lo':'i = 0','mid':'1 <= i < n-1','hi':'i = n-1'}[ip])}, 0<=j<m, {({'lo':'k = 0','mid':'1 <= k < p-1','hi':'k = p-1'}[kp])}}}"}

    all_ok = True
    checked = 0

    # spot-check every W node in the middle "slab" i (doesn't matter, W indep of i)
    i0 = n // 2
    for j in range(m):
        for k in range(p):
            node = node_name("W", i0, j, k)
            actual = kcl_equation_for_node(G, node)
            jp, kp = _classify_w(j, k, m, p)
            template = w_cases[(jp, kp)]

            # map template's generic neighbor symbols to this node's actual neighbors
            subs = {
                sp.Symbol("W_ijk"): voltage_symbol(node),
                sp.Symbol("W_i_jm1_k"): voltage_symbol(node_name("W", i0, j - 1, k)) if j > 0 else 0,
                sp.Symbol("W_i_jp1_k"): voltage_symbol(node_name("W", i0, j + 1, k)) if j < m - 1 else 0,
                sp.Symbol("W_i_j_km1"): voltage_symbol(node_name("W", i0, j, k - 1)) if k > 0 else 0,
                sp.Symbol("W_i_j_kp1"): voltage_symbol(node_name("W", i0, j, k + 1)) if k < p - 1 else 0,
                sp.Symbol("B_ijk"): voltage_symbol(node_name("B", i0, j, k)),
                sp.Symbol("I_ik"): voltage_symbol(node_name("I", i0, k)),
                sp.Symbol("G_ijk", positive=True): sp.Symbol(f"G_{i0}_{j}_{k}", positive=True),
            }
            templ_lhs = template.equation.lhs.subs(subs)
            diff = sp.simplify(templ_lhs - actual.lhs)
            checked += 1
            if diff != 0:
                all_ok = False
                print(f"MISMATCH at W({i0},{j},{k}): diff = {diff}")

    # spot-check every B node in the middle "slab" j (doesn't matter, B indep of j)
    j0 = m // 2
    for i in range(n):
        for k in range(p):
            node = node_name("B", i, j0, k)
            actual = kcl_equation_for_node(G, node)
            ip, kp = _classify_b(i, k, n, p)
            template = b_cases[(ip, kp)]

            subs = {
                sp.Symbol("B_ijk"): voltage_symbol(node),
                sp.Symbol("B_im1_jk"): voltage_symbol(node_name("B", i - 1, j0, k)) if i > 0 else 0,
                sp.Symbol("B_ip1_jk"): voltage_symbol(node_name("B", i + 1, j0, k)) if i < n - 1 else 0,
                sp.Symbol("B_i_j_km1"): voltage_symbol(node_name("B", i, j0, k - 1)) if k > 0 else 0,
                sp.Symbol("B_i_j_kp1"): voltage_symbol(node_name("B", i, j0, k + 1)) if k < p - 1 else 0,
                sp.Symbol("W_ijk"): voltage_symbol(node_name("W", i, j0, k)),
                sp.Symbol("O_jk"): voltage_symbol(node_name("O", j0, k)),
                sp.Symbol("G_ijk", positive=True): sp.Symbol(f"G_{i}_{j0}_{k}", positive=True),
            }
            templ_lhs = template.equation.lhs.subs(subs)
            diff = sp.simplify(templ_lhs - actual.lhs)
            checked += 1
            if diff != 0:
                all_ok = False
                print(f"MISMATCH at B({i},{j0},{k}): diff = {diff}")

    print(f"Checked {checked} nodes against their matching template.")
    return all_ok


# --------------------------------------------------------------------------
# 4. Sanity-check the case COUNTS sum to the right totals
# --------------------------------------------------------------------------

def check_case_counts(m: int, n: int, p: int) -> None:
    n_, m_, p_ = sp.symbols("n m p", positive=True)
    cases = generate_18_cases()
    total_w = sum(sp.sympify(c.count_formula.replace("n", str(n)).replace("m", str(m)).replace("p", str(p)))
                  for c in cases if c.node_type == "W")
    total_b = sum(sp.sympify(c.count_formula.replace("n", str(n)).replace("m", str(m)).replace("p", str(p)))
                  for c in cases if c.node_type == "B")
    print(f"For m={m}, n={n}, p={p}:")
    print(f"  Sum of W-case counts = {total_w}  (expected n*m*p = {n*m*p})")
    print(f"  Sum of B-case counts = {total_b}  (expected n*m*p = {n*m*p})")
    assert total_w == n * m * p
    assert total_b == n * m * p


# --------------------------------------------------------------------------
if __name__ == "__main__":
    cases = generate_18_cases()

    print("=" * 78)
    print("THE 18 SYMBOLIC KCL EQUATIONS")
    print("=" * 78)
    print_cases(cases)

    print("=" * 78)
    print("CASE-COUNT SANITY CHECK (m=5, n=4, p=6)")
    print("=" * 78)
    check_case_counts(m=5, n=4, p=6)
    print()

    print("=" * 78)
    print("VERIFYING TEMPLATES AGAINST A REAL INSTANTIATED GRAPH (4x4x4)")
    print("=" * 78)
    ok = verify_templates_against_graph(m=4, n=4, p=4)
    print("ALL TEMPLATES MATCH THE GRAPH-DERIVED EQUATIONS:" , ok)