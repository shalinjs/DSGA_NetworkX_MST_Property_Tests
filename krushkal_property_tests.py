# =============================================================================
# Course:            E0 251o (2026)
# Algorithm:         Kruskal's Minimum Spanning Tree (MST)
#                    via networkx.minimum_spanning_tree(G, algorithm='kruskal')
# Testing Framework: Hypothesis (property-based testing) with pytest
# Team Members:      Nitish Kumar
# Algorithms Tested: Minimum Spanning Tree — Kruskal's algorithm
#                    Cross-validated against Prim's and Borůvka's algorithms
#
# Description:
#   This file contains a comprehensive property-based test suite for NetworkX's
#   MST implementation (Kruskal's algorithm). It uses the Hypothesis library to
#   generate diverse graph structures and verify fundamental MST properties
#   including structural validity, minimality (cycle property), invariance,
#   metamorphic relationships, boundary behavior, and cross-algorithm
#   consistency. Each property test includes a detailed docstring covering
#   mathematical reasoning, graph generation strategy, assumptions, and
#   failure implications.
# =============================================================================

import math
import itertools

import networkx as nx
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Hypothesis settings profile for all property-based tests
# ---------------------------------------------------------------------------
HYPOTHESIS_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Graph Generation Strategies
# ---------------------------------------------------------------------------

@st.composite
def connected_graph(draw, min_nodes=1, max_nodes=50):
    """Generate a connected, undirected, weighted graph with random positive float weights.

    Strategy:
        1. Draw the number of nodes n.
        2. For n == 1, return a single-node graph with no edges.
        3. Build a random spanning tree by shuffling nodes and connecting consecutive pairs.
        4. Draw a density parameter to decide how many extra (non-tree) edges to add.
        5. For each possible non-tree edge, include it with probability equal to the density.
        6. Assign each edge a positive float weight drawn from [0.1, 100.0].
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    nodes = list(range(n))
    G.add_nodes_from(nodes)

    if n == 1:
        return G

    # Build a random spanning tree: shuffle nodes, connect consecutive pairs
    shuffled = draw(st.permutations(nodes))
    tree_edges = []
    for i in range(len(shuffled) - 1):
        tree_edges.append((shuffled[i], shuffled[i + 1]))

    # Determine extra edges beyond the spanning tree
    tree_edge_set = set()
    for u, v in tree_edges:
        tree_edge_set.add((min(u, v), max(u, v)))

    non_tree_edges = []
    for u, v in itertools.combinations(nodes, 2):
        if (u, v) not in tree_edge_set:
            non_tree_edges.append((u, v))

    # Draw density to control how many extra edges to add
    density = draw(st.floats(min_value=0.0, max_value=1.0))
    extra_edges = [e for e in non_tree_edges if draw(st.floats(min_value=0.0, max_value=1.0)) < density]

    all_edges = tree_edges + extra_edges

    # Assign positive float weights to all edges
    weight_strategy = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    for u, v in all_edges:
        w = draw(weight_strategy)
        G.add_edge(u, v, weight=w)

    return G

@st.composite
def unique_weight_graph(draw, min_nodes=2, max_nodes=30):
    """Generate a connected, undirected graph where all edge weights are distinct positive floats.

    Strategy:
        1. Draw the number of nodes n (at least 2 so there is at least one edge).
        2. Build a random spanning tree by shuffling nodes and connecting consecutive pairs.
        3. Draw a density parameter and add extra non-tree edges accordingly.
        4. Draw a list of unique positive floats with length equal to the total number of edges.
        5. Assign one unique weight to each edge, guaranteeing all weights are distinct.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    nodes = list(range(n))
    G.add_nodes_from(nodes)

    # Build a random spanning tree: shuffle nodes, connect consecutive pairs
    shuffled = draw(st.permutations(nodes))
    tree_edges = []
    for i in range(len(shuffled) - 1):
        tree_edges.append((shuffled[i], shuffled[i + 1]))

    # Determine extra edges beyond the spanning tree
    tree_edge_set = set()
    for u, v in tree_edges:
        tree_edge_set.add((min(u, v), max(u, v)))

    non_tree_edges = []
    for u, v in itertools.combinations(nodes, 2):
        if (u, v) not in tree_edge_set:
            non_tree_edges.append((u, v))

    # Draw density to control how many extra edges to add
    density = draw(st.floats(min_value=0.0, max_value=1.0))
    extra_edges = [e for e in non_tree_edges if draw(st.floats(min_value=0.0, max_value=1.0)) < density]

    all_edges = tree_edges + extra_edges
    num_edges = len(all_edges)

    # Draw a list of unique positive floats — one per edge
    unique_weights = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=num_edges,
            max_size=num_edges,
            unique=True,
        )
    )

    for (u, v), w in zip(all_edges, unique_weights):
        G.add_edge(u, v, weight=w)

    return G

@st.composite
def equal_weight_graph(draw, min_nodes=2, max_nodes=30):
    """Generate a connected, undirected graph where all edges share the same positive float weight.

    Strategy:
        1. Draw the number of nodes n (at least 2 so there is at least one edge).
        2. Build a random spanning tree by shuffling nodes and connecting consecutive pairs.
        3. Draw a density parameter and add extra non-tree edges accordingly.
        4. Draw a single positive float weight.
        5. Assign that same weight to every edge in the graph.
    """
    n = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    G = nx.Graph()
    nodes = list(range(n))
    G.add_nodes_from(nodes)

    # Build a random spanning tree: shuffle nodes, connect consecutive pairs
    shuffled = draw(st.permutations(nodes))
    tree_edges = []
    for i in range(len(shuffled) - 1):
        tree_edges.append((shuffled[i], shuffled[i + 1]))

    # Determine extra edges beyond the spanning tree
    tree_edge_set = set()
    for u, v in tree_edges:
        tree_edge_set.add((min(u, v), max(u, v)))

    non_tree_edges = []
    for u, v in itertools.combinations(nodes, 2):
        if (u, v) not in tree_edge_set:
            non_tree_edges.append((u, v))

    # Draw density to control how many extra edges to add
    density = draw(st.floats(min_value=0.0, max_value=1.0))
    extra_edges = [e for e in non_tree_edges if draw(st.floats(min_value=0.0, max_value=1.0)) < density]

    all_edges = tree_edges + extra_edges

    # Draw a single positive float weight and assign it to ALL edges
    w = draw(st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
    for u, v in all_edges:
        G.add_edge(u, v, weight=w)

    return G

@st.composite
def disconnected_graph(draw, min_components=2, max_components=5, min_nodes_per=2, max_nodes_per=15):
    """Generate a graph with multiple disconnected connected components.

    Strategy:
        1. Draw the number of components.
        2. For each component, draw a node count and build a connected subgraph
           using a random spanning tree plus optional extra edges.
        3. Use disjoint node labels across components (offset by cumulative count).
        4. Assign random positive float weights to all edges.
        5. Combine all components into a single nx.Graph.
    """
    num_components = draw(st.integers(min_value=min_components, max_value=max_components))
    G = nx.Graph()
    node_offset = 0

    weight_strategy = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)

    for _ in range(num_components):
        n = draw(st.integers(min_value=min_nodes_per, max_value=max_nodes_per))
        nodes = list(range(node_offset, node_offset + n))
        G.add_nodes_from(nodes)

        # Build a random spanning tree for this component
        shuffled = draw(st.permutations(nodes))
        tree_edges = []
        for i in range(len(shuffled) - 1):
            tree_edges.append((shuffled[i], shuffled[i + 1]))

        # Determine extra edges beyond the spanning tree
        tree_edge_set = set()
        for u, v in tree_edges:
            tree_edge_set.add((min(u, v), max(u, v)))

        non_tree_edges = []
        for u, v in itertools.combinations(nodes, 2):
            if (u, v) not in tree_edge_set:
                non_tree_edges.append((u, v))

        # Draw density to control how many extra edges to add
        density = draw(st.floats(min_value=0.0, max_value=1.0))
        extra_edges = [e for e in non_tree_edges if draw(st.floats(min_value=0.0, max_value=1.0)) < density]

        all_edges = tree_edges + extra_edges

        # Assign positive float weights to all edges in this component
        for u, v in all_edges:
            w = draw(weight_strategy)
            G.add_edge(u, v, weight=w)

        node_offset += n

    return G


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def mst_weight(T):
    """Return the sum of all edge weights in graph T.

    Parameters
    ----------
    T : nx.Graph
        A NetworkX graph whose edges carry a ``'weight'`` attribute.

    Returns
    -------
    float
        Total edge weight of T.
    """
    return sum(d['weight'] for u, v, d in T.edges(data=True))

def is_valid_spanning_tree(G, T):
    """Check whether T is a valid spanning tree of G.

    A valid spanning tree satisfies all of the following:
      1. T has the same vertex set as G.
      2. T has exactly n-1 edges (where n = number of vertices).
      3. T is connected.
      4. T is acyclic (i.e., T is a tree).
      5. Every edge of T also exists in G.

    Parameters
    ----------
    G : nx.Graph
        The original graph.
    T : nx.Graph
        The candidate spanning tree.

    Returns
    -------
    bool
        True if T is a valid spanning tree of G, False otherwise.
    """
    # 1. Same vertex set
    if set(T.nodes()) != set(G.nodes()):
        return False

    n = T.number_of_nodes()

    # 2. Exactly n-1 edges
    if T.number_of_edges() != n - 1:
        return False

    # 3. Connected (a single-node graph with 0 edges is trivially connected)
    if n > 0 and not nx.is_connected(T):
        return False

    # 4. Acyclic / is a tree
    if not nx.is_tree(T):
        return False

    # 5. All edges of T exist in G
    for u, v in T.edges():
        if not G.has_edge(u, v):
            return False

    return True


# ---------------------------------------------------------------------------
# Property-Based Tests
# ---------------------------------------------------------------------------

# Feature: mst-property-testing, Property 1: Spanning tree structure validity
@given(G=connected_graph())
@HYPOTHESIS_SETTINGS
def test_spanning_tree_structure(G):
    """Verify that Kruskal's algorithm returns a valid spanning tree.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, undirected, weighted graph G with n vertices, the
        output of networkx.minimum_spanning_tree(G, algorithm='kruskal') is a
        valid spanning tree: it has exactly n-1 edges, is connected, contains
        no cycles, and its vertex set equals the vertex set of G.

    Mathematical basis:
        A spanning tree of a graph G = (V, E) is a subgraph T = (V, E') that
        is a tree (connected and acyclic) and spans all vertices of G. Any tree
        on n vertices has exactly n-1 edges. Kruskal's algorithm greedily adds
        edges in weight order while avoiding cycles, which by the matroid
        theory of graphic matroids guarantees a spanning tree of minimum total
        weight for connected input graphs.

    Test strategy:
        The connected_graph Hypothesis strategy generates connected, undirected
        graphs with 1 to 50 nodes, varying densities (from tree-like to
        near-complete), and positive float edge weights in [0.1, 100.0]. This
        covers single-node graphs, two-node graphs, sparse trees, and dense
        graphs. The helper function is_valid_spanning_tree checks all four
        structural conditions in one call.

    Failure implication:
        A failure indicates that Kruskal's implementation in NetworkX produces
        an output that violates one or more fundamental spanning tree
        properties — wrong vertex set, wrong edge count, disconnected result,
        or a cycle in the output. This would represent a critical correctness
        bug in the algorithm.
    """
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')
    assert is_valid_spanning_tree(G, T)

# Feature: mst-property-testing, Property 2: Cycle property (minimality)
@given(G=connected_graph(min_nodes=3))
@HYPOTHESIS_SETTINGS
def test_cycle_property(G):
    """Verify the cycle property (minimality) of Kruskal's MST.

    **Validates: Requirements 3.1, 3.2, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, weighted graph G and for any edge (u, v) in G that
        is not in the MST T, adding (u, v) to T creates exactly one cycle, and
        the weight of (u, v) must be greater than or equal to the weight of
        every other edge in that cycle. Equivalently, the added non-MST edge is
        the heaviest (or tied-for-heaviest) edge in the unique cycle it forms.

    Mathematical basis:
        The cycle property is a fundamental characterisation of minimum
        spanning trees. Because T is a spanning tree, there is exactly one
        path between any pair of vertices in T. Adding an edge (u, v) that is
        not already in T therefore creates exactly one cycle: the path from u
        to v in T plus the edge (u, v) itself. If the weight of (u, v) were
        strictly less than some edge on that path, we could swap them to
        obtain a spanning tree of lower total weight, contradicting the
        minimality of T. Hence w(u, v) >= w(e) for every edge e on the
        u-to-v path in T.

    Test strategy:
        The connected_graph strategy generates connected, undirected graphs
        with at least 3 nodes (so non-trivial cycles are possible) and
        varying densities and positive float weights. For each generated
        graph we compute the MST via Kruskal's algorithm, identify all
        non-MST edges, and for each such edge (u, v) we find the unique
        path from u to v in T using nx.shortest_path (unweighted, since T
        is a tree and any path is the unique path). We then verify that the
        weight of (u, v) is >= the weight of every edge along that path.
        We use assume() to skip graphs that are already trees (no non-MST
        edges), since the property is vacuously true in that case.

    Failure implication:
        A failure means that a non-MST edge is strictly lighter than some
        edge on the cycle it creates in T. This would imply that swapping
        those edges yields a lighter spanning tree, proving that T is not
        a minimum spanning tree — a critical correctness bug in Kruskal's
        implementation.
    """
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')

    # Build a set of MST edges (normalised as sorted tuples for comparison)
    mst_edges = set()
    for u, v in T.edges():
        mst_edges.add((min(u, v), max(u, v)))

    # Collect non-MST edges
    non_mst_edges = []
    for u, v, d in G.edges(data=True):
        key = (min(u, v), max(u, v))
        if key not in mst_edges:
            non_mst_edges.append((u, v, d['weight']))

    # Skip if the graph is already a tree (no non-MST edges to test)
    assume(len(non_mst_edges) > 0)

    for u, v, w_uv in non_mst_edges:
        # Find the unique path from u to v in the MST
        path = nx.shortest_path(T, source=u, target=v)

        # Check that w(u, v) >= weight of every edge on the path
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            w_ab = T[a][b]['weight']
            assert w_uv >= w_ab, (
                f"Non-MST edge ({u}, {v}) with weight {w_uv} is lighter than "
                f"MST path edge ({a}, {b}) with weight {w_ab}. "
                f"This violates the cycle property."
            )

# Feature: mst-property-testing, Property 6: Idempotence / determinism
@given(G=connected_graph())
@HYPOTHESIS_SETTINGS
def test_idempotence(G):
    """Verify that Kruskal's algorithm is idempotent (deterministic).

    **Validates: Requirements 4.1, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, undirected, weighted graph G, running
        networkx.minimum_spanning_tree(G, algorithm='kruskal') twice on the
        same graph produces MSTs with the same total edge weight.

    Mathematical basis:
        A deterministic algorithm, given identical input, must produce
        identical output. Kruskal's algorithm sorts edges by weight and
        greedily selects them using a union-find structure — both operations
        are deterministic for a fixed input. Therefore the total weight of
        the MST must be the same across repeated invocations. Even when
        multiple MSTs of equal total weight exist (due to tied edge weights),
        the deterministic tie-breaking of the sort guarantees the same MST
        is selected each time.

    Test strategy:
        The connected_graph Hypothesis strategy generates connected, undirected
        graphs with 1 to 50 nodes, varying densities, and positive float edge
        weights in [0.1, 100.0]. For each generated graph we compute the MST
        twice via Kruskal's algorithm and compare the total weights using
        math.isclose to account for floating-point arithmetic.

    Failure implication:
        A failure indicates that Kruskal's implementation in NetworkX is
        non-deterministic — producing different MST weights on the same input
        across invocations. This would suggest a bug involving unstable
        sorting, randomised tie-breaking, or corrupted internal state between
        calls.
    """
    T1 = nx.minimum_spanning_tree(G, algorithm='kruskal')
    T2 = nx.minimum_spanning_tree(G, algorithm='kruskal')
    w1 = mst_weight(T1)
    w2 = mst_weight(T2)
    assert math.isclose(w1, w2, rel_tol=1e-9), (
        f"Idempotence violated: first MST weight = {w1}, second MST weight = {w2}"
    )

# Feature: mst-property-testing, Property 7: Vertex relabeling invariance
@given(G=connected_graph(min_nodes=1, max_nodes=50), data=st.data())
@HYPOTHESIS_SETTINGS
def test_vertex_relabeling_invariance(G, data):
    """Verify that relabeling vertices does not change the MST total weight.

    **Validates: Requirements 4.2, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, weighted graph G and for any permutation of vertex
        labels, the MST of the relabeled graph has the same total edge weight
        as the MST of the original graph.

    Mathematical basis:
        The minimum spanning tree depends only on the graph's structure
        (topology) and edge weights, not on the identity of vertex labels.
        Relabeling vertices produces an isomorphic graph — the same edges
        exist with the same weights, just under different names. Because
        Kruskal's algorithm operates on edge weights and connectivity (via
        union-find), the choice of vertex labels cannot affect which edges
        are selected or the resulting total weight.

    Test strategy:
        The connected_graph Hypothesis strategy generates connected, undirected
        graphs with 1 to 50 nodes, varying densities, and positive float edge
        weights in [0.1, 100.0]. For each generated graph we draw a random
        permutation of the node list, build a relabeling mapping, apply it
        via nx.relabel_nodes, compute the MST of both the original and
        relabeled graphs, and compare total weights using math.isclose.

    Failure implication:
        A failure indicates that Kruskal's implementation is sensitive to
        vertex labels — producing different MST weights for isomorphic
        graphs. This would suggest a bug where the algorithm's internal
        ordering or data structures depend on node identity rather than
        purely on edge weights and connectivity.
    """
    nodes = list(G.nodes())
    if len(nodes) <= 1:
        # Trivial case: single-node graph, MST weight is 0 regardless of labeling
        T = nx.minimum_spanning_tree(G, algorithm='kruskal')
        assert mst_weight(T) == 0.0
        return

    perm = data.draw(st.permutations(nodes))
    mapping = dict(zip(nodes, perm))
    G_relabeled = nx.relabel_nodes(G, mapping)

    T_original = nx.minimum_spanning_tree(G, algorithm='kruskal')
    T_relabeled = nx.minimum_spanning_tree(G_relabeled, algorithm='kruskal')

    w_original = mst_weight(T_original)
    w_relabeled = mst_weight(T_relabeled)

    assert math.isclose(w_original, w_relabeled, rel_tol=1e-9), (
        f"Vertex relabeling invariance violated: original MST weight = {w_original}, "
        f"relabeled MST weight = {w_relabeled}, mapping = {mapping}"
    )

# Feature: mst-property-testing, Property 4: Weight-shift invariance
@given(G=connected_graph(min_nodes=2), c=st.floats(min_value=0.1, max_value=50.0))
@HYPOTHESIS_SETTINGS
def test_weight_shift_invariance(G, c):
    """Verify that adding a constant to all edge weights preserves the MST edge set.

    **Validates: Requirements 4.3, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, weighted graph G and for any positive constant c,
        adding c to every edge weight in G produces a graph G' whose MST has
        the same edge set as the MST of G.

    Mathematical basis:
        Adding a constant c to every edge weight is a uniform shift that
        preserves the relative ordering of all edges. Since Kruskal's
        algorithm selects edges based on their sorted order (and uses
        union-find to avoid cycles), the same edges are selected in the
        same order. The total weight of the new MST will be the original
        total weight plus c × (n-1), but the edge set itself is identical.
        More formally, for any two edges e1 and e2, w(e1) <= w(e2) if and
        only if w(e1) + c <= w(e2) + c, so the sorted order is preserved.

    Test strategy:
        The connected_graph Hypothesis strategy generates connected, undirected
        graphs with at least 2 nodes (so there is at least one edge), varying
        densities, and positive float edge weights in [0.1, 100.0]. A positive
        constant c is drawn from [0.1, 50.0]. We create a shifted copy G' by
        adding c to every edge weight, compute the MST of both G and G', and
        compare edge sets. Edges are represented as frozensets of frozensets
        to ensure order-independent comparison.

    Failure implication:
        A failure indicates that a uniform weight shift causes Kruskal's
        algorithm to select a different set of edges. Since the relative
        ordering of edges is unchanged by a uniform shift, this would imply
        a bug in the sorting or edge-selection logic — perhaps involving
        numerical instability or incorrect tie-breaking that is sensitive
        to absolute weight magnitudes.
    """
    # Compute MST of original graph
    T_original = nx.minimum_spanning_tree(G, algorithm='kruskal')

    # Create shifted graph G' with all weights increased by c
    G_shifted = G.copy()
    for u, v, d in G_shifted.edges(data=True):
        d['weight'] = d['weight'] + c

    # Compute MST of shifted graph
    T_shifted = nx.minimum_spanning_tree(G_shifted, algorithm='kruskal')

    # Compare edge sets (as frozensets of frozensets for order independence)
    edges_original = frozenset(frozenset((u, v)) for u, v in T_original.edges())
    edges_shifted = frozenset(frozenset((u, v)) for u, v in T_shifted.edges())

    assert edges_original == edges_shifted, (
        f"Weight-shift invariance violated with c = {c}. "
        f"Original MST edges: {edges_original}, "
        f"Shifted MST edges: {edges_shifted}"
    )

# Feature: mst-property-testing, Property 5: Non-MST edge removal preserves MST weight
@given(G=connected_graph(min_nodes=3))
@HYPOTHESIS_SETTINGS
def test_non_mst_edge_removal(G):
    """Verify that removing a non-MST edge (if the graph stays connected) preserves MST weight.

    **Validates: Requirements 5.2, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, weighted graph G with at least one non-MST edge
        whose removal leaves G connected, removing that edge produces a graph
        G' whose MST has the same total weight as the MST of G.

    Mathematical basis:
        An edge that is not part of any MST is, by the cycle property, the
        unique maximum-weight edge in some cycle of G (or tied for maximum).
        Removing such an edge cannot disconnect the graph if an alternative
        path exists (which the connectivity check confirms). Since the edge
        was never selected by Kruskal's algorithm, its absence does not
        affect the greedy selection process — the same set of lighter edges
        remains available, and the same MST is constructed. Therefore the
        total MST weight is unchanged.

    Test strategy:
        The connected_graph strategy generates connected, undirected graphs
        with at least 3 nodes and varying densities and positive float edge
        weights in [0.1, 100.0]. We compute the MST, identify all non-MST
        edges, and for each one check whether removing it keeps G connected
        (using a copy). We use assume() to skip graphs where no non-MST edge
        can be safely removed (e.g., the graph is already a tree or every
        non-MST edge is a bridge in the non-tree sense). We then remove the
        first such safe edge, recompute the MST, and assert the total weight
        is unchanged via math.isclose.

    Failure implication:
        A failure indicates that removing a redundant (non-MST) edge from the
        graph causes Kruskal's algorithm to produce an MST with a different
        total weight. This would suggest that the algorithm incorrectly
        depends on edges that should be irrelevant to the optimal solution —
        a violation of the metamorphic relationship between input
        perturbation and output stability.
    """
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')
    original_weight = mst_weight(T)

    # Build a set of MST edges (normalised as sorted tuples)
    mst_edges = set()
    for u, v in T.edges():
        mst_edges.add((min(u, v), max(u, v)))

    # Find a non-MST edge whose removal keeps G connected
    removable_edge = None
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key not in mst_edges:
            G_copy = G.copy()
            G_copy.remove_edge(u, v)
            if nx.is_connected(G_copy):
                removable_edge = (u, v)
                break

    # Skip if no such edge exists
    assume(removable_edge is not None)

    # Remove the edge and recompute MST
    G_reduced = G.copy()
    G_reduced.remove_edge(*removable_edge)
    T_reduced = nx.minimum_spanning_tree(G_reduced, algorithm='kruskal')
    reduced_weight = mst_weight(T_reduced)

    assert math.isclose(original_weight, reduced_weight, rel_tol=1e-9), (
        f"Non-MST edge removal changed MST weight: original = {original_weight}, "
        f"after removing {removable_edge} = {reduced_weight}"
    )

# Feature: mst-property-testing, Property 3: Cross-algorithm weight consistency
@given(G=connected_graph(min_nodes=2))
@HYPOTHESIS_SETTINGS
def test_cross_algorithm_consistency(G):
    """Verify that Kruskal, Prim, and Borůvka produce MSTs with the same total weight.

    **Validates: Requirements 7.1, 7.2, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, weighted graph G, the total edge weight of the MST
        produced by Kruskal's algorithm equals the total edge weight of the
        MST produced by Prim's algorithm and the total edge weight of the MST
        produced by Borůvka's algorithm on the same graph.

    Mathematical basis:
        The minimum spanning tree of a connected, weighted graph has a unique
        total weight (even when multiple distinct MSTs exist, they all share
        the same total weight). Kruskal's, Prim's, and Borůvka's algorithms
        are all proven-correct MST algorithms — Kruskal's uses a global
        edge-sort with union-find, Prim's grows a single tree greedily from
        a start vertex, and Borůvka's merges components in parallel rounds.
        All three must therefore produce spanning trees of identical minimum
        total weight.

    Test strategy:
        The connected_graph strategy generates connected, undirected graphs
        with at least 2 nodes, varying densities, and positive float edge
        weights in [0.1, 100.0]. For each generated graph we compute the MST
        using all three algorithms available in NetworkX and compare their
        total weights pairwise using math.isclose to account for
        floating-point arithmetic differences.

    Failure implication:
        A failure indicates that at least one of the three MST algorithms in
        NetworkX produces a spanning tree with a different total weight than
        the others. Since all three are proven-correct algorithms, this would
        point to an implementation bug in one (or more) of them — possibly
        involving incorrect edge selection, faulty priority queue operations,
        or component-merging errors.
    """
    T_kruskal = nx.minimum_spanning_tree(G, algorithm='kruskal')
    T_prim = nx.minimum_spanning_tree(G, algorithm='prim')
    T_boruvka = nx.minimum_spanning_tree(G, algorithm='boruvka')

    w_kruskal = mst_weight(T_kruskal)
    w_prim = mst_weight(T_prim)
    w_boruvka = mst_weight(T_boruvka)

    assert math.isclose(w_kruskal, w_prim, rel_tol=1e-9), (
        f"Kruskal vs Prim weight mismatch: {w_kruskal} != {w_prim}"
    )
    assert math.isclose(w_kruskal, w_boruvka, rel_tol=1e-9), (
        f"Kruskal vs Borůvka weight mismatch: {w_kruskal} != {w_boruvka}"
    )

# Feature: mst-property-testing, Property 10: Unique MST with unique weights
@given(G=unique_weight_graph())
@HYPOTHESIS_SETTINGS
def test_unique_weight_unique_mst(G):
    """Verify that graphs with all unique edge weights produce a unique MST across algorithms.

    **Validates: Requirements 3.3, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected, weighted graph with all distinct edge weights,
        Kruskal's, Prim's, and Borůvka's algorithms all return MSTs with the
        exact same edge set (not just the same total weight).

    Mathematical basis:
        When all edge weights in a connected graph are distinct, the minimum
        spanning tree is unique. This follows from the cut property: for every
        cut of the graph, there is a unique lightest crossing edge, and that
        edge must belong to every MST. Since every cut has a unique minimum
        crossing edge, there is exactly one MST. Consequently, any correct
        MST algorithm — regardless of its internal strategy (greedy edge
        selection, vertex growing, component merging) — must return the same
        set of edges.

    Test strategy:
        The unique_weight_graph strategy generates connected, undirected
        graphs with at least 2 nodes (up to 30) where every edge has a
        distinct positive float weight. For each generated graph we compute
        the MST using Kruskal's, Prim's, and Borůvka's algorithms and
        compare their edge sets. Edges are represented as frozensets of
        frozensets for order-independent comparison.

    Failure implication:
        A failure indicates that two correct MST algorithms produce different
        edge sets on a graph with unique weights, where the MST is provably
        unique. This would point to a bug in at least one algorithm's edge
        selection logic — perhaps incorrect sorting, faulty priority queue
        updates, or erroneous component merging — causing it to miss the
        unique optimal solution.
    """
    T_kruskal = nx.minimum_spanning_tree(G, algorithm='kruskal')
    T_prim = nx.minimum_spanning_tree(G, algorithm='prim')
    T_boruvka = nx.minimum_spanning_tree(G, algorithm='boruvka')

    edges_kruskal = frozenset(frozenset((u, v)) for u, v in T_kruskal.edges())
    edges_prim = frozenset(frozenset((u, v)) for u, v in T_prim.edges())
    edges_boruvka = frozenset(frozenset((u, v)) for u, v in T_boruvka.edges())

    assert edges_kruskal == edges_prim, (
        f"Kruskal vs Prim edge set mismatch with unique weights: "
        f"Kruskal = {edges_kruskal}, Prim = {edges_prim}"
    )
    assert edges_kruskal == edges_boruvka, (
        f"Kruskal vs Borůvka edge set mismatch with unique weights: "
        f"Kruskal = {edges_kruskal}, Borůvka = {edges_boruvka}"
    )


# ---------------------------------------------------------------------------
# Boundary and Edge Case Tests
# ---------------------------------------------------------------------------

def test_single_vertex_graph():
    """Verify that a single-vertex graph produces an MST with 0 edges and 0 weight.

    **Validates: Requirements 6.1, 8.2**

    Boundary case:
        A graph with exactly one vertex and no edges is the smallest possible
        graph. The MST of such a graph is trivially the graph itself — a
        single node with no edges. The total weight is 0 because there are no
        edges to contribute weight. This test confirms that Kruskal's
        algorithm handles this degenerate input gracefully rather than raising
        an error or returning an incorrect result.
    """
    G = nx.Graph()
    G.add_node(0)
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')
    assert T.number_of_edges() == 0, (
        f"Expected 0 edges for single-vertex MST, got {T.number_of_edges()}"
    )
    assert mst_weight(T) == 0, (
        f"Expected 0 total weight for single-vertex MST, got {mst_weight(T)}"
    )


def test_two_vertex_graph():
    """Verify that a two-vertex, one-edge graph produces an MST containing that edge.

    **Validates: Requirements 6.2, 8.2**

    Boundary case:
        A graph with exactly two vertices and one edge is the smallest
        non-trivial connected graph. The MST must contain the single edge
        because it is the only way to span both vertices. This test confirms
        that Kruskal's algorithm correctly includes the sole available edge
        and produces a valid spanning tree with exactly 1 edge.
    """
    G = nx.Graph()
    G.add_edge(0, 1, weight=5.0)
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')
    assert T.number_of_edges() == 1, (
        f"Expected 1 edge for two-vertex MST, got {T.number_of_edges()}"
    )
    assert T.has_edge(0, 1), (
        "Expected MST to contain the edge (0, 1)"
    )


# Feature: mst-property-testing, Property 8: Equal-weight spanning tree validity
@given(G=equal_weight_graph())
@HYPOTHESIS_SETTINGS
def test_equal_weight_graph(G):
    """Verify that equal-weight graphs produce a valid spanning tree with weight (n-1) × w.

    **Validates: Requirements 6.4, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any connected graph where all edges have the same weight w, the
        MST returned by Kruskal's algorithm is a valid spanning tree and its
        total weight equals (n-1) × w.

    Mathematical basis:
        When every edge in a connected graph has the same weight w, every
        spanning tree has the same total weight: (n-1) × w, since any
        spanning tree on n vertices has exactly n-1 edges. Therefore every
        spanning tree is a minimum spanning tree. Kruskal's algorithm must
        still produce a valid spanning tree (connected, acyclic, n-1 edges,
        same vertex set), and its total weight must equal (n-1) × w.

    Test strategy:
        The equal_weight_graph Hypothesis strategy generates connected,
        undirected graphs with 2 to 30 nodes where every edge is assigned
        the same positive float weight drawn from [0.1, 100.0]. The test
        computes the MST via Kruskal's algorithm, verifies structural
        validity using is_valid_spanning_tree, extracts the common weight
        from any edge, and checks that the total MST weight equals
        (n-1) × w using math.isclose for floating-point tolerance.

    Failure implication:
        A failure indicates that Kruskal's algorithm either produces an
        invalid spanning tree or computes an incorrect total weight when
        all edges are equally weighted. This would suggest a bug in edge
        selection or weight accumulation — possibly related to tie-breaking
        logic that incorrectly skips or duplicates edges.
    """
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')
    assert is_valid_spanning_tree(G, T), (
        "MST is not a valid spanning tree of the equal-weight graph"
    )

    # Get the common weight from any edge in the original graph
    _, _, d = next(iter(G.edges(data=True)))
    w = d['weight']
    n = G.number_of_nodes()
    expected_weight = (n - 1) * w

    assert math.isclose(mst_weight(T), expected_weight, rel_tol=1e-9), (
        f"Equal-weight MST weight mismatch: expected {expected_weight}, "
        f"got {mst_weight(T)} (n={n}, w={w})"
    )


# Feature: mst-property-testing, Property 9: Minimum spanning forest for disconnected graphs
@given(G=disconnected_graph())
@HYPOTHESIS_SETTINGS
def test_disconnected_forest(G):
    """Verify that a disconnected graph produces a valid minimum spanning forest.

    **Validates: Requirements 6.5, 8.2, 8.3, 8.4, 8.5**

    Property:
        For any disconnected graph G, the output of
        networkx.minimum_spanning_tree(G, algorithm='kruskal') is a forest
        where each connected component's subtree is a valid spanning tree of
        the corresponding component in G, and the total weight equals the sum
        of MST weights of each component computed independently.

    Mathematical basis:
        A minimum spanning forest is the union of minimum spanning trees of
        each connected component. Kruskal's algorithm naturally produces a
        forest for disconnected graphs: it processes edges globally in weight
        order and uses union-find to avoid cycles, which means it builds an
        MST independently within each component without ever connecting
        different components. The total weight of the forest must therefore
        equal the sum of the individual component MST weights.

    Test strategy:
        The disconnected_graph Hypothesis strategy generates graphs with 2 to
        5 disconnected connected components, each having 2 to 15 nodes with
        disjoint labels and random positive float edge weights in
        [0.1, 100.0]. The test computes the MST (forest) of the full graph,
        verifies that the output has the same node set as the input, then
        iterates over each connected component of G. For each component, it
        extracts the corresponding subgraph of T and verifies it is a valid
        spanning tree of that component. Finally, it checks that the total
        forest weight equals the sum of per-component MST weights.

    Failure implication:
        A failure indicates that Kruskal's algorithm does not correctly handle
        disconnected graphs — either by connecting components that should
        remain separate, by producing an invalid subtree within a component,
        or by computing an incorrect total weight. This would suggest a bug
        in the union-find structure or edge-processing logic when multiple
        components are present.
    """
    T = nx.minimum_spanning_tree(G, algorithm='kruskal')

    # The forest must have the same node set as G
    assert set(T.nodes()) == set(G.nodes()), (
        f"Forest node set mismatch: expected {set(G.nodes())}, got {set(T.nodes())}"
    )

    # Verify each component's subtree and accumulate expected weight
    total_component_weight = 0.0
    for component_nodes in nx.connected_components(G):
        component_nodes = set(component_nodes)
        G_component = G.subgraph(component_nodes).copy()
        T_component = T.subgraph(component_nodes).copy()

        # Each component's subtree must be a valid spanning tree
        assert is_valid_spanning_tree(G_component, T_component), (
            f"Subtree for component {component_nodes} is not a valid spanning tree"
        )

        # Compute the independent MST weight for this component
        T_independent = nx.minimum_spanning_tree(G_component, algorithm='kruskal')
        total_component_weight += mst_weight(T_independent)

    # Total forest weight must equal sum of per-component MST weights
    forest_weight = mst_weight(T)
    assert math.isclose(forest_weight, total_component_weight, rel_tol=1e-9), (
        f"Forest weight mismatch: forest = {forest_weight}, "
        f"sum of component MSTs = {total_component_weight}"
    )
