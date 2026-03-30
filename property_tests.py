import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def connected_graphs(draw):
    """Generate a random connected undirected weighted graph."""
    n = draw(st.integers(min_value=2, max_value=50))
    nodes = list(range(n))

    # Start with a random spanning path to guarantee connectivity
    shuffled = draw(st.permutations(nodes))
    edges = [(shuffled[i], shuffled[i + 1]) for i in range(n - 1)]

    # Add some extra random edges
    num_extra = draw(st.integers(min_value=0, max_value=n * (n - 1) // 2 - (n - 1)))
    for _ in range(num_extra):
        u = draw(st.integers(min_value=0, max_value=n - 1))
        v = draw(st.integers(min_value=0, max_value=n - 1))
        if u != v:
            edges.append((u, v))

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v in edges:
        w = draw(st.integers(min_value=1, max_value=100))
        G.add_edge(u, v, weight=w)

    return G


# Postcondition: MST of a connected graph with n nodes has exactly n-1 edges
@given(G=connected_graphs())
@settings(max_examples=200)
def test_mst_has_n_minus_1_edges(G):
    """
    Property: The MST of a connected graph with n nodes must have exactly n-1 edges.

    Why it matters:
        A spanning tree is a minimal connected subgraph that includes all nodes.
        By definition, any tree with n nodes has exactly n-1 edges. Fewer edges
        would mean the graph is disconnected, and more edges would introduce a
        cycle — violating the tree property.

    Mathematical reasoning:
        A tree is a connected acyclic graph. A fundamental result in graph theory
        states that for any tree T with n nodes: |E(T)| = n - 1. This holds for
        all spanning trees, including the minimum spanning tree.

    Test input:
        Random connected undirected weighted graphs with 2 to 50 nodes. Connectivity
        is guaranteed by starting with a spanning path through all nodes, then
        optionally adding extra random edges with weights in [1, 100].

    Assumptions / Preconditions:
        - The input graph is connected (ensured by the connected_graphs strategy).
        - The input graph is undirected and simple (no self-loops or parallel edges).

    What a failure indicates:
        If this test fails, Prim's algorithm is either dropping nodes from the tree
        (too few edges), or including redundant edges that form cycles (too many
        edges). This would point to a bug in the node visitation or edge selection
        logic of the algorithm.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert mst.number_of_edges() == G.number_of_nodes() - 1


# Postcondition: Prim's MST total weight equals Kruskal's MST total weight
@given(G=connected_graphs())
@settings(max_examples=200)
def test_prim_weight_equals_kruskal_weight(G):
    """
    Property: The total weight of Prim's MST must equal the total weight of
    Kruskal's MST for the same graph.

    Why it matters:
        Both Prim's and Kruskal's are proven correct algorithms for finding a
        minimum spanning tree. While they may select different edges (when multiple
        MSTs of equal weight exist), the total weight of their outputs must always
        be identical. This cross-algorithm check validates optimality — that Prim's
        is truly finding a minimum-weight spanning tree.

    Mathematical reasoning:
        The MST of a graph is not necessarily unique — there can be multiple spanning
        trees with the same minimum total weight. However, the minimum total weight
        itself is unique. Since both algorithms are proven to produce an MST, their
        total weights must match: W(Prim's MST) = W(Kruskal's MST) = W_min(G).

    Test input:
        Random connected undirected weighted graphs with 2 to 50 nodes. Connectivity
        is guaranteed by starting with a spanning path through all nodes, then
        optionally adding extra random edges with weights in [1, 100].

    Assumptions / Preconditions:
        - The input graph is connected (ensured by the connected_graphs strategy).
        - The input graph is undirected and simple (no self-loops or parallel edges).
        - Kruskal's algorithm (used as the reference) is assumed to be correct.
          This is a reasonable assumption since it serves as an independent oracle.

    What a failure indicates:
        If this test fails, Prim's algorithm is producing a spanning tree that is
        not optimal — it is selecting a set of edges whose total weight exceeds the
        true minimum. This would point to a bug in the greedy edge selection or
        priority queue handling within Prim's algorithm.
    """
    prim_mst = nx.minimum_spanning_tree(G, algorithm="prim")
    kruskal_mst = nx.minimum_spanning_tree(G, algorithm="kruskal")
    prim_weight = sum(d["weight"] for _, _, d in prim_mst.edges(data=True))
    kruskal_weight = sum(d["weight"] for _, _, d in kruskal_mst.edges(data=True))
    assert prim_weight == kruskal_weight


# Invariant: MST is connected
@given(G=connected_graphs())
@settings(max_examples=200)
def test_mst_is_connected(G):
    """
    Property: The MST produced by Prim's algorithm must be a connected graph.

    Why it matters:
        A spanning tree must connect all nodes in the graph. If the MST is
        disconnected, it fails to serve its fundamental purpose — providing a
        path between every pair of nodes using the minimum total edge weight.

    Mathematical reasoning:
        By definition, a spanning tree T of a connected graph G is a connected
        acyclic subgraph that includes every node of G. Connectivity is one of
        the two defining properties of a tree (the other being acyclicity).
        For a graph with n nodes, a connected subgraph with n-1 edges is
        necessarily a tree.

    Test input:
        Random connected undirected weighted graphs with 2 to 50 nodes. Connectivity
        is guaranteed by starting with a spanning path through all nodes, then
        optionally adding extra random edges with weights in [1, 100].

    Assumptions / Preconditions:
        - The input graph is connected (ensured by the connected_graphs strategy).
        - The input graph is undirected and simple (no self-loops or parallel edges).

    What a failure indicates:
        If this test fails, Prim's algorithm is producing a forest (multiple
        disconnected components) instead of a single tree. This would suggest a
        bug in how the algorithm grows the tree from the starting node — likely
        failing to explore all reachable neighbors or prematurely terminating
        the frontier expansion.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert nx.is_connected(mst)


# Invariant: MST is acyclic
@given(G=connected_graphs())
@settings(max_examples=200)
def test_mst_is_acyclic(G):
    """
    Property: The MST produced by Prim's algorithm must contain no cycles.

    Why it matters:
        A spanning tree is by definition acyclic. If the MST contains a cycle,
        it means there is a redundant edge that could be removed to reduce the
        total weight while still keeping the graph connected — contradicting
        the minimality of the MST.

    Mathematical reasoning:
        A tree is a connected acyclic graph. For n nodes, a tree has exactly
        n-1 edges. If a connected graph with n nodes has more than n-1 edges,
        it must contain at least one cycle. Conversely, any cycle in the output
        means the algorithm included an unnecessary edge, violating the tree
        property. In a cycle, removing the heaviest edge would yield a lighter
        spanning subgraph, so a cyclic result cannot be a minimum spanning tree.

    Test input:
        Random connected undirected weighted graphs with 2 to 50 nodes. Connectivity
        is guaranteed by starting with a spanning path through all nodes, then
        optionally adding extra random edges with weights in [1, 100].

    Assumptions / Preconditions:
        - The input graph is connected (ensured by the connected_graphs strategy).
        - The input graph is undirected and simple (no self-loops or parallel edges).

    What a failure indicates:
        If this test fails, Prim's algorithm is adding edges that create cycles.
        This would point to a bug in the visited-node tracking — the algorithm
        is likely adding an edge to a node that has already been included in the
        growing tree, instead of skipping it.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert nx.is_tree(mst)


# Invariant: MST spans all nodes
@given(G=connected_graphs())
@settings(max_examples=200)
def test_mst_spans_all_nodes(G):
    """
    Property: The MST must contain every node present in the original graph.

    Why it matters:
        A spanning tree must "span" the entire graph — meaning it includes all
        nodes. If any node is missing from the MST, there is no path to that
        node within the tree, making it useless for applications like network
        design where every endpoint must be reachable.

    Mathematical reasoning:
        By definition, a spanning subgraph of G is a subgraph that contains
        all vertices of G. A spanning tree is a spanning subgraph that is also
        a tree. Therefore: V(MST) = V(G). The node sets must be identical.

    Test input:
        Random connected undirected weighted graphs with 2 to 50 nodes. Connectivity
        is guaranteed by starting with a spanning path through all nodes, then
        optionally adding extra random edges with weights in [1, 100].

    Assumptions / Preconditions:
        - The input graph is connected (ensured by the connected_graphs strategy).
        - The input graph is undirected and simple (no self-loops or parallel edges).

    What a failure indicates:
        If this test fails, Prim's algorithm is losing nodes during tree
        construction. This could indicate a bug where isolated or leaf nodes
        are not being added to the result, or where the algorithm's starting
        node selection causes some nodes to be skipped entirely.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    assert set(mst.nodes()) == set(G.nodes())


# Invariant: Every MST edge exists in the original graph
@given(G=connected_graphs())
@settings(max_examples=200)
def test_mst_edges_are_subset_of_original(G):
    """
    Property: Every edge in the MST must exist in the original graph.

    Why it matters:
        The MST is a subgraph of the original graph — it can only use edges
        that already exist. If the MST contains an edge not present in the
        original graph, the algorithm has fabricated a connection, producing
        an invalid result.

    Mathematical reasoning:
        A spanning tree T of graph G satisfies V(T) = V(G) and E(T) ⊆ E(G).
        The edge set of the MST must be a strict subset of the original edge
        set. Furthermore, the weight of each MST edge must match the weight
        of the corresponding edge in G.

    Test input:
        Random connected undirected weighted graphs with 2 to 50 nodes. Connectivity
        is guaranteed by starting with a spanning path through all nodes, then
        optionally adding extra random edges with weights in [1, 100].

    Assumptions / Preconditions:
        - The input graph is connected (ensured by the connected_graphs strategy).
        - The input graph is undirected and simple (no self-loops or parallel edges).

    What a failure indicates:
        If this test fails, Prim's algorithm is creating edges that don't exist
        in the input graph. This would be a severe bug — likely a corruption in
        how the algorithm reads the adjacency structure or stores edges during
        the priority queue operations.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    for u, v in mst.edges():
        assert G.has_edge(u, v)


# Invariant: Cut property — any non-MST edge is >= the heaviest edge on the path it would create
@given(G=connected_graphs())
@settings(max_examples=200)
def test_cut_property(G):
    """
    Property: For every edge (u, v) in G that is NOT in the MST, its weight
    must be >= the maximum weight edge on the unique path from u to v in the MST.

    Why it matters:
        The cut property is the fundamental optimality condition for MSTs. It
        guarantees that no edge swap can reduce the total weight. If a non-MST
        edge were lighter than the heaviest edge on the MST path it bridges,
        we could swap them to get a lighter spanning tree — contradicting the
        minimality of the MST.

    Mathematical reasoning:
        In any spanning tree T, there is exactly one path between any pair of
        nodes. Adding a non-tree edge (u, v) to T creates exactly one cycle.
        For T to be a minimum spanning tree, the cycle property must hold:
        the non-tree edge (u, v) must be the heaviest (or tied for heaviest)
        edge in that cycle. Equivalently, w(u, v) >= max edge weight on the
        unique path from u to v in T. If this were violated, removing the
        heaviest path edge and adding (u, v) would produce a lighter spanning
        tree.

    Test input:
        Random connected undirected weighted graphs with 2 to 50 nodes. Connectivity
        is guaranteed by starting with a spanning path through all nodes, then
        optionally adding extra random edges with weights in [1, 100].

    Assumptions / Preconditions:
        - The input graph is connected (ensured by the connected_graphs strategy).
        - The input graph is undirected and simple (no self-loops or parallel edges).
        - The MST is a valid tree (connected and acyclic), so a unique path
          exists between every pair of nodes in the MST.

    What a failure indicates:
        If this test fails, Prim's algorithm has chosen a suboptimal edge over
        a cheaper alternative. This is the strongest test of MST correctness —
        it directly verifies the optimality of every edge decision. A failure
        would indicate a fundamental flaw in the greedy selection logic, such
        as incorrect priority queue ordering or wrong weight comparisons.
    """
    mst = nx.minimum_spanning_tree(G, algorithm="prim")
    mst_edge_set = set(mst.edges())

    for u, v, d in G.edges(data=True):
        if (u, v) not in mst_edge_set and (v, u) not in mst_edge_set:
            # Find the unique path from u to v in the MST
            path = nx.shortest_path(mst, u, v)
            # Find the maximum weight edge on that path
            max_path_weight = max(
                mst[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1)
            )
            assert d["weight"] >= max_path_weight

