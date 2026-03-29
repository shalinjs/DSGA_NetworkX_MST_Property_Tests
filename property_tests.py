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
