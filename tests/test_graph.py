"""
Unit tests for the graph implementation.

This module provides comprehensive test cases for the AdjacencyListGraph class,
including graph operations, traversal algorithms, and edge cases.
"""

import pytest
from src.graph import AdjacencyListGraph


class TestAdjacencyListGraph:
    """Test cases for AdjacencyListGraph implementation."""
    
    def test_empty_graph_initialization(self):
        """Test that a new graph is empty."""
        graph = AdjacencyListGraph()
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0
        assert not graph.is_directed()
    
    def test_directed_graph_initialization(self):
        """Test directed graph initialization."""
        graph = AdjacencyListGraph(directed=True)
        assert graph.vertex_count() == 0
        assert graph.edge_count() == 0
        assert graph.is_directed()
    
    def test_add_vertex(self):
        """Test adding vertices to the graph."""
        graph = AdjacencyListGraph()
        
        graph.add_vertex("A")
        assert graph.vertex_count() == 1
        assert "A" in list(graph.vertices())
        
        # Adding duplicate vertex should not increase count
        graph.add_vertex("A")
        assert graph.vertex_count() == 1
        
        graph.add_vertex("B")
        assert graph.vertex_count() == 2
    
    def test_add_edge_undirected(self):
        """Test adding edges in undirected graph."""
        graph = AdjacencyListGraph(directed=False)
        
        # Add edge should automatically add vertices
        graph.add_edge("A", "B")
        assert graph.vertex_count() == 2
        assert graph.edge_count() == 1
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "A")  # Undirected
        
        # Add another edge
        graph.add_edge("B", "C", 2.5)
        assert graph.vertex_count() == 3
        assert graph.edge_count() == 2
        assert graph.get_edge_weight("B", "C") == 2.5
        assert graph.get_edge_weight("C", "B") == 2.5
    
    def test_add_edge_directed(self):
        """Test adding edges in directed graph."""
        graph = AdjacencyListGraph(directed=True)
        
        graph.add_edge("A", "B")
        assert graph.vertex_count() == 2
        assert graph.edge_count() == 1
        assert graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")  # Directed
    
    def test_remove_vertex(self):
        """Test removing vertices and their edges."""
        graph = AdjacencyListGraph()
        
        # Build a small graph
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("A", "C")
        
        # Remove vertex B
        result = graph.remove_vertex("B")
        assert result == True
        assert graph.vertex_count() == 2
        assert graph.edge_count() == 1  # Only A-C remains
        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "C")
        assert graph.has_edge("A", "C")
        
        # Try to remove non-existent vertex
        result = graph.remove_vertex("D")
        assert result == False
    
    def test_remove_edge(self):
        """Test removing edges."""
        graph = AdjacencyListGraph()
        
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        # Remove existing edge
        result = graph.remove_edge("A", "B")
        assert result == True
        assert not graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")  # Undirected
        assert graph.vertex_count() == 3  # Vertices remain
        assert graph.edge_count() == 1
        
        # Try to remove non-existent edge
        result = graph.remove_edge("A", "C")
        assert result == False
    
    def test_get_neighbors(self):
        """Test getting vertex neighbors."""
        graph = AdjacencyListGraph()
        
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        
        neighbors_a = set(graph.get_neighbors("A"))
        assert neighbors_a == {"B", "C"}
        
        neighbors_b = set(graph.get_neighbors("B"))
        assert neighbors_b == {"A", "D"}
        
        neighbors_d = set(graph.get_neighbors("D"))
        assert neighbors_d == {"B"}
        
        # Non-existent vertex
        neighbors_x = list(graph.get_neighbors("X"))
        assert neighbors_x == []
    
    def test_breadth_first_search(self):
        """Test BFS traversal."""
        graph = AdjacencyListGraph()
        
        # Build a graph: A-B-C-D with A-C connection
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        graph.add_edge("A", "C")
        
        bfs_result = graph.breadth_first_search("A")
        assert "A" in bfs_result
        assert len(bfs_result) == 4  # All vertices reachable
        assert bfs_result[0] == "A"  # Starts with source
        
        # Test with non-existent vertex
        bfs_result = graph.breadth_first_search("X")
        assert bfs_result == []
    
    def test_depth_first_search(self):
        """Test DFS traversal."""
        graph = AdjacencyListGraph()
        
        # Build a graph
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("A", "D")
        
        dfs_result = graph.depth_first_search("A")
        assert "A" in dfs_result
        assert len(dfs_result) == 4  # All vertices reachable
        assert dfs_result[0] == "A"  # Starts with source
        
        # Test with non-existent vertex
        dfs_result = graph.depth_first_search("X")
        assert dfs_result == []
    
    def test_shortest_path(self):
        """Test shortest path algorithm."""
        graph = AdjacencyListGraph()
        
        # Build a weighted graph
        graph.add_edge("A", "B", 2.0)
        graph.add_edge("B", "C", 3.0)
        graph.add_edge("A", "C", 10.0)  # Longer direct path
        graph.add_edge("C", "D", 1.0)
        
        # Shortest path A to D
        path = graph.shortest_path("A", "D")
        assert path == ["A", "B", "C", "D"]
        
        # Same vertex
        path = graph.shortest_path("A", "A")
        assert path == ["A"]
        
        # No path exists
        graph.add_vertex("E")  # Isolated vertex
        path = graph.shortest_path("A", "E")
        assert path is None
        
        # Non-existent vertices
        path = graph.shortest_path("X", "Y")
        assert path is None
    
    def test_connected_components(self):
        """Test finding connected components."""
        graph = AdjacencyListGraph()
        
        # Create two separate components
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        graph.add_edge("D", "E")
        
        graph.add_vertex("F")  # Isolated vertex
        
        components = graph.find_connected_components()
        assert len(components) == 3
        
        # Check component sizes
        component_sizes = [len(comp) for comp in components]
        component_sizes.sort()
        assert component_sizes == [1, 2, 3]
    
    def test_vertex_degree(self):
        """Test vertex degree calculation."""
        graph = AdjacencyListGraph()
        
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("A", "D")
        
        assert graph.get_vertex_degree("A") == 3
        assert graph.get_vertex_degree("B") == 1
        assert graph.get_vertex_degree("X") == 0  # Non-existent vertex
    
    def test_vertex_degree_directed(self):
        """Test vertex degree in directed graph."""
        graph = AdjacencyListGraph(directed=True)
        
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("D", "A")
        
        # In directed graph, degree returns out-degree
        assert graph.get_vertex_degree("A") == 2
        assert graph.get_vertex_degree("D") == 1
        assert graph.get_vertex_degree("B") == 0
    
    def test_edges_iteration(self):
        """Test iterating over edges."""
        graph = AdjacencyListGraph()
        
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 2.0)
        
        edges = list(graph.edges())
        assert len(edges) == 2
        
        # Check edge format (from, to, weight)
        edge_set = {(edge[0], edge[1]) for edge in edges}
        assert ("A", "B") in edge_set or ("B", "A") in edge_set
        assert ("B", "C") in edge_set or ("C", "B") in edge_set
    
    def test_from_dict_to_dict(self):
        """Test loading from and converting to dictionary."""
        graph = AdjacencyListGraph()
        
        # Test dictionary format
        graph_dict = {
            "A": ["B", "C"],
            "B": ["A", "D"],
            "C": ["A"],
            "D": ["B"]
        }
        
        graph.from_dict(graph_dict)
        assert graph.vertex_count() == 4
        assert graph.edge_count() == 3  # Undirected edges
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "D")
        
        # Convert back to dictionary
        result_dict = graph.to_dict()
        assert len(result_dict) == 4
        assert set(result_dict["A"]) == {"B", "C"}
    
    def test_large_graph(self):
        """Test with a larger graph to check performance."""
        graph = AdjacencyListGraph()
        
        # Create a graph with 100 vertices
        num_vertices = 100
        for i in range(num_vertices):
            for j in range(i + 1, min(i + 4, num_vertices)):  # Connect to next 3 vertices
                graph.add_edge(f"v{i}", f"v{j}")
        
        assert graph.vertex_count() == num_vertices
        assert graph.edge_count() > 0
        
        # Test BFS on large graph
        bfs_result = graph.breadth_first_search("v0")
        assert len(bfs_result) == num_vertices  # All should be connected
    
    def test_weighted_edges(self):
        """Test graphs with weighted edges."""
        graph = AdjacencyListGraph()
        
        graph.add_edge("A", "B", 5.5)
        graph.add_edge("B", "C", 2.3)
        
        assert graph.get_edge_weight("A", "B") == 5.5
        assert graph.get_edge_weight("B", "A") == 5.5  # Undirected
        assert graph.get_edge_weight("B", "C") == 2.3
        assert graph.get_edge_weight("A", "C") is None  # No direct edge
    
    def test_self_loops(self):
        """Test handling of self-loops."""
        graph = AdjacencyListGraph()
        
        graph.add_edge("A", "A")  # Self-loop
        assert graph.has_edge("A", "A")
        assert graph.vertex_count() == 1
        assert graph.edge_count() == 1
    
    def test_string_representation(self):
        """Test string representations."""
        graph = AdjacencyListGraph()
        graph.add_edge("A", "B")
        
        str_repr = str(graph)
        assert "Graph" in str_repr
        assert "vertices=2" in str_repr
        assert "edges=1" in str_repr
        
        repr_str = repr(graph)
        assert "AdjacencyListGraph" in repr_str


if __name__ == "__main__":
    pytest.main([__file__]) 