"""
Graph implementation for social network and web page link analysis.

This module provides a graph data structure using adjacency list representation
with support for both directed and undirected graphs.
"""

from typing import Dict, List, Set, Optional, Iterator, Tuple, Any
from collections import deque, defaultdict
import heapq
from src.base import Graph, track_performance


class AdjacencyListGraph(Graph[str]):
    """
    Graph implementation using adjacency list representation.
    
    Supports both directed and undirected graphs with weighted and unweighted edges.
    Optimized for social network analysis and web page link structures.
    """
    
    def __init__(self, directed: bool = False):
        """
        Initialize the graph.
        
        Args:
            directed: Whether the graph is directed or undirected
        """
        self._directed = directed
        self._adjacency_list: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._vertices: Set[str] = set()
        self._edge_count = 0
    
    @track_performance
    def add_vertex(self, vertex: str) -> None:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: The vertex to add
        """
        if vertex not in self._vertices:
            self._vertices.add(vertex)
            if vertex not in self._adjacency_list:
                self._adjacency_list[vertex] = {}
    
    @track_performance
    def add_edge(self, from_vertex: str, to_vertex: str, weight: float = 1.0) -> None:
        """
        Add an edge between two vertices.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            weight: Edge weight (default 1.0)
        """
        # Ensure vertices exist
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)
        
        # Check if edge already exists
        edge_exists = to_vertex in self._adjacency_list[from_vertex]
        
        # Add edge
        self._adjacency_list[from_vertex][to_vertex] = weight
        
        # For undirected graphs, add reverse edge
        if not self._directed:
            self._adjacency_list[to_vertex][from_vertex] = weight
        
        # Increment edge count only if this is a new edge
        if not edge_exists:
            self._edge_count += 1
    
    @track_performance
    def remove_vertex(self, vertex: str) -> bool:
        """
        Remove a vertex and all its edges.
        
        Args:
            vertex: The vertex to remove
            
        Returns:
            True if vertex was removed, False if it didn't exist
        """
        if vertex not in self._vertices:
            return False
        
        # Count edges to be removed
        edges_to_remove = len(self._adjacency_list[vertex])
        
        # For undirected graphs, don't double count
        if not self._directed:
            # Remove edges from other vertices to this vertex
            for v in self._vertices:
                if v != vertex and vertex in self._adjacency_list[v]:
                    del self._adjacency_list[v][vertex]
        else:
            # For directed graphs, count incoming edges separately
            for v in self._vertices:
                if v != vertex and vertex in self._adjacency_list[v]:
                    del self._adjacency_list[v][vertex]
                    edges_to_remove += 1
        
        # Update edge count
        self._edge_count -= edges_to_remove
        
        # Remove vertex
        self._vertices.remove(vertex)
        del self._adjacency_list[vertex]
        
        return True
    
    @track_performance
    def remove_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """
        Remove an edge between two vertices.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            
        Returns:
            True if edge was removed, False if it didn't exist
        """
        if (from_vertex not in self._vertices or 
            to_vertex not in self._vertices or
            to_vertex not in self._adjacency_list[from_vertex]):
            return False
        
        del self._adjacency_list[from_vertex][to_vertex]
        self._edge_count -= 1
        
        # For undirected graphs, also remove reverse edge
        if not self._directed and from_vertex in self._adjacency_list[to_vertex]:
            del self._adjacency_list[to_vertex][from_vertex]
        
        return True
    
    def get_neighbors(self, vertex: str) -> Iterator[str]:
        """
        Get all neighbors of a vertex.
        
        Args:
            vertex: The vertex to get neighbors for
            
        Yields:
            Neighbor vertices
        """
        if vertex in self._adjacency_list:
            for neighbor in self._adjacency_list[vertex]:
                yield neighbor
    
    def get_incoming_edges(self, vertex: str) -> Iterator[str]:
        """
        Get all vertices that have edges pointing to this vertex.
        Useful for directed graphs to find incoming connections.
        
        Args:
            vertex: The target vertex
            
        Yields:
            Source vertices that point to the target vertex
        """
        if vertex not in self._vertices:
            return
            
        for source_vertex in self._vertices:
            if source_vertex != vertex and vertex in self._adjacency_list[source_vertex]:
                yield source_vertex
    
    def has_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """
        Check if an edge exists between two vertices.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            
        Returns:
            True if edge exists, False otherwise
        """
        return (from_vertex in self._adjacency_list and 
                to_vertex in self._adjacency_list[from_vertex])
    
    def get_edge_weight(self, from_vertex: str, to_vertex: str) -> Optional[float]:
        """
        Get the weight of an edge.
        
        Args:
            from_vertex: Source vertex
            to_vertex: Destination vertex
            
        Returns:
            Edge weight if edge exists, None otherwise
        """
        if self.has_edge(from_vertex, to_vertex):
            return self._adjacency_list[from_vertex][to_vertex]
        return None
    
    @track_performance
    def breadth_first_search(self, start_vertex: str) -> List[str]:
        """
        Perform breadth-first search starting from a vertex.
        
        Args:
            start_vertex: Starting vertex for BFS
            
        Returns:
            List of vertices in BFS order
        """
        if start_vertex not in self._vertices:
            return []
        
        visited = set()
        queue = deque([start_vertex])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # Add unvisited neighbors to queue
                for neighbor in self.get_neighbors(vertex):
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result
    
    @track_performance
    def depth_first_search(self, start_vertex: str) -> List[str]:
        """
        Perform depth-first search starting from a vertex.
        
        Args:
            start_vertex: Starting vertex for DFS
            
        Returns:
            List of vertices in DFS order
        """
        if start_vertex not in self._vertices:
            return []
        
        visited = set()
        result = []
        
        def dfs_recursive(vertex: str):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in self.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs_recursive(neighbor)
        
        dfs_recursive(start_vertex)
        return result
    
    @track_performance
    def shortest_path(self, start_vertex: str, end_vertex: str) -> Optional[List[str]]:
        """
        Find shortest path between two vertices using Dijkstra's algorithm.
        
        Args:
            start_vertex: Starting vertex
            end_vertex: Destination vertex
            
        Returns:
            List of vertices representing the shortest path, or None if no path exists
        """
        if start_vertex not in self._vertices or end_vertex not in self._vertices:
            return None
        
        # Dijkstra's algorithm
        distances = {vertex: float('inf') for vertex in self._vertices}
        distances[start_vertex] = 0
        previous = {}
        
        # Priority queue: (distance, vertex)
        pq = [(0, start_vertex)]
        visited = set()
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_vertex in visited:
                continue
            
            visited.add(current_vertex)
            
            if current_vertex == end_vertex:
                break
            
            for neighbor in self.get_neighbors(current_vertex):
                if neighbor not in visited:
                    weight = self.get_edge_weight(current_vertex, neighbor)
                    distance = current_distance + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_vertex
                        heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        if end_vertex not in previous and start_vertex != end_vertex:
            return None
        
        path = []
        current = end_vertex
        while current is not None:
            path.append(current)
            current = previous.get(current)
        
        path.reverse()
        return path if path[0] == start_vertex else None
    
    def find_connected_components(self) -> List[Set[str]]:
        """
        Find all connected components in the graph.
        
        Returns:
            List of sets, each containing vertices in a connected component
        """
        visited = set()
        components = []
        
        for vertex in self._vertices:
            if vertex not in visited:
                # BFS to find all vertices in this component
                component = set()
                queue = deque([vertex])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        for neighbor in self.get_neighbors(current):
                            if neighbor not in visited:
                                queue.append(neighbor)
                
                components.append(component)
        
        return components
    
    def get_vertex_degree(self, vertex: str) -> int:
        """
        Get the degree of a vertex (number of edges).
        
        Args:
            vertex: The vertex to get degree for
            
        Returns:
            Degree of the vertex
        """
        if vertex not in self._vertices:
            return 0
        
        if self._directed:
            # For directed graphs, return out-degree
            return len(self._adjacency_list[vertex])
        else:
            # For undirected graphs, return total degree
            return len(self._adjacency_list[vertex])
    
    def vertices(self) -> Iterator[str]:
        """Return an iterator over all vertices."""
        for vertex in self._vertices:
            yield vertex
    
    def edges(self) -> Iterator[Tuple[str, str, float]]:
        """Return an iterator over all edges as (from, to, weight) tuples."""
        seen_edges = set()
        
        for from_vertex in self._adjacency_list:
            for to_vertex, weight in self._adjacency_list[from_vertex].items():
                edge = (from_vertex, to_vertex)
                
                # For undirected graphs, avoid duplicate edges
                if not self._directed:
                    edge = tuple(sorted([from_vertex, to_vertex]))
                    if edge in seen_edges:
                        continue
                    seen_edges.add(edge)
                
                yield (from_vertex, to_vertex, weight)
    
    def vertex_count(self) -> int:
        """Return the number of vertices."""
        return len(self._vertices)
    
    def edge_count(self) -> int:
        """Return the number of edges."""
        return self._edge_count
    
    def get_vertices(self) -> Iterator[str]:
        """
        Get all vertices in the graph.
        
        Yields:
            All vertex identifiers
        """
        for vertex in self._vertices:
            yield vertex
    
    def is_directed(self) -> bool:
        """Check if the graph is directed."""
        return self._directed
    
    def to_dict(self) -> Dict[str, List[str]]:
        """
        Convert graph to dictionary format (adjacency list without weights).
        
        Returns:
            Dictionary mapping vertices to lists of neighbors
        """
        result = {}
        for vertex in self._vertices:
            result[vertex] = list(self.get_neighbors(vertex))
        return result
    
    def from_dict(self, graph_dict: Dict[str, List[str]]) -> None:
        """
        Load graph from dictionary format.
        
        Args:
            graph_dict: Dictionary mapping vertices to lists of neighbors
        """
        # Clear existing graph
        self._adjacency_list.clear()
        self._vertices.clear()
        self._edge_count = 0
        
        # Add all vertices first
        for vertex in graph_dict:
            self.add_vertex(vertex)
        
        # Add edges
        for vertex, neighbors in graph_dict.items():
            for neighbor in neighbors:
                self.add_edge(vertex, neighbor)
    
    def __str__(self) -> str:
        """String representation of the graph."""
        return f"Graph(vertices={self.vertex_count()}, edges={self.edge_count()}, directed={self._directed})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"AdjacencyListGraph(vertices={self.vertex_count()}, edges={self.edge_count()}, directed={self._directed})" 