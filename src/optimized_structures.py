"""
Phase 3: Optimized Data Structures

Enhanced implementations with optimizations for performance and scalability:
- AVL Tree: Self-balancing BST for guaranteed O(log n) operations
- OptimizedHashTable: Enhanced hash table with improved load balancing
- OptimizedGraph: Graph with cached algorithms and optimizations
- AdvancedPriorityQueue: Priority queue with decrease-key operations
"""

from typing import Optional, Any, Dict, List, Set, Iterator, Tuple
from collections import defaultdict, deque
import heapq
import time
import math

from .base import Tree, KeyValueStore, Graph, PriorityQueue, track_performance


class AVLNode:
    """Node for AVL Tree with height balancing."""
    
    def __init__(self, key: Any, value: Any):
        self.key = key
        self.value = value
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height = 1
    
    def update_height(self):
        """Update height based on children heights."""
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        self.height = 1 + max(left_height, right_height)
    
    def get_balance(self) -> int:
        """Get balance factor (left height - right height)."""
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        return left_height - right_height


class OptimizedAVLTree(Tree[Any]):
    """
    Phase 3: Self-balancing AVL Tree for guaranteed O(log n) performance.
    Solves the recursion limit issue from Phase 2 BST implementation.
    """
    
    def __init__(self, key_func=None):
        self._root: Optional[AVLNode] = None
        self._size = 0
        self._key_func = key_func or (lambda x: x)
        
        # Performance optimization: cache frequently accessed values
        self._min_cache = None
        self._max_cache = None
        self._cache_valid = False
    
    def _get_key(self, item: Any) -> Any:
        """Extract key from item using key function."""
        return self._key_func(item)
    
    def _rotate_right(self, y: AVLNode) -> AVLNode:
        """Perform right rotation for balancing."""
        x = y.left
        T2 = x.right
        
        # Perform rotation
        x.right = y
        y.left = T2
        
        # Update heights
        y.update_height()
        x.update_height()
        
        return x
    
    def _rotate_left(self, x: AVLNode) -> AVLNode:
        """Perform left rotation for balancing."""
        y = x.right
        T2 = y.left
        
        # Perform rotation
        y.left = x
        x.right = T2
        
        # Update heights
        x.update_height()
        y.update_height()
        
        return y
    
    def _balance_node(self, node: AVLNode) -> AVLNode:
        """Balance the node using AVL rotations."""
        # Update height
        node.update_height()
        
        # Get balance factor
        balance = node.get_balance()
        
        # Left heavy
        if balance > 1:
            # Left-Right case
            if node.left.get_balance() < 0:
                node.left = self._rotate_left(node.left)
            # Left-Left case
            return self._rotate_right(node)
        
        # Right heavy
        if balance < -1:
            # Right-Left case
            if node.right.get_balance() > 0:
                node.right = self._rotate_right(node.right)
            # Right-Right case
            return self._rotate_left(node)
        
        return node
    
    @track_performance
    def insert(self, item: Any) -> None:
        """Insert item with automatic balancing."""
        key = self._get_key(item)
        self._root = self._insert_recursive(self._root, key, item)
        self._size += 1
        self._cache_valid = False
    
    def _insert_recursive(self, node: Optional[AVLNode], key: Any, item: Any) -> AVLNode:
        """Recursive insert with balancing."""
        # Base case
        if node is None:
            return AVLNode(key, item)
        
        # Standard BST insertion
        if key < node.key:
            node.left = self._insert_recursive(node.left, key, item)
        elif key > node.key:
            node.right = self._insert_recursive(node.right, key, item)
        else:
            # Update existing key
            node.value = item
            return node
        
        # Balance the node
        return self._balance_node(node)
    
    @track_performance
    def search(self, key: Any) -> Optional[Any]:
        """Search for item by key."""
        return self._search_recursive(self._root, key)
    
    def _search_recursive(self, node: Optional[AVLNode], key: Any) -> Optional[Any]:
        """Recursive search."""
        if node is None:
            return None
        
        if key == node.key:
            return node.value
        elif key < node.key:
            return self._search_recursive(node.left, key)
        else:
            return self._search_recursive(node.right, key)
    
    @track_performance
    def delete(self, key: Any) -> bool:
        """Delete item by key with automatic balancing."""
        initial_size = self._size
        self._root = self._delete_recursive(self._root, key)
        
        if self._size < initial_size:
            self._cache_valid = False
            return True
        return False
    
    def _delete_recursive(self, node: Optional[AVLNode], key: Any) -> Optional[AVLNode]:
        """Recursive delete with balancing."""
        if node is None:
            return None
        
        # Standard BST deletion
        if key < node.key:
            node.left = self._delete_recursive(node.left, key)
        elif key > node.key:
            node.right = self._delete_recursive(node.right, key)
        else:
            # Node to be deleted found
            self._size -= 1
            
            # Node with 0 or 1 child
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            
            # Node with 2 children: get inorder successor
            successor = self._find_min_node(node.right)
            node.key = successor.key
            node.value = successor.value
            # Don't decrement size again when deleting successor
            temp_size = self._size
            node.right = self._delete_recursive(node.right, successor.key)
            self._size = temp_size  # Restore size since successor deletion would decrement it again
        
        # Balance the node
        return self._balance_node(node)
    
    def _find_min_node(self, node: AVLNode) -> AVLNode:
        """Find minimum node in subtree."""
        while node.left is not None:
            node = node.left
        return node
    
    def range_search(self, min_key: Any, max_key: Any) -> List[Any]:
        """Find all items with keys in range [min_key, max_key]."""
        result = []
        self._range_search_recursive(self._root, min_key, max_key, result)
        return result
    
    def _range_search_recursive(self, node: Optional[AVLNode], min_key: Any, max_key: Any, result: List[Any]):
        """Recursive range search."""
        if node is None:
            return
        
        # If current key is in range, add to result
        if min_key <= node.key <= max_key:
            result.append(node.value)
        
        # Recursively search left and right based on range
        if node.key > min_key:
            self._range_search_recursive(node.left, min_key, max_key, result)
        if node.key < max_key:
            self._range_search_recursive(node.right, min_key, max_key, result)
    
    def min_value(self) -> Optional[Any]:
        """Get minimum value with caching optimization."""
        if self._cache_valid and self._min_cache is not None:
            return self._min_cache
        
        if self._root is None:
            return None
        
        node = self._root
        while node.left is not None:
            node = node.left
        
        self._min_cache = node.value
        return node.value
    
    def max_value(self) -> Optional[Any]:
        """Get maximum value with caching optimization."""
        if self._cache_valid and self._max_cache is not None:
            return self._max_cache
        
        if self._root is None:
            return None
        
        node = self._root
        while node.right is not None:
            node = node.right
        
        self._max_cache = node.value
        return node.value
    
    def height(self) -> int:
        """Get tree height."""
        return self._root.height if self._root else 0
    
    def size(self) -> int:
        """Get number of items in tree."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if tree is empty."""
        return self._size == 0
    
    def is_balanced(self) -> bool:
        """Check if tree is properly balanced (for testing)."""
        return self._check_balance(self._root)
    
    def _check_balance(self, node: Optional[AVLNode]) -> bool:
        """Recursively check if tree is balanced."""
        if node is None:
            return True
        
        balance = node.get_balance()
        if abs(balance) > 1:
            return False
        
        return self._check_balance(node.left) and self._check_balance(node.right)
    
    def traverse_inorder(self) -> Iterator[Any]:
        """Traverse tree in inorder."""
        return self._inorder_traversal(self._root)
    
    def _inorder_traversal(self, node: Optional[AVLNode]) -> Iterator[Any]:
        """Recursive inorder traversal."""
        if node is not None:
            yield from self._inorder_traversal(node.left)
            yield node.value
            yield from self._inorder_traversal(node.right)
    
    def traverse_preorder(self) -> Iterator[Any]:
        """Traverse tree in preorder."""
        return self._preorder_traversal(self._root)
    
    def _preorder_traversal(self, node: Optional[AVLNode]) -> Iterator[Any]:
        """Recursive preorder traversal."""
        if node is not None:
            yield node.value
            yield from self._preorder_traversal(node.left)
            yield from self._preorder_traversal(node.right)
    
    def traverse_postorder(self) -> Iterator[Any]:
        """Traverse tree in postorder."""
        return self._postorder_traversal(self._root)
    
    def _postorder_traversal(self, node: Optional[AVLNode]) -> Iterator[Any]:
        """Recursive postorder traversal."""
        if node is not None:
            yield from self._postorder_traversal(node.left)
            yield from self._postorder_traversal(node.right)
            yield node.value


class OptimizedHashTable(KeyValueStore[str, Any]):
    """
    Phase 3: Enhanced hash table with improved load balancing and performance.
    """
    
    def __init__(self, initial_capacity: int = 16):
        self._capacity = initial_capacity
        self._size = 0
        self._buckets = [[] for _ in range(self._capacity)]
        self._load_factor_threshold = 0.75
        
        # Performance optimizations
        self._resize_factor = 2
        self._min_capacity = 16
        self._access_count = 0
        self._collision_count = 0
        
        # Enhanced hashing with multiple hash functions
        self._hash_functions = [
            self._hash_djb2,
            self._hash_fnv1a,
            self._hash_murmur3_simple
        ]
        self._current_hash_func = 0
    
    def _hash_djb2(self, key: str) -> int:
        """DJB2 hash function - good general purpose hash."""
        hash_value = 5381
        for char in key:
            hash_value = ((hash_value << 5) + hash_value) + ord(char)
        return hash_value % self._capacity
    
    def _hash_fnv1a(self, key: str) -> int:
        """FNV-1a hash function - good for strings."""
        hash_value = 2166136261
        for char in key:
            hash_value ^= ord(char)
            hash_value *= 16777619
            hash_value &= 0xffffffff
        return hash_value % self._capacity
    
    def _hash_murmur3_simple(self, key: str) -> int:
        """Simplified MurmurHash3 - good distribution."""
        hash_value = 0
        for i, char in enumerate(key):
            hash_value ^= ord(char) << (i % 4 * 8)
        hash_value ^= hash_value >> 16
        hash_value *= 0x85ebca6b
        hash_value ^= hash_value >> 13
        hash_value *= 0xc2b2ae35
        hash_value ^= hash_value >> 16
        return hash_value % self._capacity
    
    def _hash(self, key: str) -> int:
        """Use current hash function."""
        return self._hash_functions[self._current_hash_func](key)
    
    def _should_resize(self) -> bool:
        """Determine if resize is needed based on load factor and collision rate."""
        load_factor = self._size / self._capacity
        collision_rate = self._collision_count / max(1, self._access_count)
        
        return (load_factor >= self._load_factor_threshold or 
                (collision_rate > 0.3 and self._size > 32))
    
    def _resize(self):
        """Resize hash table with improved algorithm."""
        old_buckets = self._buckets
        old_capacity = self._capacity
        
        # Try different hash function if too many collisions
        if self._collision_count / max(1, self._access_count) > 0.4:
            self._current_hash_func = (self._current_hash_func + 1) % len(self._hash_functions)
        
        # Increase capacity
        self._capacity = old_capacity * self._resize_factor
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
        self._collision_count = 0
        
        # Rehash all items
        for bucket in old_buckets:
            for key, value in bucket:
                self._insert_without_resize(key, value)
    
    def _insert_without_resize(self, key: str, value: Any):
        """Insert without triggering resize."""
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        # Track collision
        if bucket:
            self._collision_count += 1
        
        for i, (existing_key, existing_value) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return
        
        bucket.append((key, value))
        self._size += 1
    
    @track_performance
    def put(self, key: str, value: Any) -> None:
        """Insert with optimized resizing."""
        self._access_count += 1
        
        if self._should_resize():
            self._resize()
        
        self._insert_without_resize(key, value)
    
    @track_performance
    def get(self, key: str) -> Optional[Any]:
        """Get with access tracking."""
        self._access_count += 1
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        for existing_key, existing_value in bucket:
            if existing_key == key:
                return existing_value
        
        return None
    
    @track_performance
    def delete(self, key: str) -> bool:
        """Delete key-value pair."""
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        for i, (existing_key, existing_value) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self._size -= 1
                return True
        
        return False
    
    def keys(self) -> Iterator[str]:
        """Get all keys."""
        for bucket in self._buckets:
            for key, value in bucket:
                yield key
    
    def values(self) -> Iterator[Any]:
        """Get all values."""
        for bucket in self._buckets:
            for key, value in bucket:
                yield value
    
    def size(self) -> int:
        """Get number of items in hash table."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if hash table is empty."""
        return self._size == 0
    
    def clear(self) -> None:
        """Clear all items from hash table."""
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
        self._collision_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'size': self._size,
            'capacity': self._capacity,
            'load_factor': self._size / self._capacity,
            'collision_rate': self._collision_count / max(1, self._access_count),
            'access_count': self._access_count,
            'current_hash_function': self._current_hash_func
        }


class OptimizedSocialGraph(Graph[str]):
    """
    Phase 3: Enhanced graph with cached algorithms and optimizations.
    """
    
    def __init__(self, directed: bool = False):
        self._directed = directed
        self._adjacency_list: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._vertices: Set[str] = set()
        self._edge_count = 0
        
        # Performance optimizations
        self._centrality_cache: Dict[str, float] = {}
        self._shortest_path_cache: Dict[Tuple[str, str], Optional[List[str]]] = {}
        self._component_cache: Optional[List[Set[str]]] = None
        self._cache_valid = False
        
        # For large graphs: degree caching
        self._degree_cache: Dict[str, int] = {}
    
    def _invalidate_cache(self):
        """Invalidate all cached computations."""
        self._cache_valid = False
        self._centrality_cache.clear()
        self._shortest_path_cache.clear()
        self._component_cache = None
        self._degree_cache.clear()
    
    @track_performance
    def add_vertex(self, vertex: str) -> None:
        """Add a vertex to the graph."""
        if vertex not in self._vertices:
            self._vertices.add(vertex)
            self._adjacency_list[vertex] = {}
            self._invalidate_cache()
    
    def get_neighbors(self, vertex: str) -> Iterator[str]:
        """Get neighbors of a vertex."""
        if vertex in self._adjacency_list:
            for neighbor in self._adjacency_list[vertex]:
                yield neighbor
    
    def has_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """Check if edge exists between vertices."""
        return (from_vertex in self._adjacency_list and 
                to_vertex in self._adjacency_list[from_vertex])
    
    @track_performance
    def remove_edge(self, from_vertex: str, to_vertex: str) -> bool:
        """Remove edge between vertices."""
        if not self.has_edge(from_vertex, to_vertex):
            return False
        
        del self._adjacency_list[from_vertex][to_vertex]
        self._edge_count -= 1
        
        # For undirected graphs, also remove reverse edge
        if not self._directed and self.has_edge(to_vertex, from_vertex):
            del self._adjacency_list[to_vertex][from_vertex]
        
        self._invalidate_cache()
        return True
    
    @track_performance
    def remove_vertex(self, vertex: str) -> bool:
        """Remove vertex and all its edges."""
        if vertex not in self._vertices:
            return False
        
        # Remove all edges to this vertex
        for v in list(self._vertices):
            if v != vertex and self.has_edge(v, vertex):
                self.remove_edge(v, vertex)
        
        # Remove all edges from this vertex
        neighbors = list(self.get_neighbors(vertex))
        for neighbor in neighbors:
            self.remove_edge(vertex, neighbor)
        
        # Remove vertex
        self._vertices.remove(vertex)
        del self._adjacency_list[vertex]
        
        self._invalidate_cache()
        return True
    
    def vertex_count(self) -> int:
        """Get number of vertices."""
        return len(self._vertices)
    
    def edge_count(self) -> int:
        """Get number of edges."""
        return self._edge_count
    
    def has_vertex(self, vertex: str) -> bool:
        """Check if vertex exists in graph."""
        return vertex in self._vertices
    
    def is_directed(self) -> bool:
        """Check if graph is directed."""
        return self._directed
    
    def get_incoming_edges(self, vertex: str) -> Iterator[str]:
        """Get vertices that have edges pointing to this vertex."""
        if vertex not in self._vertices:
            return
            
        for source_vertex in self._vertices:
            if source_vertex != vertex and self.has_edge(source_vertex, vertex):
                yield source_vertex
    
    def get_vertices(self) -> Iterator[str]:
        """Get all vertices."""
        for vertex in self._vertices:
            yield vertex
    
    # Add other required methods
    def breadth_first_search(self, start_vertex: str) -> Iterator[str]:
        """BFS traversal."""
        if start_vertex not in self._vertices:
            return
        
        visited = set()
        queue = deque([start_vertex])
        visited.add(start_vertex)
        
        while queue:
            vertex = queue.popleft()
            yield vertex
            
            for neighbor in self.get_neighbors(vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    def depth_first_search(self, start_vertex: str) -> Iterator[str]:
        """DFS traversal."""
        if start_vertex not in self._vertices:
            return
        
        visited = set()
        stack = [start_vertex]
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                yield vertex
                
                # Add neighbors to stack in reverse order for consistent traversal
                neighbors = list(self.get_neighbors(vertex))
                for neighbor in reversed(neighbors):
                    if neighbor not in visited:
                        stack.append(neighbor)
    
    def shortest_path(self, start_vertex: str, end_vertex: str) -> Optional[List[str]]:
        """Find shortest path using Dijkstra's algorithm."""
        if start_vertex not in self._vertices or end_vertex not in self._vertices:
            return None
        
        distances = {vertex: float('inf') for vertex in self._vertices}
        distances[start_vertex] = 0
        previous = {}
        pq = [(0, start_vertex)]
        visited = set()
        
        while pq:
            current_distance, current_vertex = heapq.heappop(pq)
            
            if current_vertex in visited:
                continue
            
            visited.add(current_vertex)
            
            if current_vertex == end_vertex:
                # Reconstruct path
                path = []
                while current_vertex is not None:
                    path.append(current_vertex)
                    current_vertex = previous.get(current_vertex)
                return list(reversed(path))
            
            for neighbor in self.get_neighbors(current_vertex):
                if neighbor not in visited:
                    weight = self._adjacency_list[current_vertex].get(neighbor, 1.0)
                    distance = current_distance + weight
                    
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current_vertex
                        heapq.heappush(pq, (distance, neighbor))
        
        return None
    
    def find_connected_components(self) -> List[Set[str]]:
        """Find connected components."""
        visited = set()
        components = []
        
        for vertex in self._vertices:
            if vertex not in visited:
                component = set()
                stack = [vertex]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        
                        for neighbor in self.get_neighbors(current):
                            if neighbor not in visited:
                                stack.append(neighbor)
                        
                        # For undirected graphs, also check incoming edges
                        if not self._directed:
                            for incoming in self.get_incoming_edges(current):
                                if incoming not in visited:
                                    stack.append(incoming)
                
                components.append(component)
        
        return components
    
    @track_performance
    def add_edge(self, from_vertex: str, to_vertex: str, weight: float = 1.0) -> None:
        """Add edge with cache invalidation."""
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)
        
        if to_vertex not in self._adjacency_list[from_vertex]:
            self._edge_count += 1
            self._invalidate_cache()
        
        self._adjacency_list[from_vertex][to_vertex] = weight
        
        if not self._directed:
            if from_vertex not in self._adjacency_list[to_vertex]:
                self._adjacency_list[to_vertex][from_vertex] = weight
    
    @track_performance
    def shortest_path_cached(self, start_vertex: str, end_vertex: str) -> Optional[List[str]]:
        """Cached shortest path computation using Dijkstra's algorithm."""
        cache_key = (start_vertex, end_vertex)
        
        # Check cache first
        if cache_key in self._shortest_path_cache:
            return self._shortest_path_cache[cache_key]
        
        # Compute path
        path = self.shortest_path(start_vertex, end_vertex)
        
        # Cache result
        self._shortest_path_cache[cache_key] = path
        
        return path
    
    @track_performance
    def get_centrality_scores(self) -> Dict[str, float]:
        """Get cached centrality scores for all vertices."""
        if self._cache_valid and self._centrality_cache:
            return self._centrality_cache.copy()
        
        # Compute centrality for all vertices
        total_vertices = len(self._vertices)
        if total_vertices <= 1:
            return {}
        
        for vertex in self._vertices:
            in_degree = len(list(self.get_incoming_edges(vertex)))
            out_degree = len(list(self.get_neighbors(vertex)))
            
            # Normalized degree centrality
            centrality = (in_degree + out_degree) / (2 * (total_vertices - 1))
            self._centrality_cache[vertex] = centrality
        
        self._cache_valid = True
        return self._centrality_cache.copy()
    
    def get_top_central_vertices(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top k most central vertices."""
        centrality_scores = self.get_centrality_scores()
        
        # Sort by centrality score
        sorted_vertices = sorted(
            centrality_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_vertices[:k]
    
    @track_performance
    def analyze_network_properties(self) -> Dict[str, Any]:
        """Comprehensive network analysis."""
        vertex_count = len(self._vertices)
        edge_count = self._edge_count
        
        if vertex_count == 0:
            return {'vertices': 0, 'edges': 0, 'density': 0, 'components': 0}
        
        # Network density
        max_edges = vertex_count * (vertex_count - 1)
        if not self._directed:
            max_edges //= 2
        
        density = edge_count / max_edges if max_edges > 0 else 0
        
        # Connected components
        components = self.find_connected_components()
        
        # Degree statistics
        degrees = []
        for vertex in self._vertices:
            degree = len(list(self.get_neighbors(vertex)))
            if not self._directed:
                degree += len(list(self.get_incoming_edges(vertex)))
            degrees.append(degree)
        
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        
        return {
            'vertices': vertex_count,
            'edges': edge_count,
            'density': density,
            'components': len(components),
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'largest_component_size': max(len(comp) for comp in components) if components else 0
        }


class AdvancedPriorityQueue(PriorityQueue[Any]):
    """
    Phase 3: Enhanced priority queue with decrease-key operations.
    """
    
    def __init__(self, max_heap: bool = False):
        self._heap = []
        self._max_heap = max_heap
        self._item_map = {}  # Maps item ID to heap index
        self._counter = 0
        
        # Performance tracking
        self._operation_count = 0
        self._update_count = 0
    
    def _get_item_id(self, item: Any) -> str:
        """Get a hashable identifier for the item."""
        if hasattr(item, 'user_id'):
            return item.user_id
        elif hasattr(item, 'id'):
            return str(item.id)
        else:
            return str(item)
    
    @track_performance
    def enqueue(self, item: Any, priority: float) -> int:
        """Enqueue with tracking."""
        self._operation_count += 1
        
        # Adjust priority for max heap
        adjusted_priority = -priority if self._max_heap else priority
        
        heap_item = [adjusted_priority, self._counter, item]
        item_id = self._get_item_id(item)
        self._item_map[item_id] = len(self._heap)
        
        heapq.heappush(self._heap, heap_item)
        self._counter += 1
        
        # Update item map indices after heap operations
        self._rebuild_item_map()
        
        return self._counter - 1
    
    @track_performance
    def dequeue(self) -> Any:
        """Dequeue with tracking."""
        self._operation_count += 1
        
        if not self._heap:
            raise IndexError("Priority queue is empty")
        
        while self._heap:
            priority, counter, item = heapq.heappop(self._heap)
            
            # Skip lazy-deleted items (None)
            if item is not None:
                # Remove from item map
                item_id = self._get_item_id(item)
                if item_id in self._item_map:
                    del self._item_map[item_id]
                
                # Update item map indices
                self._rebuild_item_map()
                
                return item
        
        raise IndexError("Priority queue is empty")
    
    @track_performance
    def update_priority(self, item: Any, new_priority: float) -> bool:
        """Update priority with decrease/increase key operation."""
        self._update_count += 1
        
        item_id = self._get_item_id(item)
        if item_id not in self._item_map:
            return False
        
        # Remove old entry
        index = self._item_map[item_id]
        if index < len(self._heap):
            # Mark as removed (lazy deletion)
            self._heap[index][2] = None
        
        # Add new entry
        self.enqueue(item, new_priority)
        
        return True
    
    def _rebuild_item_map(self):
        """Rebuild item map after heap operations."""
        self._item_map.clear()
        for i, (priority, counter, item) in enumerate(self._heap):
            if item is not None:  # Skip lazy-deleted items
                item_id = self._get_item_id(item)
                self._item_map[item_id] = i
    
    def get_statistics(self) -> Dict[str, int]:
        """Get performance statistics."""
        return {
            'operation_count': self._operation_count,
            'update_count': self._update_count,
            'current_size': len(self._heap)
        }
    
    def size(self) -> int:
        """Get current queue size."""
        # Count non-deleted items
        return sum(1 for _, _, item in self._heap if item is not None)
    
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def peek(self) -> Any:
        """Peek at the top item without removing it."""
        if not self._heap:
            raise IndexError("Priority queue is empty")
        
        # Clean up lazy-deleted items from top and find first valid item
        while self._heap and self._heap[0][2] is None:
            heapq.heappop(self._heap)
        
        if not self._heap:
            raise IndexError("Priority queue is empty")
        
        return self._heap[0][2]


class PerformanceProfiler:
    """
    Phase 3: Advanced performance profiling and analysis tools.
    """
    
    def __init__(self):
        self.measurements = {}
        self.start_times = {}
    
    def start_measurement(self, operation_name: str):
        """Start timing an operation."""
        self.start_times[operation_name] = time.perf_counter()
    
    def end_measurement(self, operation_name: str, data_size: int = 0):
        """End timing and record measurement."""
        if operation_name not in self.start_times:
            return
        
        duration = time.perf_counter() - self.start_times[operation_name]
        
        if operation_name not in self.measurements:
            self.measurements[operation_name] = []
        
        self.measurements[operation_name].append({
            'duration': duration,
            'data_size': data_size,
            'timestamp': time.time()
        })
        
        del self.start_times[operation_name]
    
    def analyze_scaling(self, operation_name: str) -> Dict[str, Any]:
        """Analyze how operation scales with data size."""
        if operation_name not in self.measurements:
            return {}
        
        measurements = self.measurements[operation_name]
        
        # Group by data size
        size_groups = defaultdict(list)
        for measurement in measurements:
            size_groups[measurement['data_size']].append(measurement['duration'])
        
        # Calculate statistics for each size
        scaling_data = []
        for size, durations in size_groups.items():
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
            
            scaling_data.append({
                'data_size': size,
                'avg_duration': avg_duration,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'sample_count': len(durations)
            })
        
        # Sort by data size
        scaling_data.sort(key=lambda x: x['data_size'])
        
        return {
            'operation': operation_name,
            'scaling_data': scaling_data,
            'total_measurements': len(measurements)
        }
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report = ["PERFORMANCE ANALYSIS REPORT", "=" * 50, ""]
        
        for operation in self.measurements:
            analysis = self.analyze_scaling(operation)
            if not analysis:
                continue
            
            report.append(f"Operation: {operation}")
            report.append("-" * 30)
            
            for data in analysis['scaling_data']:
                size = data['data_size']
                avg = data['avg_duration']
                samples = data['sample_count']
                
                report.append(f"  Size {size:6d}: {avg:.6f}s avg ({samples} samples)")
            
            report.append("")
        
        return "\n".join(report) 