"""
Base classes and interfaces for data structures implementation.

This module provides abstract base classes and common interfaces
for the data structures used in this project.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Optional, Iterator

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class DataStructure(ABC, Generic[T]):
    """Abstract base class for all data structures."""
    
    @abstractmethod
    def size(self) -> int:
        """Return the number of elements in the data structure."""
        pass
    
    @abstractmethod
    def is_empty(self) -> bool:
        """Check if the data structure is empty."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all elements from the data structure."""
        pass


class SearchableDataStructure(DataStructure[T]):
    """Abstract base class for searchable data structures."""
    
    @abstractmethod
    def contains(self, item: T) -> bool:
        """Check if the item exists in the data structure."""
        pass
    
    @abstractmethod
    def search(self, item: T) -> Optional[T]:
        """Search for an item and return it if found."""
        pass


class MutableDataStructure(DataStructure[T]):
    """Abstract base class for mutable data structures."""
    
    @abstractmethod
    def insert(self, item: T) -> bool:
        """Insert an item into the data structure."""
        pass
    
    @abstractmethod
    def remove(self, item: T) -> bool:
        """Remove an item from the data structure."""
        pass


class KeyValueStore(ABC, Generic[K, V]):
    """Abstract base class for key-value data structures."""
    
    @abstractmethod
    def get(self, key: K) -> Optional[V]:
        """Get the value associated with the key."""
        pass
    
    @abstractmethod
    def put(self, key: K, value: V) -> None:
        """Store a key-value pair."""
        pass
    
    @abstractmethod
    def delete(self, key: K) -> bool:
        """Remove a key-value pair."""
        pass
    
    @abstractmethod
    def keys(self) -> Iterator[K]:
        """Return an iterator over all keys."""
        pass
    
    @abstractmethod
    def values(self) -> Iterator[V]:
        """Return an iterator over all values."""
        pass


class Graph(ABC, Generic[T]):
    """Abstract base class for graph data structures."""
    
    @abstractmethod
    def add_vertex(self, vertex: T) -> None:
        """Add a vertex to the graph."""
        pass
    
    @abstractmethod
    def add_edge(self, from_vertex: T, to_vertex: T, weight: float = 1.0) -> None:
        """Add an edge between two vertices."""
        pass
    
    @abstractmethod
    def remove_vertex(self, vertex: T) -> bool:
        """Remove a vertex and all its edges."""
        pass
    
    @abstractmethod
    def remove_edge(self, from_vertex: T, to_vertex: T) -> bool:
        """Remove an edge between two vertices."""
        pass
    
    @abstractmethod
    def get_neighbors(self, vertex: T) -> Iterator[T]:
        """Get all neighbors of a vertex."""
        pass
    
    @abstractmethod
    def has_edge(self, from_vertex: T, to_vertex: T) -> bool:
        """Check if an edge exists between two vertices."""
        pass


class Tree(ABC, Generic[T]):
    """Abstract base class for tree data structures."""
    
    @abstractmethod
    def insert(self, item: T) -> None:
        """Insert an item into the tree."""
        pass
    
    @abstractmethod
    def search(self, item: T) -> Optional[T]:
        """Search for an item in the tree."""
        pass
    
    @abstractmethod
    def delete(self, item: T) -> bool:
        """Delete an item from the tree."""
        pass
    
    @abstractmethod
    def traverse_inorder(self) -> Iterator[T]:
        """Traverse the tree in inorder."""
        pass
    
    @abstractmethod
    def traverse_preorder(self) -> Iterator[T]:
        """Traverse the tree in preorder."""
        pass
    
    @abstractmethod
    def traverse_postorder(self) -> Iterator[T]:
        """Traverse the tree in postorder."""
        pass


class PriorityQueue(ABC, Generic[T]):
    """Abstract base class for priority queue data structures."""
    
    @abstractmethod
    def enqueue(self, item: T, priority: float) -> None:
        """Add an item with a given priority."""
        pass
    
    @abstractmethod
    def dequeue(self) -> Optional[T]:
        """Remove and return the highest priority item."""
        pass
    
    @abstractmethod
    def peek(self) -> Optional[T]:
        """Return the highest priority item without removing it."""
        pass
    
    @abstractmethod
    def update_priority(self, item: T, new_priority: float) -> bool:
        """Update the priority of an existing item."""
        pass


# Performance tracking decorator
def track_performance(func):
    """Decorator to track performance metrics of data structure operations."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Store performance data (can be extended for more detailed tracking)
        execution_time = end_time - start_time
        if hasattr(wrapper, 'performance_data'):
            wrapper.performance_data.append(execution_time)
        else:
            wrapper.performance_data = [execution_time]
            
        return result
    
    return wrapper
