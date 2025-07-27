"""
Sample implementation of a hash table for demonstration purposes.

This module provides a basic hash table implementation that can be
used as a starting point for your chosen application.
"""

from typing import Optional, Iterator, Tuple, Any
from src.base import KeyValueStore, track_performance


class HashTable(KeyValueStore[str, Any]):
    """
    A basic hash table implementation using separate chaining.
    
    This implementation uses Python lists for collision resolution
    and demonstrates key concepts for hash-based data structures.
    """
    
    def __init__(self, initial_capacity: int = 16):
        """
        Initialize the hash table with the given capacity.
        
        Args:
            initial_capacity: Initial number of buckets
        """
        self._capacity = initial_capacity
        self._size = 0
        self._buckets = [[] for _ in range(self._capacity)]
        self._load_factor_threshold = 0.75
    
    def _hash(self, key: str) -> int:
        """
        Simple hash function using Python's built-in hash.
        
        Args:
            key: The key to hash
            
        Returns:
            Hash value modulo capacity
        """
        return hash(key) % self._capacity
    
    def _resize(self) -> None:
        """Resize the hash table when load factor exceeds threshold."""
        old_buckets = self._buckets
        self._capacity *= 2
        self._size = 0
        self._buckets = [[] for _ in range(self._capacity)]
        
        # Rehash all existing items
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)
    
    @track_performance
    def put(self, key: str, value: Any) -> None:
        """
        Insert or update a key-value pair.
        
        Args:
            key: The key to store
            value: The value to associate with the key
        """
        # Check if resize is needed
        if self._size >= self._capacity * self._load_factor_threshold:
            self._resize()
        
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        # Update existing key
        for i, (existing_key, existing_value) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)
                return
        
        # Add new key-value pair
        bucket.append((key, value))
        self._size += 1
    
    @track_performance
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve the value associated with a key.
        
        Args:
            key: The key to look up
            
        Returns:
            The value if found, None otherwise
        """
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        for existing_key, value in bucket:
            if existing_key == key:
                return value
        
        return None
    
    @track_performance
    def delete(self, key: str) -> bool:
        """
        Remove a key-value pair from the hash table.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key was found and removed, False otherwise
        """
        bucket_index = self._hash(key)
        bucket = self._buckets[bucket_index]
        
        for i, (existing_key, value) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self._size -= 1
                return True
        
        return False
    
    def keys(self) -> Iterator[str]:
        """Return an iterator over all keys."""
        for bucket in self._buckets:
            for key, _ in bucket:
                yield key
    
    def values(self) -> Iterator[Any]:
        """Return an iterator over all values."""
        for bucket in self._buckets:
            for _, value in bucket:
                yield value
    
    def items(self) -> Iterator[Tuple[str, Any]]:
        """Return an iterator over all key-value pairs."""
        for bucket in self._buckets:
            for key, value in bucket:
                yield key, value
    
    def size(self) -> int:
        """Return the number of key-value pairs."""
        return self._size
    
    def is_empty(self) -> bool:
        """Check if the hash table is empty."""
        return self._size == 0
    
    def clear(self) -> None:
        """Remove all key-value pairs."""
        self._buckets = [[] for _ in range(self._capacity)]
        self._size = 0
    
    def load_factor(self) -> float:
        """Return the current load factor."""
        return self._size / self._capacity
    
    def bucket_distribution(self) -> list[int]:
        """Return the distribution of items across buckets (for analysis)."""
        return [len(bucket) for bucket in self._buckets]
    
    def __str__(self) -> str:
        """String representation of the hash table."""
        items = list(self.items())
        return f"HashTable({dict(items)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"HashTable(size={self._size}, capacity={self._capacity}, load_factor={self.load_factor():.2f})"
