"""
Unit tests for the hash table implementation.

This module provides comprehensive test cases for the HashTable class,
including edge cases and performance considerations.
"""

import pytest
from src.hash_table import HashTable


class TestHashTable:
    """Test cases for HashTable implementation."""
    
    def test_empty_initialization(self):
        """Test that a new hash table is empty."""
        ht = HashTable()
        assert ht.is_empty()
        assert ht.size() == 0
        assert ht.get("non_existent") is None
    
    def test_single_insertion(self):
        """Test inserting a single key-value pair."""
        ht = HashTable()
        ht.put("key1", "value1")
        
        assert not ht.is_empty()
        assert ht.size() == 1
        assert ht.get("key1") == "value1"
    
    def test_multiple_insertions(self):
        """Test inserting multiple key-value pairs."""
        ht = HashTable()
        test_data = {
            "name": "John Doe",
            "age": 30,
            "city": "New York",
            "email": "john@example.com"
        }
        
        for key, value in test_data.items():
            ht.put(key, value)
        
        assert ht.size() == len(test_data)
        for key, expected_value in test_data.items():
            assert ht.get(key) == expected_value
    
    def test_update_existing_key(self):
        """Test updating an existing key's value."""
        ht = HashTable()
        ht.put("key1", "original_value")
        ht.put("key1", "updated_value")
        
        assert ht.size() == 1
        assert ht.get("key1") == "updated_value"
    
    def test_deletion(self):
        """Test deleting key-value pairs."""
        ht = HashTable()
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        
        assert ht.delete("key1") == True
        assert ht.get("key1") is None
        assert ht.get("key2") == "value2"
        assert ht.size() == 1
    
    def test_delete_non_existent(self):
        """Test deleting a non-existent key."""
        ht = HashTable()
        ht.put("key1", "value1")
        
        assert ht.delete("non_existent") == False
        assert ht.size() == 1
    
    def test_clear(self):
        """Test clearing all elements."""
        ht = HashTable()
        for i in range(10):
            ht.put(f"key{i}", f"value{i}")
        
        ht.clear()
        assert ht.is_empty()
        assert ht.size() == 0
    
    def test_keys_iteration(self):
        """Test iterating over keys."""
        ht = HashTable()
        test_keys = ["key1", "key2", "key3"]
        
        for key in test_keys:
            ht.put(key, f"value_{key}")
        
        retrieved_keys = set(ht.keys())
        assert retrieved_keys == set(test_keys)
    
    def test_values_iteration(self):
        """Test iterating over values."""
        ht = HashTable()
        test_data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        
        for key, value in test_data.items():
            ht.put(key, value)
        
        retrieved_values = set(ht.values())
        assert retrieved_values == set(test_data.values())
    
    def test_items_iteration(self):
        """Test iterating over key-value pairs."""
        ht = HashTable()
        test_data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        
        for key, value in test_data.items():
            ht.put(key, value)
        
        retrieved_items = set(ht.items())
        assert retrieved_items == set(test_data.items())
    
    def test_load_factor_and_resize(self):
        """Test that the hash table resizes when load factor exceeds threshold."""
        ht = HashTable(initial_capacity=4)  # Small capacity to trigger resize
        
        # Fill beyond load factor threshold
        for i in range(10):
            ht.put(f"key{i}", f"value{i}")
        
        # Check that all items are still accessible after resize
        for i in range(10):
            assert ht.get(f"key{i}") == f"value{i}"
        
        # Load factor should be reasonable after resize
        assert ht.load_factor() < 0.75
    
    def test_collision_handling(self):
        """Test handling of hash collisions."""
        ht = HashTable(initial_capacity=1)  # Force all items into same bucket
        
        test_data = {"key1": "value1", "key2": "value2", "key3": "value3"}
        for key, value in test_data.items():
            ht.put(key, value)
        
        # All items should be accessible despite collisions
        for key, expected_value in test_data.items():
            assert ht.get(key) == expected_value
    
    def test_different_value_types(self):
        """Test storing different types of values."""
        ht = HashTable()
        
        test_data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "boolean": True,
            "none": None
        }
        
        for key, value in test_data.items():
            ht.put(key, value)
        
        for key, expected_value in test_data.items():
            assert ht.get(key) == expected_value
    
    def test_large_dataset(self):
        """Test with a larger dataset to check performance."""
        ht = HashTable()
        num_items = 1000
        
        # Insert many items
        for i in range(num_items):
            ht.put(f"key_{i:04d}", f"value_{i}")
        
        assert ht.size() == num_items
        
        # Verify all items are accessible
        for i in range(num_items):
            assert ht.get(f"key_{i:04d}") == f"value_{i}"
    
    def test_string_representation(self):
        """Test string representation methods."""
        ht = HashTable()
        ht.put("key1", "value1")
        
        str_repr = str(ht)
        assert "key1" in str_repr
        assert "value1" in str_repr
        
        repr_str = repr(ht)
        assert "HashTable" in repr_str
        assert "size=1" in repr_str
    
    def test_bucket_distribution(self):
        """Test bucket distribution analysis."""
        ht = HashTable(initial_capacity=8)
        
        # Add some items (fewer to avoid triggering resize)
        for i in range(5):
            ht.put(f"key{i}", f"value{i}")
        
        distribution = ht.bucket_distribution()
        assert len(distribution) == ht._capacity  # Should match capacity
        assert sum(distribution) == 5  # Should equal total items


if __name__ == "__main__":
    pytest.main([__file__])
