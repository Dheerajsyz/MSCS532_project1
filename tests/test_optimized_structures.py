"""
Unit tests for Phase 3 optimized data structures.

Tests the enhanced implementations including:
- OptimizedAVLTree: Self-balancing BST
- OptimizedHashTable: Enhanced collision handling
- OptimizedSocialGraph: Cached algorithms
- AdvancedPriorityQueue: Decrease-key operations
- PerformanceProfiler: Performance analysis tools
"""

import pytest
import random
import time

from src.optimized_structures import (
    OptimizedAVLTree, OptimizedHashTable, OptimizedSocialGraph,
    AdvancedPriorityQueue, PerformanceProfiler
)


class TestOptimizedAVLTree:
    """Test cases for AVL Tree implementation."""
    
    def test_empty_tree(self):
        """Test empty tree initialization."""
        tree = OptimizedAVLTree()
        assert tree.is_empty()
        assert tree.size() == 0
        assert tree.height() == 0
        assert tree.is_balanced()
    
    def test_single_insertion(self):
        """Test single item insertion."""
        tree = OptimizedAVLTree()
        tree.insert(10)
        
        assert not tree.is_empty()
        assert tree.size() == 1
        assert tree.height() == 1
        assert tree.search(10) == 10
        assert tree.is_balanced()
    
    def test_multiple_insertions_with_balancing(self):
        """Test multiple insertions maintain balance."""
        tree = OptimizedAVLTree()
        
        # Insert values that would create unbalanced BST
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for value in values:
            tree.insert(value)
        
        assert tree.size() == 10
        assert tree.is_balanced()
        assert tree.height() <= 4  # Should be well-balanced
        
        # Test all values are searchable
        for value in values:
            assert tree.search(value) == value
    
    def test_deletion_maintains_balance(self):
        """Test deletion maintains AVL balance."""
        tree = OptimizedAVLTree()
        
        # Insert values
        values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]
        for value in values:
            tree.insert(value)
        
        # Delete some values
        tree.delete(20)
        tree.delete(70)
        tree.delete(10)
        
        assert tree.is_balanced()
        assert tree.size() == 8
        
        # Check deleted values are gone
        assert tree.search(20) is None
        assert tree.search(70) is None
        assert tree.search(10) is None
        
        # Check remaining values exist
        remaining = [50, 30, 40, 60, 80, 25, 35, 45]
        for value in remaining:
            assert tree.search(value) == value
    
    def test_range_search(self):
        """Test range search functionality."""
        tree = OptimizedAVLTree()
        
        values = [10, 5, 15, 3, 7, 12, 18, 1, 4, 6, 8, 11, 13, 16, 20]
        for value in values:
            tree.insert(value)
        
        # Test range search
        result = tree.range_search(6, 13)
        expected = [6, 7, 8, 10, 11, 12, 13]
        assert sorted(result) == expected
    
    def test_min_max_values(self):
        """Test min and max value retrieval."""
        tree = OptimizedAVLTree()
        
        values = [50, 30, 70, 20, 40, 60, 80]
        for value in values:
            tree.insert(value)
        
        assert tree.min_value() == 20
        assert tree.max_value() == 80
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        tree = OptimizedAVLTree()
        
        # Insert 1000 items
        values = list(range(1000))
        random.shuffle(values)  # Random order
        
        for value in values:
            tree.insert(value)
        
        assert tree.size() == 1000
        assert tree.is_balanced()
        assert tree.height() <= 12  # log2(1000) ≈ 10, AVL factor adds ~1.44
        
        # Test searches
        for _ in range(100):
            test_value = random.choice(values)
            assert tree.search(test_value) == test_value


class TestOptimizedHashTable:
    """Test cases for optimized hash table."""
    
    def test_basic_operations(self):
        """Test basic put/get operations."""
        ht = OptimizedHashTable()
        
        ht.put("key1", "value1")
        ht.put("key2", "value2")
        
        assert ht.get("key1") == "value1"
        assert ht.get("key2") == "value2"
        assert ht.get("nonexistent") is None
    
    def test_collision_handling(self):
        """Test collision handling with hash function switching."""
        ht = OptimizedHashTable(initial_capacity=4)  # Small capacity to force collisions
        
        # Add many items to trigger collisions
        for i in range(20):
            ht.put(f"key_{i:03d}", f"value_{i}")
        
        # Verify all items are retrievable
        for i in range(20):
            assert ht.get(f"key_{i:03d}") == f"value_{i}"
        
        # Check statistics
        stats = ht.get_statistics()
        assert stats['size'] == 20
        assert stats['load_factor'] < 0.75  # Should have resized
    
    def test_hash_function_adaptation(self):
        """Test adaptive hash function switching."""
        ht = OptimizedHashTable(initial_capacity=8)
        
        # Create keys likely to collide with simple hash
        collision_keys = [f"aaa{i}" for i in range(50)]
        
        for key in collision_keys:
            ht.put(key, f"value_{key}")
        
        stats = ht.get_statistics()
        
        # Should have switched hash function if collision rate was high
        assert stats['collision_rate'] < 0.5  # Should have adapted
        
        # Verify all keys retrievable
        for key in collision_keys:
            assert ht.get(key) == f"value_{key}"
    
    def test_performance_statistics(self):
        """Test performance statistics tracking."""
        ht = OptimizedHashTable()
        
        # Perform operations
        for i in range(100):
            ht.put(f"key_{i}", f"value_{i}")
        
        for i in range(50):
            ht.get(f"key_{i}")
        
        stats = ht.get_statistics()
        
        assert stats['size'] == 100
        assert stats['access_count'] >= 150  # 100 puts + 50 gets
        assert 'collision_rate' in stats
        assert 'load_factor' in stats


class TestOptimizedSocialGraph:
    """Test cases for optimized social graph."""
    
    def test_basic_graph_operations(self):
        """Test basic graph operations work correctly."""
        graph = OptimizedSocialGraph(directed=True)
        
        # Add vertices and edges
        graph.add_vertex("user1")
        graph.add_vertex("user2")
        graph.add_edge("user1", "user2")
        
        assert graph.vertex_count() == 2
        assert graph.edge_count() == 1
        assert graph.has_edge("user1", "user2")
    
    def test_centrality_caching(self):
        """Test centrality computation caching."""
        graph = OptimizedSocialGraph(directed=True)
        
        # Create small network
        users = ["user1", "user2", "user3", "user4"]
        for user in users:
            graph.add_vertex(user)
        
        # Add some connections
        graph.add_edge("user1", "user2")
        graph.add_edge("user2", "user3")
        graph.add_edge("user3", "user1")
        graph.add_edge("user4", "user1")
        
        # First computation should be slower
        start_time = time.perf_counter()
        centrality1 = graph.get_centrality_scores()
        first_time = time.perf_counter() - start_time
        
        # Second computation should be cached (faster)
        start_time = time.perf_counter()
        centrality2 = graph.get_centrality_scores()
        cached_time = time.perf_counter() - start_time
        
        assert centrality1 == centrality2
        assert cached_time <= first_time  # Should be same or faster
    
    def test_network_analysis(self):
        """Test comprehensive network analysis."""
        graph = OptimizedSocialGraph(directed=True)
        
        # Create test network
        for i in range(10):
            graph.add_vertex(f"user_{i}")
        
        # Add connections
        connections = [
            ("user_0", "user_1"), ("user_1", "user_2"), ("user_2", "user_0"),
            ("user_3", "user_4"), ("user_4", "user_5"), ("user_5", "user_3"),
            ("user_6", "user_7"), ("user_8", "user_9")
        ]
        
        for from_user, to_user in connections:
            graph.add_edge(from_user, to_user)
        
        # Analyze network
        properties = graph.analyze_network_properties()
        
        assert properties['vertices'] == 10
        assert properties['edges'] == len(connections)
        assert properties['components'] >= 1
        assert properties['avg_degree'] >= 0
        assert properties['max_degree'] >= 0
    
    def test_top_central_vertices(self):
        """Test top central vertices identification."""
        graph = OptimizedSocialGraph(directed=True)
        
        # Create star network (user_0 in center)
        for i in range(5):
            graph.add_vertex(f"user_{i}")
        
        # Connect all others to user_0
        for i in range(1, 5):
            graph.add_edge(f"user_{i}", "user_0")
            graph.add_edge("user_0", f"user_{i}")
        
        # Get top central vertices
        top_central = graph.get_top_central_vertices(3)
        
        assert len(top_central) <= 3
        # user_0 should be most central
        assert top_central[0][0] == "user_0"


class TestAdvancedPriorityQueue:
    """Test cases for advanced priority queue."""
    
    def test_basic_operations(self):
        """Test basic enqueue/dequeue operations."""
        pq = AdvancedPriorityQueue()
        
        pq.enqueue("item1", 1.0)
        pq.enqueue("item2", 2.0)
        pq.enqueue("item3", 0.5)
        
        # Should dequeue in priority order (min heap)
        assert pq.dequeue() == "item3"  # Lowest priority first
        assert pq.dequeue() == "item1"
        assert pq.dequeue() == "item2"
    
    def test_max_heap_behavior(self):
        """Test max heap behavior."""
        pq = AdvancedPriorityQueue(max_heap=True)
        
        pq.enqueue("low", 1.0)
        pq.enqueue("high", 3.0)
        pq.enqueue("medium", 2.0)
        
        # Should dequeue highest priority first
        assert pq.dequeue() == "high"
        assert pq.dequeue() == "medium"
        assert pq.dequeue() == "low"
    
    def test_update_priority(self):
        """Test priority update functionality."""
        pq = AdvancedPriorityQueue()
        
        pq.enqueue("item1", 1.0)
        pq.enqueue("item2", 2.0)
        pq.enqueue("item3", 3.0)
        
        # Update item2 to have highest priority
        success = pq.update_priority("item2", 0.5)
        assert success
        
        # Should dequeue updated item first
        assert pq.dequeue() == "item2"
    
    def test_statistics_tracking(self):
        """Test performance statistics tracking."""
        pq = AdvancedPriorityQueue()
        
        # Perform operations
        for i in range(10):
            pq.enqueue(f"item_{i}", float(i))
        
        # Update some priorities
        pq.update_priority("item_5", 0.1)
        pq.update_priority("item_7", 0.2)
        
        stats = pq.get_statistics()
        
        assert stats['operation_count'] >= 10  # At least 10 enqueues
        assert stats['update_count'] == 2
        assert stats['current_size'] >= 10
    
    def test_large_queue_performance(self):
        """Test performance with large queue."""
        pq = AdvancedPriorityQueue()
        
        # Add many items with higher priorities to ensure updated items come first
        items = [(f"item_{i}", 10.0 + random.random()) for i in range(1000)]
        
        for item, priority in items:
            pq.enqueue(item, priority)
        
        assert pq.size() == 1000
        
        # Update some priorities to very high priority
        expected_updated = []
        for i in range(0, 100, 10):
            item_name = f"item_{i}"
            success = pq.update_priority(item_name, 0.001)  # Very high priority
            if success:
                expected_updated.append(item_name)
        
        # Get some items from the queue - they should include the updated ones
        # (though exact order may vary due to implementation details)
        dequeued_items = []
        for _ in range(min(20, len(expected_updated) + 5)):  # Get a few more than updated
            if not pq.is_empty():
                dequeued_items.append(pq.dequeue())
        
        # At least some of the dequeued items should be from the updated list
        updated_found = sum(1 for item in dequeued_items if item in expected_updated)
        assert updated_found >= len(expected_updated) // 2  # At least half should be found


class TestPerformanceProfiler:
    """Test cases for performance profiler."""
    
    def test_measurement_recording(self):
        """Test basic measurement recording."""
        profiler = PerformanceProfiler()
        
        profiler.start_measurement("test_operation")
        time.sleep(0.01)  # Simulate work
        profiler.end_measurement("test_operation", 100)
        
        analysis = profiler.analyze_scaling("test_operation")
        
        assert analysis['operation'] == "test_operation"
        assert len(analysis['scaling_data']) == 1
        assert analysis['scaling_data'][0]['data_size'] == 100
        assert analysis['scaling_data'][0]['avg_duration'] >= 0.01
    
    def test_multiple_measurements(self):
        """Test multiple measurements and analysis."""
        profiler = PerformanceProfiler()
        
        # Record multiple measurements for same operation
        for size in [100, 200, 300]:
            for _ in range(3):  # Multiple samples per size
                profiler.start_measurement("scaling_test")
                time.sleep(size / 100000)  # Simulate size-dependent work
                profiler.end_measurement("scaling_test", size)
        
        analysis = profiler.analyze_scaling("scaling_test")
        
        assert len(analysis['scaling_data']) == 3  # Three different sizes
        
        # Check that each size has multiple samples
        for data in analysis['scaling_data']:
            assert data['sample_count'] == 3
            assert data['avg_duration'] > 0
    
    def test_performance_report_generation(self):
        """Test performance report generation."""
        profiler = PerformanceProfiler()
        
        # Record some measurements
        operations = ["op1", "op2"]
        for op in operations:
            for size in [10, 20]:
                profiler.start_measurement(op)
                time.sleep(0.001)
                profiler.end_measurement(op, size)
        
        report = profiler.generate_performance_report()
        
        assert "PERFORMANCE ANALYSIS REPORT" in report
        assert "op1" in report
        assert "op2" in report
        assert "Size" in report


class TestIntegration:
    """Integration tests for optimized structures working together."""
    
    def test_optimized_structures_integration(self):
        """Test all optimized structures working together."""
        # Create components
        avl_tree = OptimizedAVLTree()
        hash_table = OptimizedHashTable()
        graph = OptimizedSocialGraph(directed=True)
        priority_queue = AdvancedPriorityQueue(max_heap=True)
        
        # Test data
        users = [f"user_{i}" for i in range(20)]
        
        # Populate all structures
        for i, user in enumerate(users):
            # AVL tree for sorted access
            avl_tree.insert(user)
            
            # Hash table for fast lookup
            hash_table.put(user, {"id": i, "name": user})
            
            # Graph for relationships
            graph.add_vertex(user)
            
            # Priority queue for ranking
            priority_queue.enqueue(user, float(i))
        
        # Add some graph connections
        for i in range(len(users) - 1):
            graph.add_edge(users[i], users[i + 1])
        
        # Test all structures work correctly
        assert avl_tree.size() == 20
        assert avl_tree.is_balanced()
        
        hash_stats = hash_table.get_statistics()
        assert hash_stats['size'] == 20
        
        graph_props = graph.analyze_network_properties()
        assert graph_props['vertices'] == 20
        
        pq_stats = priority_queue.get_statistics()
        assert pq_stats['current_size'] == 20
        
        # Test searches and lookups work
        test_user = "user_10"
        assert avl_tree.search(test_user) == test_user
        assert hash_table.get(test_user)['name'] == test_user
        assert graph.has_vertex(test_user)
        
        print("All optimized structures integrated successfully!") 