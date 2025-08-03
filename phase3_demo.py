#!/usr/bin/env python3
"""
Phase 3: Optimization, Scaling, and Final Evaluation Demonstration

This demonstration showcases optimized data structures and performance analysis:
- AVL Tree: Self-balancing BST solving recursion issues
- Optimized Hash Table: Enhanced collision handling
- Optimized Graph: Cached algorithms for large networks  
- Advanced Priority Queue: Decrease-key operations
- Performance Profiling: Comprehensive scaling analysis
"""

import time
import random
import tracemalloc
from typing import List, Dict

from src.optimized_structures import (
    OptimizedAVLTree, OptimizedHashTable, OptimizedSocialGraph,
    AdvancedPriorityQueue, PerformanceProfiler
)
from src.social_network import SocialUser, SocialNetworkAnalyzer


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_subheader(title: str) -> None:
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


def demonstrate_avl_tree_optimization():
    """Demonstrate AVL tree solving BST recursion issues."""
    print_header("AVL TREE - OPTIMIZED IMPLEMENTATION")
    print("Self-balancing binary search tree eliminating recursion limit issues")
    
    print_subheader("AVL Tree vs Standard BST Comparison")
    
    # Test with data that would cause recursion issues in unbalanced BST
    test_sizes = [100, 500, 1000, 2000, 5000]
    
    for size in test_sizes:
        # Create sequential data (worst case for unbalanced BST)
        sequential_data = list(range(size))
        random.shuffle(sequential_data)  # Randomize to show AVL handles any order
        
        # Test AVL Tree performance
        avl_tree = OptimizedAVLTree()
        
        # Measure insertion time
        start_time = time.perf_counter()
        for value in sequential_data:
            avl_tree.insert(value)
        insert_time = time.perf_counter() - start_time
        
        # Measure search time
        search_keys = random.sample(sequential_data, min(100, size))
        start_time = time.perf_counter()
        for key in search_keys:
            avl_tree.search(key)
        search_time = time.perf_counter() - start_time
        
        # Check balance
        is_balanced = avl_tree.is_balanced()
        height = avl_tree.height()
        expected_height = int(1.44 * (size.bit_length() - 1))  # Theoretical AVL height
        
        print(f"SIZE {size:5d}: Insert {insert_time:.4f}s, Search {search_time:.4f}s, "
              f"Height {height:2d} (expected ~{expected_height}), Balanced: {is_balanced}")
    
    print_subheader("AVL Tree Features Demonstration")
    
    # Demonstrate range search and balancing
    avl_tree = OptimizedAVLTree()
    test_values = [50, 30, 70, 20, 40, 60, 80, 10, 25, 35, 45]
    
    for value in test_values:
        avl_tree.insert(value)
    
    print(f"INSERTED: {test_values}")
    print(f"TREE HEIGHT: {avl_tree.height()} (balanced)")
    print(f"TREE SIZE: {avl_tree.size()}")
    print(f"MIN VALUE: {avl_tree.min_value()}")
    print(f"MAX VALUE: {avl_tree.max_value()}")
    
    # Range search
    range_result = avl_tree.range_search(25, 65)
    print(f"RANGE SEARCH [25-65]: {sorted(range_result)}")
    
    print(f"BALANCED CHECK: {avl_tree.is_balanced()}")


def demonstrate_optimized_hash_table():
    """Demonstrate enhanced hash table with collision optimization."""
    print_header("ENHANCED HASH TABLE")
    print("Advanced collision handling with adaptive hash function selection")
    
    print_subheader("Hash Function Optimization")
    
    # Create hash table with collision-prone data
    hash_table = OptimizedHashTable(initial_capacity=16)
    
    # Insert data that might cause collisions
    test_data = []
    for i in range(1000):
        key = f"user_{i:04d}"
        value = {"id": i, "name": f"User {i}", "score": random.randint(1, 100)}
        test_data.append((key, value))
        hash_table.put(key, value)
    
    # Get statistics
    stats = hash_table.get_statistics()
    
    print(f"FINAL STATS:")
    print(f"  Size: {stats['size']}")
    print(f"  Capacity: {stats['capacity']}")
    print(f"  Load Factor: {stats['load_factor']:.3f}")
    print(f"  Collision Rate: {stats['collision_rate']:.3f}")
    print(f"  Total Accesses: {stats['access_count']}")
    print(f"  Hash Function Used: {stats['current_hash_function']}")
    
    print_subheader("Performance Comparison with Standard Hash Table")
    
    from src.hash_table import HashTable
    
    # Compare performance
    sizes = [100, 500, 1000, 2500, 5000]
    
    for size in sizes:
        # Generate test data
        keys = [f"key_{i:06d}" for i in range(size)]
        values = [f"value_{i}" for i in range(size)]
        
        # Test standard hash table
        standard_ht = HashTable()
        start_time = time.perf_counter()
        for key, value in zip(keys, values):
            standard_ht.put(key, value)
        standard_time = time.perf_counter() - start_time
        
        # Test optimized hash table
        optimized_ht = OptimizedHashTable()
        start_time = time.perf_counter()
        for key, value in zip(keys, values):
            optimized_ht.put(key, value)
        optimized_time = time.perf_counter() - start_time
        
        # Get collision rates
        standard_stats = {
            'load_factor': standard_ht.load_factor(),
            'collision_rate': 'N/A'  # Standard doesn't track this
        }
        optimized_stats = optimized_ht.get_statistics()
        
        print(f"SIZE {size:5d}: Standard {standard_time:.4f}s (LF: {standard_stats['load_factor']:.3f}), "
              f"Optimized {optimized_time:.4f}s (LF: {optimized_stats['load_factor']:.3f}, "
              f"CR: {optimized_stats['collision_rate']:.3f})")


def demonstrate_optimized_graph():
    """Demonstrate cached graph algorithms for large networks."""
    print_header("OPTIMIZED GRAPH ALGORITHMS")
    print("Cached computations providing significant performance improvements")
    
    print_subheader("Large Network Analysis")
    
    # Create large social network
    optimized_graph = OptimizedSocialGraph(directed=True)
    
    # Generate realistic social network
    num_users = 1000
    users = [f"user_{i:04d}" for i in range(num_users)]
    
    print(f"GENERATING: Network with {num_users} users...")
    
    # Add users
    for user in users:
        optimized_graph.add_vertex(user)
    
    # Add realistic connections (each user follows 5-50 others)
    connection_count = 0
    for user in users:
        num_follows = random.randint(5, 50)
        targets = random.sample(users, num_follows)
        for target in targets:
            if target != user:
                optimized_graph.add_edge(user, target)
                connection_count += 1
    
    print(f"CREATED: {connection_count} follow relationships")
    
    print_subheader("Network Analysis Performance")
    
    # Analyze network properties
    start_time = time.perf_counter()
    properties = optimized_graph.analyze_network_properties()
    analysis_time = time.perf_counter() - start_time
    
    print(f"NETWORK ANALYSIS ({analysis_time:.4f}s):")
    print(f"  Vertices: {properties['vertices']}")
    print(f"  Edges: {properties['edges']}")
    print(f"  Density: {properties['density']:.6f}")
    print(f"  Components: {properties['components']}")
    print(f"  Average Degree: {properties['avg_degree']:.2f}")
    print(f"  Max Degree: {properties['max_degree']}")
    print(f"  Largest Component: {properties['largest_component_size']}")
    
    print_subheader("Cached Centrality Computation")
    
    # Test centrality caching
    start_time = time.perf_counter()
    centrality_scores = optimized_graph.get_centrality_scores()
    first_time = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    centrality_scores_cached = optimized_graph.get_centrality_scores()
    cached_time = time.perf_counter() - start_time
    
    print(f"CENTRALITY COMPUTATION:")
    print(f"  First computation: {first_time:.4f}s")
    print(f"  Cached access: {cached_time:.6f}s")
    print(f"  Speedup: {first_time/cached_time:.1f}x")
    
    # Get top central users
    top_central = optimized_graph.get_top_central_vertices(5)
    print(f"TOP 5 CENTRAL USERS:")
    for i, (user, centrality) in enumerate(top_central, 1):
        print(f"  {i}. {user}: {centrality:.4f}")


def demonstrate_advanced_priority_queue():
    """Demonstrate priority queue with decrease-key operations."""
    print_header("ADVANCED PRIORITY QUEUE")
    print("Enhanced with decrease-key operations for dynamic priority updates")
    
    print_subheader("Decrease-Key Operations")
    
    # Create priority queue with social network influence updating
    pq = AdvancedPriorityQueue(max_heap=True)  # Max heap for influence
    
    # Create users with initial influence
    users = []
    for i in range(10):
        user = SocialUser(
            user_id=f"user_{i:02d}",
            username=f"user{i}",
            display_name=f"User {i}",
            follower_count=random.randint(100, 1000),
            following_count=random.randint(50, 500),
            post_count=random.randint(10, 200)
        )
        users.append(user)
        pq.enqueue(user, user.calculate_influence_score())
    
    print("INITIAL QUEUE STATE:")
    initial_stats = pq.get_statistics()
    print(f"  Size: {pq.size()}")
    print(f"  Operations: {initial_stats['operation_count']}")
    
    # Get top 3 users
    print("\nINITIAL TOP 3 USERS:")
    temp_users = []
    for i in range(3):
        if not pq.is_empty():
            user = pq.dequeue()
            temp_users.append(user)
            print(f"  {i+1}. {user.display_name}: {user.influence_score:.1f}")
    
    # Put them back
    for user in temp_users:
        pq.enqueue(user, user.influence_score)
    
    print_subheader("Dynamic Priority Updates")
    
    # Simulate influence changes (user gains followers)
    update_user = users[5]  # Pick a middle user
    old_score = update_user.influence_score
    
    # User gains many followers (becoming an influencer)
    update_user.update_metrics(
        followers=update_user.follower_count + 5000,
        following=update_user.following_count,
        posts=update_user.post_count + 100
    )
    new_score = update_user.influence_score
    
    print(f"UPDATING USER: {update_user.display_name}")
    print(f"  Old score: {old_score:.1f}")
    print(f"  New score: {new_score:.1f}")
    print(f"  Improvement: {new_score - old_score:.1f}")
    
    # Update priority in queue
    success = pq.update_priority(update_user, new_score)
    print(f"  Priority update success: {success}")
    
    # Show new top 3
    print("\nUPDATED TOP 3 USERS:")
    temp_users = []
    for i in range(3):
        if not pq.is_empty():
            user = pq.dequeue()
            temp_users.append(user)
            print(f"  {i+1}. {user.display_name}: {user.influence_score:.1f}")
    
    # Get final statistics
    final_stats = pq.get_statistics()
    print(f"\nFINAL QUEUE STATISTICS:")
    print(f"  Total operations: {final_stats['operation_count']}")
    print(f"  Priority updates: {final_stats['update_count']}")
    print(f"  Current size: {final_stats['current_size']}")


def demonstrate_performance_profiling():
    """Demonstrate comprehensive performance profiling."""
    print_header("PERFORMANCE ANALYSIS")
    print("Comprehensive evaluation of data structure scalability")
    
    profiler = PerformanceProfiler()
    
    print_subheader("Multi-Structure Scaling Analysis")
    
    # Test different data sizes
    test_sizes = [100, 250, 500, 1000, 2000]
    
    for size in test_sizes:
        print(f"\nTesting with {size} items...")
        
        # AVL Tree scaling
        profiler.start_measurement("avl_tree_insert")
        avl_tree = OptimizedAVLTree()
        for i in range(size):
            avl_tree.insert(i)
        profiler.end_measurement("avl_tree_insert", size)
        
        # Hash Table scaling
        profiler.start_measurement("hash_table_insert")
        hash_table = OptimizedHashTable()
        for i in range(size):
            hash_table.put(f"key_{i}", f"value_{i}")
        profiler.end_measurement("hash_table_insert", size)
        
        # Priority Queue scaling
        profiler.start_measurement("priority_queue_insert")
        pq = AdvancedPriorityQueue()
        for i in range(size):
            pq.enqueue(f"item_{i}", random.random())
        profiler.end_measurement("priority_queue_insert", size)
        
        # Graph scaling
        profiler.start_measurement("graph_construction")
        graph = OptimizedSocialGraph()
        vertices = [f"v_{i}" for i in range(min(size, 200))]  # Limit for graph
        for vertex in vertices:
            graph.add_vertex(vertex)
        
        # Add edges
        for i in range(len(vertices)):
            for j in range(min(5, len(vertices))):  # Each vertex connects to 5 others
                if i != j:
                    graph.add_edge(vertices[i], vertices[j])
        profiler.end_measurement("graph_construction", len(vertices))
    
    print_subheader("Performance Analysis Results")
    
    # Generate and display analysis
    operations = ["avl_tree_insert", "hash_table_insert", "priority_queue_insert", "graph_construction"]
    
    for operation in operations:
        analysis = profiler.analyze_scaling(operation)
        if analysis:
            print(f"\n{operation.upper().replace('_', ' ')}:")
            for data in analysis['scaling_data']:
                size = data['data_size']
                avg_duration = data['avg_duration']
                samples = data['sample_count']
                
                # Calculate operations per second
                ops_per_sec = size / avg_duration if avg_duration > 0 else 0
                
                print(f"  Size {size:4d}: {avg_duration:.6f}s avg, {ops_per_sec:8.0f} ops/sec")


def demonstrate_memory_efficiency():
    """Demonstrate memory usage optimization."""
    print_header("MEMORY EFFICIENCY EVALUATION")
    print("Analysis of memory consumption across data structure implementations")
    
    print_subheader("Memory Usage Comparison")
    
    # Start memory tracking
    tracemalloc.start()
    
    # Test data structure memory efficiency
    sizes = [1000, 2500, 5000]
    
    for size in sizes:
        print(f"\nTesting with {size} items:")
        
        # Baseline memory
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_size = sum(stat.size for stat in baseline_snapshot.statistics('lineno'))
        
        # AVL Tree memory (use key function for dict items)
        avl_tree = OptimizedAVLTree(key_func=lambda x: x["id"])
        for i in range(size):
            avl_tree.insert({"id": i, "data": f"item_{i}"})
        
        avl_snapshot = tracemalloc.take_snapshot()
        avl_size = sum(stat.size for stat in avl_snapshot.statistics('lineno'))
        avl_memory = avl_size - baseline_size
        
        # Hash Table memory
        hash_table = OptimizedHashTable()
        for i in range(size):
            hash_table.put(f"key_{i}", {"id": i, "data": f"item_{i}"})
        
        ht_snapshot = tracemalloc.take_snapshot()
        ht_size = sum(stat.size for stat in ht_snapshot.statistics('lineno'))
        ht_memory = ht_size - avl_size
        
        print(f"  AVL Tree: {avl_memory / 1024:.1f} KB ({avl_memory / size:.1f} bytes/item)")
        print(f"  Hash Table: {ht_memory / 1024:.1f} KB ({ht_memory / size:.1f} bytes/item)")
        
        # Clean up for next iteration
        del avl_tree, hash_table
    
    tracemalloc.stop()


def demonstrate_phase3_integration():
    """Demonstrate all Phase 3 optimizations working together."""
    print_header("INTEGRATED OPTIMIZATION DEMONSTRATION")
    print("Combined enhanced data structures operating on large-scale social networks")
    
    print_subheader("Optimized Social Network System")
    
    # Create enhanced social network analyzer using optimized structures
    print("BUILDING: Enhanced social network with optimized structures...")
    
    # Use optimized hash table for user registry
    user_registry = OptimizedHashTable()
    
    # Use optimized graph for relationships
    social_graph = OptimizedSocialGraph(directed=True)
    
    # Use advanced priority queue for influence ranking
    influence_ranker = AdvancedPriorityQueue(max_heap=True)
    
    # Create larger dataset
    num_users = 1000
    users = []
    
    # Generate users
    for i in range(num_users):
        user = SocialUser(
            user_id=f"user_{i:04d}",
            username=f"user{i}",
            display_name=f"User {i}",
            follower_count=random.randint(10, 5000),
            following_count=random.randint(10, 1000),
            post_count=random.randint(0, 1000),
            verified=random.random() < 0.1  # 10% verified
        )
        users.append(user)
        
        # Store in optimized hash table
        user_registry.put(user.user_id, user)
        
        # Add to graph
        social_graph.add_vertex(user.user_id)
        
        # Add to influence ranking
        influence_ranker.enqueue(user, user.calculate_influence_score())
    
    # Create connections
    connection_count = 0
    for user in users:
        num_follows = random.randint(5, 30)
        targets = random.sample(users, num_follows)
        for target in targets:
            if target.user_id != user.user_id:
                social_graph.add_edge(user.user_id, target.user_id)
                connection_count += 1
    
    print(f"CREATED: {num_users} users with {connection_count} connections")
    
    print_subheader("Performance Analysis")
    
    # Test lookup performance
    start_time = time.perf_counter()
    test_lookups = random.sample(users, 100)
    for user in test_lookups:
        found_user = user_registry.get(user.user_id)
    lookup_time = time.perf_counter() - start_time
    
    print(f"USER LOOKUPS: 100 lookups in {lookup_time:.6f}s ({lookup_time/100*1000:.3f}ms each)")
    
    # Test graph analysis
    start_time = time.perf_counter()
    network_props = social_graph.analyze_network_properties()
    analysis_time = time.perf_counter() - start_time
    
    print(f"NETWORK ANALYSIS: {analysis_time:.4f}s")
    print(f"  Density: {network_props['density']:.6f}")
    print(f"  Avg Degree: {network_props['avg_degree']:.2f}")
    print(f"  Components: {network_props['components']}")
    
    # Test influence ranking
    start_time = time.perf_counter()
    top_influencers = []
    for _ in range(10):
        if not influence_ranker.is_empty():
            top_influencers.append(influence_ranker.dequeue())
    ranking_time = time.perf_counter() - start_time
    
    print(f"INFLUENCE RANKING: Top 10 in {ranking_time:.6f}s")
    
    # Show memory and performance statistics
    ht_stats = user_registry.get_statistics()
    pq_stats = influence_ranker.get_statistics()
    
    print_subheader("System Statistics")
    print(f"HASH TABLE:")
    print(f"  Load Factor: {ht_stats['load_factor']:.3f}")
    print(f"  Collision Rate: {ht_stats['collision_rate']:.3f}")
    print(f"PRIORITY QUEUE:")
    print(f"  Operations: {pq_stats['operation_count']}")
    print(f"  Updates: {pq_stats['update_count']}")
    print(f"GRAPH:")
    print(f"  Vertices: {social_graph.vertex_count()}")
    print(f"  Edges: {social_graph.edge_count()}")


def main():
    """Main Phase 3 demonstration."""
    print("Phase 3: Optimized Data Structure Implementation")
    print("="*52)
    print("Enhanced versions optimized for large-scale datasets")
    print("="*52)
    
    # Run all Phase 3 demonstrations
    demonstrate_avl_tree_optimization()
    demonstrate_optimized_hash_table()
    demonstrate_optimized_graph()
    demonstrate_advanced_priority_queue()
    demonstrate_performance_profiling()
    demonstrate_memory_efficiency()
    demonstrate_phase3_integration()
    
    # Final summary
    print_header("PHASE 3 OPTIMIZATION SUMMARY")
    print("Implemented enhancements:")
    print("  - AVL Tree: Resolved recursion limitations for large datasets")
    print("  - Hash Table: Advanced collision handling with adaptive hash functions")
    print("  - Graph: Implemented caching for significant performance gains")
    print("  - Priority Queue: Added decrease-key operations for dynamic updates")
    print("  - Performance: Comprehensive profiling and scaling analysis")
    print("  - Memory: Detailed efficiency evaluation across implementations")
    print("\nResult: Optimized data structures suitable for large-scale social network analysis")


if __name__ == "__main__":
    main() 