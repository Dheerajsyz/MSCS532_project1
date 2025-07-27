"""
Performance benchmarking utilities for data structures.

This module provides tools for measuring and analyzing the performance
of different data structure implementations.
"""

import time
import tracemalloc
import statistics
from typing import Callable, Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import psutil
import os


class PerformanceProfiler:
    """A class for profiling data structure performance."""
    
    def __init__(self):
        self.results = {}
    
    def measure_operation(self, 
                         operation: Callable, 
                         setup_func: Callable = None,
                         teardown_func: Callable = None,
                         iterations: int = 100) -> Dict[str, float]:
        """
        Measure the performance of a single operation.
        
        Args:
            operation: The operation to measure
            setup_func: Optional setup function called before each iteration
            teardown_func: Optional teardown function called after each iteration
            iterations: Number of iterations to run
            
        Returns:
            Dictionary with performance metrics
        """
        times = []
        memory_usage = []
        
        for _ in range(iterations):
            # Setup
            if setup_func:
                setup_func()
            
            # Memory tracking
            tracemalloc.start()
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            # Time the operation
            start_time = time.perf_counter()
            result = operation()
            end_time = time.perf_counter()
            
            # Memory tracking
            memory_after = process.memory_info().rss
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
            
            # Teardown
            if teardown_func:
                teardown_func()
        
        return {
            'mean_time': statistics.mean(times),
            'median_time': statistics.median(times),
            'min_time': min(times),
            'max_time': max(times),
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
            'mean_memory': statistics.mean(memory_usage),
            'peak_memory': max(memory_usage),
            'total_iterations': iterations
        }
    
    def benchmark_scaling(self, 
                         operation_factory: Callable[[int], Callable],
                         input_sizes: List[int],
                         iterations: int = 10) -> Dict[int, Dict[str, float]]:
        """
        Benchmark how an operation scales with input size.
        
        Args:
            operation_factory: Function that creates operations for different input sizes
            input_sizes: List of input sizes to test
            iterations: Number of iterations per input size
            
        Returns:
            Dictionary mapping input sizes to performance metrics
        """
        scaling_results = {}
        
        for size in input_sizes:
            print(f"Benchmarking size {size}...")
            operation = operation_factory(size)
            results = self.measure_operation(operation, iterations=iterations)
            scaling_results[size] = results
        
        return scaling_results
    
    def plot_scaling_results(self, 
                           scaling_results: Dict[int, Dict[str, float]],
                           metric: str = 'mean_time',
                           title: str = "Performance Scaling",
                           save_path: str = None) -> None:
        """
        Plot scaling results.
        
        Args:
            scaling_results: Results from benchmark_scaling
            metric: The metric to plot
            title: Plot title
            save_path: Optional path to save the plot
        """
        sizes = list(scaling_results.keys())
        values = [scaling_results[size][metric] for size in sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Input Size')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_implementations(self, 
                              implementations: Dict[str, Callable],
                              test_case: Callable,
                              iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple implementations of the same operation.
        
        Args:
            implementations: Dictionary mapping names to implementation functions
            test_case: Function that creates test data
            iterations: Number of iterations per implementation
            
        Returns:
            Dictionary mapping implementation names to performance metrics
        """
        comparison_results = {}
        
        for name, impl in implementations.items():
            print(f"Benchmarking {name}...")
            
            def operation():
                return impl(test_case())
            
            results = self.measure_operation(operation, iterations=iterations)
            comparison_results[name] = results
        
        return comparison_results
    
    def generate_report(self, 
                       results: Dict[str, Dict[str, float]],
                       filename: str = "performance_report.txt") -> None:
        """
        Generate a performance report.
        
        Args:
            results: Performance results to include in the report
            filename: Output filename for the report
        """
        with open(filename, 'w') as f:
            f.write("Performance Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            for name, metrics in results.items():
                f.write(f"{name}:\n")
                f.write("-" * 20 + "\n")
                for metric, value in metrics.items():
                    if 'time' in metric:
                        f.write(f"  {metric}: {value:.6f} seconds\n")
                    elif 'memory' in metric:
                        f.write(f"  {metric}: {value:,.0f} bytes\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
                f.write("\n")


def benchmark_hash_table():
    """Example benchmark for hash table operations."""
    from src.hash_table import HashTable
    
    profiler = PerformanceProfiler()
    
    # Test different operations
    ht = HashTable()
    
    # Insert operation
    def insert_operation():
        for i in range(1000):
            ht.put(f"key_{i}", f"value_{i}")
    
    # Search operation
    def search_operation():
        for i in range(0, 1000, 10):  # Sample every 10th key
            ht.get(f"key_{i}")
    
    # Benchmark operations
    print("Benchmarking hash table operations...")
    
    insert_results = profiler.measure_operation(insert_operation, iterations=10)
    search_results = profiler.measure_operation(search_operation, iterations=100)
    
    results = {
        'Insert Operations': insert_results,
        'Search Operations': search_results
    }
    
    # Generate report
    profiler.generate_report(results, "hash_table_performance.txt")
    
    return results


if __name__ == "__main__":
    # Run example benchmark
    results = benchmark_hash_table()
    print("Benchmark complete. Check hash_table_performance.txt for detailed results.")
