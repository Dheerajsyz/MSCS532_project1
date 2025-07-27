"""
Data Structures Implementation Package

Social Network Analysis - Phase 2 & 3 Implementation
Focused on identifying influential users in Twitter-like platforms.

Phase 2 Components:
- Hash Table: Fast user ID to user object mapping
- Directed Graph: Models follow relationships  
- Priority Queue: Ranks users by influence metrics

Phase 3 Optimizations:
- AVL Tree: Self-balancing BST solving recursion limits
- Optimized Hash Table: Enhanced collision handling
- Optimized Graph: Cached algorithms for large networks
- Advanced Priority Queue: Decrease-key operations
"""

# Phase 2 implementations
from .hash_table import HashTable
from .graph import AdjacencyListGraph  
from .priority_queue import HeapPriorityQueue
from .social_network import SocialUser, UserRegistry, InfluenceRanker, SocialNetworkAnalyzer

# Phase 3 optimized implementations
from .optimized_structures import (
    OptimizedAVLTree, OptimizedHashTable, OptimizedSocialGraph,
    AdvancedPriorityQueue, PerformanceProfiler
)

from .base import DataStructure, KeyValueStore, Graph, Tree, PriorityQueue

__all__ = [
    # Phase 2
    'HashTable',
    'AdjacencyListGraph',
    'HeapPriorityQueue',
    'SocialUser',
    'UserRegistry', 
    'InfluenceRanker',
    'SocialNetworkAnalyzer',
    # Phase 3 optimizations
    'OptimizedAVLTree',
    'OptimizedHashTable', 
    'OptimizedSocialGraph',
    'AdvancedPriorityQueue',
    'PerformanceProfiler'
] 