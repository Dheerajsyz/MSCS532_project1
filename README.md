# Social Network Analysis - Data Structures Implementation

## Overview  
Implementation of data structures for identifying influential users in Twitter-like social network platforms. This project demonstrates the practical application of hash tables, directed graphs, and priority queues for social network analysis.

## Project Structure
```
project/
├── src/                          # Core implementations
│   ├── __init__.py
│   ├── base.py                   # Abstract base classes
│   ├── hash_table.py            # Hash table for user lookups
│   ├── graph.py                 # Directed graph for relationships
│   ├── priority_queue.py        # Priority queue for influence ranking
│   └── social_network.py        # Integrated social network analysis
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_hash_table.py
│   ├── test_graph.py
│   ├── test_priority_queue.py
│   └── test_social_network.py
├── data/                         # Data generation utilities
│   └── sample_data.py
├── performance/                  # Performance analysis tools
│   └── benchmark.py
├── social_network_demo.py        # Social network demonstration
├── social_network.json           # Sample social network data
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set Python path (if needed)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Usage
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run individual test modules
python3 -m pytest tests/test_hash_table.py -v
python3 -m pytest tests/test_graph.py -v
python3 -m pytest tests/test_priority_queue.py -v
python3 -m pytest tests/test_social_network.py -v

# Run social network analysis demonstration
python3 social_network_demo.py

# Generate performance analysis
python3 performance/benchmark.py
```

## Data Structures for Social Network Analysis

### Hash Table (UserRegistry)
- **Purpose**: Fast O(1) user ID to user object mapping
- **Features**: Separate chaining, collision handling, dual indexing
- **Use Case**: Instant user lookup by ID or username for social network operations

### Directed Graph (AdjacencyListGraph)  
- **Purpose**: Model "follows" relationships in social networks
- **Features**: Directed edges, BFS/DFS traversal, shortest path algorithms
- **Use Case**: Social connections, influence propagation analysis, network traversal

### Priority Queue (InfluenceRanker)
- **Purpose**: Rank users by influence metrics using max-heap
- **Features**: Heap-based ranking, dynamic score updates, influence distribution
- **Use Case**: Identifying top influencers, trending user analysis

## Social Network Components

### SocialUser
User representation with influence metrics including follower count, verification status, and calculated influence scores.

### UserRegistry  
Hash table-based user management system providing O(1) lookups by user ID and username.

### InfluenceRanker
Priority queue system for maintaining rankings of users by influence score with efficient top-K retrieval.

### SocialNetworkAnalyzer
Integrated system combining all data structures for comprehensive social network analysis and influential user identification.

## Sample Data
Social network analysis data:
- `social_network.json` - Sample users with influence metrics and follow relationships
- Generated data includes verified users, influencers, celebrities, and regular users
- Realistic follower/following ratios and post counts
- Diverse user types for comprehensive influence analysis

## Testing
Comprehensive test suite covering:
- Social user creation and influence calculation
- Hash table user registry operations (O(1) lookups)
- Priority queue influence ranking functionality
- Directed graph relationship modeling
- Integrated social network analysis
- Performance testing with large user datasets

## Performance Analysis
Key performance characteristics:
- **Hash Table Lookups**: O(1) user retrieval by ID or username
- **Graph Operations**: O(V+E) for traversals, efficient relationship queries
- **Priority Queue**: O(log n) for influence ranking operations
- **Centrality Calculation**: O(V²) for degree centrality analysis
- **Integrated Analysis**: Sub-second processing for networks with hundreds of users

## Dependencies
- **pytest**: Testing framework
- **matplotlib**: Performance visualization  
- **numpy**: Numerical operations
- **psutil**: System performance monitoring
