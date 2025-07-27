#!/usr/bin/env python3
"""
Social Network Analysis Demonstration - Phase 2

This demonstration showcases the implementation of data structures for
identifying influential users in a Twitter-like social network platform.

Key Data Structures Demonstrated:
1. Hash Table - Fast user ID to user object mapping (O(1) lookups)
2. Directed Graph - Models follow relationships with adjacency lists
3. Priority Queue - Ranks users by influence metrics using heaps

Matches Phase 1 Design: "Social Network Analysis: Identifying Influential Users"
"""

import json
import time
import random
from typing import List, Dict

from src.social_network import SocialNetworkAnalyzer, SocialUser


def print_header(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def print_subheader(title: str) -> None:
    """Print formatted subsection header."""
    print(f"\n--- {title} ---")


def load_social_network_data() -> Dict:
    """Load or generate social network data."""
    try:
        with open('social_network.json', 'r') as f:
            data = json.load(f)
        
        # Check if data has the expected structure
        if 'users' in data and 'connections' in data:
            print("LOADED: Existing social network data")
            return data
        else:
            print("INFO: Existing data format incompatible, generating sample data...")
            return generate_sample_network()
    except (FileNotFoundError, json.JSONDecodeError):
        print("GENERATING: Sample social network data...")
        return generate_sample_network()


def generate_sample_network() -> Dict:
    """Generate sample social network for demonstration."""
    users = []
    connections = []
    
    # Create sample users with varying influence levels
    user_profiles = [
        ("influencer_1", "TechGuru", "Alex Chen", True, 50000, 1000, 2500),
        ("influencer_2", "SocialQueen", "Maria Rodriguez", True, 75000, 500, 3200),
        ("celebrity_1", "SportsStar", "James Wilson", True, 200000, 200, 1000),
        ("user_001", "dev_sarah", "Sarah Johnson", False, 5000, 1200, 800),
        ("user_002", "music_lover", "David Kim", False, 2000, 800, 400),
        ("user_003", "foodie_blog", "Emma Thompson", False, 8000, 300, 1200),
        ("user_004", "travel_photo", "Carlos Mendez", False, 15000, 600, 900),
        ("user_005", "tech_news", "Lisa Zhang", False, 12000, 400, 600),
        ("startup_ceo", "InnovateCEO", "Michael Brown", True, 25000, 800, 1500),
        ("journalist_1", "NewsReporter", "Ana Patel", True, 30000, 1500, 2000)
    ]
    
    for user_id, username, display_name, verified, followers, following, posts in user_profiles:
        users.append({
            "user_id": user_id,
            "username": username,
            "display_name": display_name,
            "verified": verified,
            "follower_count": followers,
            "following_count": following,
            "post_count": posts
        })
    
    # Generate realistic follow relationships
    follow_patterns = [
        ("user_001", "influencer_1"),
        ("user_001", "tech_news"),
        ("user_002", "influencer_2"),
        ("user_002", "celebrity_1"),
        ("user_003", "foodie_blog"),
        ("user_003", "travel_photo"),
        ("user_004", "travel_photo"),
        ("user_004", "celebrity_1"),
        ("user_005", "influencer_1"),
        ("user_005", "tech_news"),
        ("startup_ceo", "influencer_1"),
        ("startup_ceo", "journalist_1"),
        ("journalist_1", "celebrity_1"),
        ("journalist_1", "influencer_2"),
        ("influencer_1", "startup_ceo"),
        ("influencer_2", "journalist_1"),
        ("celebrity_1", "influencer_1"),
        ("tech_news", "startup_ceo")
    ]
    
    for follower, followee in follow_patterns:
        connections.append({"follower": follower, "followee": followee})
    
    return {"users": users, "connections": connections}


def demonstrate_hash_table_user_registry():
    """Demonstrate hash table for fast user lookups."""
    print_header("HASH TABLE IMPLEMENTATION - User Registry")
    print("Demonstrating hash table performance for user lookups")
    
    from src.social_network import UserRegistry
    
    registry = UserRegistry()
    
    print_subheader("User Registration Performance")
    
    # Create sample users
    sample_users = [
        SocialUser("user001", "john_doe", "John Doe", follower_count=1000, following_count=500),
        SocialUser("user002", "jane_smith", "Jane Smith", follower_count=2500, following_count=300, verified=True),
        SocialUser("user003", "tech_guru", "Tech Guru", follower_count=50000, following_count=1000, post_count=2000)
    ]
    
    # Measure registration performance
    start_time = time.perf_counter()
    for user in sample_users:
        registry.register_user(user)
    registration_time = time.perf_counter() - start_time
    
    print(f"REGISTERED: {len(sample_users)} users in {registration_time:.6f} seconds")
    print(f"TOTAL: {registry.user_count()} users in registry")
    
    print_subheader("Fast User Lookups")
    
    # Demonstrate O(1) lookups
    start_time = time.perf_counter()
    user = registry.get_user("user002")
    lookup_time = time.perf_counter() - start_time
    
    print(f"LOOKUP: User by ID in {lookup_time:.6f} seconds: {user.display_name}")
    print(f"  - Username: @{user.username}")
    print(f"  - Followers: {user.follower_count:,}")
    print(f"  - Verified: {'Yes' if user.verified else 'No'}")
    
    # Test username lookup
    start_time = time.perf_counter()
    user = registry.get_user_by_username("tech_guru")
    username_lookup_time = time.perf_counter() - start_time
    
    print(f"LOOKUP: Username search in {username_lookup_time:.6f} seconds: {user.display_name}")
    
    print_subheader("How Fast Is It?")
    print(f"TIMING: User ID lookup: {lookup_time:.6f}s (very fast!)")
    print(f"TIMING: Username lookup: {username_lookup_time:.6f}s (also very fast!)")
    print(f"STORAGE: Handles {registry.user_count()} users efficiently")


def demonstrate_graph_social_connections():
    """Demonstrate directed graph for modeling social relationships."""
    print_header("GRAPH IMPLEMENTATION - Social Connections")
    print("Demonstrating directed graph for modeling follow relationships")
    
    from src.graph import AdjacencyListGraph
    
    # Create directed graph for follows relationships
    social_graph = AdjacencyListGraph(directed=True)
    
    print_subheader("Building Social Network Graph")
    
    # Add users to graph
    users = ["alice", "bob", "charlie", "diana", "eve", "frank"]
    for user in users:
        social_graph.add_vertex(user)
    
    # Create follow relationships (directed edges)
    follows = [
        ("alice", "bob"),      # Alice follows Bob
        ("alice", "charlie"),  # Alice follows Charlie
        ("bob", "charlie"),    # Bob follows Charlie
        ("charlie", "diana"),  # Charlie follows Diana
        ("diana", "eve"),      # Diana follows Eve
        ("eve", "alice"),      # Eve follows Alice
        ("frank", "alice"),    # Frank follows Alice
        ("frank", "bob"),      # Frank follows Bob
        ("charlie", "frank")   # Charlie follows Frank
    ]
    
    start_time = time.perf_counter()
    for follower, followee in follows:
        social_graph.add_edge(follower, followee)
    graph_build_time = time.perf_counter() - start_time
    
    print(f"BUILT: Social graph with {social_graph.vertex_count()} users")
    print(f"ADDED: {social_graph.edge_count()} follow relationships")
    print(f"TIMING: Graph construction time: {graph_build_time:.6f} seconds")
    
    print_subheader("Analyzing Social Connections")
    
    # Show who each user follows (outgoing edges)
    print("Following relationships (outgoing edges):")
    for user in users:
        following = list(social_graph.get_neighbors(user))
        print(f"  {user} follows: {following}")
    
    # Show followers (incoming edges)
    print("\nFollower relationships (incoming edges):")
    for user in users:
        followers = list(social_graph.get_incoming_edges(user))
        print(f"  {user}'s followers: {followers}")
    
    print_subheader("Graph Traversal Algorithms")
    
    # Demonstrate BFS to find reachable users
    start_user = "alice"
    start_time = time.perf_counter()
    reachable_users = social_graph.breadth_first_search(start_user)
    bfs_time = time.perf_counter() - start_time
    
    print(f"BFS: From {start_user}: {list(reachable_users)}")
    print(f"TIMING: BFS execution time: {bfs_time:.6f} seconds")
    
    # Find shortest path between users
    start_time = time.perf_counter()
    path = social_graph.shortest_path("alice", "eve")
    shortest_path_time = time.perf_counter() - start_time
    
    if path:
        print(f"PATH: Shortest path from alice to eve: {' -> '.join(path)}")
        print(f"LENGTH: Path length: {len(path) - 1} hops")
    else:
        print("RESULT: No path found from alice to eve")
    print(f"TIMING: Shortest path algorithm time: {shortest_path_time:.6f} seconds")


def demonstrate_priority_queue_influence_ranking():
    """Demonstrate priority queue for ranking users by influence."""
    print_header("PRIORITY QUEUE IMPLEMENTATION - Influence Ranking")
    print("Demonstrating priority queue for user influence ranking")
    
    from src.social_network import InfluenceRanker, SocialUser
    
    ranker = InfluenceRanker(max_heap=True)
    
    print_subheader("Creating Users with Different Influence Levels")
    
    # Create users with varying influence metrics
    users = [
        SocialUser("celeb1", "megastar", "Celebrity A", follower_count=1000000, following_count=100, 
                  post_count=5000, verified=True),
        SocialUser("influ1", "techguru", "Tech Influencer", follower_count=50000, following_count=500,
                  post_count=2000, verified=True),
        SocialUser("user1", "regularjoe", "Regular User", follower_count=500, following_count=800,
                  post_count=200, verified=False),
        SocialUser("micro1", "microinflu", "Micro Influencer", follower_count=5000, following_count=200,
                  post_count=800, verified=False),
        SocialUser("brand1", "officialco", "Brand Account", follower_count=25000, following_count=50,
                  post_count=1000, verified=True)
    ]
    
    # Add users to influence ranking
    start_time = time.perf_counter()
    for user in users:
        ranker.add_user(user)
    ranking_time = time.perf_counter() - start_time
    
    print(f"ADDED: {len(users)} users to influence ranking")
    print(f"TIMING: Ranking construction time: {ranking_time:.6f} seconds")
    
    print_subheader("Influence Score Calculation")
    
    for user in users:
        score = user.calculate_influence_score()
        print(f"  {user.display_name} (@{user.username}): {score:.1f}")
        print(f"    - Followers: {user.follower_count:,}")
        print(f"    - Following: {user.following_count:,}")
        print(f"    - Posts: {user.post_count:,}")
        print(f"    - Verified: {'Yes' if user.verified else 'No'}")
    
    print_subheader("Top Influencers Ranking")
    
    # Get top influencers using priority queue
    start_time = time.perf_counter()
    top_influencers = ranker.get_top_influencers(3)
    ranking_query_time = time.perf_counter() - start_time
    
    print(f"RANKING: Retrieved top 3 influencers in {ranking_query_time:.6f} seconds:")
    for i, user in enumerate(top_influencers, 1):
        print(f"  #{i}: {user.display_name} (@{user.username}) - Score: {user.influence_score:.1f}")
    
    # Show influence distribution
    distribution = ranker.get_influence_distribution()
    print(f"\nDISTRIBUTION: Influence Distribution:")
    print(f"  - High influence (70+): {distribution['high']} users")
    print(f"  - Medium influence (40-69): {distribution['medium']} users") 
    print(f"  - Low influence (0-39): {distribution['low']} users")


def demonstrate_integrated_social_network_analysis():
    """Demonstrate complete social network analysis system."""
    print_header("INTEGRATED SYSTEM ANALYSIS")
    print("Combining all data structures for comprehensive social network analysis")
    
    analyzer = SocialNetworkAnalyzer()
    
    print_subheader("Setting Up Social Network")
    
    # Load sample data
    network_data = load_social_network_data()
    
    # Add users to the system
    start_time = time.perf_counter()
    for user_data in network_data["users"]:
        analyzer.add_user(**user_data)
    user_setup_time = time.perf_counter() - start_time
    
    print(f"SETUP: Added {len(network_data['users'])} users to network")
    print(f"TIMING: User setup time: {user_setup_time:.4f} seconds")
    
    # Add follow relationships
    start_time = time.perf_counter()
    for connection in network_data["connections"]:
        analyzer.follow_user(connection["follower"], connection["followee"])
    connection_setup_time = time.perf_counter() - start_time
    
    print(f"CREATED: {len(network_data['connections'])} follow relationships")
    print(f"TIMING: Connection setup time: {connection_setup_time:.4f} seconds")
    
    print_subheader("Network Statistics")
    
    stats = analyzer.get_network_stats()
    print(f"STATS: Total users: {stats['total_users']}")
    print(f"STATS: Total connections: {stats['total_connections']}")
    print(f"STATS: Average connections per user: {stats['avg_connections']:.1f}")
    print(f"STATS: Influence distribution: {stats['influence_distribution']}")
    
    print_subheader("Identifying Top Influencers")
    
    # Find most influential users
    start_time = time.perf_counter()
    top_influencers = analyzer.find_influential_users(5)
    analysis_time = time.perf_counter() - start_time
    
    print(f"ANALYSIS: Influence analysis completed in {analysis_time:.4f} seconds")
    print("\nTop 5 Most Influential Users:")
    
    for i, user_data in enumerate(top_influencers, 1):
        print(f"\n#{i}: {user_data['display_name']} (@{user_data['username']})")
        print(f"    Influence Score: {user_data['influence_score']:.1f}")
        print(f"    Followers: {user_data['followers']:,}")
        print(f"    Following: {user_data['following']:,}")
        print(f"    Degree Centrality: {user_data['degree_centrality']:.3f}")
        print(f"    Verified: {'Yes' if user_data['verified'] else 'No'}")
    
    print_subheader("How Did It Perform?")
    
    # Check how fast everything was
    print("Speed Summary:")
    print(f"  Hash Table: Really fast user lookups")
    print(f"  Graph: Efficiently handles all the follow relationships")
    print(f"  Priority Queue: Quick influence ranking")
    print(f"  Total time: {analysis_time:.4f}s for {stats['total_users']} users (pretty good!)")


def main():
    """Main demonstration function."""
    print("Social Network Analysis - Phase 2 Implementation")
    print("="*55)
    print("This program analyzes social networks to identify influential users")
    print("using data structures optimized for social media platforms")
    print("\nProject: Social Network Analysis - Identifying Influential Users")
    print("="*55)
    
    print("\nImplemented Data Structures:")
    print("1. Hash Table - for fast user lookups")
    print("2. Directed Graph - to model follow relationships") 
    print("3. Priority Queue - to rank users by influence score")
    
    # Run individual component demonstrations
    demonstrate_hash_table_user_registry()
    demonstrate_graph_social_connections()
    demonstrate_priority_queue_influence_ranking()
    
    # Run integrated analysis
    demonstrate_integrated_social_network_analysis()
    
    # Summary
    print_header("IMPLEMENTATION SUMMARY")
    print("This implementation successfully demonstrates:")
    print("   - Hash Table: Efficient user lookup operations")
    print("   - Directed Graph: Accurate follow relationship modeling")
    print("   - Priority Queue: Effective influence-based ranking")
    print("\nCore functionality achieved:")
    print("   - Fast user retrieval from large datasets")
    print("   - Complete social relationship tracking")
    print("   - Identification of most influential users")
    print("   - Integration of multiple data structures")
    print("\nFulfills Phase 1 project requirements:")
    print("   Social Network Analysis for Identifying Influential Users")


if __name__ == "__main__":
    main() 