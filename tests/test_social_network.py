"""
Unit tests for social network analysis implementation.

Tests the complete system for identifying influential users including:
- SocialUser class and influence calculations
- UserRegistry (hash table for user lookups)
- InfluenceRanker (priority queue for user ranking)
- SocialNetworkAnalyzer (integrated system)
"""

import pytest
import random
from datetime import datetime

from src.social_network import SocialUser, UserRegistry, InfluenceRanker, SocialNetworkAnalyzer


class TestSocialUser:
    """Test cases for SocialUser class."""
    
    def test_user_creation(self):
        """Test basic user creation."""
        user = SocialUser(
            user_id="test001",
            username="testuser",
            display_name="Test User",
            follower_count=1000,
            following_count=500,
            post_count=200
        )
        
        assert user.user_id == "test001"
        assert user.username == "testuser"
        assert user.display_name == "Test User"
        assert user.follower_count == 1000
        assert user.following_count == 500
        assert user.post_count == 200
        assert not user.verified  # Default
    
    def test_influence_score_calculation(self):
        """Test influence score calculation algorithm."""
        # High influence user
        influencer = SocialUser(
            user_id="influencer",
            username="biginfluencer",
            display_name="Big Influencer",
            follower_count=50000,
            following_count=1000,
            post_count=2000,
            verified=True
        )
        
        score = influencer.calculate_influence_score()
        assert score > 50  # Should have high influence
        assert influencer.influence_score == score
        
        # Regular user
        regular_user = SocialUser(
            user_id="regular",
            username="regularuser",
            display_name="Regular User",
            follower_count=100,
            following_count=200,
            post_count=50,
            verified=False
        )
        
        regular_score = regular_user.calculate_influence_score()
        assert regular_score < score  # Should be less than influencer
    
    def test_update_metrics(self):
        """Test user metrics updating."""
        user = SocialUser("test", "test", "Test", follower_count=100, following_count=50)
        original_score = user.calculate_influence_score()
        
        # Update metrics
        user.update_metrics(followers=1000, following=100, posts=500)
        
        assert user.follower_count == 1000
        assert user.following_count == 100
        assert user.post_count == 500
        assert user.influence_score > original_score  # Should increase


class TestUserRegistry:
    """Test cases for UserRegistry (hash table implementation)."""
    
    def test_user_registration(self):
        """Test user registration in registry."""
        registry = UserRegistry()
        user = SocialUser("user001", "testuser", "Test User")
        
        # First registration should succeed
        assert registry.register_user(user) == True
        assert registry.user_count() == 1
        
        # Duplicate registration should fail
        duplicate = SocialUser("user001", "different", "Different User")
        assert registry.register_user(duplicate) == False
        assert registry.user_count() == 1
    
    def test_user_lookup_by_id(self):
        """Test O(1) user lookup by ID."""
        registry = UserRegistry()
        user = SocialUser("user001", "testuser", "Test User", follower_count=500)
        registry.register_user(user)
        
        # Successful lookup
        found_user = registry.get_user("user001")
        assert found_user is not None
        assert found_user.user_id == "user001"
        assert found_user.follower_count == 500
        
        # Failed lookup
        not_found = registry.get_user("nonexistent")
        assert not_found is None
    
    def test_user_lookup_by_username(self):
        """Test O(1) user lookup by username."""
        registry = UserRegistry()
        user = SocialUser("user001", "TestUser", "Test User")
        registry.register_user(user)
        
        # Case insensitive lookup
        found_user = registry.get_user_by_username("testuser")
        assert found_user is not None
        assert found_user.username == "TestUser"
        
        found_user = registry.get_user_by_username("TESTUSER")
        assert found_user is not None
    
    def test_metrics_update(self):
        """Test updating user metrics through registry."""
        registry = UserRegistry()
        user = SocialUser("user001", "testuser", "Test User", follower_count=100)
        registry.register_user(user)
        
        # Update metrics
        registry.update_user_metrics("user001", followers=500, following=200, posts=100)
        
        updated_user = registry.get_user("user001")
        assert updated_user.follower_count == 500
        assert updated_user.following_count == 200
        assert updated_user.post_count == 100
    
    def test_get_all_users(self):
        """Test retrieving all users."""
        registry = UserRegistry()
        users = [
            SocialUser("user001", "user1", "User 1"),
            SocialUser("user002", "user2", "User 2"),
            SocialUser("user003", "user3", "User 3")
        ]
        
        for user in users:
            registry.register_user(user)
        
        all_users = registry.get_all_users()
        assert len(all_users) == 3
        user_ids = {user.user_id for user in all_users}
        assert user_ids == {"user001", "user002", "user003"}


class TestInfluenceRanker:
    """Test cases for InfluenceRanker (priority queue implementation)."""
    
    def test_add_users_to_ranking(self):
        """Test adding users to influence ranking."""
        ranker = InfluenceRanker(max_heap=True)
        
        users = [
            SocialUser("user1", "low", "Low", follower_count=100, following_count=200),
            SocialUser("user2", "high", "High", follower_count=10000, following_count=500, verified=True),
            SocialUser("user3", "medium", "Medium", follower_count=1000, following_count=300)
        ]
        
        for user in users:
            ranker.add_user(user)
        
        assert ranker.size() == 3
    
    def test_top_influencers_ranking(self):
        """Test getting top influencers."""
        ranker = InfluenceRanker(max_heap=True)
        
        # Create users with different influence levels
        low_user = SocialUser("low", "low", "Low Influence", follower_count=100, following_count=500)
        high_user = SocialUser("high", "high", "High Influence", follower_count=50000, following_count=1000, verified=True)
        medium_user = SocialUser("medium", "medium", "Medium Influence", follower_count=5000, following_count=800)
        
        # Add in random order
        ranker.add_user(medium_user)
        ranker.add_user(low_user)
        ranker.add_user(high_user)
        
        # Get top 2 influencers
        top_influencers = ranker.get_top_influencers(2)
        
        assert len(top_influencers) == 2
        # Should be ordered by influence (high first, then medium)
        assert top_influencers[0].user_id == "high"
        assert top_influencers[1].user_id == "medium"
    
    def test_influence_distribution(self):
        """Test influence distribution calculation."""
        ranker = InfluenceRanker()
        
        # Create users with different influence levels
        users = [
            SocialUser("high1", "h1", "High 1", follower_count=100000, following_count=500, verified=True),
            SocialUser("high2", "h2", "High 2", follower_count=80000, following_count=300, verified=True),
            SocialUser("medium1", "m1", "Medium 1", follower_count=5000, following_count=1000),
            SocialUser("low1", "l1", "Low 1", follower_count=100, following_count=300),
            SocialUser("low2", "l2", "Low 2", follower_count=50, following_count=400)
        ]
        
        for user in users:
            ranker.add_user(user)
        
        distribution = ranker.get_influence_distribution()
        
        # Updated expectations based on new influence calculation
        # High1 & High2 should have ~75 score (high), Medium1 ~44 (medium), Low1 & Low2 <40 (low)
        assert distribution["high"] == 2  # Users with score >= 70
        assert distribution["medium"] == 1  # Users with score 40-69
        assert distribution["low"] == 2  # Users with score < 40
    
    def test_update_user_influence(self):
        """Test updating user influence in ranking."""
        ranker = InfluenceRanker()
        user = SocialUser("user1", "test", "Test", follower_count=100, following_count=200)
        
        ranker.add_user(user)
        original_top = ranker.get_top_influencers(1)[0]
        original_score = original_top.influence_score
        
        # Update user to have higher influence
        user.update_metrics(followers=50000, following=500, posts=2000)
        ranker.update_user_influence(user)
        
        new_top = ranker.get_top_influencers(1)[0]
        assert new_top.influence_score > original_score


class TestSocialNetworkAnalyzer:
    """Test cases for integrated social network analysis system."""
    
    def test_add_user_to_network(self):
        """Test adding users to the complete network."""
        analyzer = SocialNetworkAnalyzer()
        
        user = analyzer.add_user(
            user_id="test001",
            username="testuser", 
            display_name="Test User",
            follower_count=1000,
            verified=True
        )
        
        assert user.user_id == "test001"
        assert user.verified == True
        
        # Check user is in all components
        assert analyzer.user_registry.get_user("test001") is not None
        assert analyzer.social_graph.vertex_count() == 1
        assert analyzer.influence_ranker.size() == 1
    
    def test_follow_relationship(self):
        """Test creating follow relationships."""
        analyzer = SocialNetworkAnalyzer()
        
        # Add two users
        user1 = analyzer.add_user("user1", "alice", "Alice", follower_count=500, following_count=100)
        user2 = analyzer.add_user("user2", "bob", "Bob", follower_count=200, following_count=300)
        
        # Create follow relationship
        success = analyzer.follow_user("user1", "user2")  # Alice follows Bob
        
        assert success == True
        assert analyzer.social_graph.has_edge("user1", "user2")
        
        # Check metrics updated
        updated_alice = analyzer.user_registry.get_user("user1")
        updated_bob = analyzer.user_registry.get_user("user2")
        
        assert updated_alice.following_count == 101  # Increased by 1
        assert updated_bob.follower_count == 201     # Increased by 1
    
    def test_calculate_centrality_metrics(self):
        """Test degree centrality calculation."""
        analyzer = SocialNetworkAnalyzer()
        
        # Add users
        for i in range(3):
            analyzer.add_user(f"user{i}", f"user{i}", f"User {i}")
        
        # Create connections
        analyzer.follow_user("user0", "user1")
        analyzer.follow_user("user1", "user2")
        analyzer.follow_user("user2", "user0")
        
        # Calculate centrality
        centralities = analyzer.calculate_centrality_metrics()
        
        assert len(centralities) == 3
        for user_id, centrality in centralities.items():
            assert 0 <= centrality <= 1  # Normalized centrality
            
        # Check user objects updated
        user = analyzer.user_registry.get_user("user0")
        assert user.degree_centrality == centralities["user0"]
    
    def test_find_influential_users(self):
        """Test finding top influential users."""
        analyzer = SocialNetworkAnalyzer()
        
        # Add users with different influence levels - make celebrity clearly highest
        analyzer.add_user("influencer", "biginfluencer", "Big Influencer", 
                         follower_count=50000, following_count=1000, verified=True)
        analyzer.add_user("regular", "regularuser", "Regular User",
                         follower_count=500, following_count=800)
        analyzer.add_user("celebrity", "celeb", "Celebrity",
                         follower_count=200000, following_count=50, verified=True, post_count=5000)
        
        # Find top influencers
        top_influencers = analyzer.find_influential_users(2)
        
        assert len(top_influencers) == 2
        # Check that we get the expected top users (celebrity should be first now)
        top_user_ids = {user['user_id'] for user in top_influencers}
        assert "celebrity" in top_user_ids
        assert "influencer" in top_user_ids
        assert all('influence_score' in user for user in top_influencers)
        assert all('degree_centrality' in user for user in top_influencers)
    
    def test_network_statistics(self):
        """Test network statistics calculation."""
        analyzer = SocialNetworkAnalyzer()
        
        # Add users and connections
        for i in range(5):
            analyzer.add_user(f"user{i}", f"user{i}", f"User {i}")
        
        # Add some connections
        analyzer.follow_user("user0", "user1")
        analyzer.follow_user("user1", "user2")
        analyzer.follow_user("user2", "user0")
        
        stats = analyzer.get_network_stats()
        
        assert stats['total_users'] == 5
        assert stats['total_connections'] == 3
        assert stats['avg_connections'] == 0.6  # 3 connections / 5 users
        assert 'influence_distribution' in stats
    
    def test_duplicate_user_handling(self):
        """Test handling of duplicate user IDs."""
        analyzer = SocialNetworkAnalyzer()
        
        analyzer.add_user("user1", "first", "First User")
        
        # Attempt to add duplicate
        with pytest.raises(ValueError):
            analyzer.add_user("user1", "second", "Second User")
    
    def test_invalid_follow_relationship(self):
        """Test handling invalid follow relationships."""
        analyzer = SocialNetworkAnalyzer()
        
        analyzer.add_user("user1", "alice", "Alice")
        
        # Try to follow non-existent user
        success = analyzer.follow_user("user1", "nonexistent")
        assert success == False
        
        # Try non-existent user following someone
        success = analyzer.follow_user("nonexistent", "user1")
        assert success == False


class TestPerformance:
    """Performance tests for social network analysis."""
    
    def test_user_lookup_performance(self):
        """Test hash table lookup performance."""
        registry = UserRegistry()
        
        # Add many users
        num_users = 1000
        for i in range(num_users):
            user = SocialUser(f"user{i:04d}", f"user{i}", f"User {i}")
            registry.register_user(user)
        
        # Test lookup performance
        import time
        start_time = time.perf_counter()
        
        for i in range(100):  # 100 lookups
            user = registry.get_user(f"user{i:04d}")
            assert user is not None
        
        lookup_time = time.perf_counter() - start_time
        
        # Should be very fast (under 1ms for 100 lookups)
        assert lookup_time < 0.001
        
    def test_influence_ranking_performance(self):
        """Test priority queue ranking performance."""
        ranker = InfluenceRanker()
        
        # Add many users
        num_users = 1000
        for i in range(num_users):
            user = SocialUser(f"user{i:04d}", f"user{i}", f"User {i}",
                            follower_count=random.randint(10, 10000),
                            following_count=random.randint(10, 1000))
            ranker.add_user(user)
        
        # Test ranking retrieval
        import time
        start_time = time.perf_counter()
        
        top_users = ranker.get_top_influencers(10)
        
        ranking_time = time.perf_counter() - start_time
        
        assert len(top_users) == 10
        assert ranking_time < 0.01  # Should be very fast
    
    def test_large_network_analysis(self):
        """Test analysis of larger networks."""
        analyzer = SocialNetworkAnalyzer()
        
        # Create larger network
        num_users = 100
        for i in range(num_users):
            analyzer.add_user(f"user{i:03d}", f"user{i}", f"User {i}",
                            follower_count=random.randint(10, 1000),
                            following_count=random.randint(10, 500))
        
        # Add random connections
        num_connections = 200
        for _ in range(num_connections):
            follower = f"user{random.randint(0, num_users-1):03d}"
            followee = f"user{random.randint(0, num_users-1):03d}"
            if follower != followee:
                analyzer.follow_user(follower, followee)
        
        # Test analysis performance
        import time
        start_time = time.perf_counter()
        
        top_influencers = analyzer.find_influential_users(10)
        stats = analyzer.get_network_stats()
        
        analysis_time = time.perf_counter() - start_time
        
        assert len(top_influencers) == 10
        assert stats['total_users'] == num_users
        assert analysis_time < 1.0  # Should complete in under 1 second 