"""
Social Network Analysis - Phase 2 Implementation

This module implements data structures for analyzing social networks
and identifying influential users in a Twitter-like platform.

Key Components:
- SocialUser: Represents a user with influence metrics
- UserRegistry: Hash table for fast user ID lookups  
- InfluenceRanker: Priority queue for ranking users by influence
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

from .base import track_performance
from .hash_table import HashTable
from .graph import AdjacencyListGraph
from .priority_queue import HeapPriorityQueue


@dataclass
class SocialUser:
    """
    Represents a user in the social network with influence metrics.
    """
    user_id: str
    username: str
    display_name: str
    follower_count: int = 0
    following_count: int = 0
    post_count: int = 0
    join_date: datetime = field(default_factory=datetime.now)
    verified: bool = False
    
    # Influence metrics
    degree_centrality: float = 0.0
    influence_score: float = 0.0
    reach_score: float = 0.0
    
    def calculate_influence_score(self) -> float:
        """
        Calculate overall influence score based on multiple metrics.
        
        Returns:
            Combined influence score (0-100)
        """
        # Base score from follower count (logarithmic scale to handle large numbers)
        import math
        follower_score = min(40, math.log10(max(1, self.follower_count)) * 10)
        
        # Follower-to-following ratio bonus
        if self.following_count > 0:
            ratio = self.follower_count / self.following_count
            ratio_bonus = min(20, math.log10(max(1, ratio)) * 10)
        else:
            ratio_bonus = 20  # Maximum bonus if following no one
        
        # Activity score from posts
        activity_score = min(15, self.post_count / 1000 * 15)
        
        # Verification bonus
        verification_bonus = 15 if self.verified else 0
        
        # Centrality bonus (will be 0 initially, updated later)
        centrality_bonus = self.degree_centrality * 10
        
        self.influence_score = follower_score + ratio_bonus + activity_score + verification_bonus + centrality_bonus
        return self.influence_score
    
    def update_metrics(self, followers: int, following: int, posts: int = None):
        """Update user metrics and recalculate influence."""
        self.follower_count = followers
        self.following_count = following
        if posts is not None:
            self.post_count = posts
        self.calculate_influence_score()


class UserRegistry:
    """
    Hash table for fast user ID to user object mapping.
    Optimized for O(1) user lookups in social network analysis.
    """
    
    def __init__(self):
        self._user_table = HashTable()
        self._username_table = HashTable()  # Secondary index for username lookups
        
    @track_performance
    def register_user(self, user: SocialUser) -> bool:
        """
        Register a new user in the system.
        
        Args:
            user: SocialUser object to register
            
        Returns:
            True if successful, False if user already exists
        """
        if self.get_user(user.user_id) is not None:
            return False
            
        self._user_table.put(user.user_id, user)
        self._username_table.put(user.username.lower(), user)
        return True
    
    @track_performance
    def get_user(self, user_id: str) -> Optional[SocialUser]:
        """Get user by ID with O(1) lookup."""
        return self._user_table.get(user_id)
    
    @track_performance
    def get_user_by_username(self, username: str) -> Optional[SocialUser]:
        """Get user by username with O(1) lookup."""
        return self._username_table.get(username.lower())
    
    def update_user_metrics(self, user_id: str, followers: int, following: int, posts: int = None):
        """Update user metrics and recalculate influence."""
        user = self.get_user(user_id)
        if user:
            user.update_metrics(followers, following, posts)
    
    def get_all_users(self) -> List[SocialUser]:
        """Get all registered users."""
        return list(self._user_table.values())
    
    def user_count(self) -> int:
        """Get total number of registered users."""
        return self._user_table.size()


class InfluenceRanker:
    """
    Priority queue for ranking users by influence metrics.
    Maintains top influencers efficiently using heap operations.
    """
    
    def __init__(self, max_heap: bool = True):
        self._ranker = HeapPriorityQueue(max_heap=max_heap)
        self._user_scores = {}  # Track current scores for updates
    
    @track_performance
    def add_user(self, user: SocialUser) -> None:
        """Add user to influence ranking."""
        influence_score = user.calculate_influence_score()
        self._ranker.enqueue(user, influence_score)
        self._user_scores[user.user_id] = influence_score
    
    @track_performance
    def update_user_influence(self, user: SocialUser) -> None:
        """Update user's influence score in the ranking."""
        new_score = user.calculate_influence_score()
        
        # For now, re-add with new score (in real implementation, would use decrease-key)
        self._ranker.enqueue(user, new_score)
        self._user_scores[user.user_id] = new_score
    
    @track_performance
    def get_top_influencers(self, count: int = 10) -> List[SocialUser]:
        """
        Get top N influential users.
        
        Args:
            count: Number of top users to return
            
        Returns:
            List of top influential users
        """
        top_users = []
        temp_users = []
        
        # Extract top users
        for _ in range(min(count, self._ranker.size())):
            if not self._ranker.is_empty():
                user = self._ranker.dequeue()
                top_users.append(user)
                temp_users.append(user)
        
        # Re-add users back to maintain ranking
        for user in temp_users:
            self._ranker.enqueue(user, self._user_scores[user.user_id])
        
        return top_users
    
    def get_influence_distribution(self) -> Dict[str, int]:
        """Get distribution of users by influence level."""
        distribution = {"high": 0, "medium": 0, "low": 0}
        
        for user_id, score in self._user_scores.items():
            if score >= 70:
                distribution["high"] += 1
            elif score >= 40:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def size(self) -> int:
        """Get number of users in ranking."""
        return self._ranker.size()


class SocialNetworkAnalyzer:
    """
    Main class that combines all data structures for social network analysis.
    Implements the complete system for identifying influential users.
    """
    
    def __init__(self):
        self.user_registry = UserRegistry()
        self.social_graph = AdjacencyListGraph(directed=True)  # Directed for "follows" relationships
        self.influence_ranker = InfluenceRanker(max_heap=True)
        self._analysis_cache = {}
    
    @track_performance
    def add_user(self, user_id: str, username: str, display_name: str, **kwargs) -> SocialUser:
        """Add a new user to the social network."""
        user = SocialUser(
            user_id=user_id,
            username=username,
            display_name=display_name,
            **kwargs
        )
        
        # Register in hash table
        if self.user_registry.register_user(user):
            # Add to graph
            self.social_graph.add_vertex(user_id)
            
            # Add to influence ranking
            self.influence_ranker.add_user(user)
            
            return user
        else:
            raise ValueError(f"User {user_id} already exists")
    
    @track_performance
    def follow_user(self, follower_id: str, followee_id: str) -> bool:
        """Create a follow relationship (directed edge)."""
        follower = self.user_registry.get_user(follower_id)
        followee = self.user_registry.get_user(followee_id)
        
        if not follower or not followee:
            return False
        
        # Add directed edge in graph
        self.social_graph.add_edge(follower_id, followee_id)
        
        # Update metrics
        follower.following_count += 1
        followee.follower_count += 1
        
        # Update influence rankings
        self.influence_ranker.update_user_influence(follower)
        self.influence_ranker.update_user_influence(followee)
        
        # Clear analysis cache
        self._analysis_cache.clear()
        
        return True
    
    @track_performance
    def calculate_centrality_metrics(self) -> Dict[str, float]:
        """Calculate degree centrality for all users."""
        centralities = {}
        total_users = self.social_graph.vertex_count()
        
        if total_users <= 1:
            return centralities
        
        for user_id in self.social_graph.get_vertices():
            # In-degree (followers) + out-degree (following)
            in_degree = len(list(self.social_graph.get_incoming_edges(user_id)))
            out_degree = len(list(self.social_graph.get_neighbors(user_id)))
            
            # Normalize by max possible connections
            centrality = (in_degree + out_degree) / (2 * (total_users - 1))
            centralities[user_id] = centrality
            
            # Update user object
            user = self.user_registry.get_user(user_id)
            if user:
                user.degree_centrality = centrality
        
        return centralities
    
    @track_performance
    def find_influential_users(self, count: int = 10) -> List[Dict]:
        """
        Find top influential users with detailed metrics.
        
        Args:
            count: Number of top users to return
            
        Returns:
            List of user influence data
        """
        # Update centrality metrics
        self.calculate_centrality_metrics()
        
        # Get top influencers
        top_users = self.influence_ranker.get_top_influencers(count)
        
        result = []
        for user in top_users:
            user_data = {
                'user_id': user.user_id,
                'username': user.username,
                'display_name': user.display_name,
                'followers': user.follower_count,
                'following': user.following_count,
                'influence_score': user.influence_score,
                'degree_centrality': user.degree_centrality,
                'verified': user.verified
            }
            result.append(user_data)
        
        return result
    
    def get_network_stats(self) -> Dict[str, any]:
        """Get overall network statistics."""
        return {
            'total_users': self.user_registry.user_count(),
            'total_connections': self.social_graph.edge_count(),
            'influence_distribution': self.influence_ranker.get_influence_distribution(),
            'avg_connections': self.social_graph.edge_count() / max(1, self.user_registry.user_count())
        } 