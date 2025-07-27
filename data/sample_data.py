"""
Sample dataset generators for testing data structures.

This module provides functions to generate various types of test data
for different application scenarios.
"""

import random
import string
from typing import List, Dict, Tuple, Set
import json


def generate_web_pages(num_pages: int) -> List[Dict[str, str]]:
    """
    Generate sample web page data for search engine testing.
    
    Args:
        num_pages: Number of web pages to generate
        
    Returns:
        List of dictionaries representing web pages
    """
    pages = []
    
    # Sample content themes
    themes = [
        "technology", "science", "health", "sports", "entertainment",
        "education", "business", "travel", "food", "lifestyle"
    ]
    
    # Sample words for content generation
    tech_words = [
        "algorithm", "database", "network", "security", "software", "hardware",
        "programming", "coding", "development", "framework", "library", "API",
        "server", "client", "application", "system", "cloud", "computing",
        "artificial", "intelligence", "machine", "learning", "data", "analytics"
    ]
    science_words = [
        "research", "experiment", "theory", "discovery", "analysis", "hypothesis",
        "methodology", "observation", "evidence", "conclusion", "study", "findings",
        "investigation", "scientific", "empirical", "statistical", "quantitative",
        "qualitative", "peer", "review", "publication", "journal", "conference"
    ]
    general_words = [
        "information", "system", "process", "method", "solution", "development",
        "implementation", "optimization", "performance", "efficiency", "quality",
        "management", "strategy", "planning", "execution", "evaluation", "assessment",
        "improvement", "innovation", "technology", "digital", "modern", "advanced"
    ]
    
    for i in range(num_pages):
        theme = random.choice(themes)
        
        # Generate URL
        url = f"https://example.com/{theme}/page_{i}"
        
        # Generate title
        title_words = random.sample(tech_words + science_words + general_words, 3)
        title = " ".join(title_words).title()
        
        # Generate content
        all_words = tech_words + science_words + general_words
        word_count = min(random.randint(20, 50), len(all_words))
        content_words = random.sample(all_words, word_count)
        content = " ".join(content_words)
        
        pages.append({
            "id": i,
            "url": url,
            "title": title,
            "content": content,
            "theme": theme,
            "word_count": len(content_words)
        })
    
    return pages


def generate_social_network(num_users: int, avg_connections: int = 5) -> Dict[str, List[str]]:
    """
    Generate a social network graph.
    
    Args:
        num_users: Number of users in the network
        avg_connections: Average number of connections per user
        
    Returns:
        Dictionary representing adjacency list of the social network
    """
    # Generate user IDs
    users = [f"user_{i:04d}" for i in range(num_users)]
    
    # Initialize adjacency list
    network = {user: [] for user in users}
    
    # Add connections
    for user in users:
        num_connections = random.randint(1, avg_connections * 2)
        potential_friends = [u for u in users if u != user and u not in network[user]]
        
        if potential_friends:
            friends = random.sample(
                potential_friends, 
                min(num_connections, len(potential_friends))
            )
            
            for friend in friends:
                if friend not in network[user]:
                    network[user].append(friend)
                if user not in network[friend]:
                    network[friend].append(user)
    
    return network


def generate_product_catalog(num_products: int) -> List[Dict[str, any]]:
    """
    Generate an e-commerce product catalog.
    
    Args:
        num_products: Number of products to generate
        
    Returns:
        List of product dictionaries
    """
    categories = [
        "Electronics", "Books", "Clothing", "Home & Garden", 
        "Sports & Outdoors", "Health & Beauty", "Toys & Games"
    ]
    
    brands = [
        "TechCorp", "GlobalBrand", "QualityMade", "InnovateCo",
        "PremiumLine", "ValueChoice", "TrustedName"
    ]
    
    products = []
    
    for i in range(num_products):
        category = random.choice(categories)
        brand = random.choice(brands)
        
        # Generate product attributes
        product = {
            "id": i,
            "name": f"{brand} {category} Product {i}",
            "category": category,
            "brand": brand,
            "price": round(random.uniform(10.0, 500.0), 2),
            "rating": round(random.uniform(1.0, 5.0), 1),
            "num_reviews": random.randint(0, 1000),
            "in_stock": random.choice([True, False]),
            "stock_quantity": random.randint(0, 100) if random.choice([True, False]) else 0,
            "description": f"High-quality {category.lower()} product from {brand}",
            "tags": random.sample(
                ["popular", "sale", "new", "premium", "eco-friendly", "bestseller"], 
                random.randint(1, 3)
            )
        }
        
        products.append(product)
    
    return products


def generate_user_behavior_data(num_users: int, num_products: int) -> List[Dict[str, any]]:
    """
    Generate user behavior data for recommendation systems.
    
    Args:
        num_users: Number of users
        num_products: Number of products
        
    Returns:
        List of user interaction records
    """
    interactions = []
    
    for user_id in range(num_users):
        # Each user interacts with 5-20 products
        num_interactions = random.randint(5, 20)
        
        for _ in range(num_interactions):
            interaction = {
                "user_id": user_id,
                "product_id": random.randint(0, num_products - 1),
                "interaction_type": random.choice(["view", "like", "purchase", "add_to_cart"]),
                "rating": random.randint(1, 5) if random.choice([True, False]) else None,
                "timestamp": f"2025-07-{random.randint(1, 31):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00"
            }
            interactions.append(interaction)
    
    return interactions


def generate_inventory_data(num_items: int) -> List[Dict[str, any]]:
    """
    Generate inventory management data.
    
    Args:
        num_items: Number of inventory items
        
    Returns:
        List of inventory item dictionaries
    """
    categories = [
        "Raw Materials", "Finished Goods", "Work in Progress", 
        "Maintenance Items", "Office Supplies"
    ]
    
    suppliers = [
        "Global Supply Co", "Quality Materials Inc", "Reliable Vendors LLC",
        "Premier Suppliers", "Trusted Partners Corp"
    ]
    
    items = []
    
    for i in range(num_items):
        item = {
            "sku": f"SKU-{i:06d}",
            "name": f"Item {i}",
            "category": random.choice(categories),
            "supplier": random.choice(suppliers),
            "unit_cost": round(random.uniform(1.0, 100.0), 2),
            "current_stock": random.randint(0, 1000),
            "reorder_point": random.randint(10, 100),
            "max_stock": random.randint(500, 2000),
            "location": f"Warehouse-{random.randint(1, 5)}-Shelf-{random.randint(1, 100)}",
            "last_updated": f"2025-07-{random.randint(1, 31):02d}",
            "is_active": random.choice([True, False])
        }
        
        items.append(item)
    
    return items


def save_sample_data(filename: str, data: any) -> None:
    """
    Save sample data to a JSON file.
    
    Args:
        filename: Output filename
        data: Data to save
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Sample data saved to {filename}")


def generate_search_queries(num_queries: int) -> List[str]:
    """
    Generate sample search queries.
    
    Args:
        num_queries: Number of queries to generate
        
    Returns:
        List of search query strings
    """
    query_templates = [
        "how to {} {}",
        "best {} for {}",
        "what is {} {}",
        "why {} {}",
        "where to find {}",
        "{} tutorial",
        "{} guide",
        "{} tips",
        "{} examples"
    ]
    
    topics = [
        "programming", "data structures", "algorithms", "database",
        "machine learning", "web development", "mobile apps",
        "cybersecurity", "cloud computing", "artificial intelligence"
    ]
    
    modifiers = [
        "beginners", "advanced", "professional", "students",
        "developers", "engineers", "researchers", "experts"
    ]
    
    queries = []
    
    for _ in range(num_queries):
        template = random.choice(query_templates)
        topic = random.choice(topics)
        modifier = random.choice(modifiers)
        
        if "{}" in template:
            if template.count("{}") == 2:
                query = template.format(topic, modifier)
            else:
                query = template.format(topic)
        else:
            query = template
        
        queries.append(query)
    
    return queries


if __name__ == "__main__":
    # Generate sample data for different applications
    print("Generating sample data...")
    
    # Web pages for search engine
    web_pages = generate_web_pages(100)
    save_sample_data("web_pages.json", web_pages)
    
    # Social network
    social_network = generate_social_network(50, 5)
    save_sample_data("social_network.json", social_network)
    
    # Product catalog
    products = generate_product_catalog(200)
    save_sample_data("products.json", products)
    
    # User behavior data
    user_behavior = generate_user_behavior_data(100, 200)
    save_sample_data("user_behavior.json", user_behavior)
    
    # Inventory data
    inventory = generate_inventory_data(150)
    save_sample_data("inventory.json", inventory)
    
    # Search queries
    queries = generate_search_queries(50)
    save_sample_data("search_queries.json", queries)
    
    print("Sample data generation complete!")
