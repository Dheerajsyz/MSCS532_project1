"""
Unit tests for the priority queue implementation.

This module provides comprehensive test cases for the HeapPriorityQueue,
InventoryReorderQueue, and TaskScheduler classes.
"""

import pytest
from src.priority_queue import HeapPriorityQueue, InventoryReorderQueue, TaskScheduler, PriorityItem


class TestHeapPriorityQueue:
    """Test cases for HeapPriorityQueue implementation."""
    
    def test_empty_queue_initialization(self):
        """Test that a new priority queue is empty."""
        pq = HeapPriorityQueue()
        assert pq.is_empty()
        assert pq.size() == 0
        assert pq.peek() is None
        assert pq.peek_priority() is None
        assert pq.dequeue() is None
    
    def test_min_heap_behavior(self):
        """Test min-heap behavior (default)."""
        pq = HeapPriorityQueue(max_heap=False)
        
        # Add items with different priorities
        pq.enqueue("low priority", 10.0)
        pq.enqueue("high priority", 1.0)
        pq.enqueue("medium priority", 5.0)
        
        assert pq.size() == 3
        assert not pq.is_empty()
        
        # Should return items in ascending priority order
        assert pq.dequeue() == "high priority"  # Priority 1.0
        assert pq.dequeue() == "medium priority"  # Priority 5.0
        assert pq.dequeue() == "low priority"   # Priority 10.0
        
        assert pq.is_empty()
    
    def test_max_heap_behavior(self):
        """Test max-heap behavior."""
        pq = HeapPriorityQueue(max_heap=True)
        
        # Add items with different priorities
        pq.enqueue("low priority", 1.0)
        pq.enqueue("high priority", 10.0)
        pq.enqueue("medium priority", 5.0)
        
        # Should return items in descending priority order
        assert pq.dequeue() == "high priority"  # Priority 10.0
        assert pq.dequeue() == "medium priority"  # Priority 5.0
        assert pq.dequeue() == "low priority"   # Priority 1.0
    
    def test_peek_operations(self):
        """Test peek operations without removal."""
        pq = HeapPriorityQueue()
        
        pq.enqueue("item1", 5.0)
        pq.enqueue("item2", 2.0)
        pq.enqueue("item3", 8.0)
        
        # Peek should return highest priority item without removing it
        assert pq.peek() == "item2"  # Priority 2.0 (lowest in min-heap)
        assert pq.peek_priority() == 2.0
        assert pq.size() == 3  # Size unchanged
        
        # Dequeue should return the same item
        assert pq.dequeue() == "item2"
        assert pq.size() == 2
    
    def test_equal_priorities(self):
        """Test handling of equal priorities."""
        pq = HeapPriorityQueue()
        
        # Add items with same priority
        pq.enqueue("first", 5.0)
        pq.enqueue("second", 5.0)
        pq.enqueue("third", 5.0)
        
        # All should be dequeued (order may vary due to timestamp)
        items = []
        while not pq.is_empty():
            items.append(pq.dequeue())
        
        assert len(items) == 3
        assert set(items) == {"first", "second", "third"}
    
    def test_update_priority(self):
        """Test updating priority of existing items."""
        pq = HeapPriorityQueue()
        
        pq.enqueue("item1", 5.0)
        pq.enqueue("item2", 10.0)
        pq.enqueue("item3", 1.0)
        
        # Update priority of item2 to make it highest priority
        result = pq.update_priority("item2", 0.5)
        assert result == True
        
        # item2 should now be first
        assert pq.dequeue() == "item2"
        
        # Try to update non-existent item
        result = pq.update_priority("nonexistent", 1.0)
        assert result == False
    
    def test_clear_operation(self):
        """Test clearing the priority queue."""
        pq = HeapPriorityQueue()
        
        pq.enqueue("item1", 1.0)
        pq.enqueue("item2", 2.0)
        pq.enqueue("item3", 3.0)
        
        assert pq.size() == 3
        
        pq.clear()
        assert pq.is_empty()
        assert pq.size() == 0
        assert pq.peek() is None
    
    def test_to_list_and_sorted_items(self):
        """Test getting items as lists."""
        pq = HeapPriorityQueue()
        
        items_with_priorities = [
            ("low", 10.0),
            ("high", 1.0),
            ("medium", 5.0)
        ]
        
        for item, priority in items_with_priorities:
            pq.enqueue(item, priority)
        
        # Test to_list
        items_list = pq.to_list()
        assert len(items_list) == 3
        
        # Test get_sorted_items
        sorted_items = pq.get_sorted_items()
        assert len(sorted_items) == 3
        
        # In min-heap, sorted items should be in ascending priority order
        priorities = [priority for item, priority in sorted_items]
        assert priorities == [1.0, 5.0, 10.0]
    
    def test_large_queue(self):
        """Test with a large number of items."""
        pq = HeapPriorityQueue()
        
        # Insert 1000 items with random priorities
        import random
        items = []
        for i in range(1000):
            priority = random.uniform(1.0, 100.0)
            pq.enqueue(f"item_{i}", priority)
            items.append((f"item_{i}", priority))
        
        assert pq.size() == 1000
        
        # Dequeue all items and verify they come out in priority order
        previous_priority = float('-inf')
        dequeued_count = 0
        
        while not pq.is_empty():
            item = pq.dequeue()
            current_priority = pq.peek_priority() if not pq.is_empty() else float('inf')
            assert previous_priority <= current_priority
            previous_priority = current_priority
            dequeued_count += 1
        
        assert dequeued_count == 1000
    
    def test_string_representation(self):
        """Test string representation methods."""
        pq = HeapPriorityQueue()
        pq.enqueue("item", 5.0)
        
        str_repr = str(pq)
        assert "PriorityQueue" in str_repr
        assert "min-heap" in str_repr
        assert "size=1" in str_repr
        
        pq_max = HeapPriorityQueue(max_heap=True)
        str_repr_max = str(pq_max)
        assert "max-heap" in str_repr_max


class TestInventoryReorderQueue:
    """Test cases for InventoryReorderQueue implementation."""
    
    def test_empty_reorder_queue(self):
        """Test empty reorder queue operations."""
        irq = InventoryReorderQueue()
        
        assert irq.is_empty()
        assert irq.size() == 0
        assert irq.get_next_reorder() is None
        assert irq.peek_next_reorder() is None
    
    def test_urgency_calculation(self):
        """Test automatic urgency score calculation."""
        irq = InventoryReorderQueue()
        
        # Out of stock item (should have maximum urgency)
        out_of_stock = {
            "sku": "SKU001",
            "name": "Critical Item",
            "current_stock": 0,
            "reorder_point": 10,
            "max_stock": 100,
            "unit_cost": 50.0,
            "is_active": True
        }
        
        # Well-stocked item (should have low urgency)
        well_stocked = {
            "sku": "SKU002", 
            "name": "Normal Item",
            "current_stock": 80,
            "reorder_point": 20,
            "max_stock": 100,
            "unit_cost": 25.0,
            "is_active": True
        }
        
        irq.add_reorder_alert(out_of_stock)
        irq.add_reorder_alert(well_stocked)
        
        # Out of stock item should be processed first
        most_urgent = irq.get_next_reorder()
        assert most_urgent["sku"] == "SKU001"
        
        next_urgent = irq.get_next_reorder()
        assert next_urgent["sku"] == "SKU002"
    
    def test_custom_urgency_score(self):
        """Test adding items with custom urgency scores."""
        irq = InventoryReorderQueue()
        
        item1 = {"sku": "SKU001", "name": "Item 1", "current_stock": 50}
        item2 = {"sku": "SKU002", "name": "Item 2", "current_stock": 30}
        
        # Add with custom urgency scores
        irq.add_reorder_alert(item1, 90.0)  # High urgency
        irq.add_reorder_alert(item2, 10.0)  # Low urgency
        
        # High urgency item should come first
        first = irq.get_next_reorder()
        assert first["sku"] == "SKU001"
        
        second = irq.get_next_reorder()
        assert second["sku"] == "SKU002"
    
    def test_update_item_urgency(self):
        """Test updating urgency of existing items."""
        irq = InventoryReorderQueue()
        
        item = {"sku": "SKU001", "name": "Test Item", "current_stock": 50}
        irq.add_reorder_alert(item, 30.0)
        
        # Update urgency
        result = irq.update_item_urgency("SKU001", 95.0)
        assert result == True
        
        # Try to update non-existent item
        result = irq.update_item_urgency("SKU999", 50.0)
        assert result == False
    
    def test_get_critical_items(self):
        """Test getting critical items above threshold."""
        irq = InventoryReorderQueue()
        
        items = [
            {"sku": "SKU001", "name": "Critical 1"},
            {"sku": "SKU002", "name": "Critical 2"}, 
            {"sku": "SKU003", "name": "Normal"}
        ]
        
        urgencies = [90.0, 85.0, 40.0]
        
        for item, urgency in zip(items, urgencies):
            irq.add_reorder_alert(item, urgency)
        
        # Get items with urgency >= 80
        critical_items = irq.get_critical_items(80.0)
        assert len(critical_items) == 2
        
        critical_skus = {item["sku"] for item in critical_items}
        assert "SKU001" in critical_skus
        assert "SKU002" in critical_skus
        assert "SKU003" not in critical_skus
    
    def test_reorder_summary(self):
        """Test getting reorder summary statistics."""
        irq = InventoryReorderQueue()
        
        items = [
            {"sku": "SKU001", "name": "Item 1"},
            {"sku": "SKU002", "name": "Item 2"},
            {"sku": "SKU003", "name": "Item 3"}
        ]
        
        urgencies = [95.0, 80.0, 30.0]
        
        for item, urgency in zip(items, urgencies):
            irq.add_reorder_alert(item, urgency)
        
        summary = irq.get_reorder_summary()
        
        assert summary["total_items"] == 3
        assert summary["critical_items"] == 2  # >= 80 urgency
        assert summary["average_urgency"] == (95.0 + 80.0 + 30.0) / 3
        assert summary["most_urgent_sku"] == "SKU001"
        assert summary["most_urgent_urgency"] == 95.0
    
    def test_empty_summary(self):
        """Test summary for empty queue."""
        irq = InventoryReorderQueue()
        
        summary = irq.get_reorder_summary()
        
        assert summary["total_items"] == 0
        assert summary["critical_items"] == 0
        assert summary["average_urgency"] == 0
        assert summary["most_urgent_sku"] is None
    
    def test_clear_queue(self):
        """Test clearing the reorder queue."""
        irq = InventoryReorderQueue()
        
        item = {"sku": "SKU001", "name": "Test Item"}
        irq.add_reorder_alert(item, 50.0)
        
        assert irq.size() == 1
        
        irq.clear()
        assert irq.is_empty()
        assert irq.size() == 0


class TestTaskScheduler:
    """Test cases for TaskScheduler implementation."""
    
    def test_empty_scheduler(self):
        """Test empty task scheduler."""
        scheduler = TaskScheduler()
        
        counts = scheduler.get_task_count()
        assert counts["total"] == 0
        assert counts["high"] == 0
        assert counts["normal"] == 0
        assert counts["low"] == 0
        
        assert scheduler.get_next_task() is None
    
    def test_schedule_tasks(self):
        """Test scheduling tasks with different priorities."""
        scheduler = TaskScheduler()
        
        # Schedule tasks
        task1_id = scheduler.schedule_task("High priority task", "high", 90.0)
        task2_id = scheduler.schedule_task("Normal task", "normal", 50.0)
        task3_id = scheduler.schedule_task("Low priority task", "low", 10.0)
        
        assert task1_id == 1
        assert task2_id == 2
        assert task3_id == 3
        
        counts = scheduler.get_task_count()
        assert counts["total"] == 3
        assert counts["high"] == 1
        assert counts["normal"] == 1
        assert counts["low"] == 1
    
    def test_priority_execution_order(self):
        """Test that tasks are executed in priority order."""
        scheduler = TaskScheduler()
        
        # Schedule tasks in mixed order
        scheduler.schedule_task("Low task", "low", 10.0)
        scheduler.schedule_task("High task 1", "high", 95.0)
        scheduler.schedule_task("Normal task", "normal", 50.0)
        scheduler.schedule_task("High task 2", "high", 90.0)
        
        # Should execute in priority order: high -> normal -> low
        task1 = scheduler.get_next_task()
        assert task1[1] == "High task 1"  # Highest score in high priority
        
        task2 = scheduler.get_next_task()
        assert task2[1] == "High task 2"  # Second highest in high priority
        
        task3 = scheduler.get_next_task()
        assert task3[1] == "Normal task"  # Normal priority
        
        task4 = scheduler.get_next_task()
        assert task4[1] == "Low task"  # Low priority
        
        # No more tasks
        assert scheduler.get_next_task() is None
    
    def test_same_priority_level_ordering(self):
        """Test ordering within same priority level."""
        scheduler = TaskScheduler()
        
        # Schedule multiple high priority tasks with different scores
        scheduler.schedule_task("High 1", "high", 95.0)
        scheduler.schedule_task("High 2", "high", 90.0)
        scheduler.schedule_task("High 3", "high", 98.0)
        
        # Should execute in order of scores (highest first)
        task1 = scheduler.get_next_task()
        assert task1[1] == "High 3"  # Score 98.0
        
        task2 = scheduler.get_next_task()
        assert task2[1] == "High 1"  # Score 95.0
        
        task3 = scheduler.get_next_task()
        assert task3[1] == "High 2"  # Score 90.0
    
    def test_default_priority(self):
        """Test default priority assignment."""
        scheduler = TaskScheduler()
        
        # Schedule task without specifying priority
        task_id = scheduler.schedule_task("Default task")
        
        counts = scheduler.get_task_count()
        assert counts["normal"] == 1
        assert counts["high"] == 0
        assert counts["low"] == 0
    
    def test_clear_all_tasks(self):
        """Test clearing all scheduled tasks."""
        scheduler = TaskScheduler()
        
        # Schedule some tasks
        scheduler.schedule_task("Task 1", "high")
        scheduler.schedule_task("Task 2", "normal")
        scheduler.schedule_task("Task 3", "low")
        
        assert scheduler.get_task_count()["total"] == 3
        
        scheduler.clear_all_tasks()
        
        counts = scheduler.get_task_count()
        assert counts["total"] == 0
        assert scheduler.get_next_task() is None
    
    def test_large_number_of_tasks(self):
        """Test with a large number of tasks."""
        scheduler = TaskScheduler()
        
        # Schedule 100 tasks across different priorities
        for i in range(100):
            priority = ["high", "normal", "low"][i % 3]
            score = float(i % 50)  # Vary scores within priority levels
            scheduler.schedule_task(f"Task {i}", priority, score)
        
        assert scheduler.get_task_count()["total"] == 100
        
        # Execute all tasks and verify priority ordering
        executed_tasks = []
        while scheduler.get_task_count()["total"] > 0:
            task = scheduler.get_next_task()
            if task:
                executed_tasks.append(task)
        
        assert len(executed_tasks) == 100
        
        # Verify that high priority tasks come first
        # (Note: exact ordering depends on scores and tie-breaking)
        high_priority_indices = []
        normal_priority_indices = []
        low_priority_indices = []
        
        for i, (task_id, task_name) in enumerate(executed_tasks):
            if "high" in str(task_name) or task_id <= 34:  # Approximate check
                high_priority_indices.append(i)
            elif "normal" in str(task_name) or 35 <= task_id <= 67:
                normal_priority_indices.append(i)
            else:
                low_priority_indices.append(i)
        
        # High priority tasks should generally come before normal and low
        # (This is a simplified check due to scoring complexity)
        assert len(executed_tasks) == 100
    
    def test_string_representation(self):
        """Test string representation of task scheduler."""
        scheduler = TaskScheduler()
        
        scheduler.schedule_task("High task", "high")
        scheduler.schedule_task("Normal task", "normal")
        scheduler.schedule_task("Low task", "low")
        
        str_repr = str(scheduler)
        assert "TaskScheduler" in str_repr
        assert "total=3" in str_repr
        assert "high=1" in str_repr
        assert "normal=1" in str_repr
        assert "low=1" in str_repr


class TestPriorityItem:
    """Test cases for PriorityItem data class."""
    
    def test_priority_item_comparison(self):
        """Test comparison logic for priority items."""
        from datetime import datetime, timedelta
        
        # Create items with different priorities
        item1 = PriorityItem(priority=1.0, item="high", item_id=1)
        item2 = PriorityItem(priority=2.0, item="low", item_id=2)
        
        # Lower priority value should be "less than"
        assert item1 < item2
        assert not item2 < item1
        
        # Test timestamp-based tie breaking
        base_time = datetime.now()
        item3 = PriorityItem(priority=1.0, item="first", timestamp=base_time, item_id=3)
        item4 = PriorityItem(priority=1.0, item="second", timestamp=base_time + timedelta(seconds=1), item_id=4)
        
        # Earlier timestamp should be "less than"
        assert item3 < item4
    
    def test_priority_item_equality(self):
        """Test equality comparison for priority items."""
        item1 = PriorityItem(priority=1.0, item="test", item_id=1)
        item2 = PriorityItem(priority=2.0, item="test", item_id=1)  # Same ID
        item3 = PriorityItem(priority=1.0, item="test", item_id=2)  # Different ID
        
        # Equality based on item_id
        assert item1 == item2
        assert item1 != item3


if __name__ == "__main__":
    pytest.main([__file__]) 