"""
Priority Queue implementation for inventory management and task scheduling.

This module provides a heap-based priority queue optimized for inventory
reorder alerts and priority-based processing systems.
"""

import heapq
from typing import Any, Optional, List, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from src.base import PriorityQueue, track_performance


@dataclass
class PriorityItem:
    """
    Wrapper class for items in the priority queue.
    
    Uses negative priority for max-heap behavior with Python's min-heap.
    """
    priority: float
    item: Any
    timestamp: datetime = field(default_factory=datetime.now)
    item_id: int = field(default=0)
    
    def __lt__(self, other: 'PriorityItem') -> bool:
        """
        Compare items by priority, then by timestamp.
        Lower priority values have higher precedence.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp
    
    def __eq__(self, other: 'PriorityItem') -> bool:
        """Check equality based on item_id."""
        return self.item_id == other.item_id


class HeapPriorityQueue(PriorityQueue[Any]):
    """
    Priority Queue implementation using a binary heap.
    
    Supports both min-heap (default) and max-heap operations.
    Optimized for inventory management and task scheduling.
    """
    
    def __init__(self, max_heap: bool = False):
        """
        Initialize the priority queue.
        
        Args:
            max_heap: If True, highest priority items are returned first.
                     If False, lowest priority items are returned first.
        """
        self._heap: List[PriorityItem] = []
        self._max_heap = max_heap
        self._item_map = {}  # Maps item_id to PriorityItem for quick lookup
        self._counter = 0  # For unique item IDs
    
    def _adjust_priority(self, priority: float) -> float:
        """Adjust priority for max-heap behavior if needed."""
        return -priority if self._max_heap else priority
    
    @track_performance
    def enqueue(self, item: Any, priority: float) -> None:
        """
        Add an item with a given priority.
        
        Args:
            item: The item to add
            priority: Priority value (lower values = higher priority for min-heap)
        """
        adjusted_priority = self._adjust_priority(priority)
        self._counter += 1
        priority_item = PriorityItem(
            priority=adjusted_priority,
            item=item,
            item_id=self._counter
        )
        
        heapq.heappush(self._heap, priority_item)
        self._item_map[self._counter] = priority_item
    
    @track_performance
    def dequeue(self) -> Optional[Any]:
        """
        Remove and return the highest priority item.
        
        Returns:
            The highest priority item, or None if queue is empty
        """
        if not self._heap:
            return None
        
        priority_item = heapq.heappop(self._heap)
        if priority_item.item_id in self._item_map:
            del self._item_map[priority_item.item_id]
        
        return priority_item.item
    
    @track_performance
    def peek(self) -> Optional[Any]:
        """
        Return the highest priority item without removing it.
        
        Returns:
            The highest priority item, or None if queue is empty
        """
        if not self._heap:
            return None
        return self._heap[0].item
    
    def peek_priority(self) -> Optional[float]:
        """
        Return the priority of the highest priority item.
        
        Returns:
            Priority value or None if queue is empty
        """
        if not self._heap:
            return None
        
        priority = self._heap[0].priority
        return -priority if self._max_heap else priority
    
    @track_performance
    def update_priority(self, item: Any, new_priority: float) -> bool:
        """
        Update the priority of an existing item.
        
        Args:
            item: The item to update
            new_priority: New priority value
            
        Returns:
            True if item was found and updated, False otherwise
        """
        # Find the item in the heap
        for i, priority_item in enumerate(self._heap):
            if priority_item.item == item:
                # Remove the old item
                old_priority_item = self._heap[i]
                if old_priority_item.item_id in self._item_map:
                    del self._item_map[old_priority_item.item_id]
                
                # Replace with last item and heapify
                self._heap[i] = self._heap[-1]
                self._heap.pop()
                
                if i < len(self._heap):
                    heapq._siftup(self._heap, i)
                    heapq._siftdown(self._heap, 0, i)
                
                # Add item with new priority
                self.enqueue(item, new_priority)
                return True
        
        return False
    
    def size(self) -> int:
        """Return the number of items in the queue."""
        return len(self._heap)
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._heap) == 0
    
    def clear(self) -> None:
        """Remove all items from the queue."""
        self._heap.clear()
        self._item_map.clear()
        self._counter = 0
    
    def to_list(self) -> List[Tuple[Any, float]]:
        """
        Return all items as a list of (item, priority) tuples.
        
        Returns:
            List of (item, priority) tuples in heap order
        """
        result = []
        for priority_item in self._heap:
            priority = -priority_item.priority if self._max_heap else priority_item.priority
            result.append((priority_item.item, priority))
        return result
    
    def get_sorted_items(self) -> List[Tuple[Any, float]]:
        """
        Return all items sorted by priority.
        
        Returns:
            List of (item, priority) tuples sorted by priority
        """
        items = self.to_list()
        return sorted(items, key=lambda x: x[1], reverse=self._max_heap)
    
    def __str__(self) -> str:
        """String representation of the priority queue."""
        heap_type = "max-heap" if self._max_heap else "min-heap"
        return f"PriorityQueue({heap_type}, size={self.size()})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        heap_type = "max-heap" if self._max_heap else "min-heap"
        return f"HeapPriorityQueue({heap_type}, size={self.size()})"


class InventoryReorderQueue:
    """
    Specialized priority queue for inventory reorder management.
    
    Prioritizes items based on urgency of reorder needs.
    """
    
    def __init__(self):
        """Initialize the inventory reorder queue."""
        self._queue = HeapPriorityQueue(max_heap=True)  # Higher priority = more urgent
        self._items = {}  # Maps SKU to item data
    
    def add_reorder_alert(self, item: dict, urgency_score: float = None) -> None:
        """
        Add an item to the reorder queue.
        
        Args:
            item: Inventory item dictionary
            urgency_score: Custom urgency score (if None, calculated automatically)
        """
        sku = item.get('sku')
        if urgency_score is None:
            urgency_score = self._calculate_urgency(item)
        
        self._queue.enqueue(item, urgency_score)
        self._items[sku] = item
    
    def _calculate_urgency(self, item: dict) -> float:
        """
        Calculate urgency score for an inventory item.
        
        Higher scores indicate more urgent reorder needs.
        
        Args:
            item: Inventory item dictionary
            
        Returns:
            Urgency score (0-100)
        """
        current_stock = item.get('current_stock', 0)
        reorder_point = item.get('reorder_point', 0)
        max_stock = item.get('max_stock', 100)
        unit_cost = item.get('unit_cost', 0)
        is_active = item.get('is_active', True)
        
        # Base urgency based on stock level
        if current_stock <= 0:
            stock_urgency = 100  # Out of stock - maximum urgency
        elif current_stock <= reorder_point:
            # Linear scale from reorder point to 80% urgency
            ratio = 1 - (current_stock / reorder_point)
            stock_urgency = 50 + (ratio * 30)
        else:
            # Below reorder point but not critical
            ratio = current_stock / max_stock
            stock_urgency = max(0, 50 - (ratio * 30))
        
        # Adjust for item cost (expensive items get higher priority)
        cost_factor = min(unit_cost / 100, 1.0)  # Normalize to 0-1
        cost_urgency = cost_factor * 20
        
        # Active items get higher priority
        active_bonus = 10 if is_active else 0
        
        total_urgency = stock_urgency + cost_urgency + active_bonus
        return min(total_urgency, 100)  # Cap at 100
    
    def get_next_reorder(self) -> Optional[dict]:
        """
        Get the next item that needs to be reordered.
        
        Returns:
            Most urgent inventory item or None if queue is empty
        """
        item = self._queue.dequeue()
        if item:
            sku = item.get('sku')
            if sku in self._items:
                del self._items[sku]
        return item
    
    def peek_next_reorder(self) -> Optional[dict]:
        """
        Peek at the next item without removing it.
        
        Returns:
            Most urgent inventory item or None if queue is empty
        """
        return self._queue.peek()
    
    def update_item_urgency(self, sku: str, new_urgency: float) -> bool:
        """
        Update the urgency score for an item.
        
        Args:
            sku: Item SKU
            new_urgency: New urgency score
            
        Returns:
            True if item was found and updated
        """
        if sku not in self._items:
            return False
        
        item = self._items[sku]
        return self._queue.update_priority(item, new_urgency)
    
    def get_critical_items(self, threshold: float = 80.0) -> List[dict]:
        """
        Get all items with urgency above threshold.
        
        Args:
            threshold: Minimum urgency score
            
        Returns:
            List of critical items
        """
        critical_items = []
        for item, urgency in self._queue.to_list():
            if urgency >= threshold:
                critical_items.append(item)
        
        return sorted(critical_items, key=lambda x: self._calculate_urgency(x), reverse=True)
    
    def get_reorder_summary(self) -> dict:
        """
        Get a summary of reorder status.
        
        Returns:
            Dictionary with reorder statistics
        """
        items = self._queue.to_list()
        
        if not items:
            return {
                'total_items': 0,
                'critical_items': 0,
                'average_urgency': 0,
                'most_urgent_sku': None,
                'least_urgent_sku': None
            }
        
        urgencies = [urgency for _, urgency in items]
        critical_count = sum(1 for urgency in urgencies if urgency >= 80)
        
        most_urgent_item = max(items, key=lambda x: x[1])
        least_urgent_item = min(items, key=lambda x: x[1])
        
        return {
            'total_items': len(items),
            'critical_items': critical_count,
            'average_urgency': sum(urgencies) / len(urgencies),
            'most_urgent_sku': most_urgent_item[0].get('sku'),
            'most_urgent_urgency': most_urgent_item[1],
            'least_urgent_sku': least_urgent_item[0].get('sku'),
            'least_urgent_urgency': least_urgent_item[1]
        }
    
    def size(self) -> int:
        """Return the number of items in the queue."""
        return self._queue.size()
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return self._queue.is_empty()
    
    def clear(self) -> None:
        """Clear all items from the queue."""
        self._queue.clear()
        self._items.clear()
    
    def __str__(self) -> str:
        """String representation."""
        return f"InventoryReorderQueue(items={self.size()})"


class TaskScheduler:
    """
    Task scheduling system using priority queues.
    
    Manages different types of tasks with varying priorities.
    """
    
    def __init__(self):
        """Initialize the task scheduler."""
        self._high_priority = HeapPriorityQueue(max_heap=True)
        self._normal_priority = HeapPriorityQueue(max_heap=True)
        self._low_priority = HeapPriorityQueue(max_heap=True)
        self._task_counter = 0
    
    def schedule_task(self, task: Any, priority: str = "normal", score: float = 50.0) -> int:
        """
        Schedule a task with given priority level.
        
        Args:
            task: Task to schedule
            priority: Priority level ("high", "normal", "low")
            score: Fine-grained priority score within the level
            
        Returns:
            Task ID for tracking
        """
        self._task_counter += 1
        
        if priority == "high":
            self._high_priority.enqueue((self._task_counter, task), score)
        elif priority == "low":
            self._low_priority.enqueue((self._task_counter, task), score)
        else:  # normal
            self._normal_priority.enqueue((self._task_counter, task), score)
        
        return self._task_counter
    
    def get_next_task(self) -> Optional[Tuple[int, Any]]:
        """
        Get the next task to execute.
        
        Returns:
            (task_id, task) tuple or None if no tasks available
        """
        # Check high priority first
        if not self._high_priority.is_empty():
            return self._high_priority.dequeue()
        
        # Then normal priority
        if not self._normal_priority.is_empty():
            return self._normal_priority.dequeue()
        
        # Finally low priority
        if not self._low_priority.is_empty():
            return self._low_priority.dequeue()
        
        return None
    
    def get_task_count(self) -> dict:
        """
        Get count of tasks by priority level.
        
        Returns:
            Dictionary with task counts
        """
        return {
            'high': self._high_priority.size(),
            'normal': self._normal_priority.size(),
            'low': self._low_priority.size(),
            'total': (self._high_priority.size() + 
                     self._normal_priority.size() + 
                     self._low_priority.size())
        }
    
    def clear_all_tasks(self) -> None:
        """Clear all scheduled tasks."""
        self._high_priority.clear()
        self._normal_priority.clear()
        self._low_priority.clear()
        self._task_counter = 0
    
    def __str__(self) -> str:
        """String representation."""
        counts = self.get_task_count()
        return f"TaskScheduler(total={counts['total']}, high={counts['high']}, normal={counts['normal']}, low={counts['low']})" 