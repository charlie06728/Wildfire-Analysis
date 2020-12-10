"""This file contain the class of PriorityQueue which is specially designed for data system."""

from typing import Any, List, Tuple
import datetime


class PriorityQueue:
    """A queue of items that can be dequeued in priority order.

    When removing an item from the queue, the highest-priority item is the one
    that is removed.
    """
    # Private Instance Attributes:
    #   - _items: a list of the items in this priority queue
    _items: List[Tuple[datetime.date, Any]]

    def __init__(self) -> None:
        """Initialize a new and empty priority queue."""
        self._items = []

    def is_empty(self) -> bool:
        """Return whether this priority queue contains no items.
        """
        return self._items == []

    def dequeue(self, date: datetime.date) -> Tuple[datetime.date, float]:
        """Remove and return the item with the highest priority.

        Raise an EmptyPriorityQueueError when the priority queue is empty.
        """
        dequeued = None
        if self.is_empty():
            raise EmptyPriorityQueueError
        else:
            for item in self._items:
                if item[0] == date:
                    index = self._items.index(item)
                    dequeued = self._items.pop(index)
                    
        return dequeued

    def enqueue(self, priority: datetime.date, item: float) -> None:
        """Add the given item with the given priority(date) to this priority queue.
        """
        i = 0
        while i < len(self._items) and self._items[i][0] < priority:
            # Loop invariant: all items in self._items[0:i]
            # have a lower priority than <priority>.
            i = i + 1

        self._items.insert(i, (priority, item))


class EmptyPriorityQueueError(Exception):
    """Exception raised when calling pop on an empty stack."""

    def __str__(self) -> str:
        """Return a string representation of this error."""
        return 'You called dequeue on an empty priority queue.'
