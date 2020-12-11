"""This file contain the class of PriorityQueue which is specially designed for data system."""

from typing import Any, List, Tuple, Optional
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

    def get_queue(self) -> List[Tuple]:
        """Return a priority queue with tuple(date, float)"""
        return self._items

    def get_list(self) -> list:
        """Return the list that containing the wanted data."""
        # ACCUMULATOR:
        list_so_far = []

        for item in self._items:
            list_so_far.append(item[1])

        return list_so_far

    def access_data(self, date: datetime.date) -> Any:
        """Return the data corresponding to the given date."""
        for item in self._items:
            if item[0] == date:
                return item[1]
            else:
                return None


class FireEvent:
    """A class that represents a single wild fire event

    Instance Attributes:
     - fod_id: Global unique identifier(FOD_ID) of the fire event           'FOD_ID'
     - date: the date that this fire event was spotted and reported.    'DISCOVERY_DOY '
     - size: Estimate of acres within the final perimeter of the fire.  'FIRE_SIZE'
     - duration: Time of day that the fire was declared contained      'CONT_TIME'
            or otherwise controlled (hhmm where hh=hour, mm=minutes). # TODO: change the unit
     - year: Calendar year in which the fire was discovered or confirmed to exist. 'FIRE_YEAR'
     - times: the times of event, default 1 and will be changed when merging data.
     """
    fod_id: int
    date: datetime.date
    size: float
    duration: int
    times: int

    def __init__(self, fod_id: int, date: datetime.date, size: float, duration: int) -> None:
        """Initialize a a FireEvent class """
        self.fod_id = fod_id
        self.date = date
        self.size = size
        self.duration = duration
        self.times = 1


class ClimateQueue(PriorityQueue):
    """A queue of items that can be dequeued in priority order.

    When removing an item from the queue, the highest-priority item is the one
    that is removed.
    """
    # Private Instance Attributes:
    #   - _items: a list of the items in this priority queue
    _items: List[Tuple[datetime.date, Any]]

    def __init__(self) -> None:
        """Initialize a new and empty priority queue."""
        PriorityQueue.__init__(self)
        self._items = []

    def enqueue(self, priority: datetime.date, item: Any) -> None:
        """Add the given item with the given priority(date) to this priority queue.

        >>> me = ClimateQueue()
        >>> me.enqueue(datetime.date(2020, 1, 1), 20)
        >>> me.enqueue(datetime.date(2020, 1, 1), 40)
        >>> me.enqueue(datetime.date(2020, 1, 1), 30)
        >>> me.enqueue(datetime.date(2020, 2, 1), 40)
        >>> me._items
        [(datetime.date(2020, 1, 1), 30), (datetime.date(2020, 1, 2), 40)]
        """
        i = 0
        while i < len(self._items) and self._items[i][0] < priority:
            # Loop invariant: all items in self._items[0:i]
            # have a lower priority than <priority>.
            i = i + 1

        self._items.insert(i, (priority, item))

    def eliminate_duplicate(self) -> None:
        """Take average of all the duplicate data

        Should be used when all the data were extracted."""
        i = 0
        j = 1
        length = len(self._items)
        dup_length = 0
        flag = True
        while flag:
            if self._items[j][0] == self._items[j - 1][0]:
                j += 1
                dup_length += 1
                flag = j < len(self._items)
            else:
                if (j - i) > 1:
                    self._take_average(i, j)   # TODO: potential bugs.
                length = length - dup_length + 1
                dup_length = 0
                i += 1
                j = j - dup_length + 1
                flag = j < len(self._items)

    def _take_average(self, i, j) -> None:
        """"""
        average = sum(x[1] for x in self._items[i:j - 1]) / (j - i)
        self._items[j - 1] = (self._items[j - 1][0], average)
        for _ in range(i, j - 1):
            self._items.pop(i)


class FireQueue(PriorityQueue):
    """A priority queue designed for fire events.

    When all the data are loaded, the function eliminate_duplicate and complete_timeline should be
    called respectively.

    Instance Attributes:
     - _items: private attribute that store the fire events in form of (datetime.date, FireEvent)
     """
    _items: List[Tuple[datetime.date, FireEvent]]

    def __init__(self) -> None:
        """Initialize the class with empty items."""
        PriorityQueue.__init__(self)
        self._items = []

    def enqueue(self, priority: datetime.date, item: Optional[FireEvent]) -> None:
        """Enqueue"""
        i = 0
        while i < len(self._items) and self._items[i][0] < priority:
            # Loop invariant: all items in self._items[0:i]
            # have a lower priority than <priority>.
            i = i + 1

        self._items.insert(i, (priority, item))

    def eliminate_duplicate(self) -> None:
        """Take average of all the duplicate data

        Should be used when all the data were extracted."""
        i = 0
        j = 1
        length = len(self._items)
        dup_length = 0
        flag = True
        while flag:
            if self._items[j][0] == self._items[j - 1][0]:
                j += 1
                dup_length += 1
                flag = j < len(self._items)
            else:
                if (j - i) > 1:
                    self._take_average(i, j)  # TODO: potential bugs.
                length = length - dup_length + 1
                dup_length = 0
                i += 1
                j = j - dup_length + 1
                flag = j < len(self._items)
        """
        i = 0
        j = 1
        while j <= len(self._items):
            if self._items[j][0] == self._items[j - 1][0]:
                j += 1
            else:
                if (j - i) > 1:
                    self._take_average(i, j)  # TODO: potential bugs.
                i = j
                j += 1
        """

    def _take_average(self, i, j) -> None:
        """"""
        average_size = sum(x[1].size for x in self._items[i:j - 1]) / (j - i)
        total_times = j - i
        self._items[j - 1][1].size = average_size
        self._items[j - 1][1].times = total_times   # TODO: potential bugs.
        for _ in range(i, j - 1):
            self._items.pop(i)

    def complete_timeline(self) -> None:
        """To make the queue has a complete record of fire events from
         start time to end time inclusive no matter whether there is a fire event"""
        start_year = self._items[0][0].year
        start_month = self._items[0][0].month
        end_year = self._items[-1][0].year
        end_month = self._items[-1][0].month

        months = (end_year - start_year) * 12 + end_month - start_month + 1
        for i in range(months):
            # Month range from 0 to 11 inclusive in one year
            current_year = (i + start_month) // 13 + start_year
            current_month = (i + start_month - 1) % 12 + 1
            # if current_month > 12:
            #    current_month = (current_month - 1) % 12 + 1
            current_date = datetime.date(current_year, current_month, 1)

            dates = self._get_dates()
            if current_date not in dates:
                self.enqueue(current_date, None)

    def _get_dates(self) -> list:
        """Helper function"""
        # ACCUMULATOR:
        dates_so_far = []
        for item in self._items:
            dates_so_far.append(item[0])

        return dates_so_far


class EmptyPriorityQueueError(Exception):
    """Exception raised when calling pop on an empty stack."""

    def __str__(self) -> str:
        """Return a string representation of this error."""
        return 'You called dequeue on an empty priority queue.'


