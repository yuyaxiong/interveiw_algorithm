# 1109. 航班预订统计
from typing import List

# 差分数组
class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        delta_seat_list = [0 for _ in range(n+1)]
        for booking in bookings:
            first, last, seats = booking[0], booking[1], booking[2]
            delta_seat_list[first] += seats
            if last+1 < len(delta_seat_list):
                delta_seat_list[last+1] -= seats
        all_seat_list = []
        for delta in delta_seat_list[1:]:
            alls = all_seat_list[-1] + delta  if len(all_seat_list) > 0 else delta
            all_seat_list.append(alls)
        return all_seat_list

