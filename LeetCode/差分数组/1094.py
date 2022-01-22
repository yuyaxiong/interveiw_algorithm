
# 1094. 拼车
from typing import List

# 差值数组
class Solution:
    def carPooling(self, trips: List[List[int]], capacity: int) -> bool:
        max_stop = max([trip[2] for trip in trips])
        delta_cap_list = [0 for _ in range(max_stop+1)]
        for trip in trips:
            cap, start, stop = trip[0], trip[1], trip[2]
            delta_cap_list[start] += cap
            delta_cap_list[stop] -= cap
        cap_list = []
        status = True
        for delta in delta_cap_list:
            cap = cap_list[-1] + delta if len(cap_list) > 0 else delta
            if cap > capacity:
                status = False
                break
            else:
                cap_list.append(cap)
        return status