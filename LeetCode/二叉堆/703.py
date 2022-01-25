

from typing import List
import heapq


# 703. 数据流中的第 K 大元素
class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        class SmallHeap:
            def __init__(self, k):
                self.k = k
                self.small_heap = []

            def add(self, val):
                if len(self.small_heap) == k:
                    top_val = heapq.heappop(self.small_heap)
                    heapq.heappush(self.small_heap, max(val, top_val))
                else:
                    heapq.heappush(self.small_heap, val)
                return self.small_heap[0]

        self.s_heap = SmallHeap(k)
        for n in nums:
            self.s_heap.add(n)

    def add(self, val: int) -> int:
        return self.s_heap.add(val)





# Your KthLargest object will be instantiated and called as such:
# obj = KthLargest(k, nums)
# param_1 = obj.add(val)