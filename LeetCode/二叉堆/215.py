
from typing import List
import heapq
# 215.数组中的第K大元素
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        class SmallHeap():
            def __init__(self, k: int):
                self.k = k
                self.small_heap = []

            def add(self, n: int):
                if len(self.small_heap) == k:
                    val = heapq.heappop(self.small_heap)
                    heapq.heappush(self.small_heap, max(val, n))
                else:
                    heapq.heappush(self.small_heap, n)

            def getTop(self):
                return self.small_heap[0] 

        s_heap = SmallHeap(k)
        for n in nums:
            s_heap.add(n)
        print(s_heap.small_heap)
        return s_heap.getTop()

if __name__ == "__main__":
    nums = [3,2,1,5,6,4]
    k = 2
    sol = Solution()
    ret = sol.findKthLargest(nums, k)
    print(ret)


        