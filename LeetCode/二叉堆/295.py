

import heapq
# 295.数据流的中位数
class MedianFinder:
    def __init__(self):
        class SmallHeap:
            def __init__(self):
                self.small_heap = []

            def add(self, n):
                heapq.heappush(self.small_heap, n)

            def getTop(self):
                if self.num() > 0:
                    return self.small_heap[0]
                else:
                    return None

            def pop(self):
                if self.num() > 0:
                    return heapq.heappop(self.small_heap)
                else:
                    return None
            
            def num(self):
                return len(self.small_heap)

        class BigHeap:
            def __init__(self):
                self.big_heap = []
            
            def add(self, n):
                heapq.heappush(self.big_heap, n * (-1))

            def getTop(self):
                if self.num() > 0:
                    return self.big_heap[0] * -1
                else:
                    return None

            def pop(self):
                if self.num() > 0:
                    return heapq.heappop(self.big_heap) * (-1)
                else:
                    return None

            def num(self):
                return len(self.big_heap)
            
        self.s_heap = SmallHeap()
        self.b_heap = BigHeap()

    def addNum(self, num: int) -> None:
        small_top, big_top = self.s_heap.pop(), self.b_heap.pop()
        small_count, big_count = self.s_heap.num(), self.b_heap.num()
        sorted_list = []
        if small_top is not None:
            sorted_list.append(small_top)
        if big_top is not None:
            sorted_list.append(big_top)
        sorted_list.append(num)
        sorted_list = sorted(sorted_list)
        if small_count > big_count:
            if len(sorted_list) == 3:
                self.s_heap.add(sorted_list[-1])
                self.b_heap.add(sorted_list[0])
                self.b_heap.add(sorted_list[1])
            else:
                self.s_heap.add(sorted_list[1])
                self.b_heap.add(sorted_list[0])
        elif small_count < big_count:
            if len(sorted_list) == 3:
                self.s_heap.add(sorted_list[-2])
                self.s_heap.add(sorted_list[-1])
                self.b_heap.add(sorted_list[0])
            else:
                self.s_heap.add(sorted_list[1])
                self.b_heap.add(sorted_list[0])
        elif small_count == big_count:
            if len(sorted_list) == 1:
                self.s_heap.add(sorted_list[0])
            elif len(sorted_list) == 2:
                self.s_heap.add(sorted_list[1])
                self.b_heap.add(sorted_list[0])
            else:
                self.s_heap.add(sorted_list[-2])
                self.s_heap.add(sorted_list[-1])
                self.b_heap.add(sorted_list[0])
            
    def findMedian(self) -> float:
        small_count, big_count = self.s_heap.num(), self.b_heap.num()
        small_top, big_top = self.s_heap.getTop(), self.b_heap.getTop()
        if big_count > small_count:
            return big_top
        elif big_count < small_count:
            return small_top
        elif big_count == small_count:
            if small_count == 0:
                return None
            else:
                return (small_top + big_top)/2.0


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()


if __name__ == "__main__":
    sol = MedianFinder