# 875. 爱吃香蕉的珂珂
from typing import List


class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        max_val = max(piles)
        if max_val <= 0:
            return 0
        return self.minHelp(piles, 1, max_val, h)

    def minHelp(self, piles: List[int], s: int, e: int, h: int):
        if s == e:
            return s
        mid = (s+e)//2
        if self.eatCount(piles, mid) > h:
            return self.minHelp(piles, mid+1, e, h)
        else:
            return self.minHelp(piles, s, mid, h)
        
    def eatCount(self, piles: List[int], k: int):
        counter = 0
        for n in piles:
            if n % k == 0:
                counter += n // k
            else:
                counter += n // k + 1
        return counter