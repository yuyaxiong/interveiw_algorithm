# 793. 阶乘函数后 K 个零
import sys
class Solution:
    def preimageSizeFZF(self, k: int) -> int:
        return self.rightBound(k) - self.leftBound(k) + 1

    def trailingZeros(self, n):
        res = 0
        d = n
        while d //5 > 0:
            res += d//5
            d = d//5
        return res

    def leftBound(self, target):
        low, hight = 0 , sys.maxsize
        while low < hight:
            mid = int((low + hight)/2)
            mid_zero = self.trailingZeros(mid)
            if mid_zero < target:
                low = mid+1
            elif mid_zero > target:
                hight = mid
            else:
                hight = mid
        return low

    def rightBound(self, target):
        low, hight = 0, sys.maxsize
        while low < hight:
            mid = int((low+hight)//2)
            mid_zero = self.trailingZeros(mid)
            if mid_zero < target:
                low = mid+1
            elif mid_zero > target:
                hight = mid
            else:
                low = mid+1
        return low-1
