# 704. 二分查找
from typing import List


class Solution:
    def search(self, nums: List[int], target: int) -> int:
        return self.findN(nums, target, 0, len(nums)-1)

    def findN(self, nums, target, s, e):
        if s > e:
            return -1
        mid = int((s+e)/2)
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            return self.findN(nums, target, s, mid-1)
        elif nums[mid] < target:
            return self.findN(nums, target, mid+1, e)
        

