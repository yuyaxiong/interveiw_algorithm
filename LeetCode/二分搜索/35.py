
# 35. 搜索插入位置
from typing import List


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return self.findN(nums, target, 0, len(nums)-1)

    def findN(self, nums, target, s, e):
        if s > e:
            return s 
        mid = int((s + e)/2)
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            return self.findN( nums,target, s, mid-1)
        elif nums[mid] < target:
            return self.findN(nums, target, mid+1, e)
