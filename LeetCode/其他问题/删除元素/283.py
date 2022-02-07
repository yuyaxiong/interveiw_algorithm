# 283. 移动零
from typing import List

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        s_p = 0
        for i, n in enumerate(nums):
            if n != 0:
                nums[s_p] = nums[i]
                s_p += 1
        while s_p < len(nums):
            nums[s_p] = 0
            s_p += 1
        return nums