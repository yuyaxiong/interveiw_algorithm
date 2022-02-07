# 55. 跳跃游戏
from typing import List


class Solution:
    def canJump(self, nums: List[int]) -> bool:
        farthest = 0
        end = 0
        for i in range(len(nums)-1):
            farthest = max(nums[i] + i, farthest)
            # 碰上0值就直接返回为False
            if farthest <= i:
                return False
            if end == i:
                end = farthest
        return farthest >= len(nums) - 1