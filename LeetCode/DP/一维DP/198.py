# 198. 打家劫舍
from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        dp = [ 0 for _ in range(len(nums))]
        if len(nums) == 0:
            return 0
        if len(nums) == 1:
            return nums[0]
        dp[0] = nums[0]
        dp[1] = max(dp[0], nums[1])
        if len(nums) == 1:
            return dp[0]
        if len(nums) == 2:
            return dp[1]
        for i in range(2, len(nums)):
            dp[i] = max(nums[i] + dp[i-2], dp[i-1])
        return max(dp)