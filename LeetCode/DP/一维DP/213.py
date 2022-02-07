# 213. 打家劫舍 II
from typing import List


class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        val_1 = self.dp(nums, 1, len(nums)-1)
        val_2 = self.dp(nums, 0, len(nums)-2)
        return max(val_1, val_2)

    def dp(self, nums, start, end):
        if end-start == 0:
            return nums[start]
        dp = [0 for _ in range(len(nums))]
        dp[start] = nums[start]
        dp[start+1] = max(nums[start+1], dp[start])
        for i in range(start+2, end+1):
            dp[i] = max(nums[i] + dp[i-2], dp[i-1])
        return max(dp)



class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        memo1 = [-1 for _ in range(n)]
        memo2 = [-1 for _ in range(n)]
        return max(self.dp(nums, 0, n-2, memo1), self.dp(nums, 1, n-1, memo2))

    def dp(self, nums, start, end, memo):
        if start > end:
            return 0
        if memo[start] != -1:
            return memo[start]
        res = max(self.dp(nums, start+2, end, memo) + nums[start], self.dp(nums, start+1, end, memo))
        memo[start] = res 
        return res 


def testCase():
    nums = [1,2,3,1]
    sol = Solution()
    ret = sol.rob(nums)
    print(ret)

def testCase1():
    nums = [1,2,3]
    sol = Solution()
    ret = sol.rob(nums)
    print(ret)

if __name__ == "__main__":
    testCase()
    testCase1()

