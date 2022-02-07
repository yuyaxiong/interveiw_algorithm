# 494. 目标和
from typing import List


class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        result = 0
        result = self.trac(nums, 0, result, 0, len(nums) , target)
        return result

    def trac(self, nums, base, result, i, n, target):
        if i == n:
            if base == target:
                result += 1
            return result
        result = self.trac(nums, base + nums[i], result, i+1, n, target)
        result = self.trac(nums, base - nums[i], result, i+1, n, target)
        return result

    def calc(self, n_list):
        base = 0
        for i in range(int(len(n_list)/2)):
            symbol, val = n_list[i*2], n_list[i*2 + 1]
            if symbol == '+':
                base += int(val)
            else:
                base -= int(val)
        return base

class Solution1:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return 0
        self.memo = dict()
        return self.dp(nums, 0, target)

    def dp(self, nums, i, rest):
        if len(nums) == i:
            if rest == 0:
                return 1
            return 0
        key = str(i) + "," + str(rest)
        # 自底向上的过程中这个剪枝很关键
        if self.memo.get(key) is not None:
            return self.memo.get(key)
        result = self.dp(nums, i + 1, rest - nums[i]) + self.dp(nums, i+1, rest + nums[i])
        self.memo[key] = result
        return result



def testCase():
    nums = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    target = 0 
    sol = Solution1()
    ret = sol.findTargetSumWays(nums, target) 
    print(ret)  


def testCase1():
    nums = [25,14,16,44,9,22,15,27,23,10,41,25,14,35,28,47,39,26,11,38]
    target = 43
    sol = Solution1()
    ret = sol.findTargetSumWays(nums, target)
    print(ret)

if __name__ == "__main__":
    testCase1()

