# 45. 跳跃游戏 II
from typing import List


class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        end, farthest = 0, 0
        jump_count = 0
        for i in range(n-1):
            farthest = max(nums[i]+i, farthest)
            if end == i:
                jump_count += 1
                end = farthest
        return jump_count

def testCase():
    nums = [3,2,1]
    sol = Solution()
    ret = sol.jump(nums)
    print(ret)

def testCase1():
    nums = [2,3,1,1,4]
    sol = Solution()
    ret = sol.jump(nums)
    print(ret)

if __name__ == "__main__":
    testCase()
    testCase1()

