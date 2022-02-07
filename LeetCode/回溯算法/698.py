#698. 划分为k个相等的子集
# leetcode 暴力搜索会超时
from typing import List

class Solution:
    def canPartitionKSubsets(self, nums: List[int], k: int) -> bool:
        if k > len(nums):
            return False
        sum_val = sum(nums)
        if sum_val % k != 0:
            return False
        used_list = [False for _ in nums]
        target = sum_val/k
        return self.backtrack(k, 0, nums, 0, used_list, target)

    def backtrack(self, k, bucket, nums, start, used_list, target):
        status = False
        if k == 0:
            status = True
            return status
        if bucket == target:
            return self.backtrack(k -1, 0, nums, 0, used_list, target)
        for i in range(start, len(nums)):
            if used_list[i]:
                continue
            if nums[i] + bucket > target:
                continue

            used_list[i] = True
            bucket += nums[i]
            if self.backtrack(k, bucket, nums, i+1, used_list, target):
                status = True
                return status
            used_list[i] = False
            bucket -= nums[i]
        return False




def testCase():
    nums = [3,9,4,5,8,8,7,9,3,6,2,10,10,4,10,2]
    k = 10
    sol = Solution()
    ret = sol.canPartitionKSubsets(nums, k)
    print(ret)

def testCase1():
    nums = [10,5,5,4,3,6,6,7,6,8,6,3,4,5,3,7]
    k = 8
    sol = Solution()
    ret = sol.canPartitionKSubsets(nums, k)
    print(ret)

def testCase2():
    nums = [4,3,2,3,5,2,1]
    k = 4
    sol = Solution()
    ret = sol.canPartitionKSubsets(nums, k)
    print(ret)

def testCase3():
    nums = [3522,181,521,515,304,123,2512,312,922,407,146,1932,4037,2646,3871,269]
    k = 5
    sol = Solution()
    ret = sol.canPartitionKSubsets(nums, k)
    print(ret)

if __name__ == "__main__":

    # testCase()
    # testCase1()
    # testCase2()
    testCase3()
