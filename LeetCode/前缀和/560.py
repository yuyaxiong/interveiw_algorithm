

from typing import List

# 560. 和为 K 的子数组
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        acc_list, acc_dict = self.getAccList(nums)
        base = acc_dict[k] if acc_dict.get(k) is not None else 0
        count = 0
        for n in acc_list[::-1]:
            acc_dict[n] -= 1
            if acc_dict.get(n - k) is not None and acc_dict.get(n-k) > 0:
                count += acc_dict.get(n-k)
        return count + base

    def getAccList(self, nums):
        acc_dict = {}
        acc_list = []
        for n in nums:
            val = acc_list[-1] + n if len(acc_list) > 0 else  n
            acc_list.append(val)
            if acc_dict.get(val) is None:
                acc_dict[val] = 1
            else:
                acc_dict[val] += 1
        return acc_list, acc_dict


class Solution1:
    def subarraySum(self, nums: List[int], k: int) -> int:
        preSumDict = dict()
        preSumDict[0] = 1
        res, sum0_i = 0, 0
        for i in range(len(nums)):
            sum0_i += nums[i]
            sum0_j = sum0_i - k
            if preSumDict.get(sum0_j) is not None:
                res += preSumDict.get(sum0_j)
            if preSumDict.get(sum0_i) is None:
                preSumDict[sum0_i] = 1
            else:
                preSumDict[sum0_i] += 1
        return res