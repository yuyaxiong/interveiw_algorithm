# 78. 子集
from typing import List


class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = []
        self.sub_list(nums, result, [])
        return result

    def sub_list(self, nums, result, pre_list):
        if len(nums) == 0:
            result.append(pre_list[::])
            return 
        result.append(pre_list[::])
        for i, n in enumerate(nums):
            tmp_list = pre_list[::]
            tmp_list.append(n)
            tmp_nums = nums[i+1:]
            self.sub_list(tmp_nums, result, tmp_list)


        
