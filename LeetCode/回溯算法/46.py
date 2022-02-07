# 46. 全排列
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result, pre_list = [], []
        self.permute_help(nums, result, pre_list)
        return result

    def permute_help(self, nums, result, pre_list):
        if len(nums) == 0:
            result.append(pre_list)
            return 
        for n in nums:
            tmp = pre_list[::]
            tmp.append(n)
            tmp_nums = nums[::]
            tmp_nums.remove(n)
            self.permute_help(tmp_nums, result, tmp)
        return 
