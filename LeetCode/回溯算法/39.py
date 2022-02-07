# 39. 组合总和
from typing import List


class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result_list = []
        pre_list = []
        self.getSum(candidates, result_list, target, pre_list)
        return result_list


    def getSum(self, candidates, result_list, target, pre_list):
        if target == 0:
            result_list.append(pre_list):
            return 

        for n in candidates:
            if target - n >= 0:
                tmp_list = pre_list[::]
                tmp_list.append(n)
                self.getSum(cand_dict, result_list, target-n, pre_list)
        return 