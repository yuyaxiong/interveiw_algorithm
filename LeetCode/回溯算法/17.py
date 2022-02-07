# 17. 电话号码的字母组合
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        alpha_dict = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno","7":"pqrs","8":"tuv","9":"wxyz"}
        result_list = []
        if len(digits) == 0:
            return []
        for alpha in alpha_dict.get(digits[0]):
            self.letterHelp(alpha_dict, alpha, result_list, digits[1:])
        return result_list

    def letterHelp(self, alpha_dict: dict, cur_alpha: str, result_list, digits: str):
        if len(digits) == 0:
            result_list.append(cur_alpha)
            return None
        for alpha in alpha_dict.get(digits[0]):
            self.letterHelp(alpha_dict, cur_alpha + alpha, result_list, digits[1:])