#   77. 组合

from typing import List


class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        result, pre_list = [], []
        self.combine_help(1, n+1, k, result, pre_list)
        return result

    def combine_help(self, i,j, k, result, pre_list):
        if k == 0:
            result.append(pre_list)
            return 
        for n in range(i, j):
            tmp_list = pre_list[::]
            tmp_list.append(n)
            self.combine_help(n+1, j, k-1 ,  result, tmp_list)
        return 




def testCase():
    n , k = 13, 13
    sol = Solution()
    ret = sol.combine(n, k)
    print(ret)

if __name__ == "__main__":
    testCase()

