# 931. 下降路径最小和
import sys
from typing import List
class Solution:
    def minFallingPathSum(self, matrix: List[List[int]]) -> int:
        row ,col = len(matrix), len(matrix[0])
        self.memo = [[float("inf") for _ in range(col)] for _ in range(row)]
        res_list = []
        for j in range(col):
            res = self.minPath(matrix, 0, j, row, col)
            res_list.append(res)
        return min(res_list)

    def minPath(self, matrix, i, j, row, col):
        if i == row - 1:
            self.memo[i][j] = matrix[i][j]
            return matrix[i][j]

        if self.memo[i][j] != float("inf"):
            return self.memo[i][j]
        res = float("INF")
        res = min(res, self.minPath(matrix, i+1, j, row, col))
        if j - 1 >= 0:
            res = min(res, self.minPath(matrix, i+1, j-1, row, col))
        if j + 1 < col:
            print(j)
            res = min(res, self.minPath(matrix, i+1, j+1, row, col))
        res += matrix[i][j]
        self.memo[i][j] = res
        return res


def testCase():
    matrix = [[17,82],[1,-44]]
    sol = Solution()
    ret = sol.minFallingPathSum(matrix)
    print(ret)

if __name__ == "__main__":
    testCase()
    