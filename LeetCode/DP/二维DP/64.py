# 64. 最小路径和
from typing import List


class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        dp = [[0 for _ in range(col)] for _ in range(row)]
        for j in range(col):
            if j - 1 >= 0:
                dp[0][j] =  grid[0][j] + dp[0][j-1]
            else:
                dp[0][j] =  grid[0][j]

        for i in range(row):
            if i - 1 >= 0:
                dp[i][0] = grid[i][0] + dp[i-1][0]
            else:
                dp[i][0] = grid[i][0]

        for i in range(1, row):
            for j in range(1, col):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[row-1][col-1]