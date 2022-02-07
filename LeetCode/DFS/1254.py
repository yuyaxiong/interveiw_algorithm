# 1254. 统计封闭岛屿的数目
from typing import List


class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        for j in range(col):
            if grid[0][j] == 0:
                self.dfs(grid, 0, j, row, col)
            if grid[row-1][j] == 0:
                self.dfs(grid, row-1, j, row, col)

        for i in range(row):
            if grid[i][0] ==  0:
                self.dfs(grid, i, 0, row, col)
            if grid[i][col-1] == 0:
                self.dfs(grid, i, col-1, row, col)

        counter = 0
        for i in range(1, row-1):
            for j in range(1, col-1):
                if grid[i][j] == 0:
                    self.dfs(grid, i, j, row, col)
                     counter += 1
        return counter 

    def dfs(self, grid, i, j, row, col):
        if grid[i][j] == 0:
            grid[i][j] = 1
        else:
            return 
        if i + 1 < row:
            self.dfs(grid, i+1, j, row, col)
        if i - 1 >= 0:
            self.dfs(grid, i-1, j, row, col)
        if j + 1 < col:
            self.dfs(grid, i, j+1, row, col)
        if j - 1 >= 0:
            self.dfs(grid, i, j-1, row, col)
        return