# 200. 岛屿数量
from typing import List


class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        row, col = len(grid), len(grid[0])
        counter = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == '1':
                    counter += 1
                    self.dfs(grid, i, j, row, col)
                    print(i, j, grid)
        return counter 

    def dfs(self, grid, i, j, row, col):
        if grid[i][j] == '1':
            grid[i][j] = '0'
        else:
            return 
        if i + 1 < row:
            self.dfs(grid, i+1, j, row, col)
        if i -1 >= 0:
            self.dfs(grid, i-1, j, row, col)
        if j + 1 < col:
            self.dfs(grid, i, j+1, row, col)
        if j - 1 >= 0:
            self.dfs(grid, i, j-1, row, col)
        return 

def testCase():
    sol = Solution()
    grid = [["1","0","1","1","0","1","1"]]
    ret = sol.numIslands(grid)
    print(ret)

if __name__ == "__main__":
    testCase()