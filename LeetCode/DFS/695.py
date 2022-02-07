# 695. 岛屿的最大面积
from typing import List


class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        row, col = len(grid), len(grid[0])
        area_list = []
        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    area = self.dfs(grid, i, j, row, col, 0)
                    area_list.append(area)
        return max(area_list)

    def dfs(self, grid, i, j, row, col, area):
        if grid[i][j] == 1:
            grid[i][j] = 0
            area += 1
        else:
            return area
        if i + 1 < row:
            area = self.dfs(grid, i+1, j, row, col, area)
        if i - 1 >= 0:
            area = self.dfs(grid, i-1, j, row, col, area)
        if j + 1 < col:
            area = self.dfs(grid, i, j+1, row, col, area)
        if j - 1 >= 0:
            area = self.dfs(grid, i, j-1, row, col, area)
        return area


def testCase():
    grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
    sol = Solution()
    ret = sol.maxAreaOfIsland(grid)
    print(ret)

if __name__ == "__main__":
    testCase()