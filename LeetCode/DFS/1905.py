
# 1905. 统计子岛屿
from typing import List


class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        row, col = len(grid1), len(grid1[0])
        counter = 0
        for i in range(row):
            for j in range(col):
                if grid2[i][j] == 1:
                    status = self.findIsland(grid1, grid2,i, j, row, col, True)
                    print("mark1")
                    if status:
                        counter += 1
        return counter 

    def findIsland(self, grid1, grid2, i, j, row, col, status):
        if grid2[i][j] == 1:
            grid2[i][j] = 0
            if grid1[i][j] == 0:
                # print("grid1:", i, j)
                status = status and False
        else:
            return status

        if i + 1 < row:
            status = self.findIsland(grid1, grid2, i+1, j, row, col, status)
        if i -1 >= 0:
            status = self.findIsland(grid1, grid2, i-1, j, row, col, status)
        if j + 1 < col:
            status = self.findIsland(grid1, grid2, i, j+1, row, col, status)
        if j - 1 >= 0:
            status = self.findIsland(grid1, grid2, i, j-1, row, col, status)
        return status

def testCase():
    grid1 = [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]]
    grid2 = [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]   
    sol = Solution()
    ret = sol.countSubIslands(grid1, grid2)
    print(ret)

def testCase1():
    grid1 = [[1,1,1,1,0,0],[1,1,0,1,0,0],[1,0,0,1,1,1],[1,1,1,0,0,1],
    [1,1,1,1,1,0],[1,0,1,0,1,0],[0,1,1,1,0,1],[1,0,0,0,1,1],[1,0,0,0,1,0],[1,1,1,1,1,0]]
    grid2 = [[1,1,1,1,0,1],[0,0,1,0,1,0],[1,1,1,1,1,1],[0,1,1,1,1,1],
    [1,1,1,0,1,0],[0,1,1,1,1,1],[1,1,0,1,1,1],[1,0,0,1,0,1],[1,1,1,1,1,1],[1,0,0,1,0,0]]
    sol = Solution()
    ret = sol.countSubIslands(grid1, grid2)
    print(ret)


if __name__ == "__main__":
    testCase1()
