# 694.不同的岛屿数量
class Solution:
    def numDistinctIslands(self, grid):
        islands_dict = dict()
        row, col = len(grid), len(grid[0])
        for i in range(row):
            for j in range(col):
                if grid[i][j] == "1":
                    res = []
                    self.dfs(grid, i, j, res, 666, row, col)
                    islands_dict[",".join(res)] = 1
        return len(islands_dict)

    def dfs(self, grid, i, j, res, dir, row, col):
        if grid[i][j] == '1':
            grid[i][j] = '0'
            res.append(str(dir))
        else:
            return 
        if i + 1 < row:
            self.dfs(grid, i+1, j, res, 1, row, col)
        if i - 1 >= 0:
            self.dfs(grid, i-1, j, res, 2, row, col)
        if j + 1 < col:
            self.dfs(grid, i, j+1, res, 3, row, col)
        if j - 1 >= 0:
            self.dfs(grid, i, j-1, res, 4, row, col)
        return 

def testCase():
    grid = [["1", "1", "0", "0", "0"],  ["1", "1", "0", "0", "0"], ["0", "0", "0", "1", "1"], ["0", "0", "0", "1", "1"]]
    sol = Solution()
    ret = sol.numDistinctIslands(grid)

    
    print(ret)


if __name__ == "__main__":
    testCase()