#  51. N 皇后
from typing import List

class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        self.res = []
        board = [['.' for _ in range(n)] for _ in range(n)]
        self.backtrack(board, 0)
        return self.res


    def backtrack(self, board, row):
        if row == len(board):
            tmp_list = ["".join(ns) for ns in board]
            self.res.append(tmp_list)
            return 
        n = len(board[row])
        for col in range(n):
            if not self.isValid(board, row, col):
                continue
            board[row][col] = 'Q'
            self.backtrack(board, row + 1)
            board[row][col] = '.'

    def isValid(self, board, row, col):
        n = len(board)
        # 列检查
        for i in range(n):
            if board[i][col] == 'Q':
                return False
        # 右上方
        i, j = row -1 , col + 1 
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        # 左上方
        i , j = row-1, col-1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        return True




def testCase():
    n = 4
    sol = Solution()
    ret = sol.solveNQueens(n)
    # print(len(ret))
    # print(ret)
    # ret_list = [['.' for _ in range(n)] for _ in range(n)]
    # ret_list[0][0] = 'Q'
    # ret = sol.isValid(ret_list,1, 0)
    print(ret)

if __name__ == "__main__":
    testCase()