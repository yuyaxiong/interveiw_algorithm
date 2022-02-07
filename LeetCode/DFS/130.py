# 130. 被围绕的区域
from typing import List

class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if board is None or len(board) == 0:
            return 
        row, col = len(board), len(board[0])
        for i in range(row):
            self.dfs(board, i, 0)
            self.dfs(board, i, col - 1)
        for j in range(col):
            self.dfs(board, 0, j)
            self.dfs(board, row-1, j)
        for i in range(row):
            for j in range(col):
                if board[i][j] == 'O':
                    board[i][j] = 'X'
                if board[i][j] == '-':
                    board[i][j] = 'O'

    def dfs(self, board, i, j):
        if (i< 0 or j < 0 or i >= len(board) or j >= len(board[0]) or board[i][j] != "O"):
            return 
        board[i][j] = '-'
        self.dfs(board, i-1, j)
        self.dfs(board, i+1, j)
        self.dfs(board, i, j-1)
        self.dfs(board, i, j+1)
        return 